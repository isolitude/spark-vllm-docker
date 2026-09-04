# mods/fix-deepseek-v4-orphan-tail

Absorb an **orphan DSML closing tail** and emit a synthetic `bash` tool call whose
stdout prompts the model to re-issue the dropped tool call, so an agent keeps
working.

## Problem

DeepSeek-V4-Flash-0731 intermittently emits a turn whose content is *only* an
orphan closing tail — stray `</parameter>`, `</invoke>`, `</tool_calls>` with no
opening marker and no real tool call. This appears *after* a tool call has already
succeeded normally (vllm-project/vllm#51914, comment #5354570612).

The parser has no transition for `PARAM_CLOSE` / `INVOKE_END` / `TOOL_END` from
`CONTENT`, so the closers fall through and are emitted as `TEXT_CHUNK` into
assistant `content`. The client (an agent harness) reads that turn as its final JSON
answer and fails with `Expecting value: line 1 column 1`, interrupting agent work.
The residual DSML also contaminates context and feeds a self-reinforcing leak.

## Fix

Two rules in `vllm/parser/engine/streaming_parser_engine.py::_on_terminal`, active
only when `self.state == ParserState.CONTENT` (which the state machine already
proves is "no wrapper open this turn"):

- On `INVOKE_END` / `TOOL_END` in CONTENT: emit a synthetic `bash` tool call whose
  `command` prints a message telling the model its last tool call was dropped and
  to re-issue it, and never emit the closer text.
- On a lone `PARAM_CLOSE` in CONTENT: silently absorb it (no spurious regen; keeps
  the leak-safety without a false-positive `bash` call).

A small special case in `parser_engine.py::_handle_tool_end` makes the `bash` call
carry the exact JSON args verbatim (DeepSeek's `_dsml_arg_converter` would otherwise
strip them to `{}`). The special case uses a **discriminator** so it cannot hijack a
real bash call:

- **Synthetic** (args is a complete JSON dict, starts with `{`): emitted verbatim as a
  `DeltaToolCall`, then `return` — never falls through to the arg-converter flush, so the
  output is exactly one clean JSON object with **no trailing `{}`** (double-emit guard
  asserted in `test_tail.py` case A).
- **Real** bash call (args is DSML `<parameter>` markup, does *not* start with `{`):
  falls through to the normal `_flush_arg_converter`, byte-identical to the unpatched
  engine (`['bash', '{}']`).
- **Ineligible** request (`bash` not declared, or `tool_choice=none`, recorded from the
  *request* in the base `_check_skip_tool_parsing` as `_tail_regenerate_eligible`): drop,
  no spurious call.

The synthetic call is gated by `find_tool_name(tools, "bash")`, so if `bash` is not
declared the residual is still dropped but no synthetic call is emitted.

## Design notes / constraints

- **Narrow detector**: only stray closers in `CONTENT`. The state machine already
  consumes every legitimate wrapper inside a tool state, so a closer reaching CONTENT
  is definitionally orphaned — no per-turn bookkeeping required.
- **V3.2 `function_calls` byte-for-byte**: the foreign wrapper is routed through the
  `FOREIGN_BLOCK` states (in the full mod) or is not one of the three target terminals
  (raw/light), so this patch never rewrites it.
- **Ordering**: run this mod **after** `fix-deepseek-v4-orphan-invoke` and
  `fix-deepseek-v4-orphan-invoke-full` (the patch inserts before the transition lookup
  in `_on_terminal`, which the full mod rewrites). The `_recovery_outer_closer_pending`
  `getattr` guard ensures it never regens on the full mod's legitimate recovered-call
  boundary closer.

## Deployment

Mods run inside the container at launch. Reapply mods and rolling-restart both nodes
for the change to take effect (the running container loads the old `.py` at process
start).

## Validation

Offline harness (`test_tail.py`, run against a `patch_vllm.py verify`-patched copy of
`vllm/parser/`). All 4 cases pass:

- **A**: orphan closing tail -> exactly one synthetic bash re-issue call whose `command`
  is the clean re-issue prompt (JSON round-trips, no trailing `{}` — regression guard for
  the double-emit fix).
- **B**: prose merely quoting the closers -> absorbed, no regen.
- **C**: `bash` not declared -> residual absorbed, no synthetic call.
- **D**: `tool_choice=none` request -> no crash, no synthetic call.

Real-call parity is verified separately against the unpatched site-packages engine in
the container: a real bash call yields `['bash', '{}']` on both — byte-identical.

`patch_vllm.py` is idempotent (marker `__spark_orphan_tail__`) and supports a `verify`
target-prefix mode for offline runs against a copy of `vllm/parser/`.
