#!/usr/bin/env python3
"""
Inline patch for vLLM DeepSeek-V4: absorb orphan DSML closing tails.

Why this exists
---------------
DeepSeek-V4-Flash-0731 intermittently emits a turn whose content is *only* an
orphan DSML closing tail -- stray </parameter>, </invoke>, </tool_calls> markers
with no opening wrapper and no real tool call, appearing after a tool call has
already succeeded normally (vllm-project/vllm#51914, comment #5354570612).
``deepseek_v4.py`` defines no transition for PARAM_CLOSE / INVOKE_END / TOOL_END
from ParserState.CONTENT, so StreamingParserEngine._on_terminal falls through to
_emit_for_state and the closers leak into assistant content as TEXT_CHUNK. The
client (an agent harness) reads that turn as its final JSON answer and fails,
interrupting the agent.

The state machine already proves "orphaning": every legitimate wrapper closes
while still in a *tool state*, never in CONTENT. So a PARAM_CLOSE / INVOKE_END /
TOOL_END reaching _on_terminal while ``self.state == ParserState.CONTENT`` is
definitionally orphaned -- no per-turn bookkeeping is required.

Fix (two rules in _on_terminal, active only in CONTENT):
  * On INVOKE_END / TOOL_END: emit a synthetic ``bash`` tool call whose args are
    ``{"command": "echo \\"...\\""}`` telling the model that its latest tool call
    was dropped (orphan DSML closing tail) and asking it to re-issue the tool
    call. The agent executes it, the echo output feeds back into the turn, and
    the model regenerates a properly-wrapped call; the residual closers are
    never emitted as text.
  * On a lone PARAM_CLOSE: silently absorb it (leak-safety without a spurious
    regen / false-positive bash call).

A small special case in parser_engine.ParserEngine._handle_tool_end makes the
``bash`` call carry the exact JSON args verbatim (DeepSeek's *dsml_arg_converter
would otherwise strip them to ``{}``). The synthetic call is gated by
``find_tool_name(self._tools, "bash")`` so if ``bash`` is not declared the
residual is still dropped but no synthetic call is emitted.

Idempotent (marker ``__spark_orphan_tail__``), safe to re-run, tolerates the raw
engine, the light orphan-invoke mod, or the full orphan-invoke-full mod (guards on
``_recovery_outer_closer_pending`` so it never regens on the full mod's legitimate
recovered-call boundary closer).
"""

from __future__ import annotations

import glob
import sys

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SITE = "/usr/local/lib/python3.12/dist-packages"
VLLM_PARSER = f"{_SITE}/vllm/parser"

if len(sys.argv) > 1 and sys.argv[1] == "verify":
    # Called with a target prefix under which a copy of vllm/ sits.
    if len(sys.argv) > 2:
        _SITE = sys.argv[2]
        VLLM_PARSER = f"{_SITE}/vllm/parser"


def _read(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def _write(path: str, content: str) -> None:
    with open(path, "w") as f:
        f.write(content)


def _abort(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def _require_replace(
    content: str, anchor: str, replacement: str, what: str
) -> str:
    if anchor not in content:
        _abort(f"could not find anchor for {what}")
    if content.count(anchor) != 1:
        _abort(f"anchor for {what} is not unique ({content.count(anchor)} hits)")
    return content.replace(anchor, replacement)


_TAIL = "__spark_orphan_tail__"

# Strong tail anchors: a real turn-leak always includes the deeper closers.
_CLOSERS = "('INVOKE_END', 'TOOL_END')"

# The synthetic bash command the agent executes when an orphan DSML closing tail
# is detected. Its stdout is the tool result fed back to the model, so the text
# doubles as the prompt that asks the model to re-issue the last tool call.
_REGEN_MESSAGE = (
    "The last tool call was dropped and needs to be re-issued."
    " Please retry the previous tool call now."
)

# ---------------------------------------------------------------------------
# 1. vllm/parser/engine/streaming_parser_engine.py
# ---------------------------------------------------------------------------
def patch_streaming_engine(path: str) -> None:
    content = _read(path)
    if _TAIL in content:
        print(f"  [skipped] {path} (already patched)")
        return

    # 1a. Ensure `json` and `find_tool_name` are imported (needed by
    #     _spark_regenerate_tail and the regen gate).
    if not any(
        line.strip() == "import json" for line in content.splitlines()
    ):
        anchor_import = (
            "from __future__ import annotations\n"
        )
        content = _require_replace(
            content,
            anchor_import,
            "from __future__ import annotations\n"
            "import json\n",
            "streaming engine json import",
        )

    # 1b. Module constants, inserted immediately before the class declaration.
    anchor_class = (
        "class StreamingParserEngine:\n"
    )
    repl_class = (
        "# Absorb an orphan DeepSeek V4 DSML closing tail in CONTENT and emit a\n"
        "# synthetic bash regenerate tool call (see mods/fix-deepseek-v4-orphan-tail).\n"
        f"_SPARK_ORPHAN_TAIL_CLOSERS = {_CLOSERS}\n"
        f"_REGEN_TAIL_MESSAGE = {_REGEN_MESSAGE!r}\n"
        "\n"
        "class StreamingParserEngine:\n"
    )
    content = _require_replace(
        content, anchor_class, repl_class, "streaming engine class anchor"
    )

    # 1c. _on_terminal head: prepend the regenerate helper + CONTENT-closer guards.
    #     Anchor on the single unique method header line so this works on BOTH the
    #     raw/light shape (body starts with `key = ...`) and the full-mod shape
    #     (body starts with the recovery `if` block). Whatever follows is left
    #     intact; the `_recovery_outer_closer_pending` getattr guard lets a
    #     recovered-call boundary closer fall through to the full mod's absorb.
    anchor_onterm = (
        "    def _on_terminal(self, terminal: str, value: str) -> list[SemanticEvent]:\n"
    )
    repl_onterm = (
        "    def _spark_regenerate_tail(self) -> list[SemanticEvent]:\n"
        "        idx = self.tool_index + 1\n"
        "        self.tool_index = idx\n"
        "        args = json.dumps({'command': _REGEN_TAIL_MESSAGE})\n"
        "        return [\n"
        "            SemanticEvent(EventType.TOOL_CALL_START, tool_index=idx),\n"
        "            SemanticEvent(\n"
        "                EventType.ARG_VALUE_CHUNK,\n"
        "                value=args,\n"
        "                tool_index=idx,\n"
        "            ),\n"
        "            SemanticEvent(EventType.TOOL_NAME, value=\"bash\", tool_index=idx),\n"
        "            SemanticEvent(EventType.TOOL_CALL_END, tool_index=idx),\n"
        "        ]\n"
        "\n"
        "    def _on_terminal(self, terminal: str, value: str) -> list[SemanticEvent]:\n"
        "        if (\n"
        "            not getattr(self, \"_recovery_outer_closer_pending\", False)\n"
        "            and self.state == ParserState.CONTENT\n"
        "            and terminal in _SPARK_ORPHAN_TAIL_CLOSERS\n"
        "            and not getattr(self, \"_tail_regen_done\", False)\n"
        "            and not getattr(self, \"_tail_content_since_reset\", False)\n"
        "        ):\n"
        "            self._tail_regen_done = True\n"
        "            return self._spark_regenerate_tail()\n"
        "        if (\n"
        "            not getattr(self, \"_recovery_outer_closer_pending\", False)\n"
        "            and self.state == ParserState.CONTENT\n"
        "            and (terminal in _SPARK_ORPHAN_TAIL_CLOSERS\n"
        "                 or terminal == \"PARAM_CLOSE\")\n"
        "        ):\n"
        "            return []  # absorb stray closer text (no regen: prose/bad tools)\n"
    )
    content = _require_replace(
        content, anchor_onterm, repl_onterm, "streaming engine _on_terminal head"
    )

    # 1d. _on_content: mark that real (non-whitespace) text was emitted this turn,
    #     which disables the orphan-tail regen (prose quoting closers is not an
    #     orphan tail). Whitespace-only deltas do not set the flag, so a turn whose
    #     only "content" is stray closers + newlines still regens.
    anchor_on_content = (
        "    def _on_content(self, text: str) -> list[SemanticEvent]:\n"
        "        if not text:\n"
        "            return []\n"
    )
    repl_on_content = (
        "    def _on_content(self, text: str) -> list[SemanticEvent]:\n"
        "        if not text:\n"
        "            return []\n"
        "        if text.strip():\n"
        "            self._tail_content_since_reset = True\n"
    )
    content = _require_replace(
        content, anchor_on_content, repl_on_content, "streaming engine _on_content flag"
    )

    # 1e. reset(): per-request flags.
    anchor_reset = (
        "        self._in_skipped_tool_span = False\n"
        "        self._reset_args_state()\n"
    )
    repl_reset = (
        "        self._in_skipped_tool_span = False\n"
        "        self._reset_args_state()\n"
        "        self._tail_regen_done = False\n"
        "        self._tail_content_since_reset = False\n"
    )
    content = _require_replace(
        content, anchor_reset, repl_reset, "streaming engine reset flags"
    )

    content = content + f"\n# {_TAIL}\n"
    _write(path, content)
    print(f"  patched {path}")


# ---------------------------------------------------------------------------
# 2. vllm/parser/engine/parser_engine.py
# ---------------------------------------------------------------------------
def patch_parser_engine(path: str) -> None:
    content = _read(path)
    if _TAIL in content:
        print(f"  [skipped] {path} (already patched)")
        return

    # 2a. Exact-args bash special case in _handle_tool_end, before the args flush.
    #     Only the *synthetic* regenerate (args is a complete JSON dict) is emitted
    #     verbatim; a *real* bash call (args is DSML <parameter> markup) falls
    #     through to the normal converter flush so it still yields real JSON args.
    anchor_end = (
        "        idx = event.tool_index\n"
        "        if idx >= len(self._tool_slots):\n"
        "            return\n"
        "\n"
        "        remaining = self._flush_arg_converter(idx)\n"
    )
    repl_end = (
        "        idx = event.tool_index\n"
        "        if idx >= len(self._tool_slots):\n"
        "            return\n"
        "\n"
        "        slot_before_flush = self._tool_slots[idx]\n"
        "        if slot_before_flush.name == \"bash\":\n"
        "            _spark_args_json = slot_before_flush.args.strip()\n"
        "            _spark_is_synthetic = bool(\n"
        "                _spark_args_json.startswith(\"{\")\n"
        "                and getattr(self, \"_tail_regenerate_eligible\", True)\n"
        "                and find_tool_name(self._tools, \"bash\")\n"
        "            )\n"
        "            if _spark_is_synthetic:\n"
        "                # Synthetic orphan-tail regenerate: the args are already the\n"
        "                # exact JSON dictated by this mod, so emit them verbatim -- the\n"
        "                # DeepSeek arg converter would otherwise strip bare JSON to {}.\n"
        "                self._ensure_tool_id(slot_before_flush, \"bash\")\n"
        "                slot_before_flush.name_sent = True\n"
        "                deltas.append(\n"
        "                    DeltaToolCall(\n"
        "                        index=idx,\n"
        "                        id=slot_before_flush.id,\n"
        "                        type=\"function\",\n"
        "                        function=DeltaFunctionCall(\n"
        "                            name=\"bash\",\n"
        "                            arguments=slot_before_flush.args,\n"
        "                        ),\n"
        "                    )\n"
        "                )\n"
        "                return\n"
        "            elif not getattr(self, \"_tail_regenerate_eligible\", True):\n"
        "                # bash not declared on the request (or tool_choice=none): the\n"
        "                # residual was already absorbed by the streaming engine; drop\n"
        "                # the call so it never reaches the generic commit path (where\n"
        "                # an empty _tools would let _is_valid_tool_name accept any\n"
        "                # name and emit a spurious {}).\n"
        "                return\n"
        "            # else: a REAL bash call with DSML markup args -- fall through to\n"
        "            # the normal converter flush below so it still produces real JSON.\n"
        "\n"
        "        remaining = self._flush_arg_converter(idx)\n"
    )
    content = _require_replace(
        content, anchor_end, repl_end, "parser engine _handle_tool_end bash special case"
    )

    # 2b. Record regen eligibility from the REQUEST (not the possibly-stale
    #     `self._tools`): the synthetic bash re-issue must not fire when the
    #     request declares no tools or tool_choice=none. Runs in the base
    #     _check_skip_tool_parsing, which the full mod's override calls via
    #     super() -- so this stays correct on raw/light/full hosts alike.
    anchor_check = (
        "        tools = getattr(request, \"tools\", None)\n"
        "        if tools:\n"
        "            self._tools = tools\n"
    )
    repl_check = (
        "        tools = getattr(request, \"tools\", None)\n"
        "        if tools:\n"
        "            self._tools = tools\n"
        "        self._tail_regenerate_eligible = bool(\n"
        "            tools and find_tool_name(tools, \"bash\")\n"
        "            and getattr(request, \"tool_choice\", None) != \"none\"\n"
        "        )\n"
    )
    content = _require_replace(
        content, anchor_check, repl_check, "parser engine _check_skip_tool_parsing regen eligibility"
    )

    content = content + f"\n# {_TAIL}\n"
    _write(path, content)
    print(f"  patched {path}")


# ---------------------------------------------------------------------------
def main() -> None:
    files = [
        f"{VLLM_PARSER}/engine/streaming_parser_engine.py",
        f"{VLLM_PARSER}/engine/parser_engine.py",
    ]
    ok = True
    for fp in files:
        p = glob.glob(fp)
        if not p:
            print(f"  MISSING {fp}", file=sys.stderr)
            ok = False
            continue
        patch_fns = {
            "streaming_parser_engine.py": patch_streaming_engine,
            "parser_engine.py": patch_parser_engine,
        }
        name = fp.split("/")[-1]
        try:
            patch_fns[name](p[0])
        except SystemExit:
            ok = False
    if not ok:
        print("WARNING: some files could not be patched.", file=sys.stderr)
    print("Patch complete.")


if __name__ == "__main__":
    main()
