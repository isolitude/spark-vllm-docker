# fix-deepseek-v4-dsml-recovery

Consolidated DeepSeek-V4-Flash-0731 DSML tool-call recovery. **Supersedes**
`mods/fix-deepseek-v4-orphan-invoke`, `fix-deepseek-v4-orphan-invoke-full`, and
`fix-deepseek-v4-orphan-tail` — do not run this alongside any of them.

## Why

Direct review of ~90 captured live failures (`analyze-dsml-calls.py --dump-raw`
against DeepSeek-V4-Flash-0731 traffic) shows three distinct, real failure
shapes:

1. **Dialect confusion / missing wrapper** (~65% of failures): a complete
   invoke block is emitted, either in the fullwidth DSML dialect
   (`<｜DSML｜invoke name="...">`) or the plain-ASCII dialect
   (`<invoke name="...">` — the same tag shape Claude Code's own tool-call
   protocol uses), but the outer `<｜DSML｜tool_calls>` wrapper is missing or
   corrupted. **Fully recoverable**: tool name and arguments are intact.
2. **Bare parameter fragments**: `<｜DSML｜parameter>` / `<parameter>`
   elements with no invoke opener at all. Unrecoverable — no tool name to
   work with.
3. **Orphan closing tail**: stray closers (`</invoke>`, `</tool_calls>`,
   `</parameter>`) landing in plain content after a call has already
   resolved normally, no real content. Unrecoverable.

The three older mods only recognize the fullwidth `INVOKE_PREFIX` terminal,
so they miss the ASCII dialect and both unrecoverable shapes entirely — most
real failures went uncovered. They also patch by anchor-based string
replacement (`_require_replace`) across up to 5 files, which breaks outright
on any upstream formatting drift.

## What this mod does

Ships **complete replacement files** under `files/` and writes them out
wholesale (no anchor matching) to:

- `vllm/parser/deepseek_v4.py`
- `vllm/parser/engine/parser_engine_config.py`
- `vllm/parser/engine/streaming_parser_engine.py`
- `vllm/parser/engine/parser_engine.py`

**Recoverable case (shape 1):** a provisional-hold mechanism (ported from
the design proven in `fix-deepseek-v4-orphan-invoke-full` / upstream vLLM PR
#52645) buffers the candidate invoke — from either dialect — and validates
the completed tool name against `request.tools`. Valid → commits as a real
tool call. Invalid, or the held name looks nothing like a real name (too
long / contains a newline) → rolls back, replaying the buffered text as
ordinary content. Nothing is silently dropped either way.

**Unrecoverable cases (shapes 2 and 3):** a bare parameter fragment or a
stray closer with no invoke content is absorbed (never leaked into content)
and a synthetic `bash` tool call is emitted once per turn:
`{"command": "The last tool call was dropped and needs to be re-issued. Please retry the previous tool call now."}`.
Gated on `find_tool_name(request.tools, "bash")`; if `bash` isn't declared
the residual is still absorbed but no call is emitted. The prompt text
**never mentions "DSML"** — doing so was observed to increase the frequency
of this exact failure mode.

Also fixes a related corpus finding: the ASCII dialect commonly omits the
`string="true|false"` attribute on `<parameter>` (unlike the fullwidth
dialect, which always carries it); the arg converter now treats a missing
attribute as an implicit string value instead of silently dropping the
parameter.

## Idempotency / safety

Each replacement file starts with a `# __spark_dsml_recovery__` marker; if
the installed file already has it, `run.sh` skips that file. The first time
a file is replaced, the original is preserved as `<name>.orig-spark` next to
it for diffing/rollback.

## Testing

Verified against `eugr/spark-vllm-b12x:latest`:

```
docker run -d --name dsml-test --entrypoint sleep eugr/spark-vllm-b12x:latest infinity
docker cp mods/fix-deepseek-v4-dsml-recovery dsml-test:/tmp/mod
docker exec dsml-test /tmp/mod/run.sh
docker cp mods/fix-deepseek-v4-dsml-recovery/test_dsml_recovery.py dsml-test:/tmp/
docker exec dsml-test python3 /tmp/test_dsml_recovery.py
```

`test_dsml_recovery.py` (9 cases): fullwidth + ASCII orphan-invoke recovery,
invalid-name rollback, prose false-positive guard, DSML + ASCII orphan
closing tail regenerate, recovered-call-then-leftover-closer (no double
regen), undeclared-bash guard, bare-parameter-fragment regenerate. All pass.

Also replayed all 96 captured real failure samples end-to-end through the
patched parser: 31/31 orphan-invoke samples recovered as real tool calls,
39/42 orphan-parameter and 17/17 dropped-stray-closer samples correctly
triggered the bash regenerate fallback. The remaining 3 orphan-parameter
samples were truncated mid-generation with no closing terminal at all (no
detectable signal — not a mod defect).

`patch_vllm.py verify <prefix>` supports dry-run verification against a
scratch copy of the `vllm/` tree instead of the real site-packages.
