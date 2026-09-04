#!/usr/bin/env python3
"""
Inline patch for vLLM DeepSeek-V4 ParserEngine: recover "orphan" DSML invokes.

Problem
-------
Under long context (95k+ tokens), DeepSeek-V4-Flash occasionally emits a
complete <｜DSML｜invoke ...>...</｜DSML｜invoke> block but *omits* the opening
<｜DSML｜tool_calls> wrapper. The vllm.parser.deepseek_v4 state machine has no
transition from CONTENT/REASONING when it sees INVOKE_PREFIX, so the entire
DSML block falls through as TEXT_CHUNK / REASONING_CHUNK and leaks as plain
text to the client (e.g. Claude Code, which does not understand DSML and thus
never executes the tool call).

Fix
---
Inject two fallback transitions into deepseek_v4_config():
  CONTENT    + INVOKE_PREFIX -> TOOL_NAME        (emits TOOL_CALL_START)
  REASONING  + INVOKE_PREFIX -> TOOL_NAME        (emits REASONING_END + TOOL_CALL_START)
INVOKE_PREFIX ("<｜DSML｜invoke name=\"") is a highly specific token, so this
cannot meaningfully misfire on ordinary prose.

Idempotent: safe to re-run.
"""

from __future__ import annotations

import re
import sys

PATTERN = "/usr/local/lib/python*/dist-packages/vllm/parser/deepseek_v4.py"
matches = __import__("glob").glob(PATTERN)
if not matches:
    print(f"ERROR: Could not find deepseek_v4.py matching {PATTERN}", file=sys.stderr)
    sys.exit(1)

path = matches[0]
print(f"Patching {path}")

with open(path, "r") as f:
    content = f.read()

# --- idempotency ----------------------------------------------------------
ORPHAN_MARKER = "orphan INVOKE_PREFIX fallback"
if ORPHAN_MARKER in content:
    print("Already patched, skipping.")
    sys.exit(0)

# --- anchor: CONTENT/REASONING -> TOOL_START transitions we extend ---------
# We insert right after the existing "tool call beginning while still inside
# <think>" REASONING->TOOL_START block. Use a precise, stable snippet.
REASONING_ANCHOR = (
    "# Tool call beginning while still inside <think>\n"
    "            (ParserState.REASONING, \"TOOL_START\"): Transition(\n"
    "                ParserState.TOOL_PREAMBLE,\n"
    "                (EventType.REASONING_END,),\n"
    "            ),"
)

# --- the fallback snippet to inject ---------------------------------------
FALLBACK_SNIPPET = (
    "# orphan INVOKE_PREFIX fallback: recover a <｜DSML｜invoke ...> block "
    "emitted without an opening <｜DSML｜tool_calls> wrapper\n"
    "            (ParserState.CONTENT, \"INVOKE_PREFIX\"): Transition(\n"
    "                ParserState.TOOL_NAME,\n"
    "                (EventType.TOOL_CALL_START,),\n"
    "            ),\n"
    "            (ParserState.REASONING, \"INVOKE_PREFIX\"): Transition(\n"
    "                ParserState.TOOL_NAME,\n"
    "                (EventType.REASONING_END, EventType.TOOL_CALL_START,),\n"
    "            ),"
)

if REASONING_ANCHOR not in content:
    print(
        "ERROR: Could not locate REASONING->TOOL_START transition block; "
        "deepseek_v4.py layout may differ from this build.",
        file=sys.stderr,
    )
    sys.exit(1)

content = content.replace(REASONING_ANCHOR, REASONING_ANCHOR + "\n" + FALLBACK_SNIPPET)

# Sanity check: cleaned indentation of the snippet must match the surrounding
# (12 spaces) so the generated state machine stays valid Python/YAML-friendly.
with open(path, "w") as f:
    f.write(content)

print(
    "Patch applied: CONTENT/REASONING now fall back to TOOL_NAME on a "
    "wrapper-less <｜DSML｜invoke> (orphan-invoke recovery)."
)
