#!/usr/bin/env python3
"""
Offline unit test for mods/fix-deepseek-v4-trailing-system-genprompt.

Imports the patched files/deepseek_v4_encoding.py directly (no vllm install,
no container needed) and checks that every prompt ends with the
"<｜Assistant｜>" generation-prompt marker, regardless of what role the last
input message has.

Run:
    python3 mods/fix-deepseek-v4-trailing-system-genprompt/test_trailing_system_genprompt.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_MOD_DIR = Path(__file__).resolve().parent
_ENCODING_PATH = _MOD_DIR / "files" / "deepseek_v4_encoding.py"

_spec = importlib.util.spec_from_file_location("_dsv4_encoding_patched", _ENCODING_PATH)
_encoding = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_encoding)

encode_messages = _encoding.encode_messages
ASSISTANT_SP_TOKEN = _encoding.ASSISTANT_SP_TOKEN

FAILURES: list[str] = []


def check(name: str, messages: list[dict], expect_suffix: bool = True) -> None:
    prompt = encode_messages(
        messages, thinking_mode="thinking", add_default_bos_token=False, reasoning_effort="high"
    )
    has_suffix = ASSISTANT_SP_TOKEN in prompt[-len(ASSISTANT_SP_TOKEN) - 20 :]
    ok = has_suffix == expect_suffix
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: suffix_present={has_suffix} (expected {expect_suffix})")
    if not ok:
        FAILURES.append(name)
        print(f"       tail: {prompt[-120:]!r}")


# --- The bug: request ends on a system message ---------------------------
check(
    "trailing system, single turn",
    [
        {"role": "user", "content": "question"},
        {"role": "system", "content": "<total_tokens>100 tokens left</total_tokens>"},
    ],
)

check(
    "trailing system, after a completed assistant turn (the high-repro-rate shape)",
    [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "real question"},
        {"role": "system", "content": "<total_tokens>100 tokens left</total_tokens>"},
    ],
)

check(
    "trailing developer message",
    [
        {"role": "user", "content": "question"},
        {"role": "developer", "content": "extra instructions"},
    ],
)

# --- Control: request ends on a user message (never broken) --------------
check(
    "trailing user (control, always worked)",
    [{"role": "user", "content": "question"}],
)

# --- Control: request ends on an assistant message (continuation) --------
# Must NOT get a generation-prompt suffix -- that would corrupt continuation.
check(
    "trailing assistant, wo_eos (continuation, must stay unsuffixed)",
    [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "partial answer", "wo_eos": True},
    ],
    expect_suffix=False,
)

if FAILURES:
    print(f"\n{len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)

print("\nAll cases passed.")
