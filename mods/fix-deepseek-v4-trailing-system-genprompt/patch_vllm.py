#!/usr/bin/env python3
"""
Whole-file-replacement patch for vLLM's DeepSeek-V4 prompt encoding.

Why this exists
----------------
When a chat-completions request's LAST message has role=system (e.g. Claude
Code's "<total_tokens>N tokens left</total_tokens>" low-context-budget
reminder, injected as the final message of a turn), the stock
`render_message()` in vllm/tokenizers/deepseek_v4_encoding.py never appends
the "<｜Assistant｜><think>" generation-prompt suffix -- that branch only
fires for role in ("user", "developer"). The rendered prompt ends on raw
system-message text with no generation-prompt suffix, and the model responds
by emitting EOS as its very first token: a deterministic empty-output bug,
confirmed by direct encode_messages() comparison and by live replay
(~77% empty rate on affected traffic shapes, 0% after this fix). This was
initially mistaken for a Claude-Code-version-correlated issue; it isn't --
it reproduces for any client whose request happens to end on a system
message, independent of client version.

See mods/fix-deepseek-v4-trailing-system-genprompt/README.md for the full
investigation writeup.

Idempotent: the replacement file starts with a
``# __spark_trailing_system_genprompt__`` marker line; if the installed file
already has it, the file is skipped. The first time it's replaced, the
pre-patch original is preserved next to it as ``<name>.orig-spark`` (only if
that backup does not already exist), so stock behavior can always be diffed
or restored.
"""

from __future__ import annotations

import glob
import shutil
import sys
from pathlib import Path

_MARKER = "# __spark_trailing_system_genprompt__"

_SITE = "/usr/local/lib/python3.12/dist-packages"
# A positional argument overrides the site-packages prefix -- used by
# run.sh (real PYTHON_ROOT) and by the "verify" mode below (a scratch
# tree containing a copy of vllm/) alike.
_args = [a for a in sys.argv[1:] if a != "verify"]
if _args:
    _SITE = _args[0]

VLLM_TOKENIZERS = f"{_SITE}/vllm/tokenizers"

_MOD_DIR = Path(__file__).resolve().parent
_FILES_DIR = _MOD_DIR / "files"

_TARGET_SRC = "deepseek_v4_encoding.py"
_TARGET_DST = "deepseek_v4_encoding.py"


def _abort(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def _replace_file(src: Path, dst: Path) -> None:
    if not dst.exists():
        _abort(f"target file does not exist: {dst}")

    existing = dst.read_text()
    if _MARKER in existing:
        print(f"  [skipped] {dst} (already patched)")
        return

    backup = dst.with_suffix(dst.suffix + ".orig-spark")
    if not backup.exists():
        shutil.copy2(dst, backup)
        print(f"  [backup] {dst} -> {backup}")

    new_content = src.read_text()
    if _MARKER not in new_content:
        _abort(f"replacement source missing idempotency marker: {src}")

    dst.write_text(new_content)
    print(f"  patched {dst}")


def main() -> None:
    tokenizers_dir = glob.glob(VLLM_TOKENIZERS)
    if not tokenizers_dir:
        _abort(f"vllm/tokenizers not found at {VLLM_TOKENIZERS}")
    tokenizers_dir = Path(tokenizers_dir[0])

    src = _FILES_DIR / _TARGET_SRC
    dst = tokenizers_dir / _TARGET_DST
    try:
        _replace_file(src, dst)
    except SystemExit:
        print("WARNING: file could not be patched.", file=sys.stderr)
        sys.exit(1)

    print("Patch complete.")


if __name__ == "__main__":
    main()
