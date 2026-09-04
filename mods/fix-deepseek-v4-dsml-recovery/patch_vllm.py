#!/usr/bin/env python3
"""
Whole-file-replacement patch for vLLM DeepSeek-V4 DSML tool-call recovery.

Why this exists
----------------
DeepSeek-V4-Flash-0731 intermittently drops or corrupts its DSML tool-call
markup under long context. Direct review of ~90 captured failures shows
three distinct, real shapes (not overlapping, not test noise):

1. Dialect confusion / missing wrapper (~65% of failures): a complete
   invoke block -- either the fullwidth ``<｜DSML｜invoke ...>`` form or the
   plain-ASCII ``<invoke name="...">`` form (the same dialect Claude Code's
   own tool-call protocol uses) -- with the outer ``<｜DSML｜tool_calls>``
   wrapper missing or corrupted. Fully recoverable: tool name and arguments
   are intact.
2. Bare parameter fragments (no invoke opener at all): unrecoverable, no
   tool name to work with.
3. Orphan closing tail: stray closers with no real invoke content,
   appearing after a call already resolved normally: unrecoverable.

Unlike the three older `mods/fix-deepseek-v4-orphan-*` mods (anchor-based
`_require_replace` surgical patching across up to 5 files, each patch
fragile to any upstream formatting drift), this mod ships complete
replacement file contents under files/ and writes them out wholesale. This
supersedes and replaces all three -- do not run them together.

Files replaced under the target vllm/parser tree:
  - deepseek_v4.py                    (ASCII-dialect terminals, provisional
                                        recovery transitions, validator hook)
  - engine/parser_engine_config.py    (Transition recovery flags)
  - engine/streaming_parser_engine.py (provisional hold/commit/abort +
                                        orphan-tail bash regenerate)
  - engine/parser_engine.py           (verbatim args for the synthetic
                                        bash regenerate call)

Idempotent: each replacement file starts with a ``# __spark_dsml_recovery__``
marker line; if the installed file already has it, that file is skipped.
The first time a file is replaced, the pre-patch original is preserved next
to it as ``<name>.orig-spark`` (only if that backup does not already exist),
so the stock behavior can always be diffed or restored.
"""

from __future__ import annotations

import glob
import shutil
import sys
from pathlib import Path

_MARKER = "# __spark_dsml_recovery__"

_SITE = "/usr/local/lib/python3.12/dist-packages"
# A positional argument overrides the site-packages prefix -- used by
# run.sh (real PYTHON_ROOT) and by the "verify" mode below (a scratch
# tree containing a copy of vllm/) alike.
_args = [a for a in sys.argv[1:] if a != "verify"]
if _args:
    _SITE = _args[0]

VLLM_PARSER = f"{_SITE}/vllm/parser"

_MOD_DIR = Path(__file__).resolve().parent
_FILES_DIR = _MOD_DIR / "files"

# (source file under files/, destination relative to vllm/parser/)
_TARGETS = [
    ("deepseek_v4.py", "deepseek_v4.py"),
    ("parser_engine_config.py", "engine/parser_engine_config.py"),
    ("streaming_parser_engine.py", "engine/streaming_parser_engine.py"),
    ("parser_engine.py", "engine/parser_engine.py"),
]


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
    parser_dir = glob.glob(VLLM_PARSER)
    if not parser_dir:
        _abort(f"vllm/parser not found at {VLLM_PARSER}")
    parser_dir = Path(parser_dir[0])

    ok = True
    for src_name, rel_dst in _TARGETS:
        src = _FILES_DIR / src_name
        dst = parser_dir / rel_dst
        try:
            _replace_file(src, dst)
        except SystemExit:
            ok = False

    if not ok:
        print("WARNING: some files could not be patched.", file=sys.stderr)
        sys.exit(1)
    print("Patch complete.")


if __name__ == "__main__":
    main()
