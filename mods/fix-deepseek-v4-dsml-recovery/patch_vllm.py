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
replacement file contents and writes them out wholesale. This supersedes
and replaces all three -- do not run them together.

Versioned file sets
--------------------
Upstream vLLM refactored the parser engine between image generations:
``:0823`` and earlier ship the pre-refactor engine, ``:0904`` (and later)
ship a ``token_count`` / ``reasoning_token_count`` refactor in the two
engine files. The recovery logic is orthogonal to that refactor, so each
version gets its own replacement set:

  - files-0823/   -- for images whose engine lacks the 0904 token_count
                    refactor (the original mod content, byte-identical to
                    the legacy files/ directory).
  - files-0904/   -- for images whose engine has the 0904 refactor: keeps
                    every token_count/reasoning_token_count hook while
                    adding the same recovery branches.

The version is detected automatically from the installed
``engine/streaming_parser_engine.py``: the 0904 refactor is present iff
that file references ``_reasoning_token_count`` / ``reasoning_token_count``.
Detection can be bypassed for unknown images with ``--files-dir``.

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

# Versioned replacement sets, newest first. Each value's first element is the
# canonical directory; legacy "files" is kept as an alias of 0823.
_FILES_DIRS = {
    "0904": _MOD_DIR / "files-0904",
    "0823": _MOD_DIR / "files-0823",
}
_LEGACY_DIR = _MOD_DIR / "files"


def _detect_version(parser_dir: Path) -> str:
    """Detect which upstream parser-engine generation is installed."""
    spse = parser_dir / "engine" / "streaming_parser_engine.py"
    if not spse.exists():
        _abort(f"cannot find {spse}")
    text = spse.read_text()
    # 0904+ refactor: token_count plumbing + reasoning_token_count property.
    if "_reasoning_token_count" in text or "reasoning_token_count" in text:
        return "0904"
    return "0823"


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


# (source file, destination relative to vllm/parser/)
_TARGETS = [
    ("deepseek_v4.py", "deepseek_v4.py"),
    ("parser_engine_config.py", "engine/parser_engine_config.py"),
    ("streaming_parser_engine.py", "engine/streaming_parser_engine.py"),
    ("parser_engine.py", "engine/parser_engine.py"),
]


def main() -> None:
    parser_dir = glob.glob(VLLM_PARSER)
    if not parser_dir:
        _abort(f"vllm/parser not found at {VLLM_PARSER}")
    parser_dir = Path(parser_dir[0])

    # Version selection (overrideable with --files-dir).
    files_dir: Path | None = None
    version: str | None = None
    if "--files-dir" in sys.argv:
        idx = sys.argv.index("--files-dir")
        if idx + 1 >= len(sys.argv):
            _abort("--files-dir requires a directory argument")
        files_dir = Path(sys.argv[idx + 1])
        if not files_dir.is_dir():
            _abort(f"--files-dir directory does not exist: {files_dir}")
    else:
        version = _detect_version(parser_dir)
        files_dir = _FILES_DIRS.get(version)
        if files_dir is None or not files_dir.is_dir():
            files_dir = _LEGACY_DIR
            version = "legacy(0823 alias)"
        if not files_dir.is_dir():
            _abort(f"no replacement set found under {_MOD_DIR}")

    print(f"Detected parser-engine version: {version}  ({files_dir.name})")

    ok = True
    for src_name, rel_dst in _TARGETS:
        src = files_dir / src_name
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
