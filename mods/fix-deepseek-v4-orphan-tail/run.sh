#!/bin/bash
set -euo pipefail

# Absorb an orphan DeepSeek DSML closing tail after a successful tool call and
# emit a synthetic `bash` tool call that prompts the model to re-issue the last
# tool call so the agent keeps working
# instead of parsing residual DSML markup as its final answer.
# Coexists with mods/fix-deepseek-v4-orphan-invoke and -full (run AFTER them):
# this patch only guards CONTENT-state stray closers; it does not touch the
# provisional-recovery machinery of the full mod.
# NB: purely local file patch — no network, no proxy required.

PYTHON_ROOT="${VLLM_SITE_PACKAGES:-${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}}"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Fixing DeepSeek-V4 ParserEngine: absorb orphan DSML closing tail + bash regenerate"

if [ ! -f "$PYTHON_ROOT/vllm/parser/engine/streaming_parser_engine.py" ]; then
  echo "[fix-deepseek-v4-orphan-tail] streaming_parser_engine.py not found at $PYTHON_ROOT" >&2
  exit 1
fi

python3 "$MOD_DIR/patch_vllm.py" "$PYTHON_ROOT"
echo "Orphan-tail recovery applied."
