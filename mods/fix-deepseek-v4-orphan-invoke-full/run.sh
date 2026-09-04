#!/bin/bash
set -euo pipefail

# Full orphan-invoke recovery for DeepSeek-V4, ported from vLLM PR #52645.
# Coexists with the lighter mods/fix-deepseek-v4-orphan-invoke: this patch
# strips the older mod's plain INVOKE_PREFIX transitions, then installs the
# provisional (validated / rollback-aware) variants.
# NB: purely local file patch — no network, no proxy required.

PYTHON_ROOT="${VLLM_SITE_PACKAGES:-${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}}"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Fixing DeepSeek-V4 ParserEngine: full orphan <DSML|invoke> recovery (provisional hold + tool-name validation + rollback)"

if [ ! -f "$PYTHON_ROOT/vllm/tool_parsers/utils.py" ]; then
  echo "[fix-deepseek-v4-orphan-invoke-full] vLLM tool_parsers/utils.py not found at $PYTHON_ROOT" >&2
  exit 1
fi

python3 "$MOD_DIR/patch_vllm.py" "$PYTHON_ROOT"
echo "Full orphan-invoke fallback applied."
