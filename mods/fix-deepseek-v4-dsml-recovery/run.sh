#!/bin/bash
set -euo pipefail

# Consolidated DeepSeek-V4 DSML recovery: dialect-agnostic orphan-invoke
# recovery (fullwidth + ASCII, missing/corrupted outer wrapper) plus
# orphan-closing-tail absorption with synthetic bash regenerate.
#
# Supersedes mods/fix-deepseek-v4-orphan-invoke, -invoke-full, and -tail:
# do not run this alongside any of them. Uses whole-file replacement
# (files/*.py copied verbatim) instead of anchor-based string patching, so
# it is not fragile to upstream formatting drift.
# NB: purely local file patch — no network, no proxy required.

PYTHON_ROOT="${VLLM_SITE_PACKAGES:-${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}}"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Fixing DeepSeek-V4 ParserEngine: consolidated DSML recovery (dialect-agnostic orphan-invoke + orphan-tail regenerate)"

if [ ! -f "$PYTHON_ROOT/vllm/parser/deepseek_v4.py" ]; then
  echo "[fix-deepseek-v4-dsml-recovery] vllm/parser/deepseek_v4.py not found at $PYTHON_ROOT" >&2
  exit 1
fi

python3 "$MOD_DIR/patch_vllm.py" "$PYTHON_ROOT"
echo "Consolidated DSML recovery applied."
