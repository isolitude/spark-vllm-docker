#!/bin/bash
set -euo pipefail

# Fixes empty-output bug in DeepSeek-V4 prompt encoding: a request whose
# last message has role=system got no "<｜Assistant｜><think>"
# generation-prompt suffix, leaving the model an unterminated prompt that
# it resolves by emitting EOS immediately. See README.md for the writeup.
# NB: purely local file patch -- no network, no proxy required.

PYTHON_ROOT="${VLLM_SITE_PACKAGES:-${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}}"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Fixing DeepSeek-V4 encoding: generation-prompt suffix on trailing system messages"

if [ ! -f "$PYTHON_ROOT/vllm/tokenizers/deepseek_v4_encoding.py" ]; then
  echo "[fix-deepseek-v4-trailing-system-genprompt] vllm/tokenizers/deepseek_v4_encoding.py not found at $PYTHON_ROOT" >&2
  exit 1
fi

python3 "$MOD_DIR/patch_vllm.py" "$PYTHON_ROOT"
echo "Trailing-system generation-prompt fix applied."
