#!/bin/bash
set -e

export http_proxy=http://192.168.4.157:7890
export https_proxy=http://192.168.4.157:7890

echo "Fixing DeepSeek-V4 ParserEngine: orphan <｜DSML｜invoke> recovery (fallback when the opening <｜DSML｜tool_calls> wrapper is omitted at long context)"

python3 "$(dirname "$0")/patch_deepseek_v4.py"
echo "Orphan-invoke fallback applied."
