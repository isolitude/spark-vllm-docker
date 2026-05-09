#!/bin/bash
set -e

export http_proxy=http://192.168.4.157:7890
export https_proxy=http://192.168.4.157:7890

cd /usr/local/lib/python3.12/dist-packages
echo "Downloading PR #38909"
PATCH_FILE=$(mktemp /tmp/pr38909.XXXXXX.diff)
if curl -fsL https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/38909.diff -o "$PATCH_FILE"; then
  echo "- Download succeeded ($(wc -c < "$PATCH_FILE") bytes)"
else
  echo "- Download FAILED"
  rm -f "$PATCH_FILE"
  exit 1
fi

echo "Applying PR #38909"
if git apply --exclude="tests/*" "$PATCH_FILE"; then
  echo "- PR #38909 applied successfully"
else
  echo "- PR #38909 can't be applied, skipping"
fi
rm -f "$PATCH_FILE"
