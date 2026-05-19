#!/bin/bash
set -e

# Fix: checkpoint stores split q/k/v_proj weights but vLLM's
# MiniCPMV4_6ViTWindowAttentionSelfAttn only has a fused qkv_proj.
# Adds a load_weights() method to the class that merges them correctly.
cp minicpmv4_6.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/minicpmv4_6.py
echo "Successfully applied MiniCPM-V 4.6 qkv_proj merge patch"
