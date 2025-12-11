#!/bin/bash

export HF_HOME="/data"
# uv pip list

# Start vLLM server
vllm serve \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000

VLLM_FLASHINFER_MOE_BACKEND=throughput vllm serve \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    --enable-expert-parallel \
    --tensor-parallel-size 1 \
    --data-parallel-size 2
