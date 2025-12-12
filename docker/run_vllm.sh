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

VLLM_HAS_FLASHINFER_CUBIN=1 \
VLLM_USE_FLASHINFER_MOE_FP8=1 \
VLLM_FLASHINFER_MOE_BACKEND=throughput \
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --enable-expert-parallel \
  --tensor-parallel-size 1 \
  --data-parallel-size 2

# Most verbose logging
VLLM_LOGGING_LEVEL=DEBUG \
VLLM_TRACE_FUNCTION=1 \
VLLM_HAS_FLASHINFER_CUBIN=1 \
VLLM_USE_FLASHINFER_MOE_FP8=1 \
VLLM_FLASHINFER_MOE_BACKEND=throughput \
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --enable-expert-parallel \
  --tensor-parallel-size 1 \
  --data-parallel-size 2 \
  --all2all-backend deepep_high_throughput \
  --compilation-mode NONE \
  --cudagraph-mode FULL

VLLM_LOGGING_LEVEL=DEBUG \
VLLM_TRACE_FUNCTION=1 \
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --enable-expert-parallel \
  --tensor-parallel-size 1 \
  --data-parallel-size 2 \
  --compilation-config '{"mode": 0, "cudagraph_mode": "full"}' \
  --all2all-backend deepep_high_throughput