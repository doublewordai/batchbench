#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME=${IMAGE_NAME:-tytn/batchbench:cu126}

# Example offline run parameters (override as needed)
export BATCHBENCH_MODE=offline
export OFFLINE_MODEL=${OFFLINE_MODEL:-Qwen/Qwen3-0.6B}
export OFFLINE_NUM_REQS=${OFFLINE_NUM_REQS:-128}
export OFFLINE_ICL=${OFFLINE_ICL:-512}
export OFFLINE_OCL=${OFFLINE_OCL:-1}
export OFFLINE_MAX_NUM_BATCHED_TOKENS=${OFFLINE_MAX_NUM_BATCHED_TOKENS:-512}
export OFFLINE_GPU_MEMORY_UTILIZATION=${OFFLINE_GPU_MEMORY_UTILIZATION:-0.75}
export OFFLINE_THROUGHPUT_DIR=${OFFLINE_THROUGHPUT_DIR:-/outputs}

# Directory to receive output CSVs
OUTPUT_DIR=${OUTPUT_DIR:-$(pwd)/outputs}
mkdir -p "$OUTPUT_DIR"

docker run --rm -it \
  --gpus all \
  -e BATCHBENCH_MODE \
  -e OFFLINE_MODEL \
  -e OFFLINE_NUM_REQS \
  -e OFFLINE_ICL \
  -e OFFLINE_OCL \
  -e OFFLINE_MAX_NUM_BATCHED_TOKENS \
  -e OFFLINE_GPU_MEMORY_UTILIZATION \
  -e OFFLINE_THROUGHPUT_DIR \
  -v "$OUTPUT_DIR":/outputs \
  "$IMAGE_NAME"
