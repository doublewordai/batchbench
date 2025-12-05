#!/usr/bin/env bash
# Example: Running BatchBench by overriding the entrypoint to use launch_batchbench.sh
#
# This approach is useful when:
# - You want to customize the startup script at runtime
# - You're deploying to a compute provider that requires a custom bash entrypoint
# - You need to add pre/post benchmark steps in the startup script

set -euo pipefail

# Example 1: Mount launch_batchbench.sh and override entrypoint
# The script generates its own YAML config internally, so no separate config file needed
docker run --gpus all \
  --entrypoint /bin/bash \
  -v "$(pwd)/run_vllm.sh:/workspace/run_vllm.sh:ro" \
  -v "$(pwd)/results:/workspace/results" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  tytn/batchbench:cu126 \
  /workspace/run_vllm.sh

