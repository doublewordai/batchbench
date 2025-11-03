#!/usr/bin/env bash
# Simple sanity check runner for the offline benchmark CLI.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON:-python}

batchbench.offline \
  --model facebook/opt-125m \
  --num_reqs 8 \
  --icl 1024 \
  --ocl 8192 \
  --tensor_parallel_size 1 \
  --pipeline_parallel_size 1 \
  --max_num_batched_tokens 512