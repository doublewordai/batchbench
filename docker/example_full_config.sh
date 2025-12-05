#!/usr/bin/env bash
# Example: Running BatchBench with full configuration including
# lognormal output sampling and server result reporting

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

# Docker image to use
IMAGE="${BATCHBENCH_IMAGE:-tytn/batchbench:testbaked}"

# Config file path (relative to script directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/config.full.yaml}"

# Optional: Results server URL (can also be set in config file)
RESULTS_SERVER="${RESULTS_SERVER:-}"

# ============================================================================
# Example 1: Run with mounted config file
# ============================================================================

echo "=== Example 1: Basic run with full config ==="
echo "Using config: ${CONFIG_FILE}"

docker run --gpus all --rm \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  --network host \
  "${IMAGE}"
#   -v "${CONFIG_FILE}:/etc/batchbench/config.yaml" \