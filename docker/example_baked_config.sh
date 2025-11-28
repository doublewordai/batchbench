#!/usr/bin/env bash
# Example: Building a Docker image with a baked-in config file
# This is useful when deploying to environments where mounting files is difficult

set -euo pipefail

# Example 1: Build with offline benchmark config baked in
echo "Building image with offline config baked in..."
docker build \
  --build-arg BAKED_CONFIG=docker/config.offline.yaml \
  -f docker/Dockerfile.cu126 \
  -t batchbench:offline-baked \
  .

# Example 2: Build with online benchmark config baked in
echo "Building image with online config baked in..."
docker build \
  --build-arg BAKED_CONFIG=docker/config.online.yaml \
  -f docker/Dockerfile.cu126 \
  -t batchbench:online-baked \
  .

# Example 3: Build with custom config baked in
echo "Building image with custom config baked in..."
# First create a custom config
cat > /tmp/my-custom-config.yaml <<EOF
mode: offline

offline:
  model: "meta-llama/Llama-2-7b"
  num_reqs: 500
  icl: 2048
  ocl: 512
  tensor_parallel_size: 2
  gpu_memory_utilization: 0.95
EOF

docker build \
  --build-arg BAKED_CONFIG=/tmp/my-custom-config.yaml \
  -f docker/Dockerfile.cu126 \
  -t batchbench:custom-baked \
  .

echo "Done! You can now run without mounting:"
echo "  docker run --gpus all batchbench:offline-baked"
echo ""
echo "Or override the baked config at runtime:"
echo "  docker run --gpus all -v ./override.yaml:/etc/batchbench/config.yaml batchbench:offline-baked"
