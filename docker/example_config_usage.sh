#!/usr/bin/env bash
# Example: Running BatchBench with YAML configuration

# Example 1: Offline benchmark with mounted config
docker run --gpus all \
  -v $(pwd)/config.offline.yaml:/etc/batchbench/config.yaml \
  -v $(pwd)/results:/workspace/results \
  batchbench:latest

# Example 2: Online benchmark with custom config path
docker run --gpus all \
  -v $(pwd)/config.online.yaml:/config/my-config.yaml \
  -e CONFIG_FILE=/config/my-config.yaml \
  -v $(pwd)/results:/workspace/results \
  batchbench:latest

# Example 3: With model cache mounted
docker run --gpus all \
  -v $(pwd)/config.offline.yaml:/etc/batchbench/config.yaml \
  -v $(pwd)/results:/workspace/results \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  batchbench:latest

