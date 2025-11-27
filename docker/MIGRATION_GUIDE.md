# Migration Guide: Environment Variables to YAML Config

This guide helps you migrate from the environment variable-based configuration (using `run_docker.sh`) to the new YAML-based configuration (using `run_docker.py`).

## What Changed?

- **Old**: Configuration via environment variables (e.g., `OFFLINE_MODEL`, `ONLINE_USERS`)
- **New**: Configuration via YAML file mounted into the container

## Benefits of YAML Configuration

1. **Easier to read and maintain** - All settings in one structured file
2. **Version control friendly** - Track config changes in git
3. **Less error-prone** - YAML syntax validation catches mistakes
4. **Better organization** - Grouped settings (server, client, generate)
5. **Reusable** - Share configs across environments

## Quick Start

### 1. Choose or create a config file

```bash
# Copy an example config
cp docker/config.offline.yaml my-config.yaml

# Or create your own from the template
cp docker/config.example.yaml my-config.yaml
```

### 2. Edit the config file

```yaml
mode: offline

offline:
  model: "Qwen/Qwen3-0.6B"
  num_reqs: 100
  icl: 512
  ocl: 128
  # ... other settings
```

### 3. Run with Docker

```bash
# Default location
docker run --gpus all \
  -v $(pwd)/my-config.yaml:/etc/batchbench/config.yaml \
  batchbench:latest

# Custom location
docker run --gpus all \
  -v $(pwd)/my-config.yaml:/config/bench.yaml \
  -e CONFIG_FILE=/config/bench.yaml \
  batchbench:latest
```

## Conversion Examples

### Example 1: Offline Benchmark

**Old (environment variables):**
```bash
docker run --gpus all \
  -e BATCHBENCH_MODE=offline \
  -e OFFLINE_MODEL="Qwen/Qwen3-0.6B" \
  -e OFFLINE_NUM_REQS=100 \
  -e OFFLINE_ICL=512 \
  -e OFFLINE_OCL=128 \
  -e OFFLINE_TENSOR_PARALLEL_SIZE=1 \
  -e OFFLINE_GPU_MEMORY_UTILIZATION=0.9 \
  batchbench:latest
```

**New (YAML config):**
```yaml
# config.yaml
mode: offline

offline:
  model: "Qwen/Qwen3-0.6B"
  num_reqs: 100
  icl: 512
  ocl: 128
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.9
```

```bash
docker run --gpus all \
  -v $(pwd)/config.yaml:/etc/batchbench/config.yaml \
  batchbench:latest
```

### Example 2: Online Benchmark

**Old (environment variables):**
```bash
docker run --gpus all \
  -e BATCHBENCH_MODE=online \
  -e ONLINE_MODEL="Qwen/Qwen3-0.6B" \
  -e ONLINE_SERVER_HOST=0.0.0.0 \
  -e ONLINE_SERVER_PORT=8000 \
  -e ONLINE_USERS=10 \
  -e ONLINE_REQUESTS_PER_USER=5 \
  -e ONLINE_OUTPUT_TOKENS=128 \
  -e ONLINE_VERBOSE=true \
  -e ONLINE_GENERATE_COUNT=100 \
  -e ONLINE_GENERATE_APPROX_INPUT_TOKENS=512 \
  batchbench:latest
```

**New (YAML config):**
```yaml
# config.yaml
mode: online

online:
  model: "Qwen/Qwen3-0.6B"
  
  server:
    host: "0.0.0.0"
    port: 8000
  
  client:
    users: 10
    requests_per_user: 5
    output_tokens: 128
    verbose: true

generate:
  count: 100
  approx_input_tokens: 512
```

```bash
docker run --gpus all \
  -v $(pwd)/config.yaml:/etc/batchbench/config.yaml \
  batchbench:latest
```

## Complete Mapping Table

### General

| Environment Variable | YAML Path | Type |
|---------------------|-----------|------|
| `BATCHBENCH_MODE` | `mode` | string |

### Offline Settings

| Environment Variable | YAML Path | Type |
|---------------------|-----------|------|
| `OFFLINE_MODEL` | `offline.model` | string |
| `OFFLINE_NUM_REQS` | `offline.num_reqs` | integer |
| `OFFLINE_ICL` | `offline.icl` | integer |
| `OFFLINE_OCL` | `offline.ocl` | integer |
| `OFFLINE_THROUGHPUT_DIR` | `offline.throughput_dir` | string |
| `OFFLINE_TENSOR_PARALLEL_SIZE` | `offline.tensor_parallel_size` | integer |
| `OFFLINE_PIPELINE_PARALLEL_SIZE` | `offline.pipeline_parallel_size` | integer |
| `OFFLINE_GPU_MEMORY_UTILIZATION` | `offline.gpu_memory_utilization` | float |
| `OFFLINE_MAX_NUM_BATCHED_TOKENS` | `offline.max_num_batched_tokens` | integer |
| `OFFLINE_EXTRA_ARGS` | `offline.extra_args` | list |

### Online Server Settings

| Environment Variable | YAML Path | Type |
|---------------------|-----------|------|
| `ONLINE_MODEL` | `online.model` | string |
| `ONLINE_CLIENT_MODEL` | `online.client_model` | string |
| `ONLINE_SERVER_HOST` | `online.server.host` | string |
| `ONLINE_SERVER_PORT` | `online.server.port` | integer |
| `ONLINE_TENSOR_PARALLEL_SIZE` | `online.server.tensor_parallel_size` | integer |
| `ONLINE_PIPELINE_PARALLEL_SIZE` | `online.server.pipeline_parallel_size` | integer |
| `ONLINE_MAX_NUM_BATCHED_TOKENS` | `online.server.max_num_batched_tokens` | integer |
| `ONLINE_GPU_MEMORY_UTILIZATION` | `online.server.gpu_memory_utilization` | float |
| `ONLINE_SERVER_EXTRA_ARGS` | `online.server.extra_args` | list |
| `ONLINE_SERVER_WAIT_RETRIES` | `online.server.wait_retries` | integer |
| `ONLINE_SERVER_WAIT_DELAY_SECS` | `online.server.wait_delay_secs` | integer |
| `ONLINE_SERVER_HEALTH_URL` | `online.server.health_url` | string |

### Online Client Settings

| Environment Variable | YAML Path | Type |
|---------------------|-----------|------|
| `ONLINE_HOST` | `online.client.host` | string |
| `ONLINE_ENDPOINT` | `online.client.endpoint` | string |
| `ONLINE_USERS` | `online.client.users` | integer |
| `ONLINE_REQUESTS_PER_USER` | `online.client.requests_per_user` | integer |
| `ONLINE_API_KEY` | `online.client.api_key` | string |
| `ONLINE_API_KEY_ENV` | `online.client.api_key_env` | string |
| `ONLINE_REQUEST_TIMEOUT_SECS` | `online.client.request_timeout_secs` | integer |
| `ONLINE_MAX_RETRIES` | `online.client.max_retries` | integer |
| `ONLINE_RETRY_DELAY_MS` | `online.client.retry_delay_ms` | integer |
| `ONLINE_OUTPUT_TOKENS` | `online.client.output_tokens` | integer |
| `ONLINE_OUTPUT_VARY` | `online.client.output_vary` | float |
| `ONLINE_RANDOM_REQUESTS` | `online.client.random_requests` | boolean |
| `ONLINE_VERBOSE` | `online.client.verbose` | boolean |
| `ONLINE_CLIENT_EXTRA_ARGS` | `online.client.extra_args` | list |

### Dataset Generation Settings

| Environment Variable | YAML Path | Type |
|---------------------|-----------|------|
| `ONLINE_DATASET_PATH` | `generate.dataset_path` | string |
| `ONLINE_GENERATE_COUNT` | `generate.count` | integer |
| `ONLINE_GENERATE_PREFIX_OVERLAP` | `generate.prefix_overlap` | float |
| `ONLINE_GENERATE_APPROX_INPUT_TOKENS` | `generate.approx_input_tokens` | integer |
| `ONLINE_GENERATE_TOKENIZER_MODEL` | `generate.tokenizer_model` | string |
| `ONLINE_GENERATE_TOKEN_TOLERANCE` | `generate.token_tolerance` | integer |
| `ONLINE_GENERATE_HUGGINGFACE_TOKEN` | `generate.huggingface_token` | string |
| `ONLINE_GENERATE_EXTRA_ARGS` | `generate.extra_args` | list |

## Tips

1. **Use null for optional values**: Instead of omitting env vars, use `null` in YAML
2. **Boolean values**: Use `true`/`false`, not `"true"`/`"false"`
3. **Lists for extra args**: 
   - Old: `OFFLINE_EXTRA_ARGS="--arg1 value1 --flag"`
   - New: `extra_args: ["--arg1", "value1", "--flag"]`
4. **Comments**: YAML supports comments with `#`
5. **Version control**: Add your config files to `.gitignore` if they contain secrets

## Backward Compatibility

The old `run_docker.sh` script is still available if you need it, but it's recommended to migrate to the YAML-based configuration for better maintainability.
