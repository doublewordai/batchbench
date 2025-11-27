# BatchBench Docker Configuration

The `run_docker.py` script uses YAML configuration files instead of environment variables for easier configuration management.

## Usage

### 1. Create or customize a configuration file

Choose one of the example configs or create your own:
- `config.example.yaml` - Template with all available options
- `config.offline.yaml` - Example offline benchmark configuration
- `config.online.yaml` - Example online benchmark configuration

### 2. Mount the config file and run the container

```bash
docker run -v $(pwd)/config.yaml:/etc/batchbench/config.yaml your-image
```

Or specify a custom path using the `CONFIG_FILE` environment variable:

```bash
docker run \
  -v $(pwd)/my-config.yaml:/config/bench.yaml \
  -e CONFIG_FILE=/config/bench.yaml \
  your-image
```

## Configuration Structure

### Mode Selection

```yaml
mode: offline  # or 'online'
```

### Offline Configuration

```yaml
offline:
  model: "Qwen/Qwen3-0.6B"              # Model name or path
  num_reqs: 100                          # Number of requests
  icl: 512                               # Input context length
  ocl: 128                               # Output context length
  throughput_dir: "/workspace/results"   # Results directory
  tensor_parallel_size: 1                # Tensor parallelism
  pipeline_parallel_size: 1              # Pipeline parallelism
  gpu_memory_utilization: 0.9            # GPU memory (0.0-1.0)
  max_num_batched_tokens: 8192          # Max batched tokens
  extra_args: []                         # Additional CLI args as list
```

### Online Configuration

```yaml
online:
  model: "Qwen/Qwen3-0.6B"      # Model for server
  client_model: null             # Override for client (optional)
  
  server:
    host: "0.0.0.0"
    port: 8000
    tensor_parallel_size: 1
    pipeline_parallel_size: 1
    max_num_batched_tokens: 8192
    gpu_memory_utilization: 0.9
    extra_args: []
    
    # Health check settings
    wait_retries: 60
    wait_delay_secs: 1
    health_url: null              # Override default health endpoint
  
  client:
    host: null                    # Override server URL
    endpoint: null                # API endpoint path
    users: 10                     # Concurrent users
    requests_per_user: 5          # Requests per user
    api_key: null                 # API key
    api_key_env: null             # Env var with API key
    request_timeout_secs: 120     # Request timeout
    max_retries: 3                # Max retry attempts
    retry_delay_ms: 100           # Retry delay
    output_tokens: 128            # Output token count
    output_vary: 0.2              # Token variation
    random_requests: false        # Randomize order
    verbose: true                 # Verbose output
    extra_args: []
```

### Dataset Generation (for online mode)

```yaml
generate:
  dataset_path: /tmp/batchbench_requests.jsonl
  count: 100                      # Number of requests
  prefix_overlap: 0.15            # Prefix overlap ratio
  approx_input_tokens: 512        # Approx input tokens
  tokenizer_model: "Qwen/Qwen3-0.6B"
  token_tolerance: 50             # Token tolerance
  huggingface_token: null         # HF token if needed
  extra_args: []
```

## Examples

### Offline Benchmark

```yaml
mode: offline

offline:
  model: "meta-llama/Llama-2-7b"
  num_reqs: 200
  icl: 1024
  ocl: 256
  tensor_parallel_size: 2
  gpu_memory_utilization: 0.95
```

### Online Benchmark with Custom Dataset

```yaml
mode: online

online:
  model: "mistralai/Mistral-7B-v0.1"
  
  server:
    port: 8080
    tensor_parallel_size: 2
  
  client:
    users: 20
    requests_per_user: 10
    output_tokens: 200
    verbose: true

generate:
  dataset_path: /data/my-requests.jsonl
  # If file exists, it will be used instead of generating
```

## Docker Compose Example

```yaml
version: '3.8'

services:
  batchbench:
    image: your-batchbench-image
    volumes:
      - ./config.yaml:/etc/batchbench/config.yaml
      - ./results:/workspace/results
      - ./models:/workspace/models
    environment:
      - CONFIG_FILE=/etc/batchbench/config.yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

## Migration from Environment Variables

If you were using the shell script with environment variables, here's how to migrate:

| Old Environment Variable | New YAML Path |
|-------------------------|---------------|
| `BATCHBENCH_MODE` | `mode` |
| `OFFLINE_MODEL` | `offline.model` |
| `OFFLINE_NUM_REQS` | `offline.num_reqs` |
| `ONLINE_MODEL` | `online.model` |
| `ONLINE_SERVER_HOST` | `online.server.host` |
| `ONLINE_SERVER_PORT` | `online.server.port` |
| `ONLINE_USERS` | `online.client.users` |
| `ONLINE_DATASET_PATH` | `generate.dataset_path` |
| `ONLINE_GENERATE_COUNT` | `generate.count` |

## Notes

- Set values to `null` to omit that parameter
- `extra_args` should be a list: `["--arg1", "value1", "--flag"]`
- Boolean values use YAML booleans: `true` or `false`
- If a config file exists at the dataset path, generation is skipped
