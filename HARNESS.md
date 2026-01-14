# Harness Usage

## Setup

```bash
pip install -e ".[harness]"

export PRIME_API_KEY=...
export PRIME_TEAM_ID=...
export PRIME_SSH_KEY_PATH=...
```

- API key: https://app.primeintellect.ai/dashboard/tokens
- Team ID: https://app.primeintellect.ai/dashboard/team-profile

## Running

### Single Config

```bash
python -m batchbench.harness configs/harness.yaml
```

By default, the instance is automatically terminated after the benchmark completes. With a single config, you have the option to:

**Keep the instance running** for another benchmark:
```bash
python -m batchbench.harness --keep-alive configs/harness.yaml
```

**Resume** an existing instance:
```bash
python -m batchbench.harness --resume <pod-id> configs/harness.yaml
```

### Directory of Configs

```bash
python -m batchbench.harness configs/my-benchmarks/
```

Instances are automatically terminated after processing completes. The `--resume` and `--keep-alive` options are not available in this mode.

## Configuration

The config file is the interface for setting instance, vLLM, and benchmark args. Use the exact syntax of the relevant API or CLI:

- `instance.availability`: https://docs.primeintellect.ai/api-reference/availability/get-gpu-availability
- `instance.create`: https://docs.primeintellect.ai/api-reference/pods/create-pod
- `vllm.args`: `vllm serve --help`
- `benchmark`: `batchbench --help`

## Architecture

The harness accepts either a single config file or a directory of configs (all `*.yaml` files in that directory).

**Entry point**: `run_queues()` orchestrates the full pipeline. First, `build_queues()` groups configs by instance type (determined by `gpu_type` and `gpu_count` from the availability settings). Each group becomes a queue processed by a `QueueWorker`. Multiple queues run in parallel via `ThreadPoolExecutor`.

**Provisioning**: For each queue, the worker queries the availability API endpoint, selects the cheapest GPU from results, then POSTs to the create API endpoint. The `Instance` class holds SSH connection info (host, port, user) in its fields. API interaction is abstracted through `PrimeIntellectClient`. Once provisioned, all configs in that queue run sequentially on the same instance. After processing completes, the instance is automatically terminated by default.

**Environment setup**: For a freshly provisioned GPU, the harness pulls the docker image, starts the container, and clones the batchbench repository. When resuming (`--resume`), setup continues from where it was interrupted or skips entirely if already complete. These actions are executed via `docker pull`/`run`/`exec` wrapped in `SSHSession.run()`, which uses paramiko to maintain an SSH connection and execute commands on the remote instance. `RemoteEnvironment` wraps `SSHSession` and provides an `exec()` method that runs `docker exec` inside the SSH session.

**Pipeline stages**: After setup, two stages run sequentially: start vLLM server, run benchmark (via Rust binary). Each stage builds a CLI command from config and executes it via `RemoteEnvironment.exec()`.
