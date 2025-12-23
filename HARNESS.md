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

```bash
python -m batchbench.harness configs/harness.yaml
```

## Cleanup

The GPU instance is **not** automatically deleted after benchmarking. Terminate manually:

```bash
prime pods terminate <pod-id>
```

## Resuming

Reuse an existing instance with `--resume`:

```bash
python -m batchbench.harness --resume <pod-id> configs/harness.yaml
```

Use `--resume` when:
1. Finished benchmarking and want to reuse the same instance for another run
2. Script exited early due to an error and you want to retry after fixing

When changing instance config (docker-image, availability, create), spin up a new instance instead.

## Configuration

The config file is the interface for setting instance, vLLM, generate, and benchmark args. Use the exact syntax of the relevant API or CLI:

- `instance.availability`: https://docs.primeintellect.ai/api-reference/availability/get-gpu-availability
- `instance.create`: https://docs.primeintellect.ai/api-reference/pods/create-pod
- `vllm.args`: `vllm serve --help`
- `generate`: `python -m batchbench.generate --help`
- `benchmark`: see `rust-bench/README.md`

## Architecture

Entry point is `run_harness()` which orchestrates the full pipeline. The first step is either provisioning a new GPU or connecting to an already provisioned one.

**Provisioning**: Query the availability API endpoint per instance config, select the cheapest GPU from results, then POST to the create API endpoint. The `Instance` class holds SSH connection info (host, port, user) in its fields. API interaction is abstracted through `PrimeIntellectClient`.

**Environment setup**: For a freshly provisioned GPU, pull the docker image, start the container, clone batchbench, activate the virtual environment, and install dependencies. When resuming (`--resume`), setup continues from where it was interrupted or skips entirely if already complete. These actions are executed via `docker pull`/`run`/`exec` wrapped in `SSHSession.run()`, which uses paramiko to maintain an SSH connection and execute commands on the remote instance. `RemoteEnvironment` wraps `SSHSession` and provides an `exec()` method that runs `docker exec` inside the SSH session.

**Pipeline stages**: After setup, three stages run sequentially: start vLLM server, generate synthetic data (`batchbench.generate`), run benchmark (`batchbench.online`). Each stage builds a CLI command from config and executes it via `RemoteEnvironment.exec()`.
