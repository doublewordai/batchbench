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
