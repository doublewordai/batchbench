# BatchBench Copilot Instructions

## Project Overview

BatchBench is now a Rust-only benchmarking suite for LLM inference workloads. The single CLI binary `batchbench` generates synthetic request datasets and drives parallel online requests to OpenAI-compatible endpoints.

## Architecture

- **Rust core** (`rust-bench/`): Async HTTP client and CLI (`batchbench.rs`) built with Tokio and `clap`.
- **Build script**: `build_rust.sh` builds the release binary; no Python packaging remains.
- **Invocation**: Run the Rust binary directly (from `target/release/batchbench`) with inline dataset generation sized to `users * requests_per_user`.

## Critical Workflows

- **Build**: `./build_rust.sh` or `cd rust-bench && cargo build --release --bin batchbench`.
- **Run**: `./rust-bench/target/release/batchbench --model <model> --users <n> --requests-per-user <m> [other flags]`.
- **Tests**: `cd rust-bench && cargo test`.

## Key Conventions

- Requests are generated inline; JSONL inputs and Python entrypoints have been removed.
- Request selection is deterministic: request `m` from user `n` maps to index `m * N + n` in the generated list (N = users).
- Output length options: fixed (`--output-tokens` with optional `--output-vary`, default 0) or log-normal sampling (mutually exclusive with fixed tokens).

## Docker Configuration

Docker configs remain under `docker/`; mount YAML configs as needed (see `docker/CONFIG.md`).

## Common Pitfalls

1. Rebuild after Rust changes (`cargo build --release`).
2. Ensure API keys are provided via `--api-key` or the env var named by `--api-key-env`.
3. `--output-vary` defaults to 0; set `--output-tokens` to enable fixed/varied lengths.
