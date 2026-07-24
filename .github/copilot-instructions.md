# BatchBench Copilot Instructions

## Project Overview

BatchBench is a Rust benchmarking suite for LLM inference workloads. The `batchbench` CLI generates synthetic request datasets and drives parallel online requests to OpenAI-compatible endpoints. The separate `batchbench-agent` CLI drives concurrent, stateful agent loops with growing message histories for prefix-cache benchmarking.

## Architecture

- **Rust core** (`rust/`): Async HTTP clients and CLIs built with Tokio and `clap`.
- **Python package** (`src/batchbench/`): Thin PyO3 wrappers and installed console entrypoints.
- **Invocation**: Run `batchbench` for independent batched requests or `batchbench-agent` for stateful tool loops.

## Critical Workflows

- **Build**: `cargo build --release --manifest-path rust/Cargo.toml`.
- **Run**: `./rust/target/release/batchbench --model <model> --users <n> --requests-per-user <m> [other flags]`.
- **Run agent loops**: `./rust/target/release/batchbench-agent --model <model> --agents <n> --tool-invocations <m> [other flags]`.
- **Tests**: `cargo test --manifest-path rust/Cargo.toml`.

## Key Conventions

- Requests are generated inline; JSONL input has been removed.
- Request selection is deterministic: request `m` from user `n` maps to index `m * N + n` in the generated list (N = users).
- Output length options: fixed (`--output-tokens` with optional `--output-vary`, default 0) or log-normal sampling (mutually exclusive with fixed tokens).
- Agent-loop input, output, environment, tool-invocation counts, and simulated tool-call latency each support fixed or independent log-normal sampling.

## Docker Configuration

Docker configs remain under `docker/`; mount YAML configs as needed (see `docker/CONFIG.md`).

## Common Pitfalls

1. Rebuild after Rust changes (`cargo build --release`).
2. Ensure API keys are provided via `--api-key` or the env var named by `--api-key-env`.
3. `--output-vary` defaults to 0; set `--output-tokens` to enable fixed/varied lengths.
