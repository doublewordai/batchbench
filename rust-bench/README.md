# batchbench-rs

An asynchronous benchmarking library for exercising text-generation style HTTP endpoints with multiple concurrent users. It orchestrates request workers, aggregates run-wide metrics, and emits a rich report summarising throughput, latency, and failures.

## Getting Started

### Command-Line Runner

The repository includes a ready-to-go CLI that reads prompts from a JSONL file (each record must expose the request payload in its `text` field) and fans them out across concurrent users. Configure it with flags for host, endpoint, model name, user count, and retry behaviour. After each run, the CLI prints aggregate request counts, token throughput, and p50/p90/p99 latency percentiles:

```
cargo run --bin batchbench -- \
  --jsonl data/requests.jsonl \
  --users 32 \
  --model gpt-4o-mini \
  --host https://api.openai.com \
  --endpoint /v1/chat/completions
```

By default every user issues a single request, so make sure the JSONL file contains at least as many entries as the `--users` flag. Override `--requests-per-user` if you want multiple iterations per worker. Authentication is optional—supply `--api-key`, set `OPENAI_API_KEY`, or omit both for anonymous endpoints.

The CLI reuses the library entry points under the hood and prints a `BenchmarkReport` when the run completes.

- `--jsonl <path>` Path to the JSONL input whose objects expose a `text` field (required).
- `--users <num>` Number of concurrent workers; defaults to the JSONL record count.
- `--model <name>` Model identifier injected into every request; default `gpt-4o-mini`.
- `--host <url>` Base host for the API, including scheme; default `https://api.openai.com`.
- `--endpoint <path-or-url>` Endpoint path or full URL; default `/v1/chat/completions`.
- `--requests-per-user <num>` Iterations each worker performs before stopping; default `1`.
- `--api-key <token>` Optional API key for the Authorization header; takes precedence over `--api-key-env`.
- `--api-key-env <name>` Environment variable to consult when `--api-key` is omitted; default `OPENAI_API_KEY`. Ignored if not set.
- `--request-timeout-secs <seconds>` Per-request timeout; default `60` seconds.
- `--max-retries <num>` Retry attempts before a request is marked failed; default `2`.
- `--retry-delay-ms <milliseconds>` Base delay between retries; default `250` milliseconds.
- `--output-tokens <num>` Force completions to emit exactly `<num>` tokens (sets `max_completion_tokens` and enables `nvext.ignore_eos`).
- `--output-vary <num>` When used with `--output-tokens`, add a per-request uniform variation in `[-num, +num]` tokens (lower bounded at 1).

### Library Usage

1. Install the Rust toolchain with [`rustup`](https://rustup.rs/) if you have not already.
2. Add this crate as a dependency in the binary that will orchestrate the benchmark. For a sibling project, point to the path that contains this repository:

   ```toml
   [dependencies]
   batchbench-rs = { path = "../rust-bench" }
   tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
   anyhow = "1"
   serde_json = "1"
   ```

3. In your binary, construct a `BenchmarkConfig` and call `run_benchmark`. Because the library is async, wrap the call in a Tokio runtime:

   ```rust
   use std::time::Duration;

   use anyhow::Result;
   use batchbench_rs::{run_benchmark, BenchmarkConfig, RunMode};
   use serde_json::json;

   #[tokio::main]
   async fn main() -> Result<()> {
       let api_key = std::env::var("OPENAI_API_KEY").ok();

       let config = BenchmarkConfig::try_new(
           "https://api.openai.com/v1/chat/completions",
           api_key,
           8,
           RunMode::Finite { requests_per_user: 25 },
           json!({
               "model": "gpt-4o-mini",
               "messages": [{"role": "user", "content": "Ping"}]
           }),
       )?
       .with_request_timeout(Duration::from_secs(30))
       .with_retry(3, Duration::from_millis(400));

       let report = run_benchmark(config).await?;
       println!("{:#?}", report);

       Ok(())
   }
   ```

4. Run your driver with `cargo run`. When the future resolves you receive a `BenchmarkReport` that you can log, persist, or feed into another system.

## Configuring Benchmarks

- **Target endpoint**: `BenchmarkConfig::try_new` validates and stores the URL you pass as `endpoint` and, when supplied, attaches the provided API key as a `Bearer` token. Supply any additional headers with `BenchmarkConfig::add_header` or by mutating `headers_mut()`.
- **Concurrent users**: `user_count` controls how many independent worker tasks are spawned.
- **Run modes**:
  - `RunMode::Finite { requests_per_user }` submits a fixed number of requests from each user before shutting down.
  - `RunMode::LongRunning { duration }` loops until the wall-clock duration elapses.
- **Request payload**: Provide any JSON value. The body is sent to the target verbatim on every request. Use `BenchmarkConfig::with_per_user_bodies` when each worker should send a distinct payload (e.g. sourced from a JSONL file).
- **Timeouts and retries**: `with_request_timeout` adjusts the per-request deadline. Use `with_retry(max_retries, retry_delay)` to customise exponential-style backoff between attempts.

## Understanding Reports

`run_benchmark` returns a `BenchmarkReport` (see `src/report.rs`) containing:

- Aggregate request counts, token totals, and total elapsed time.
- Derived throughput metrics, such as tokens-per-second and requests-per-second.
- Latency percentiles (p50/p90/p99) across all successful requests.
- A list of `FailureRecord` items capturing the last error seen per failed request.

Use these fields to build dashboards, generate CSVs, or trigger alerts after a run.

## Developing In-Repo

- Run `cargo fmt` to format changes.
- Execute `cargo test` to validate library behaviour.
- Add integration drivers under `examples/` or an external binary crate to exercise the benchmark end-to-end.

For the main scheduling loop and request orchestration logic, refer to `src/runner.rs`.
