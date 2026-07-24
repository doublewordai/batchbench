# BatchBench

BatchBench ships a Rust benchmarking core with a thin Python wrapper.

You can install it with:

```bash
uv pip install batchbench
```

The Python package exposes Rust functionality for request generation and benchmark execution.

## Python API

```python
import batchbench

config = {
    "endpoint": "https://example.com/v1/chat/completions",
    "user_count": 1,
    "mode": batchbench.finite_mode(requests_per_user=1),
    "request_body": batchbench.request_entry(
        {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "ping"}],
        },
        line_idx=0,
        input_tokens=1,
    ),
    "requests": [
        batchbench.request_entry(
            {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 4,
            },
            line_idx=0,
            input_tokens=1,
        )
    ],
    "dry_run": True,
}

report = batchbench.run_benchmark(config)
print(report)
```

Request generation:

```python
requests = batchbench.generate_requests(
    {
        "count": 16,
        "prefix_overlap": 0.2,
        "target_tokens": 128,
        "tokenizer_model": "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8",
        "dist_mode": "fixed",
    },
    model="Qwen/Qwen3-VL-235B-A22B-Instruct-FP8",
)
```

## Python CLI

The package installs `batchbench`, which forwards directly to the Rust CLI implementation.
Use the same flags as the Rust binary:

```bash
batchbench \
  --model gpt-4o-mini \
  --users 8 \
  --requests-per-user 2 \
  --input-tokens 256 \
  --output-tokens 64 \
  --output-vary 0
```

Use `--sglang` to apply output token constraints via `min_new_tokens`/`max_new_tokens`
instead of `min_tokens`/`max_tokens`.

Press `Ctrl+C` during a run to cancel active requests and print a partial summary.

## Agent-loop benchmark

`batchbench-agent` is a separate entrypoint for stateful agent workloads. It starts
`--agents` independent loops concurrently. Within each agent, model requests remain
sequential: the returned assistant message and a synthetic tool/environment response
are appended to `messages`, then the complete growing conversation is sent again.
This makes every request after the first one reuse that agent's previous prompt as a
prefix and exercises server-side KV caching.

Fixed-length example:

```bash
batchbench-agent \
  --model Qwen/Qwen3-8B \
  --host http://localhost:8000 \
  --agents 16 \
  --input-tokens 256 \
  --output-tokens 64 \
  --environment-tokens 128 \
  --tool-invocations 8 \
  --tool-call-latency-ms 250
```

Log-normal example:

```bash
batchbench-agent \
  --model Qwen/Qwen3-8B \
  --host http://localhost:8000 \
  --agents 16 \
  --input-lognorm-median 256 \
  --input-lognorm-sigma 0.5 \
  --input-lognorm-max 2048 \
  --output-lognorm-median 64 \
  --output-lognorm-sigma 0.4 \
  --output-lognorm-max 512 \
  --environment-lognorm-median 128 \
  --environment-lognorm-sigma 0.6 \
  --environment-lognorm-max 1024 \
  --tool-invocations-lognorm-median 8 \
  --tool-invocations-lognorm-sigma 0.3 \
  --tool-invocations-lognorm-max 32 \
  --tool-call-latency-lognorm-median-ms 250 \
  --tool-call-latency-lognorm-sigma 0.5 \
  --tool-call-latency-lognorm-max-ms 2000 \
  --seed 42
```

Each log-normal family accepts either `*-lognorm-median` or `*-lognorm-mu`,
requires `*-lognorm-sigma`, and optionally accepts `*-lognorm-max`. Samples are
independent between agents and turns; `--seed` makes the sampled workload
reproducible. `--tokenizer-model` can be supplied when the endpoint's model name is
not also a Hugging Face tokenizer identifier.

Tool-call latency defaults to zero. Set a fixed delay with
`--tool-call-latency-ms`, or sample milliseconds independently for every invocation
with `--tool-call-latency-lognorm-median-ms` (or
`--tool-call-latency-lognorm-mu`), `--tool-call-latency-lognorm-sigma`, and the
optional `--tool-call-latency-lognorm-max-ms`. After a model response succeeds, the
agent sleeps for the sampled duration before making the environment result
available and submitting its next model request.

One tool invocation means one model request followed by one synthetic environment
response. The benchmark asks the model for an `environment` tool call and preserves
the returned assistant message in history. If an endpoint returns plain assistant
content instead, BatchBench wraps it in a valid synthetic tool call before appending
the environment response so the loop can continue.

The final report includes:

- total input tokens sent, from successful responses' `usage.prompt_tokens`;
- total output tokens generated, from `usage.completion_tokens`;
- estimated cached input tokens under perfect prefix caching;
- total simulated tool-call latency across all agents.

For each successful request after an agent's first, the cache estimate adds that
same agent's preceding prompt-token count (capped by the current prompt count).
Retries and failed requests are excluded because they do not provide reliable usage
data. Use `--results-csv <path>` to persist the same totals, or `--dry-run` to inspect
the independently sampled workload without sending requests.

## Rust CLI

The existing Rust CLI is unchanged:

```bash
cargo build --release --manifest-path rust/Cargo.toml --bin batchbench
./rust/target/release/batchbench --help
```

The agent-loop binary can likewise be run directly:

```bash
cargo build --release --manifest-path rust/Cargo.toml --bin batchbench-agent
./rust/target/release/batchbench-agent --help
```

## Releases and PyPI

- CI (`.github/workflows/ci.yaml`) checks Rust build/test, builds a wheel, and runs smoke tests.
- Release Please (`.github/workflows/release-please.yaml`) opens/updates release PRs and, on merge, creates `v*` tags/releases.
- Python release workflow (`.github/workflows/python-release.yaml`) builds and publishes prebuilt platform wheels to PyPI on `v*` tag pushes.
