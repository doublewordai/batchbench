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

To benchmark a JSONL dataset instead of generated prompts, pass `--dataset-jsonl`.
Each non-empty line may be one of:

- `{"text": "prompt text", "input_tokens": 123}` to build a chat completion request.
- `{"body": {...}, "input_tokens": 123}` to send `body` as the request payload.
- A full request body object, such as `{"model": "...", "messages": [...]}`.

```bash
batchbench \
  --dataset-jsonl dataset.jsonl \
  --model gpt-4o-mini \
  --users 8 \
  --requests-per-user 2
```

The dataset must contain at least `users * requests-per-user` request entries.
If `--users` is omitted for a dataset run, BatchBench uses as many complete
request rounds as the dataset can provide for the selected `--requests-per-user`.

Use `--sglang` to apply output token constraints via `min_new_tokens`/`max_new_tokens`
instead of `min_tokens`/`max_tokens`.

Press `Ctrl+C` during a run to cancel active requests and print a partial summary.

## Rust CLI

The existing Rust CLI is unchanged:

```bash
cargo build --release --manifest-path rust/Cargo.toml --bin batchbench
./rust/target/release/batchbench --help
```

## Releases and PyPI

- CI (`.github/workflows/ci.yaml`) checks Rust build/test, builds a wheel, and runs smoke tests.
- Release Please (`.github/workflows/release-please.yaml`) opens/updates release PRs and, on merge, creates `v*` tags/releases.
- Python release workflow (`.github/workflows/python-release.yaml`) builds and publishes prebuilt platform wheels to PyPI on `v*` tag pushes.
