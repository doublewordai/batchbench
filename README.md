# BatchBench

Load generation for OpenAI-compatible chat endpoints. A Rust core with a thin Python
wrapper, published as prebuilt wheels:

```bash
uv pip install batchbench
```

Three entrypoints:

| Command | What it does |
|---|---|
| `batchbench` | Independent requests at a fixed concurrency with controlled token shapes |
| `batchbench-agent` | Stateful multi-turn sessions, sampled or replayed from a manifest |
| `batchbench-export-plans` | Builds a replay manifest from prompt-chain records in ClickHouse |

## Independent requests

```bash
batchbench \
  --model gpt-4o-mini \
  --users 8 \
  --requests-per-user 2 \
  --input-tokens 256 \
  --output-tokens 64 \
  --output-vary 0
```

Output length is enforced with `min_tokens`/`max_tokens`; pass `--sglang` for
backends that use `min_new_tokens`/`max_new_tokens`. `Ctrl+C` cancels active
requests and prints a partial summary. The same run is available from Python:

```python
import batchbench

report = batchbench.run_benchmark({
    "endpoint": "https://example.com/v1/chat/completions",
    "user_count": 1,
    "mode": batchbench.finite_mode(requests_per_user=1),
    "requests": [batchbench.request_entry(
        {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 4},
        line_idx=0, input_tokens=1)],
    "dry_run": True,
})
```

## Multi-turn sessions

`batchbench-agent` runs concurrent agent loops. Within a loop, each request appends
the model's reply and a synthetic tool result to the conversation and sends it again,
so every turn reuses the previous prompt as a prefix. Token shapes are fixed or
log-normal:

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

Each `--*-tokens` flag has a `--*-lognorm-median` / `--*-lognorm-sigma` /
`--*-lognorm-max` family; `--seed` makes sampling reproducible. `--tokenizer-model`
names the tokenizer when the endpoint's model name is not a Hugging Face identifier
(a model id, a `tokenizer.json`, or a directory containing one). Every request carries a
per-agent `user` field and routing header so a gateway can keep a session on one
worker; `--user-prefix`, `--disable-user-tagging`, and
`--dp-rank-perfect-routing` control that.

## Replaying a manifest

A manifest is a JSONL file with one trajectory per line: its requests, their prompt
and output token counts, the gaps between them, and optionally the prompt's content
blocks. The agent binary replays it instead of sampling:

```bash
batchbench-agent \
  --model Qwen/Qwen3-8B \
  --host http://localhost:8000 \
  --agent-plans-jsonl plans.jsonl \
  --admission open-loop \
  --time-scale 2 \
  --agent-events-jsonl events.jsonl \
  --results-csv results.csv
```

`--admission open-loop` starts each trajectory at its recorded offset; the default
closed-loop admission fills `--max-active-agents` slots in file order. `--time-scale N`
replays a recorded hour in `1/N` hours. Block text is generated from each block's seed
at exactly the requested token count, so blocks with equal seeds send identical bytes
and shared prefixes exercise prefix caching. `--dry-run` checks a manifest without
sending requests. The format is specified in [docs/manifest.md](docs/manifest.md);
`examples/trajectory-replay/` has one file per schema version.

## Building a manifest from records

`batchbench-export-plans` reads a window of prompt-chain records from ClickHouse,
rebuilds sessions from the chains, and writes a schema version 2 manifest:

```bash
uv pip install "batchbench[export]"

CLICKHOUSE_URL=https://warehouse.example.com:8443 CLICKHOUSE_USER=... CLICKHOUSE_PASSWORD=... \
batchbench-export-plans \
  --chains-table chains_current --analytics-table requests \
  --start 2026-09-01T09:00:00Z --end 2026-09-01T10:00:00Z \
  --model vendor/model-a \
  --sample 0.1 --seed 7 --stratify-by-session-length \
  --output plans.jsonl
```

The input tables it expects, and how sessions are reconstructed, are described in
[docs/export-plans.md](docs/export-plans.md).

## Report

Each run prints total prompt and completion tokens from the responses' `usage`,
estimated cached prompt tokens under perfect prefix caching, request latency p50/p90/p99,
and for agent runs the end-to-end session latency. `--results-csv` persists the same
figures; `--agent-events-jsonl` records per-trajectory admission and completion times.

`batchbench-harness` drives runs on remote GPU hosts; see [HARNESS.md](HARNESS.md).

## Building from source

```bash
cargo build --release --manifest-path rust/Cargo.toml --bin batchbench
cargo build --release --manifest-path rust/Cargo.toml --bin batchbench-agent
```

Releases are cut by Release Please; a `v*` tag builds platform wheels and publishes
them to PyPI.
