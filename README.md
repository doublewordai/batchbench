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
not also a Hugging Face tokenizer identifier. It accepts a Hugging Face model ID,
a local `tokenizer.json` file, or a local directory containing `tokenizer.json`.

### Trajectory replay and rolling admission

Use `--agent-plans-jsonl` to replay complete empirical token shapes instead of
sampling initial prompts, outputs, environment results, and invocation counts
independently. Each non-empty JSONL line is one trajectory, and file order is the
FIFO admission order:

```json
{"schema_version":1,"trajectory_id":"agent-001","requests":[{"prompt_tokens":13381,"output_tokens":288},{"prompt_tokens":14800,"output_tokens":512,"delay_after_ms":25},{"prompt_tokens":7000,"output_tokens":128,"reset_before":true}]}
```

`prompt_tokens` and `output_tokens` are required and must be positive.
`delay_after_ms` defaults to zero. For a normal transition, BatchBench infers the
synthetic environment growth as:

```text
next prompt - current prompt - current output
```

If a trajectory compacts or resets such that this value would be negative, mark the
next request with `"reset_before": true`. BatchBench then starts that request from a
fresh synthetic prompt while preserving the trajectory's user/routing identity. A
reset deliberately claims no prefix-cache reuse from the preceding request.

Absolute prompt counts commonly include chat-template and tool-envelope tokens that
are not part of the synthetic message content. After calibrating those values against
the target backend, use `--replay-initial-overhead-tokens` for a first request or
reset, and `--replay-turn-overhead-tokens` for each normal appended turn. BatchBench
subtracts these values when generating content while retaining the manifest's
absolute prompt targets for reporting.

`--max-active-agents` bounds simultaneous trajectories while retaining every plan
in the manifest. BatchBench initially admits up to that limit and immediately
admits the next queued trajectory whenever any active trajectory terminates. The
replacement inherits the freed scheduler/routing slot, keeping active load balanced
across data-parallel ranks when perfect routing is enabled.

```bash
batchbench-agent \
  --model Qwen/Qwen3-8B \
  --host http://localhost:8000 \
  --agent-plans-jsonl examples/trajectory-replay/plans.jsonl \
  --max-active-agents 2 \
  --replay-initial-overhead-tokens 0 \
  --replay-turn-overhead-tokens 0 \
  --agent-events-jsonl agent-events.jsonl \
  --seed 42 \
  --dry-run
```

Do not combine `--agent-plans-jsonl` with `--agents` or the synthetic workload-shape
flags. `--seed` remains useful because it makes generated placeholder text
reproducible.

`--agent-events-jsonl` writes each trajectory's queue/admission time, reusable
routing slot, optional DP rank, finish time, runtime, and completion status. The
summary separately reports the final admission time and final drain duration.

Trajectory replay reproduces the joint request-count and nominal token-shape plan,
not the original text. Prompt targets include workload-specific framing, while
BatchBench generates its own chat/tool framing and exact-length synthetic message
content. Uncalibrated framing can therefore make live prompts drift systematically
from their targets over a long trajectory. Live `usage.prompt_tokens` is
authoritative and should be compared with the plan before a capacity result is
accepted. Likewise, exact output lengths require a backend that honors the requested
minimum/maximum token constraints.

Every request includes a `user` field and an `X-SMG-Routing-Key` header containing
the same UUID, which remains stable for the life of that agent and differs between
agents. Use `--disable-user-tagging` to omit both, or `--user-prefix <prefix>` to
use deterministic `<prefix>-<agent_id>` values instead of UUIDs.

For deterministic data-parallel routing, pass both
`--dp-rank-perfect-routing` and `--dp-rank-perfect-routing-num <ranks>`.
Each request then includes `X-SMG-Target-Worker: <routing_slot % ranks>`. Initial
agents occupy consecutive slots; under rolling admission, a replacement inherits
the slot freed by the trajectory that just terminated. This preserves the active
per-rank balance and is independent of user tagging.

Tool-call latency defaults to zero. Set a fixed delay with
`--tool-call-latency-ms`, or sample milliseconds independently for every invocation
with `--tool-call-latency-lognorm-median-ms` (or
`--tool-call-latency-lognorm-mu`), `--tool-call-latency-lognorm-sigma`, and the
optional `--tool-call-latency-lognorm-max-ms`. After a model response succeeds, the
agent sleeps for the sampled duration before making the environment result
available and submitting its next model request.

One tool invocation means one unconstrained model request followed by one synthetic
environment response. BatchBench treats generated assistant output as opaque state:
it preserves content and reasoning output but ignores any model-generated tool calls.
It then adds its own valid synthetic `environment` tool call and appends the sampled
environment response. This keeps the trajectory protocol-valid without assuming
anything about the generated output.

The final report includes:

- total input tokens sent, from successful responses' `usage.prompt_tokens`;
- total output tokens generated, from `usage.completion_tokens`;
- estimated cached input tokens under perfect prefix caching;
- total simulated tool-call latency across all agents;
- request-latency p50, p90, and p99 across successful requests;
- end-to-end p50, p90, and p99 across completed agents, measured from the start
  of each agent loop through its final synthetic tool delay.

Failed agents are excluded from the end-to-end latency distribution because their
lifetimes end early. Their request failures are still included in the failure report.

For each successful request after an agent's first, the cache estimate adds that
same agent's preceding prompt-token count (capped by the current prompt count).
Retries and failed requests are excluded because they do not provide reliable usage
data. Use `--results-csv <path>` to persist the same totals, or `--dry-run` to inspect
the sampled or replayed plans without sending requests.

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
