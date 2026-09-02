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
in the manifest. BatchBench initially admits up to that limit. Whenever an active
trajectory terminates, the next queued trajectory prepares its initial prompt and
then inherits the freed scheduler/routing slot. Multiple replacements prepare
concurrently through the same bounded worker pool while preserving FIFO admission
order, keeping active load balanced across data-parallel ranks when perfect routing
is enabled.

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
reproducible. Generated content is accepted only when it re-encodes to the requested
token count; an unconstructable field fails with its trajectory context instead of
silently changing the workload.

Replay manifests remain compact in memory: BatchBench materializes the initially
active prompts before the benchmark clock starts, then generates only the current
environment response and the next reset prompt while the current HTTP request is in
flight. Token generation uses a bounded worker pool, and its per-agent/per-turn random
streams are independent of asynchronous scheduling. Request JSON is serialized from
the live conversation without first cloning the complete message tree.

`--agent-events-jsonl` writes each trajectory's scheduled and actual admission time,
queue wait, reusable routing slot, optional DP rank, finish time, runtime, and
completion status. The summary separately reports the final admission time and final
drain duration.

### Schema version 2: content blocks and open-loop admission

Schema version 1 manifests keep working unchanged. Version 2 adds per-request
content structure, per-request overrides, and a start offset per trajectory. Lines
of both versions may be mixed in one file.

```json
{"schema_version":2,"trajectory_id":"session-7f3a-0","start_after_ms":1250,"requests":[
  {"prompt_tokens":1340,"output_tokens":96,"overhead_tokens":41,"stream":true,"max_tokens":512,"delay_after_ms":830,
   "blocks":[{"seed":"9c1d…","tokens":611,"role":"tool_definition"},{"seed":"2b77…","tokens":420,"role":"system"},{"seed":"e04a…","tokens":268,"role":"user"}]},
  {"prompt_tokens":1612,"output_tokens":40,"overhead_tokens":52,"stream":true,"max_tokens":512,
   "blocks":[{"seed":"9c1d…","tokens":611,"role":"tool_definition"},{"seed":"2b77…","tokens":420,"role":"system"},{"seed":"e04a…","tokens":268,"role":"user"},
             {"seed":"51f0…","tokens":96,"role":"assistant","live":true},{"seed":"c9aa…","tokens":80,"role":"tool_call"},{"seed":"77d2…","tokens":55,"role":"tool"}]}]}
```

(Shown wrapped for readability; each trajectory is one JSONL line. See
`examples/trajectory-replay/plans-v2.jsonl` for two trajectories that share their
tool-definition and system blocks.)

`blocks` describes the prompt as ordered content blocks. `role` is one of
`tool_definition`, `system`, `user`, `assistant`, `tool`, and `tool_call`.
Tool definitions become entries in the request's `tools` array (synthetic function
schemas whose serialized JSON re-encodes to `tokens`); `system`, `user`, `assistant`,
and `tool` blocks become messages of that role; a `tool_call` block becomes a
synthetic `tool_calls` entry on the preceding assistant message (or on a new
assistant message when none precedes it), and the next `tool` block references its
id. Block text is generated from the seed alone, at exactly `tokens` tokens, so
equal seeds produce identical bytes in every trajectory, request, and run. Shared
prefixes across sessions (the same system prompt, the same tool set) therefore
replay as identical bytes and exercise cross-session prefix caching. Within a
trajectory, a seed that was already sent reuses the text already sent.

A block with `"live": true` and role `assistant` is the model's own previous reply in
this conversation: BatchBench substitutes the assistant message returned by the
previous request instead of generating text. When there is no previous reply (the
first request of a trajectory, a second live block in the same request, or a
request after a reset), the block is generated from its seed and counted in the
report as a live block fallback.

When `blocks` is present the environment-growth inference of version 1 is not used:
each request's blocks define its prompt, so there is no negative-growth case.
`reset_before` still starts a fresh conversation (the per-trajectory block cache is
dropped) and, in version 2, is accepted on the first request to mark a session
whose earlier turns predate the exported window. `sum(blocks.tokens)` is the
content target and `prompt_tokens` remains the reporting target; they should agree
with `prompt_tokens == sum(blocks.tokens) + overhead_tokens`. A mismatch is logged
once per trajectory and the blocks are replayed as written.

`overhead_tokens` overrides `--replay-initial-overhead-tokens` /
`--replay-turn-overhead-tokens` for that request: it is the number of chat-template
and tool-envelope tokens the backend adds on top of the content, so the content
target is `prompt_tokens - overhead_tokens`. Requests without it fall back to the
global flags (initial overhead for a first or reset request, plus the turn overhead
for each appended turn).

`stream` sends `"stream": true` with `stream_options.include_usage` and consumes the
server-sent events into the assistant message. `max_tokens` sets the output cap the
same way the run-level output-token fields do (honouring `--sglang`); the planned
`output_tokens` remains the floor, clamped to the cap.

`start_after_ms` is the trajectory's start offset from benchmark start. It is used by
`--admission open-loop`, which admits every trajectory at its offset regardless of
free slots; `--max-active-agents` then becomes a hard cap that delays admission
when reached, and each delayed trajectory is counted as a late admission. The
default `--admission closed-loop` is the existing behaviour (manifest-order
admission into `--max-active-agents` slots), where offsets are ignored.
`--time-scale <factor>` divides every `start_after_ms` and `delay_after_ms` value,
so `--time-scale 4` replays a recorded hour in fifteen minutes. Under open-loop
admission the first requests of the earliest trajectories are prepared before the
clock starts, and later trajectories are prepared ahead of their offsets through a
bounded lookahead; the report's maximum admission lag shows how far any admission
slipped behind its schedule for any reason (cap or preparation).

```bash
batchbench-agent \
  --model Qwen/Qwen3-8B \
  --host http://localhost:8000 \
  --agent-plans-jsonl examples/trajectory-replay/plans-v2.jsonl \
  --admission open-loop \
  --time-scale 2 \
  --max-active-agents 64 \
  --agent-events-jsonl agent-events.jsonl \
  --dry-run
```

The summary, `--results-csv`, and `--agent-events-jsonl` gain `live_block_fallbacks`,
`late_admissions`, `max_admission_lag_ms`, `admission_mode`, `time_scale`, and the
per-trajectory `scheduled_at_seconds`. Under open-loop admission the reported
maximum active agents is the peak concurrency actually observed. Unknown fields are
rejected in both schema versions, and version 1 lines reject the version 2 fields.

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

## Exporting plans from production records

`batchbench export-plans` (also `python -m batchbench.export_plans`) turns a window of
ClickHouse prompt-chain records into a schema version 2 manifest. Each
`clay.prompt_chains` row is one chat-completions request as a chain of keyed content
hashes, one per prompt block, joined with `clay.http_analytics` on
`(instance_id, correlation_id)` for its token counts and request scalars. The chains
are read through the deduplicating `clay.prompt_chains_current` view by default
(`--chains-table` / `--chains-final` select the base table with `FINAL` instead), and
both sides of the join are bounded to the window before joining. Install the
ClickHouse client with `uv pip install "batchbench[export]"`.

```bash
CLICKHOUSE_URL=https://user:password@warehouse.example.com:8443 \
batchbench export-plans \
  --start 2026-09-01T09:00:00Z --end 2026-09-01T10:00:00Z \
  --model vendor/model-a \
  --sample 0.1 --seed 7 --stratify-by-session-length \
  --time-scale 4 \
  --output plans.jsonl
```

Sessions are reconstructed from the chains themselves: within a principal, a request
continues the latest earlier request (within `--link-window-hours`, default 24) whose
full chain is a strict prefix of its own chain, and each root with its descendants
forms one trajectory ordered by timestamp. The query starts one link window before
`--start` so that sessions which began earlier are recognised; their first in-window
request carries `reset_before`. Per request, `prompt_tokens` and `output_tokens` come
from the analytics row, `delay_after_ms` is the gap between this request's end and
the next request's start, `blocks` zips the chain hashes (as seeds) with the block
roles and token counts, `overhead_tokens` is `prompt_tokens - sum(block_tokens)`, and
the block that is the reply to the previous request in the trajectory is marked
`live`. Requests whose block token counts are missing become a single `user` block of
`prompt_tokens`. `stream` and `max_tokens` are copied when recorded. Requests without
a positive prompt or completion count are dropped and counted in the summary.

`--sample` keeps a deterministic fraction of trajectories (seeded by `--seed`);
`--stratify-by-session-length` applies the fraction within power-of-two
session-length buckets so short and long sessions are both represented.
`--time-scale` divides the start offsets and delays. Filters: `--principal-id`,
`--model`, `--served-by`. `--dump-rows-jsonl` saves the fetched rows and
`--rows-jsonl` re-exports from such a file without ClickHouse. An export with no
trajectories is an error, and the output is validated against the schema version 2
rules before it is written. It replays with:

```bash
batchbench-agent --agent-plans-jsonl plans.jsonl --admission open-loop --model <model> --host <host>
```

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
