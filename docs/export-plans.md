# Exporting a manifest from prompt-chain records

`batchbench-export-plans` reads prompt-chain records from ClickHouse, or Dynamo
request traces from local files ([below](#dynamo-request-traces)), and writes a
[schema version 2 manifest](manifest.md). Reading ClickHouse needs the `export` extra:

```bash
uv pip install "batchbench[export]"
```

## Input

A prompt-chain record describes one chat-completions request without its content:
one cumulative hash per prompt block, with the block's role and token count. Because
the hashes are cumulative, two requests whose prompts share a prefix share a chain
prefix, which is enough to rebuild sessions.

`--chains-table` names a table or view with these columns:

| Column | Type | Meaning |
|---|---|---|
| `ts` | DateTime64 | When the request was recorded |
| `instance_id` | UUID | Gateway instance that served it |
| `correlation_id` | Int64 | Request id within that instance |
| `principal_id` | UUID | Whose request it was; sessions never cross principals |
| `model` | String | Model alias |
| `chain` | Array(FixedString(32)) | Cumulative hash after each block |
| `block_roles` | Array(String) | Role per block: `tool_definition`, `system`, `user`, `assistant`, `tool`, `tool_call` |
| `block_tokens` | Array(UInt32) | Untemplated token count per block |

`--analytics-table` names a per-request usage table joined on
`(instance_id, correlation_id)`, with `prompt_tokens`, `completion_tokens`, `stream`,
`max_tokens`, `finish_reason`, `served_by`, `request_origin`, `user_id`, `api_key_id`,
`duration_ms`, and `duration_to_first_byte_ms`. `--analytics-ts-column` (default
`timestamp`) is the column used to bound that side of the join; it is widened by a day
on each end of the window because a request's start time and its chain's capture time
differ.

If the chains table is a ReplacingMergeTree with duplicates, read it through a
deduplicating view, or pass `--chains-final` to read the base table with `FINAL`.

## Output

```bash
CLICKHOUSE_URL=https://warehouse.example.com:8443 CLICKHOUSE_USER=... CLICKHOUSE_PASSWORD=... \
batchbench-export-plans \
  --chains-table chains_current --analytics-table requests \
  --start 2026-09-01T09:00:00Z --end 2026-09-01T10:00:00Z \
  --model vendor/model-a \
  --sample 0.1 --seed 7 --stratify-by-session-length \
  --time-scale 4 \
  --output plans.jsonl
```

Sessions are reconstructed from the chains: within a principal, a request continues
the latest earlier request (within `--link-window-hours`, default 24) whose full chain
is a strict prefix of its own. Each root and its descendants form one trajectory in
timestamp order. The query starts one link window before `--start` so that sessions
which began earlier are recognised; their first in-window request carries
`reset_before`.

Per request, `prompt_tokens` and `output_tokens` come from the usage row,
`delay_after_ms` is the gap from this request's end to the next request's start,
`blocks` pairs each chain hash (as the seed) with its role and token count,
`overhead_tokens` is `prompt_tokens - sum(block_tokens)`, and the block that is the
reply to the previous request in the trajectory is marked `live`. `stream` and
`max_tokens` are copied when recorded. A request with no block counts becomes a single
`user` block of `prompt_tokens`. Requests without a positive prompt or completion
count are dropped and counted in the summary line.

Filters: `--principal-id`, `--model`, `--served-by`. `--sample` keeps a deterministic
fraction of trajectories (`--seed`); `--stratify-by-session-length` applies it within
power-of-two session-length buckets so short and long sessions are both kept.
`--time-scale` divides offsets and delays. `--dump-rows-jsonl` saves the fetched rows
and `--rows-jsonl` re-exports from such a file without ClickHouse. An export with no
trajectories is an error, and the manifest is validated before it is written.

Replay with:

```bash
batchbench-agent --agent-plans-jsonl plans.jsonl --admission open-loop --model <model> --host <host>
```

## Dynamo request traces

`--dynamo-traces` replaces the prompt-chain source with the `request_end` rows a Dynamo
frontend writes when request tracing is enabled (schema `dynamo.request.trace.v1`,
gzipped or plain JSONL). It takes a directory (read recursively for `*.jsonl.gz` and
`*.jsonl`), a file, or a glob, and may be repeated. No ClickHouse is needed:

```bash
batchbench-export-plans \
  --dynamo-traces ./traces \
  --model example-org/model-a \
  --start 2026-09-01T09:00:00Z --end 2026-09-01T10:00:00Z \
  --output plans.jsonl
```

Fields read from each row:

| Field | Use |
|---|---|
| `request.model` | Dynamo model name; `--model` matches it exactly |
| `request.request_received_ms` | Request time, used for the window, ordering and offsets |
| `request.total_time_ms` | Duration, for `delay_after_ms` |
| `request.input_tokens` | `prompt_tokens` (`replay.input_length` when absent or short of the hashed blocks) |
| `request.output_tokens` | `output_tokens`; rows without a positive count are dropped |
| `request.replay.trace_block_size` | Tokens per hashed block; varies by deployment |
| `request.replay.input_length` | Hashed prompt length |
| `request.replay.input_sequence_hashes` | Cumulative hash of each full block of the prompt, then of the partial last block when one remains |
| `request.request_id` | Dynamo's request id, kept in trajectory `metadata` |
| `request.x_request_id` | The caller's `x-request-id` header, when it sent one |
| `agent_context.session_id` | Agent session, when present |
| `request.finish_reason_metadata.finish_reason` | Picks among retried attempts |

There are `ceil(input_length / trace_block_size)` hashes. When the length is not a
multiple of the block size, the last hash covers the partial last block; it changes as
soon as a later turn fills that block, so it is split off and never used for linking or
cutting, only to seed the tail block. Rows that hash only the full blocks are read the
same way. A row with more hashes than `ceil(input_length / trace_block_size)`, or whose full
blocks hold more tokens than the prompt, is counted as `malformed_hashes`.

Trace files are named by upload time, and a file can hold rows received long before it
was written, so every file is read and rows are windowed by `request_received_ms`:
rows in `[--start - --link-window-hours, --end)` are loaded, and those before `--start`
only serve as link targets. Rows with the same `x_request_id` are attempts of one
request (a gateway retry or failover reuses the header); the last attempt that finished
is kept and the others are counted as `duplicate_attempts`.

Every loaded row stays in memory until the manifest is written, and a long prompt's hash
list is large, so a busy model over a long window can need more memory than `--sample`,
which thins trajectories after they are built, can save. `--sample-sessions FRACTION`
keeps that fraction of sessions instead, deciding as each row is read and before its
hashes are parsed; dropped rows count as `sampled_out`. A row is kept or dropped with its
agent session id, so sessions stay whole. A row without one goes with its `x_request_id`
(its retried attempts together), else its `request_id`; such a row can lose the earlier
request it extends, and then replays as the start of a new trajectory. Each key's draw comes
from `--seed`, so for one seed a smaller fraction keeps a subset of what a larger one keeps,
which makes manifests at several loads comparable.

### Sessions and blocks

Requests link with the same strict-prefix rule as prompt chains, applied to the block
hashes, and only within the same model, block size, principal and agent session id.
Rows with different block sizes are never linked or merged, even within one model.

Each prompt is cut after every hash where some chain of the same model and block size
ends or where two such chains continue differently. Each piece becomes one `user` block
whose seed is the piece's last hash and whose size is the block size times its hash
count, so a prefix shared by many requests (a system prompt, a common document) replays
with identical bytes everywhere. The tokens past the last full block
(`input_tokens - block_size * full blocks`) become a final `user` block, seeded by the
partial-block hash (so identical prompts share it) or, without one, by the request, so `sum(blocks.tokens) = prompt_tokens` and `overhead_tokens` is 0. A final block
rather than overhead keeps the replayed prompt the recorded length, which matters for
block sizes in the thousands.

A request that extends the previous request of its trajectory repeats the blocks
emitted for that request (its replies and tail keep their seeds and roles), and the
tokens that follow start with a `live` assistant block of the previous request's
`output_tokens`, then the rest as `user` blocks. A live block replays the whole real
reply, so when the extension holds fewer tokens than the previous output, the reply is
generated at the size that fits instead (split where segments end), with no live
block, and counted as `live_clamped`. Only the newest reply in a request is `live`: earlier replies
repeat as plain assistant blocks under the seeds they were sent with, and the agent
replays a seed it has already sent in the trajectory with the same bytes, a real reply
included. Any other request in a trajectory (a branch, which carries `reset_before`)
starts with the blocks already sent for the longest prefix it shares with an earlier
request of the trajectory. A reset clears the agent's cache, so the branch's generated
blocks come out the same from their seeds, but its copies of earlier replies are
generated rather than the model's real text. A live reply cannot be split, so a
branch that diverges inside one reuses the blocks only up to the reply's start. Each trajectory carries
`metadata` with the model name, block size and, when joined, the alias and principal.
Its `requests` list names the source row of each request, in request order:
`request_id`, `x_request_id` (lower-cased as for the join; null when the row has none)
and `received_ms` (the row's `request_received_ms`), so replay results can be joined
back to the trace rows.

### Joining analytics on `x_request_id`

With `--analytics-table` (and `--clickhouse-url`) the rows that carry an `x_request_id`
are joined to a gateway analytics table on `--analytics-id-column` (default
`x_request_id`). The query reads that table's id, `instance_id`, `correlation_id`,
`model`, `user_id` and `api_key_id`, bounded by `--analytics-ts-column` (default
`timestamp`) to the time span of the rows with an id, widened by 15 minutes on each
side. The id is compared as a string, case-insensitively. A matched row's `user_id`
(else its `api_key_id`) becomes the principal, which links sessions and is what
`--principal-id` filters on, and its `model` becomes the trajectory's `alias`. Principals come only from the join, so
`--principal-id` needs one.
Unmatched rows keep no principal and link by agent session id alone, if they have one.
`--dump-rows-jsonl` saves the analytics rows and `--analytics-rows-jsonl` joins against
such a file instead of ClickHouse:

```bash
CLICKHOUSE_URL=https://warehouse.example.com:8443 CLICKHOUSE_USER=... CLICKHOUSE_PASSWORD=... \
batchbench-export-plans \
  --dynamo-traces ./traces --model example-org/model-a \
  --analytics-table requests \
  --start 2026-09-01T09:00:00Z --end 2026-09-01T10:00:00Z \
  --output plans.jsonl
```

Besides the usual counts, the summary line reports, for traces: `files`,
`unreadable_files` (a truncated file keeps the rows before the damage), `lines`,
`bad_lines`, `not_request_end`, `no_replay`, `other_model`, `outside_window`,
`malformed_hashes`, `loaded`, `with_x_request_id`, `duplicate_attempts`,
`analytics_rows`, `analytics_matched`, `with_session_id`, `live_blocks`,
`live_clamped`, and `block_sizes` (rows loaded per block size).

### Fidelity lost against prompt chains

- **No roles.** Every block is `user` except the previous requests' replies, which are
  `assistant` (live, repeated, or generated): system prompts, tool results and tool
  calls are not told apart.
- **No tools array.** Tool definitions are part of the hashed prompt, so they replay as
  user text and the replayed request sends no `tools`.
- **Template overhead is merged and double counted.** Trace token counts include the
  chat template. The replay sends every block as its own message and the backend adds
  its template on top, so the replayed prompt exceeds `prompt_tokens` by the replay's
  per-message template cost, and the recorded per-message overhead cannot be separated
  from content.
- **Granularity is one trace block.** Cuts fall only on block boundaries, so a block
  straddles message boundaries, and the live reply starts where the previous prompt's
  tokens end rather than at a message boundary.
- **Large blocks hide divergence.** Two prompts that differ only inside their last
  partial block have the same full-block hashes. A turn that adds less than one block keeps its
  chain unchanged, so it is not a strict extension and starts a new trajectory; with
  block sizes in the thousands, many sessions fall apart into single requests. Without
  a principal or session id, a short request whose distinctive content sits in its tail
  can be linked to an unrelated longer request that shares its full blocks.
- **Reply length is overstated for reasoning models.** The reply is sized from the
  previous `output_tokens`, which for a reasoning model include the reasoning tokens.
  Clients usually do not send reasoning back, so the copy of the reply in the next
  prompt is shorter than that. The trace has no reasoning token count to subtract, so
  the reply block is oversized and eats into the new user or tool content that follows.
  When the reply does not fit in the extension at all, it is generated at the size that
  fits rather than replayed live. `live_clamped` in the summary counts those cases. When it is
  a large share of `live_clamped + live_blocks`, as is typical for a reasoning model,
  most replies are not replayed live and each turn's new content is undercounted.
- **No `stream` or `max_tokens`.** Traces do not record them, so they are omitted.
- **Timing starts at Dynamo.** `request_received_ms` is when the frontend received the
  request, after any gateway queueing.
