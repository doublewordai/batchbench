# Exporting a manifest from prompt-chain records

`batchbench-export-plans` reads prompt-chain records from ClickHouse and writes a
[schema version 2 manifest](manifest.md). It needs the `export` extra:

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
