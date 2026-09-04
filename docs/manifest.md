# Replay manifest format

A manifest is a JSONL file. Each non-empty line is one trajectory: a sequence of
requests sent one after another on the same conversation. File order is the admission
order under closed-loop admission. Lines of both schema versions may be mixed in one
file. Unknown fields are rejected.

## Schema version 1: token counts only

```json
{"schema_version":1,"trajectory_id":"agent-001","requests":[
  {"prompt_tokens":13381,"output_tokens":288},
  {"prompt_tokens":14800,"output_tokens":512,"delay_after_ms":25},
  {"prompt_tokens":7000,"output_tokens":128,"reset_before":true}]}
```

| Field | Meaning |
|---|---|
| `prompt_tokens` | Prompt length the request should reach. Required, positive. |
| `output_tokens` | Completion length to request. Required, positive. |
| `delay_after_ms` | Pause before the next request. Default 0. |
| `reset_before` | Start this request from a fresh conversation instead of appending. |

Between two appended requests the synthetic tool result is sized so the conversation
grows to the next prompt target: `next prompt - current prompt - current output`. If
that would be negative, mark the next request `reset_before`. A reset claims no
prefix-cache reuse from the previous request.

Prompt counts from a real backend include chat-template and tool-envelope tokens that
are not message content. `--replay-initial-overhead-tokens` (first or reset request)
and `--replay-turn-overhead-tokens` (each appended turn) are subtracted when generating
content; the manifest's absolute counts remain the reporting target.

## Schema version 2: content blocks and start offsets

```json
{"schema_version":2,"trajectory_id":"session-7f3a-0","start_after_ms":1250,"requests":[
  {"prompt_tokens":1340,"output_tokens":96,"overhead_tokens":41,"stream":true,"max_tokens":512,"delay_after_ms":830,
   "blocks":[{"seed":"9c1d…","tokens":611,"role":"tool_definition"},
             {"seed":"2b77…","tokens":420,"role":"system"},
             {"seed":"e04a…","tokens":268,"role":"user"}]},
  {"prompt_tokens":1612,"output_tokens":40,"overhead_tokens":52,"stream":true,"max_tokens":512,
   "blocks":[{"seed":"9c1d…","tokens":611,"role":"tool_definition"},
             {"seed":"2b77…","tokens":420,"role":"system"},
             {"seed":"e04a…","tokens":268,"role":"user"},
             {"seed":"51f0…","tokens":96,"role":"assistant","live":true},
             {"seed":"c9aa…","tokens":80,"role":"tool_call"},
             {"seed":"77d2…","tokens":55,"role":"tool"}]}]}
```

Version 2 adds, per trajectory, `start_after_ms`, and per request:

| Field | Meaning |
|---|---|
| `blocks` | Ordered content blocks that make up the prompt. |
| `overhead_tokens` | Template and envelope tokens the backend adds on top of the content. Overrides the global overhead flags for this request. |
| `stream` | Send `"stream": true` with usage in the final chunk. |
| `max_tokens` | Output cap; `output_tokens` remains the floor, clamped to the cap. |

### Blocks

Each block is `{seed, tokens, role}` plus an optional `live` flag. `role` is one of
`tool_definition`, `system`, `user`, `assistant`, `tool`, `tool_call`.

- Text is generated from `seed` alone at exactly `tokens` tokens, so equal seeds
  produce identical bytes across trajectories, requests, and runs. Two sessions with
  the same system prompt replay with the same system-prompt bytes. Within a trajectory
  a seed that was already sent reuses the bytes already sent. Generation fails
  loudly if a block cannot be constructed at the requested count.
- `tool_definition` blocks become entries in the request's `tools` array. `system`,
  `user`, `assistant`, and `tool` blocks become messages of that role. A `tool_call`
  block becomes a `tool_calls` entry on the preceding assistant message, and the
  following `tool` block references its id.
- An `assistant` block with `"live": true` is the model's own reply from the previous
  request in this trajectory, and replays with the text the model actually returned.
  When no previous reply exists (first request, second live block in one request,
  request after a reset) the block is generated from its seed and counted as a live
  block fallback in the report.

`sum(blocks.tokens) + overhead_tokens` should equal `prompt_tokens`. A mismatch is
logged once per trajectory and the blocks are replayed as written. `reset_before` on
the first request marks a session whose earlier turns predate the exported window.

### Admission

`--admission closed-loop` (default) ignores `start_after_ms` and admits trajectories
in file order into `--max-active-agents` slots; a finishing trajectory frees its slot,
and the next one inherits it, which keeps per-rank balance stable under perfect
routing.

`--admission open-loop` admits every trajectory at `start_after_ms` regardless of free
slots. `--max-active-agents` then becomes a hard cap; a trajectory held back by it is
counted as a late admission, and the report's maximum admission lag shows how far any
admission slipped behind its schedule. `--time-scale N` divides every
`start_after_ms` and `delay_after_ms`.

The first requests of the earliest trajectories are generated before the clock starts;
later ones are generated ahead of their offsets, and the current tool result and the
next reset prompt are generated while the previous request is in flight.

### Fidelity

Replay reproduces the request count, the token shape of every prompt and completion,
prefix sharing, and arrival timing. It does not reproduce the original text: block
content is placeholder text, so the model's replies, tokenization of real content,
and content-dependent effects such as expert routing are not reproduced. Compare live
`usage.prompt_tokens` against the plan before accepting a capacity result, and use a
backend that honours the requested minimum and maximum output tokens.
