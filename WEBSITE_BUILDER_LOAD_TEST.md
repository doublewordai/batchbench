# Website-builder sustained-load extension

This branch extends `batchbench-agent` for large-context, OpenAI-compatible
website-builder workloads. It was built to exercise persistent agent
conversations under a controlled logical-token schedule while retaining
request-level streaming metrics.

## Added capabilities

- `--system-prompt-file`: prepends a fixed system message to every agent
  conversation, producing a stable shared prefix.
- Twenty OpenAI function-tool definitions representing common website-builder
  environment operations.
- Stateful multi-turn histories that preserve assistant messages and synthetic
  tool results per agent.
- Stable per-agent `user` values and routing headers (existing behavior), with
  deterministic values available through `--user-prefix`.
- `--stream`: requests OpenAI-compatible SSE responses with
  `stream_options: {"include_usage": true}` and parses content, reasoning,
  final usage, and `[DONE]` events.
- `--duration-secs`, `--target-logical-tpm`, and
  `--logical-tokens-per-request`: pace request starts at a specified logical
  token rate for a fixed duration.
- `--requests-csv`: emits one row per request, including latency, TTFT, stream
  delivery duration, and client-observed output delivery TPS.

TTFT is measured from request start to the first non-empty content or reasoning
delta. Output delivery TPS is `completion_tokens / (last_delta - first_delta)`.
Because proxies may buffer SSE events, this is a client-observed delivery metric,
not guaranteed backend decode TPS.

## Build and test

```bash
cargo fmt --manifest-path rust/Cargo.toml --check
cargo test --manifest-path rust/Cargo.toml
cargo build --release --manifest-path rust/Cargo.toml --bin batchbench-agent
```

## Generate the fixed system prompt

The included helper uses a local `tokenizer.json` and writes a deterministic
34,400-token website-builder prompt:

```bash
cargo run --manifest-path rust/Cargo.toml \
  --bin generate_website_prompt -- \
  /path/to/tokenizer.json \
  /tmp/website_builder_system_prompt.txt \
  34400
```

The prompt is synthetic and contains no customer prompt or customer identity.

## Example sustained worker

The command below runs one 60-agent worker at 6M logical TPM. Two workers run
concurrently produce the 120-agent, 12M logical-TPM workload.

```bash
OPENAI_API_KEY=... ./rust/target/release/batchbench-agent \
  --model MODEL_ID \
  --host https://openai-compatible.example.com \
  --agents 60 \
  --system-prompt-file /tmp/website_builder_system_prompt.txt \
  --tokenizer-model /path/to/tokenizer.json \
  --input-lognorm-median 22000 \
  --input-lognorm-sigma 0.58 \
  --input-lognorm-max 70000 \
  --output-tokens 1360 \
  --environment-lognorm-median 370 \
  --environment-lognorm-sigma 1.93 \
  --environment-lognorm-max 135000 \
  --tool-invocations 10000 \
  --tool-call-latency-lognorm-median-ms 1400 \
  --tool-call-latency-lognorm-sigma 2.0 \
  --tool-call-latency-lognorm-max-ms 120000 \
  --duration-secs 600 \
  --target-logical-tpm 6000000 \
  --logical-tokens-per-request 80000 \
  --request-timeout-secs 180 \
  --max-retries 0 \
  --stream \
  --user-prefix readiness-worker-a \
  --results-csv /tmp/worker-a-summary.csv \
  --requests-csv /tmp/worker-a-requests.csv
```

Start the two workers behind a common barrier if the scored window must begin at
full rate. Give the second worker a distinct `--user-prefix` and output paths.

## Workload interpretation

`--target-logical-tpm` schedules request starts using
`--logical-tokens-per-request`; it does not predict the endpoint's returned usage.
Use the final usage events and request CSV to calculate actual logical TPM. For
capacity analysis, report cached input separately from uncached input and output.
