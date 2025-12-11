# BatchBench

BatchBench bundles two benchmarking utilities behind installable Python entrypoints:

- `batchbench.generate` produces JSONL request corpora with controllable prefix overlap and approximate token counts.
- `batchbench.online` launches the packaged Rust binary that fans requests out to OpenAI-compatible endpoints in parallel.

## Installation

```bash
pip install batchbench
```

Optional extras install tool-specific dependencies:

```bash
pip install "batchbench[generate]"   # adds transformers for prompt sizing
```

## Generating Requests

```bash
batchbench.generate \
  --count 100 \
  --prefix-overlap 0.3 \
  --approx-input-tokens 512 \
  --tokenizer-model gpt-3.5-turbo \
  --output data
```

Each row in the resulting JSONL file has a `text` field. The filename embeds run metadata (count, tokens, prefix, tokenizer) to keep runs distinct.

## Online Benchmarking

`batchbench.online` wraps the Rust executable that used to live under `rust-bench/`. The binary ships inside the wheel, so Cargo is not required on the host.

```bash
batchbench.online \
  --jsonl data/requests.jsonl \
  --model gpt-4o-mini \
  --host https://api.openai.com \
  --endpoint /v1/chat/completions \
  --users 8 \
  --requests-per-user 1
```

Provide an API key via `--api-key` or the environment variable named by `--api-key-env` (defaults to `OPENAI_API_KEY`).

## Development Notes

The project now follows a `src/` layout. Run `pip install -e .[generate]` during development to work against the editable package. The Rust binary can be rebuilt with `cargo build --release` inside `rust-bench/`; copy the resulting executable to `src/batchbench/bin/` if you need to refresh it.
