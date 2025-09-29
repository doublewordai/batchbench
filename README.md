# Batch Benchmark Toolkit

This repository provides two utilities for benchmarking OpenAI-compatible services:

- `generate_requests.py` – creates prompt corpora in JSON Lines (JSONL) format with configurable prefix overlap and approximate token lengths using Hugging Face tokenizers.
- `locust_batch.py` – drives a Locust load test where each simulated user submits a single non-streaming chat completion and records end-to-end latency plus token usage to CSV.

## Requirements

- Python 3.9+
- [`uv`](https://github.com/astral-sh/uv) for lightweight virtualenv and lock-free installs (optional but recommended)
- `locust`
- `transformers` (only required when using token-aware prompt generation)
- Optional: tokenizer model weights cached via `huggingface_hub` or environment configuration suitable for offline tokenizers

### Installing Dependencies with `uv`

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

If you prefer `pip`, you can instead run:

```bash
pip install locust transformers
```

If you intend to benchmark an OpenAI-hosted endpoint, set `OPENAI_API_KEY` in your environment before running Locust. For fully offline testing you can omit it and target your own compatible service.

## Generating Prompts

Run the generator to produce a JSONL file containing one prompt per line inside a `text` field. Point `--output` at a directory (it defaults to `outputs`) and the script will append metadata to the filename automatically:

```bash
python generate_requests.py \
  --count 100 \
  --output data
```

Key options:

- `--prefix-overlap FLOAT` – share a fraction of prefix tokens across prompts (default `0.0`).
- `--approx-input-tokens INT` – approximate the number of input tokens per prompt. Requires `transformers` and a tokenizer.
- `--tokenizer-model NAME` – Hugging Face tokenizer identifier (defaults to `gpt2` when unspecified).
- `--token-tolerance INT` – allows fine-tuning the acceptable +/- range for token counts (defaults to `max(5, 5% of target)`).
- `--huggingface-token TOKEN` – supply a Hugging Face Hub access token (falls back to `HUGGINGFACE_TOKEN` / `HUGGING_FACE_HUB_TOKEN`).

Example: generate 200 prompts that share 30% of their prefix and target roughly 512 tokens using the `gpt-3.5-turbo` tokenizer:

```bash
python generate_requests.py \
  --count 200 \
  --prefix-overlap 0.3 \
  --approx-input-tokens 512 \
  --tokenizer-model gpt-3.5-turbo \
  --output data
```

> **Note:** The first time you reference a tokenizer, the weights will download from Hugging Face. Pre-cache them if you plan to run in an offline environment.

Generated filenames include the prompt count, target token size, prefix overlap, and tokenizer identifier. If a matching file already exists the generator aborts rather than overwriting it.

If the tokenizer repository requires authentication, either pass `--huggingface-token` or set `export HUGGINGFACE_TOKEN=your_pat` (or `HUGGING_FACE_HUB_TOKEN`).

## Running the Locust Benchmark

Locust consumes the JSONL prompts and launches one user per request so that every prompt is processed simultaneously. Each user:

1. Builds a chat completion payload with the configured model/system prompt.
2. Submits a single non-streaming request to the endpoint.
3. Captures prompt tokens, completion tokens, total tokens, and end-to-end latency.
4. Appends the metrics to a CSV file (default `results.csv`).
5. Contributes to a run-wide summary that reports total time, token throughput, and latency percentiles (p50/p90/p99) once all prompts finish.

### Environment Variables

- `REQUESTS_FILE` – path to the prompt JSONL file (default `requests.jsonl`).
- `OPENAI_ENDPOINT_PATH` – API path (default `/v1/chat/completions`). Combine with `--host` in Locust to set the base URL.
- `OPENAI_API_KEY` – Bearer token used in the `Authorization` header.
- `OPENAI_MODEL` – model name injected into each request (default `gpt-4o-mini`).
- `SYSTEM_PROMPT` – system role content (default `You are a helpful assistant.`).
- `RESULTS_CSV` – destination for the metrics file (default `results.csv`).

### Launching Locust Headlessly

1. Ensure the number of Locust users equals the number of prompts so all requests are issued at once.
2. Choose a spawn rate that ramps up quickly (often equal to the user count).

Example:

```bash
export REQUESTS_FILE=data/requests.jsonl
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-mini
export SYSTEM_PROMPT="You are a helpful assistant."
export RESULTS_CSV=out/results.csv

locust -f locust_batch.py --headless -u 200 -r 200 --host https://your-openai-compatible-host
```

Locust writes per-request metrics to the specified CSV while its standard statistics still surface in the console (or the web UI if you run without `--headless`). After the queue is drained, each simulated user stops automatically.

### Inspecting Results

The CSV file contains columns:

- `index` – zero-based prompt index.
- `prompt_tokens`, `completion_tokens`, `total_tokens` – token usage reported by the service.
- `latency_ms` – request latency in milliseconds.
- `status` – `success`, `http_error`, or `decode_error`.
- `error` – error detail when status is not `success`.

You can load the file into your preferred analysis tool, e.g. Pandas:

```python
import pandas as pd

df = pd.read_csv("out/results.csv")
print(df.describe())
```

## Tips

- For strictly offline execution, pre-download tokenizer assets and disable network access before running.
- When calibrating token counts, review a few prompts and use the same tokenizer locally to verify the lengths meet your expectations.
- Locust’s standard command-line options (like `--run-time`) remain available if you need to limit runtime or integrate into automated pipelines.
