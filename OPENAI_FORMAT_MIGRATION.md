# Converting to OpenAI Batch API Format

This document explains how to convert existing JSONL files and generate new ones in the OpenAI Batch API compliant format.

## New Format

The JSONL files now follow the OpenAI Batch API format:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Your prompt text here..."
    }
  ],
  "model": "Qwen/Qwen3-30B-A3B-FP8"
}
```

## Converting Existing Files

Use the `convert_to_openai_format.py` script to convert old format files:

```bash
# Convert a single file
python convert_to_openai_format.py data/requests_count-200_tokens-512_prefix-0p00_tokenizer-Qwen-Qwen3-0.6B.jsonl \
    --model "Qwen/Qwen3-30B-A3B-FP8" \
    --output data/requests_count-200_tokens-512_prefix-0p00_tokenizer-Qwen-Qwen3-0.6B_openai.jsonl

# Convert all JSONL files in a directory
python convert_to_openai_format.py data/ \
    --model "Qwen/Qwen3-30B-A3B-FP8" \
    --recursive \
    --output data_openai/
```

### Options

- `-m, --model`: Model identifier to include in the output (optional but recommended)
- `-o, --output`: Output file or directory (defaults to input with `_openai` suffix)
- `--overwrite`: Overwrite existing output files
- `-r, --recursive`: Process all `.jsonl` files in a directory recursively

## Generating New Files

When generating new JSONL files with `generate.py`, you can optionally include the model field:

```bash
python -m batchbench.generate \
    --count 200 \
    --approx-input-tokens 512 \
    --tokenizer-model "Qwen/Qwen3-0.6B" \
    --model "Qwen/Qwen3-30B-A3B-FP8" \
    --output data/
```

The `--model` parameter is optional. If omitted, the JSONL will only contain the `messages` field.

## Backward Compatibility

The Rust benchmark tool (`batchbench`) now supports both formats:

1. **New OpenAI format** (preferred): `{"messages": [...], "model": "..."}`
2. **Legacy format**: `{"text": "..."}`

This means:
- Old JSONL files will continue to work without conversion
- The `--model` CLI argument is used as a fallback if the JSONL doesn't specify a model
- If the JSONL contains a `model` field, it takes precedence over the CLI argument

## Examples

### Example: Convert all data files

```bash
python convert_to_openai_format.py data/ \
    --model "Qwen/Qwen3-30B-A3B-FP8" \
    --recursive \
    --output data/ \
    --overwrite
```

This will convert all `.jsonl` files in the `data/` directory in place.

### Example: Generate new OpenAI-compliant files

```bash
# Generate with model field
python -m batchbench.generate \
    --count 1024 \
    --approx-input-tokens 1024 \
    --prefix-overlap 0.15 \
    --tokenizer-model "Qwen/Qwen3-0.6B" \
    --model "Qwen/Qwen3-30B-A3B-FP8" \
    --output data/

# Generate without model field (will be added by batchbench CLI)
python -m batchbench.generate \
    --count 200 \
    --approx-input-tokens 512 \
    --tokenizer-model "Qwen/Qwen3-0.6B" \
    --output data/
```

### Example: Run benchmark with new format

```bash
cargo run --release --bin batchbench -- \
    --jsonl data/requests_count-200_tokens-512_prefix-0p00_tokenizer-Qwen-Qwen3-0.6B_openai.jsonl \
    --model "Qwen/Qwen3-30B-A3B-FP8" \
    --host "https://api.example.com" \
    --endpoint "/v1/chat/completions" \
    --users 10 \
    --requests-per-user 5
```

If the JSONL file contains a `model` field, it will be used automatically. The `--model` CLI argument serves as a fallback.
