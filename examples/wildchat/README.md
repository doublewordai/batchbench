# WildChat Dataset Download

This directory contains a script to download and convert the WildChat dataset to OpenAI Batch API compliant JSONL format for use with batchbench.

## Overview

The WildChat dataset contains 650K+ real conversations between users and ChatGPT. This script downloads a random sample and converts it to the full OpenAI Batch API format, which includes `custom_id`, `method`, `url`, and `body` fields with the conversation messages.

## Requirements

```bash
pip install datasets tqdm
```

## Usage

### Basic usage (download 50k conversations):

```bash
python download_wildchat.py
```

This will create `wildchat_50k.jsonl` in the current directory.

### Custom options:

```bash
# Download 10k conversations
python download_wildchat.py --num-samples 10000 --output wildchat_10k.jsonl

# Use a different model identifier
python download_wildchat.py --num-samples 50000 --model "gpt-4-turbo"

# Use a custom endpoint URL
python download_wildchat.py --url "/v1/chat/completions" --model "claude-3-opus"

# Download only conversations with 2-5 turns
python download_wildchat.py --min-turns 2 --max-turns 5 --num-samples 10000

# Use different random seed
python download_wildchat.py --seed 123
```

### All options:

- `--output`: Output JSONL file path (default: `wildchat_50k.jsonl`)
- `--num-samples`: Number of samples to download (default: 50000)
- `--seed`: Random seed for reproducibility (default: 42)
- `--min-turns`: Minimum number of conversation turns (default: 1)
- `--max-turns`: Maximum number of conversation turns (default: None)
- `--model`: Model identifier to include in requests (default: `gpt-3.5-turbo-0125`)
- `--url`: API endpoint URL (default: `/v1/chat/completions`)

## Output Format

Each line in the output JSONL file follows the full OpenAI Batch API format:

```json
{
  "custom_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "method": "POST",
  "url": "/v1/chat/completions",
  "body": {
    "model": "gpt-3.5-turbo-0125",
    "messages": [
      {
        "role": "user",
        "content": "What is the capital of France?"
      },
      {
        "role": "assistant",
        "content": "The capital of France is Paris."
      }
    ]
  }
}
```

Each record contains:
- `custom_id`: A unique UUID identifier for the request
- `method`: HTTP method (always "POST")
- `url`: API endpoint path (default: `/v1/chat/completions`)
- `body`: The request body containing:
  - `model`: The model identifier
  - `messages`: List of message objects with `role` and `content` fields

## Dataset Information

- **Source**: [allenai/WildChat](https://huggingface.co/datasets/allenai/WildChat)
- **Size**: 529K conversations
- **Languages**: 66 languages
- **License**: ODC-BY

## Notes

- The dataset is downloaded from Hugging Face and cached locally on first run
- Subsequent runs will use the cached version
- The script filters out toxic content (dataset already cleaned)
- Random sampling ensures diverse conversations
