# WildChat Dataset Download

This directory contains a script to download and convert the WildChat dataset to JSONL format for use with batchbench.

## Overview

The WildChat dataset contains 650K+ real conversations between users and ChatGPT. This script downloads a random sample and converts it to the format expected by batchbench, where each conversation's history is stored as a list of messages with `content` and `role` fields.

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

## Output Format

Each line in the output JSONL file contains:

```json
{
  "text": [
    {
      "content": "What is the capital of France?",
      "role": "user"
    },
    {
      "content": "The capital of France is Paris.",
      "role": "assistant"
    }
  ],
  "conversation_id": "abc123...",
  "model": "gpt-4",
  "turn": 1,
  "language": "English"
}
```

The `text` field contains the conversation history as a list of messages, where each message has:
- `content`: The message content
- `role`: Either "user" or "assistant"

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
