#!/usr/bin/env python3
"""Download WildChat dataset and convert to OpenAI Batch API format."""

import json
import random
import argparse
import uuid
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


def convert_conversation(conversation):
    """Remove trailing assistant message and format for OpenAI."""
    if conversation and conversation[-1].get("role") == "assistant":
        conversation = conversation[:-1]
    return [{"role": turn["role"], "content": turn["content"]} for turn in conversation]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="wildchat_50k.jsonl")
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="gpt-3.5-turbo-0125")
    parser.add_argument("--url", default="/v1/chat/completions")
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Load dataset and filter for English
    dataset = load_dataset("allenai/WildChat", split="train")
    english_indices = [i for i in range(len(dataset)) if dataset[i]["language"] == "English"]
    
    # Sample
    sampled = random.sample(english_indices, min(args.num_samples, len(english_indices)))
    
    # Convert to JSONL
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.output, "w", encoding="utf-8") as f:
        for idx in tqdm(sampled):
            messages = convert_conversation(dataset[idx]["conversation"])
            record = {
                "custom_id": str(uuid.uuid4()),
                "method": "POST",
                "url": args.url,
                "body": {"model": args.model, "messages": messages}
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()