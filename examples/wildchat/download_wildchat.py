#!/usr/bin/env python3
"""
Download WildChat dataset and convert to OpenAI Batch API compliant JSONL format.

This script downloads a random sample of 50k conversations from the WildChat dataset
and converts them to the full OpenAI Batch API format with custom_id, method, url, and body.

Each record includes:
- custom_id: A unique UUID for the request
- method: POST
- url: /v1/chat/completions
- body: Contains the model and messages
"""

import json
import random
import argparse
import uuid
from pathlib import Path
from typing import List, Dict, Any

try:
    from datasets import load_dataset
    from tqdm import tqdm
except ImportError:
    print("Error: Required packages not installed.")
    print("Please install: pip install datasets tqdm")
    exit(1)


def convert_conversation(conversation: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Convert WildChat conversation format to OpenAI Batch API messages format.
    
    Args:
        conversation: List of conversation turns from WildChat
        
    Returns:
        List of dicts with 'role' and 'content' fields (OpenAI format)
    """
    # Ignore the last message if it's from the assistant
    if conversation and conversation[-1].get("role") == "assistant":
        conversation = conversation[:-1]
    
    converted = []
    for turn in conversation:
        converted.append({
            "role": turn["role"],
            "content": turn["content"]
        })
    return converted


def main():
    parser = argparse.ArgumentParser(
        description="Download and convert WildChat dataset to JSONL format"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="wildchat_50k.jsonl",
        help="Output JSONL file path (default: wildchat_50k.jsonl)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50000,
        help="Number of samples to download (default: 50000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)"
    )
    parser.add_argument(
        "--min-turns",
        type=int,
        default=1,
        help="Minimum number of turns in conversation (default: 1)"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Maximum number of turns in conversation (default: None)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo-0125",
        help="Model identifier to include in output records (default: gpt-3.5-turbo-0125)"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="/v1/chat/completions",
        help="API endpoint URL (default: /v1/chat/completions)"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    print(f"Loading WildChat dataset...")
    print("This may take a few minutes on first run (dataset will be cached)...")
    
    # Load the dataset
    dataset = load_dataset("allenai/WildChat", split="train")
    
    print(f"Total conversations in dataset: {len(dataset)}")
    
    # Filter for English language conversations
    print("Filtering for English language conversations...")
    english_indices = [i for i in range(len(dataset)) if dataset[i]["language"] == "English"]
    print(f"Found {len(english_indices)} English conversations")
    
    # Sample random indices from English conversations
    if len(english_indices) > args.num_samples:
        sampled_indices = random.sample(english_indices, args.num_samples)
    else:
        print(f"Warning: Only {len(english_indices)} English conversations available.")
        sampled_indices = english_indices
    
    print(f"Sampling {len(sampled_indices)} English conversations...")
    
    # Convert and write to JSONL
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Writing to {output_path}...")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for idx in tqdm(sampled_indices, desc="Converting English conversations"):
            example = dataset[idx]
            
            # Convert conversation to OpenAI Batch API format
            messages = convert_conversation(example["conversation"])
            
            # Create output record in full OpenAI Batch API format
            record = {
                "custom_id": str(uuid.uuid4()),
                "method": "POST",
                "url": args.url,
                "body": {
                    "model": args.model,
                    "messages": messages
                }
            }
            
            # Write as JSONL
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"\nDone! Wrote {len(sampled_indices)} conversations to {output_path}")
    print(f"\nExample record format:")
    
    # Show an example
    example = dataset[sampled_indices[0]]
    messages = convert_conversation(example["conversation"])
    example_record = {
        "custom_id": str(uuid.uuid4()),
        "method": "POST",
        "url": args.url,
        "body": {
            "model": args.model,
            "messages": messages[:2] if len(messages) > 2 else messages  # Show first 2 turns only
        }
    }
    
    # Add metadata for display purposes only (not in actual output)
    example_metadata = {
        "conversation_id": example["conversation_id"],
        "original_model": example["model"],
        "turn": example["turn"],
        "language": example["language"]
    }
    
    print("Output record:")
    print(json.dumps(example_record, indent=2, ensure_ascii=False))
    print("\nOriginal metadata (for reference only):")
    print(json.dumps(example_metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
