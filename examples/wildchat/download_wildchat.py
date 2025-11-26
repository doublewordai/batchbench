#!/usr/bin/env python3
"""
Download WildChat dataset and convert to JSONL format.

This script downloads a random sample of 50k conversations from the WildChat dataset
and converts them to JSONL format where each line contains a 'text' field with the
conversation history in the format expected by batchbench.

The 'text' field will be a list of maps with 'content' and 'role' fields.
"""

import json
import random
import argparse
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
    Convert WildChat conversation format to the expected format.
    
    Args:
        conversation: List of conversation turns from WildChat
        
    Returns:
        List of dicts with 'content' and 'role' fields
    """
    converted = []
    for turn in conversation:
        converted.append({
            "content": turn["content"],
            "role": turn["role"]
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
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    print(f"Loading WildChat dataset...")
    print("This may take a few minutes on first run (dataset will be cached)...")
    
    # Load the dataset
    dataset = load_dataset("allenai/WildChat", split="train")
    
    print(f"Total conversations in dataset: {len(dataset)}")
    
    # Sample random indices from entire dataset
    if len(dataset) > args.num_samples:
        sampled_indices = random.sample(range(len(dataset)), args.num_samples)
    else:
        print(f"Warning: Dataset only has {len(dataset)} conversations.")
        sampled_indices = list(range(len(dataset)))
    
    print(f"Sampling {len(sampled_indices)} conversations...")
    
    # Convert and write to JSONL
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Writing to {output_path}...")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for idx in tqdm(sampled_indices, desc="Converting conversations"):
            example = dataset[idx]
            
            # Convert conversation to expected format
            text_field = convert_conversation(example["conversation"])
            
            # Create output record
            record = {
                "text": text_field,
            }
            
            # Write as JSONL
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"\nDone! Wrote {len(sampled_indices)} conversations to {output_path}")
    print(f"\nExample record format:")
    
    # Show an example
    example = dataset[sampled_indices[0]]
    text_field = convert_conversation(example["conversation"])
    example_record = {
        "text": text_field[:2] if len(text_field) > 2 else text_field,  # Show first 2 turns only
        "conversation_id": example["conversation_id"],
        "model": example["model"],
        "turn": example["turn"],
        "language": example["language"]
    }
    print(json.dumps(example_record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
