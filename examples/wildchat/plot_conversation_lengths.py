#!/usr/bin/env python3
"""
Plot histogram of conversation lengths from WildChat JSONL file.
Uses tokenization to measure conversation lengths in tokens.
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import tiktoken
TIKTOKEN_AVAILABLE = True


def calculate_conversation_length(text_field, tokenizer=None, use_tokens=True):
    """
    Calculate total length of a conversation.
    
    Args:
        text_field: Either a string or list of message dicts
        tokenizer: HuggingFace tokenizer instance (required if use_tokens=True)
        use_tokens: If True, use tokenizer; otherwise count characters
        
    Returns:
        Total token count (if use_tokens=True) or character count of all messages
    """
    if isinstance(text_field, str):
        if use_tokens and tokenizer:
            return len(tokenizer.encode(text_field, allowed_special={'<|endoftext|>'}))
        return len(text_field)
    elif isinstance(text_field, list):
        total = 0
        for message in text_field:
            if isinstance(message, dict) and "content" in message:
                content = message["content"]
                if use_tokens and tokenizer:
                    total += len(tokenizer.encode(content, allowed_special={'<|endoftext|>'}))
                else:
                    total += len(content)
        return total
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Plot histogram of conversation lengths"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Input JSONL file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image file (default: show plot interactively)"
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins (default: 50)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="cl100k_base",
        help="Tokenizer to use (default: cl100k_base for GPT-4). Options: gpt2, r50k_base, p50k_base, cl100k_base, o200k_base"
    )
    parser.add_argument(
        "--use-chars",
        action="store_true",
        help="Use character count instead of tokens"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return 1
    
    # Load tokenizer
    tokenizer = None
    use_tokens = not args.use_chars
    
    if use_tokens:
        if not TIKTOKEN_AVAILABLE:
            print("tiktoken not available, falling back to character count...")
            use_tokens = False
        else:
            print(f"Loading tokenizer: {args.tokenizer}...")
            try:
                # Try to get encoding by name or model
                try:
                    tokenizer = tiktoken.get_encoding(args.tokenizer)
                except KeyError:
                    tokenizer = tiktoken.encoding_for_model(args.tokenizer)
                print(f"Tokenizer loaded successfully!")
            except Exception as e:
                print(f"Error loading tokenizer: {e}")
                print("Falling back to character count...")
                use_tokens = False
    
    print(f"Reading {input_path}...")
    
    lengths = []
    turn_counts = []
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in tqdm(enumerate(f, 1)):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
                text_field = record.get("text")
                
                if text_field is not None:
                    length = calculate_conversation_length(text_field, tokenizer, use_tokens)
                    lengths.append(length)
                    
                    # Also track number of turns if available
                    if isinstance(text_field, list):
                        turn_counts.append(len(text_field))
                    elif "turn" in record:
                        turn_counts.append(record["turn"])
                        
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON on line {line_num}: {e}")
                continue
    
    if not lengths:
        print("Error: No valid conversations found")
        return 1
    
    unit = "tokens" if use_tokens else "characters"
    
    print(f"\nAnalyzed {len(lengths)} conversations")
    print(f"Statistics ({unit}):")
    print(f"  Min length: {min(lengths):,} {unit}")
    print(f"  Max length: {max(lengths):,} {unit}")
    print(f"  Mean length: {np.mean(lengths):,.1f} {unit}")
    print(f"  Median length: {np.median(lengths):,.1f} {unit}")
    print(f"  Std dev: {np.std(lengths):,.1f} {unit}")
    
    if turn_counts:
        print(f"\nTurn statistics:")
        print(f"  Min turns: {min(turn_counts)}")
        print(f"  Max turns: {max(turn_counts)}")
        print(f"  Mean turns: {np.mean(turn_counts):.1f}")
        print(f"  Median turns: {np.median(turn_counts):.1f}")
    
    # Create the histogram
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Histogram 1: Total conversation length
    ax1.hist(lengths, bins=args.bins, edgecolor='black', alpha=0.7)
    ax1.set_xlabel(f'Total Conversation Length ({unit})', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    title_suffix = f" ({args.tokenizer})" if use_tokens else ""
    ax1.set_title(f'Distribution of Conversation Lengths{title_suffix}\n({len(lengths)} conversations)', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'Mean: {np.mean(lengths):,.0f}\nMedian: {np.median(lengths):,.0f}\nStd: {np.std(lengths):,.0f}'
    ax1.text(0.97, 0.97, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=10)
    
    # Histogram 2: Number of turns (if available)
    if turn_counts:
        ax2.hist(turn_counts, bins=min(50, max(turn_counts)), edgecolor='black', alpha=0.7, color='orange')
        ax2.set_xlabel('Number of Turns', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Distribution of Conversation Turns', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        stats_text = f'Mean: {np.mean(turn_counts):.1f}\nMedian: {np.median(turn_counts):.1f}'
        ax2.text(0.97, 0.97, stats_text, transform=ax2.transAxes,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                 fontsize=10)
    else:
        ax2.text(0.5, 0.5, 'Turn count data not available', 
                transform=ax2.transAxes, ha='center', va='center',
                fontsize=14)
        ax2.set_xticks([])
        ax2.set_yticks([])
    
    plt.tight_layout()
    
    if args.output:
        output_path = Path(args.output)
        print(f"\nSaving plot to {output_path}...")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved successfully!")
    else:
        print("\nDisplaying plot...")
        plt.show()
    
    return 0


if __name__ == "__main__":
    exit(main())
