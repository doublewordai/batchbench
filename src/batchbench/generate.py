"""Generate batchbench request payloads."""

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer

def print_histogram(sequence_lengths, num_bins=20):
    """Print an ASCII histogram of token length distribution."""
    if not sequence_lengths:
        return
    
    min_len = min(sequence_lengths)
    max_len = max(sequence_lengths)
    median_len = sorted(sequence_lengths)[len(sequence_lengths) // 2]
    mean_len = sum(sequence_lengths) / len(sequence_lengths)
    
    # Calculate percentiles
    sorted_lens = sorted(sequence_lengths)
    p50 = sorted_lens[int(len(sorted_lens) * 0.50)]
    p95 = sorted_lens[int(len(sorted_lens) * 0.95)]
    p99 = sorted_lens[int(len(sorted_lens) * 0.99)] if len(sorted_lens) > 100 else sorted_lens[-1]
    
    # Create bins
    bin_width = (max_len - min_len) / num_bins
    if bin_width == 0:
        bin_width = 1
    
    bins = [0] * num_bins
    for length in sequence_lengths:
        bin_idx = min(int((length - min_len) / bin_width), num_bins - 1)
        bins[bin_idx] += 1
    
    # Print statistics
    print("\n" + "="*70, file=sys.stderr)
    print("Token Length Distribution Statistics:", file=sys.stderr)
    print("="*70, file=sys.stderr)
    print(f"  Count:  {len(sequence_lengths):,}", file=sys.stderr)
    print(f"  Min:    {min_len:,} tokens", file=sys.stderr)
    print(f"  Mean:   {mean_len:,.1f} tokens", file=sys.stderr)
    print(f"  Median: {median_len:,} tokens", file=sys.stderr)
    print(f"  P95:    {p95:,} tokens", file=sys.stderr)
    print(f"  P99:    {p99:,} tokens", file=sys.stderr)
    print(f"  Max:    {max_len:,} tokens", file=sys.stderr)
    print("="*70, file=sys.stderr)
    
    # Print histogram
    max_count = max(bins) if bins else 1
    bar_width = 50
    
    print("\nHistogram:", file=sys.stderr)
    for i, count in enumerate(bins):
        bin_start = min_len + i * bin_width
        bin_end = bin_start + bin_width
        bar_length = int((count / max_count) * bar_width) if max_count > 0 else 0
        bar = "█" * bar_length
        print(f"  {bin_start:>7.0f}-{bin_end:>7.0f} │{bar} {count}", file=sys.stderr)
    print("="*70 + "\n", file=sys.stderr)

def assemble_prompts(
    count,
    prefix_overlap,
    target_tokens,
    tokenizer,
    tolerance,
    dist_mode,
    dist_median,
    dist_sigma,
    dist_max,
    seed,
):
    """Create prompts with specified prefix overlap."""
    rng = random.Random(seed)
    sequence_lengths = []
    
    for _ in tqdm(range(count), desc="Sampling sequence lengths"):
        if dist_mode == "lognormal":
            mu = math.log(dist_median)
            length = max(1, int(round(rng.lognormvariate(mu, dist_sigma))))
            length = min(length, dist_max)
        else:
            lower = max(1, target_tokens - tolerance)
            upper = target_tokens + tolerance
            length = rng.randint(lower, upper)
        
        sequence_lengths.append(length)
    
    min_length = min(sequence_lengths)
    prefix_length = int(min_length * prefix_overlap)
    prefix_ids = [rng.randint(0, tokenizer.vocab_size - 1) for _ in range(prefix_length)]

    prompts = []
    for seq_length in tqdm(sequence_lengths, desc="Generating"):
        token_ids = [rng.randint(0, tokenizer.vocab_size - 1) for _ in range(seq_length)]
        unique_ids = token_ids[prefix_length:]
        final_ids = prefix_ids + unique_ids        
        prompt = tokenizer.decode(final_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        prompt = re.sub(r'[\ud800-\udfff]', '', prompt)  # Remove surrogate characters
        prompts.append(prompt)
    
    return prompts, sequence_lengths


def build_output_path(
    count,
    prefix_overlap,
    tokenizer_label,
    seed,
    dist_mode,
    dist_median=None,
    dist_sigma=None,
    dist_max=None,
    target_tokens=None,
    tolerance=None,
):
    """Generate output path based on generation parameters."""
    base_dir = Path("prompts")

    prefix_label = f"{prefix_overlap:.2f}".replace(".", "p")
    tokenizer_component = re.sub(r"[^0-9A-Za-z._-]+", "-", tokenizer_label).strip("-._") or "none"

    tokens_label = (f"lognorm-{int(dist_median)}-{dist_sigma:.2f}-{dist_max}".replace(".", "p")
                    if dist_mode == "lognormal" else f"fixed-{target_tokens}-{tolerance}")

    metadata = f"n{count}_prefix{prefix_label}_{tokens_label}_tok-{tokenizer_component}_seed{seed}"
    filename = f"{metadata}.jsonl"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / filename

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", "-n", type=int, default=10, help="Number of requests")
    parser.add_argument("--prefix-overlap", type=float, default=0.0)
    parser.add_argument("--approx-input-tokens", type=int, default=0, help="Target tokens per prompt")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--token-tolerance", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--dist-mode", choices=["fixed", "lognormal"], default="fixed")
    parser.add_argument("--dist-median", type=float, default=None)
    parser.add_argument("--dist-sigma", type=float, default=0.5)
    parser.add_argument("--dist-max", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    assert 0 <= args.prefix_overlap <= 1
    
    # Validate distribution mode requirements
    if args.dist_mode == "lognormal":
        assert args.dist_median is not None, "--dist-median required for lognormal mode"
        assert args.dist_sigma is not None, "--dist-sigma required for lognormal mode"
        assert args.dist_max is not None, "--dist-max required for lognormal mode"
    else:
        assert args.approx_input_tokens > 0, "--approx-input-tokens must be > 0 for fixed mode"
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        print(f"Tokenizer loaded successfully. Vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"Error loading tokenizer: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Calculate tolerance
    target_tokens = args.approx_input_tokens
    tolerance = args.token_tolerance if args.token_tolerance is not None else max(5, int(target_tokens * 0.05))
    
    # Generate prompts
    prompts, sequence_lengths = assemble_prompts(
        count=args.count,
        prefix_overlap=args.prefix_overlap,
        target_tokens=target_tokens,
        tokenizer=tokenizer,
        tolerance=tolerance,
        dist_mode=args.dist_mode,
        dist_median=args.dist_median,
        dist_sigma=args.dist_sigma,
        dist_max=args.dist_max,
        seed=args.seed,
    )

    print_histogram(sequence_lengths)
    
    # Build output path
    output_path = build_output_path(
        count=args.count,
        prefix_overlap=args.prefix_overlap,
        tokenizer_label=args.tokenizer,
        seed=args.seed,
        dist_mode=args.dist_mode,
        dist_median=args.dist_median,
        dist_sigma=args.dist_sigma,
        dist_max=args.dist_max,
        target_tokens=target_tokens,
        tolerance=tolerance,
    )
    
    # Write output
    with output_path.open("w", encoding="utf-8") as f:
        for prompt in prompts:
            record = {"messages": [{"role": "user", "content": prompt}]}
            if args.model:
                record["model"] = args.model
            json.dump(record, f)
            f.write("\n")
    
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())