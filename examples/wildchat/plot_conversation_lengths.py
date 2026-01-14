#!/usr/bin/env python3
"""Plot histogram of conversation lengths from JSONL file."""

import json
import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from tqdm import tqdm
from transformers import AutoTokenizer


def get_conversation_length(messages, tokenizer, apply_template):
    """Calculate total length of conversation in tokens."""
    if apply_template:
        token_ids = tokenizer.apply_chat_template(messages, tokenize=True)
        return len(token_ids)
    else:
        return sum(len(tokenizer.encode(msg["content"])) for msg in messages)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("--output", default=None)
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--output-stats", default=None)
    args = parser.parse_args()

    # Load tokenizer (required)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        print(f"Tokenizer loaded successfully. Vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"Error loading tokenizer: {e}", file=sys.stderr)
        sys.exit(1)

    # Read and process
    lengths = []
    turn_counts = []

    with open(args.input_file, "r", encoding="utf-8") as f:
        for line_num, line in tqdm(enumerate(f, 1)):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
                messages = record["body"]["messages"]
                length = get_conversation_length(messages, tokenizer, not args.no_chat_template)
                lengths.append(length)
                turn_counts.append(sum(1 for msg in messages if msg["role"] == "user"))

            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON on line {line_num}: {e}")
                continue

    if not lengths:
        return

    lengths = np.array(lengths)

    # Stats
    min_val = min(lengths)
    max_val = max(lengths)
    mean_val = np.mean(lengths)
    median_val = np.median(lengths)
    std_val = np.std(lengths)
    p5 = np.percentile(lengths, 5)
    p95 = np.percentile(lengths, 95)
    p99 = np.percentile(lengths, 99)

    # Fit log-normal distribution
    lognorm_params = stats.lognorm.fit(lengths, floc=0)
    # scipy's lognorm uses shape (sigma), loc, scale (exp(mu)) parameterization
    # floc=0 forces the distribution to start at 0 (standard log-normal)
    shape, loc, scale = lognorm_params
    # Convert to more interpretable parameters
    # For lognormal: mu = log(scale), sigma = shape
    mu = np.log(scale)
    sigma = shape
    # For LogNormal(μ, σ): median = exp(μ), mean = exp(μ + σ²/2)
    fitted_median = np.exp(mu)
    fitted_mean = np.exp(mu + sigma**2 / 2)

    # Save stats if requested
    if args.output_stats:
        stats_data = {
            "dist_median": float(fitted_median),
            "dist_sigma": float(sigma),
            "dist_max": int(p99),
            "sample_count": len(lengths),
            "sample_median": float(median_val),
            "sample_mean": float(mean_val),
            "sample_min": float(min_val),
            "sample_max": float(max_val),
            "sample_p99": int(p99),
        }
        with open(args.output_stats, "w") as f:
            json.dump(stats_data, f, indent=2)

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Linear scale histogram
    ax1.hist(lengths, bins=args.bins, edgecolor='black', alpha=0.7, density=True)
    ax1.set_xlabel('Total Conversation Length (tokens)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title(f'Distribution of Conversation Lengths ({args.tokenizer})\n({len(lengths)})', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Overlay fitted distribution
    x = np.linspace(min_val, max_val, 1000)
    pdf = stats.lognorm.pdf(x, shape, loc, scale)
    ax1.plot(x, pdf, 'r-', linewidth=2, label='Fitted Log-Normal')
    ax1.axvline(fitted_median, color='red', linestyle='--', linewidth=1.5, label=f'Median = {fitted_median:.0f}')
    ax1.legend(loc='upper right', fontsize=10)

    # Stats boxes
    stats_text = (f'Mean: {mean_val:,.0f}\n'
                  f'Median: {median_val:,.0f}\n'
                  f'Std: {std_val:,.0f}\n'
                  f'Min: {min_val:,}\n'
                  f'Max: {max_val:,}\n'
                  f'P5: {p5:,.0f}\n'
                  f'P95: {p95:,.0f}')

    ax1.text(0.97, 0.70, stats_text, transform=ax1.transAxes, va='top', ha='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fit_text = (f'Fitted Log-Normal:\n'
            f'μ = {mu:.3f}\n'
            f'σ = {sigma:.3f}\n'
            f'Inferred median = {fitted_median:,.0f}\n'
            f'Inferred mean = {fitted_mean:,.0f}')
    ax1.text(0.97, 0.97, fit_text, transform=ax1.transAxes, va='top', ha='right',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5),
             fontsize=10)

    # Log scale histogram
    log_lengths = np.log10(lengths + 1)
    log_min = np.min(log_lengths)
    log_max = np.max(log_lengths)
    log_mean = np.mean(log_lengths)
    log_median = np.median(log_lengths)
    log_p5 = np.percentile(log_lengths, 5)
    log_p95 = np.percentile(log_lengths, 95)


    ax2.hist(log_lengths, bins=args.bins, edgecolor='black', alpha=0.7, color='orange', density=True)
    ax2.set_xlabel('Conversation Length (tokens, log scale)', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title(f'Distribution of Conversation Lengths (Log Scale) ({args.tokenizer})', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Overlay fitted normal distribution on log-transformed data
    # If X ~ LogNormal(mu, sigma), then log(X) ~ Normal(mu, sigma)
    # We're using log10, so we need to convert: log10(X) = ln(X)/ln(10) ~ Normal(mu/ln(10), sigma/ln(10))
    mu_log10 = mu / np.log(10)
    sigma_log10 = sigma / np.log(10)
    x_log = np.linspace(log_min, log_max, 1000)
    pdf_log = stats.norm.pdf(x_log, mu_log10, sigma_log10)
    ax2.plot(x_log, pdf_log, 'r-', linewidth=2, label='Fitted Normal (from Log-Normal)')
    ax2.axvline(np.log10(fitted_median), color='red', linestyle='--', linewidth=1.5, label=f'Inferred median = {fitted_median:,.0f}')
    ax2.legend(loc='upper right', fontsize=10)

    # Fix x-axis labels to show actual values
    tick_values = [10, 100, 1000, 10000, 100000]
    tick_positions = [np.log10(v) for v in tick_values]
    valid_ticks = [(pos, val) for pos, val in zip(tick_positions, tick_values)
                   if log_min - 0.5 <= pos <= log_max + 0.5]

    if valid_ticks:
        ax2.set_xticks([t[0] for t in valid_ticks])
        ax2.set_xticklabels([f'{t[1]:,}' for t in valid_ticks])

    # Add statistics for log lengths (show actual values, not log)
    log_stats_text = (f'Mean (geometric): {10**log_mean:,.0f}\n'
                      f'Median: {10**log_median:,.0f}\n'
                      f'Min: {10**log_min:,.0f}\n'
                      f'Max: {10**log_max:,.0f}\n'
                      f'P5: {10**log_p5:,.0f}\n'
                      f'P95: {10**log_p95:,.0f}')
    ax2.text(0.97, 0.70, log_stats_text, transform=ax2.transAxes, va='top', ha='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax2.text(0.97, 0.97, fit_text, transform=ax2.transAxes, va='top', ha='right',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5),
             fontsize=10)

    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
    else:
        plt.show()


if __name__ == "__main__":
    main()
