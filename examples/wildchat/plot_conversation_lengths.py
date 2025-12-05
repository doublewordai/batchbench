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
from scipy import stats
from tqdm import tqdm

import tiktoken
TIKTOKEN_AVAILABLE = True


def apply_chat_template(messages, tokenizer_name="cl100k_base"):
    """
    Apply a simple chat template to format messages for token counting.
    This mimics the ChatML format used by OpenAI models.
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys
        tokenizer_name: Name of the tokenizer (used to select appropriate template)
        
    Returns:
        Formatted string with chat template applied
    """
    # Use ChatML-style template (used by GPT-3.5/4, many open models)
    formatted_parts = []
    for message in messages:
        if isinstance(message, dict):
            role = message.get("role", "user")
            content = message.get("content", "")
            # ChatML format: <|im_start|>role\ncontent<|im_end|>
            formatted_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    
    return "\n".join(formatted_parts)


def calculate_output_length(record, tokenizer=None, use_tokens=True):
    """
    Calculate the output/response length from a batch API response record.
    
    Args:
        record: A dict from the batch API output JSONL file
        tokenizer: tiktoken tokenizer instance (required if use_tokens=True and completion_tokens unavailable)
        use_tokens: If True, use token count; otherwise count characters
        
    Returns:
        Tuple of (length, content) where length is token count or character count,
        and content is the response text (or None if not available)
    """
    # Try to extract from OpenAI Batch API response format
    # Format: {"response": {"body": {"choices": [...], "usage": {...}}}}
    response = record.get("response", {})
    body = response.get("body", {})
    usage = body.get("usage", {})
    choices = body.get("choices", [])
    
    # Get the response content
    content = None
    if choices and isinstance(choices, list) and len(choices) > 0:
        message = choices[0].get("message", {})
        content = message.get("content", "")
    
    # If use_tokens and completion_tokens is available, use it directly
    if use_tokens and "completion_tokens" in usage:
        return usage["completion_tokens"], content
    
    # Otherwise, tokenize or count characters of the content
    if content:
        if use_tokens and tokenizer:
            return len(tokenizer.encode(content, allowed_special={'<|endoftext|>', '<|im_start|>', '<|im_end|>'})), content
        else:
            return len(content), content
    
    return 0, None


def calculate_conversation_length(text_field, tokenizer=None, use_tokens=True, apply_template=True):
    """
    Calculate total length of a conversation.
    
    Args:
        text_field: Either a string or list of message dicts
        tokenizer: tiktoken tokenizer instance (required if use_tokens=True)
        use_tokens: If True, use tokenizer; otherwise count characters
        apply_template: If True, apply chat template to message lists
        
    Returns:
        Total token count (if use_tokens=True) or character count of all messages
    """
    if isinstance(text_field, str):
        if use_tokens and tokenizer:
            return len(tokenizer.encode(text_field, allowed_special={'<|endoftext|>', '<|im_start|>', '<|im_end|>'}))
        return len(text_field)
    elif isinstance(text_field, list):
        if apply_template:
            # Apply chat template and tokenize the whole thing
            formatted_text = apply_chat_template(text_field)
            if use_tokens and tokenizer:
                return len(tokenizer.encode(formatted_text, allowed_special={'<|endoftext|>', '<|im_start|>', '<|im_end|>'}))
            return len(formatted_text)
        else:
            # Legacy behavior: sum up content lengths
            total = 0
            for message in text_field:
                if isinstance(message, dict) and "content" in message:
                    content = message["content"]
                    if use_tokens and tokenizer:
                        total += len(tokenizer.encode(content, allowed_special={'<|endoftext|>', '<|im_start|>', '<|im_end|>'}))
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
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Don't apply chat template to messages (just sum content lengths)"
    )
    parser.add_argument(
        "--output-lengths",
        action="store_true",
        help="Plot output/response lengths from batch API output file instead of input conversation lengths"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return 1
    
    # Load tokenizer
    tokenizer = None
    use_tokens = not args.use_chars
    apply_template = not args.no_chat_template
    
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
    output_mode = args.output_lengths
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in tqdm(enumerate(f, 1)):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
                
                if output_mode:
                    # Parse batch API output format for response lengths
                    length, content = calculate_output_length(record, tokenizer, use_tokens)
                    if length > 0:
                        lengths.append(length)
                else:
                    # Support OpenAI Batch API format: {"body": {"messages": [...]}}
                    # as well as legacy format: {"text": ...} or {"messages": [...]}
                    if "body" in record and "messages" in record["body"]:
                        text_field = record["body"]["messages"]
                    elif "messages" in record:
                        text_field = record["messages"]
                    else:
                        text_field = record.get("text")
                    
                    if text_field is not None:
                        length = calculate_conversation_length(text_field, tokenizer, use_tokens, apply_template)
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
        if output_mode:
            print("Error: No valid responses found in output file")
        else:
            print("Error: No valid conversations found")
        return 1
    
    unit = "tokens" if use_tokens else "characters"
    length_type = "Output/Response" if output_mode else "Conversation"
    
    # Calculate statistics
    min_val = min(lengths)
    max_val = max(lengths)
    mean_val = np.mean(lengths)
    median_val = np.median(lengths)
    std_val = np.std(lengths)
    p5_val = np.percentile(lengths, 5)
    p95_val = np.percentile(lengths, 95)
    
    item_name = "responses" if output_mode else "conversations"
    print(f"\nAnalyzed {len(lengths)} {item_name}")
    print(f"Statistics ({unit}):")
    print(f"  Min length: {min_val:,} {unit}")
    print(f"  Max length: {max_val:,} {unit}")
    print(f"  Mean length: {mean_val:,.1f} {unit}")
    print(f"  Median length: {median_val:,.1f} {unit}")
    print(f"  Std dev: {std_val:,.1f} {unit}")
    print(f"  P5: {p5_val:,.1f} {unit}")
    print(f"  P95: {p95_val:,.1f} {unit}")
    
    if turn_counts and not output_mode:
        print(f"\nTurn statistics:")
        print(f"  Min turns: {min(turn_counts)}")
        print(f"  Max turns: {max(turn_counts)}")
        print(f"  Mean turns: {np.mean(turn_counts):.1f}")
        print(f"  Median turns: {np.median(turn_counts):.1f}")
        print(f"  Std dev: {np.std(turn_counts):.1f}")
        print(f"  P5: {np.percentile(turn_counts, 5):.1f}")
        print(f"  P95: {np.percentile(turn_counts, 95):.1f}")
    
    # Create the histogram
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Fit log-normal distribution to the data
    lengths_array = np.array(lengths)
    # scipy's lognorm uses shape (sigma), loc, scale (exp(mu)) parameterization
    # floc=0 forces the distribution to start at 0 (standard log-normal)
    lognorm_params = stats.lognorm.fit(lengths_array, floc=0)
    shape, loc, scale = lognorm_params
    # Convert to more interpretable parameters
    # For lognormal: mu = log(scale), sigma = shape
    mu = np.log(scale)
    sigma = shape
    
    # Histogram 1: Total conversation length
    counts, bin_edges, patches = ax1.hist(lengths, bins=args.bins, edgecolor='black', alpha=0.7, density=True)
    ax1.set_xlabel(f'Total {length_type} Length ({unit})', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    title_suffix = f" ({args.tokenizer})" if use_tokens else ""
    ax1.set_title(f'Distribution of {length_type} Lengths{title_suffix}\n({len(lengths)} {item_name})', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Overlay fitted log-normal distribution on linear scale
    x_linear = np.linspace(min_val, max_val, 1000)
    pdf_linear = stats.lognorm.pdf(x_linear, shape, loc, scale)
    ax1.plot(x_linear, pdf_linear, 'r-', linewidth=2, label='Fitted Log-Normal')
    
    # Add statistics text with all requested metrics
    stats_text = (f'Mean: {mean_val:,.0f}\n'
                  f'Median: {median_val:,.0f}\n'
                  f'Std: {std_val:,.0f}\n'
                  f'Min: {min_val:,}\n'
                  f'Max: {max_val:,}\n'
                  f'P5: {p5_val:,.0f}\n'
                  f'P95: {p95_val:,.0f}')
    ax1.text(0.97, 0.70, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=10)
    
    # Add fitted parameters text box
    # For LogNormal(μ, σ): median = exp(μ), mean = exp(μ + σ²/2)
    fitted_median = np.exp(mu)
    fitted_mean = np.exp(mu + sigma**2 / 2)
    fit_text = (f'Fitted Log-Normal:\n'
                f'μ = {mu:.3f}\n'
                f'σ = {sigma:.3f}\n'
                f'Inferred median = {fitted_median:,.0f}\n'
                f'Inferred mean = {fitted_mean:,.0f}')
    ax1.text(0.97, 0.97, fit_text, transform=ax1.transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5),
             fontsize=10)
    
    # Add vertical line for inferred median
    ax1.axvline(x=fitted_median, color='red', linestyle='--', linewidth=1.5, 
                label=f'Inferred median = {fitted_median:,.0f}')
    ax1.legend(loc='upper right', fontsize=10)
    
    # Histogram 2: Log of conversation lengths
    log_lengths = np.log10(np.array(lengths) + 1)  # +1 to avoid log(0)
    log_min = np.min(log_lengths)
    log_max = np.max(log_lengths)
    log_mean = np.mean(log_lengths)
    log_median = np.median(log_lengths)
    log_std = np.std(log_lengths)
    log_p5 = np.percentile(log_lengths, 5)
    log_p95 = np.percentile(log_lengths, 95)
    
    ax2.hist(log_lengths, bins=args.bins, edgecolor='black', alpha=0.7, color='orange', density=True)
    ax2.set_xlabel(f'{length_type} Length ({unit}, log scale)', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title(f'Distribution of {length_type} Lengths (Log Scale){title_suffix}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Overlay fitted normal distribution on log-transformed data
    # If X ~ LogNormal(mu, sigma), then log(X) ~ Normal(mu, sigma)
    # We're using log10, so we need to convert: log10(X) = ln(X)/ln(10) ~ Normal(mu/ln(10), sigma/ln(10))
    mu_log10 = mu / np.log(10)
    sigma_log10 = sigma / np.log(10)
    x_log = np.linspace(log_min, log_max, 1000)
    pdf_log = stats.norm.pdf(x_log, mu_log10, sigma_log10)
    ax2.plot(x_log, pdf_log, 'r-', linewidth=2, label='Fitted Normal (from Log-Normal)')
    
    # Add vertical line for inferred median (in log10 scale)
    ax2.axvline(x=np.log10(fitted_median), color='red', linestyle='--', linewidth=1.5, 
                label=f'Inferred median = {fitted_median:,.0f}')
    ax2.legend(loc='upper right', fontsize=10)
    
    # Set x-axis ticks to show actual values instead of log values
    # Generate nice tick positions in log space
    tick_values = [10, 100, 1000, 10000, 100000, 1000000]
    tick_positions = [np.log10(v) for v in tick_values]
    # Filter to only show ticks within the data range
    valid_ticks = [(pos, val) for pos, val in zip(tick_positions, tick_values) 
                   if log_min - 0.5 <= pos <= log_max + 0.5]
    if valid_ticks:
        ax2.set_xticks([t[0] for t in valid_ticks])
        ax2.set_xticklabels([f'{t[1]:,}' for t in valid_ticks])
    
    # Add statistics for log lengths (show actual values, not log)
    log_stats_text = (f'Mean: {10**log_mean:,.0f}\n'
                      f'Median: {10**log_median:,.0f}\n'
                      f'Min: {10**log_min:,.0f}\n'
                      f'Max: {10**log_max:,.0f}\n'
                      f'P5: {10**log_p5:,.0f}\n'
                      f'P95: {10**log_p95:,.0f}')
    ax2.text(0.97, 0.70, log_stats_text, transform=ax2.transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=10)
    
    # Add fitted parameters text box for log scale
    fit_text_log = (f'Fitted Log-Normal:\n'
                    f'μ = {mu:.3f} (ln scale)\n'
                    f'σ = {sigma:.3f}\n'
                    f'Inferred median = {fitted_median:,.0f}\n'
                    f'Inferred mean = {fitted_mean:,.0f}')
    ax2.text(0.97, 0.97, fit_text_log, transform=ax2.transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5),
             fontsize=10)
    
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
