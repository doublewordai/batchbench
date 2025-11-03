"""Utilities and CLI entrypoint for generating batchbench request payloads."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, List

DEFAULT_PREFIX_TEXT = (
    "In this experiment we explore the capability of large language models "
    "to adapt their narrative based on subtle contextual variations. The "
    "following prompt requests creative output across a range of scenarios."
)


def load_tokenizer(model_name: str, token: str | None = None):
    """Load a Hugging Face tokenizer and configure pad token if needed."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Approximate token sizing requires the 'transformers' package. "
            "Install it with `pip install transformers`."
        ) from exc

    load_kwargs = {"use_fast": True}
    if token:
        load_kwargs["token"] = token
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"Failed to load tokenizer '{model_name}': {exc}") from exc

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def assemble_prompts(
    count: int,
    prefix_overlap: float,
    *,
    target_tokens: int | None = None,
    tokenizer: Any | None = None,
    tolerance: int = 5,
) -> List[str]:
    """Create prompt strings whose prefixes overlap by the requested fraction."""
    if tokenizer is None:
        raise ValueError("A tokenizer is required to detokenize random token ids.")

    prompts: List[str] = []
    rng = random.Random()
    sequence_lengths: List[int] = []
    for _ in range(count):
        sequence_length = 1
        if target_tokens and target_tokens > 0:
            lower = max(1, target_tokens - (tolerance if tolerance else 0))
            upper = target_tokens + (tolerance if tolerance else 0)
            if lower > upper:
                lower = upper
            sequence_length = rng.randint(lower, upper) if lower != upper else lower
        sequence_lengths.append(sequence_length)

    prefix_ratio = max(0.0, min(prefix_overlap, 1.0))
    min_length = min(sequence_lengths) if sequence_lengths else 0
    prefix_length = int(math.floor(min_length * prefix_ratio)) if min_length else 0
    if prefix_ratio > 0.0 and prefix_length == 0 and min_length > 0:
        prefix_length = 1

    prefix_ids = (
        [rng.randint(1, 10000) for _ in range(prefix_length)] if prefix_length else []
    )

    for seq_length in sequence_lengths:
        token_ids = [rng.randint(1, 10000) for _ in range(seq_length)]
        unique_ids = token_ids[prefix_length:]
        final_ids = prefix_ids + unique_ids

        prompt_text = tokenizer.decode(
            final_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        if not prompt_text.strip():
            try:
                tokens = tokenizer.convert_ids_to_tokens(final_ids)
                prompt_text = " ".join(tokens).strip()
            except Exception:
                prompt_text = " ".join(str(tid) for tid in final_ids)

        prompts.append(prompt_text)

    return prompts


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--count",
        "-n",
        type=int,
        default=10,
        help="Number of requests to generate (default: 10)",
    )
    parser.add_argument(
        "--prefix-overlap",
        type=float,
        default=0.0,
        help="Fraction of tokens shared as a prefix across requests (0.0-1.0, default: 0.0)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("outputs"),
        help=(
            "Destination directory or base filename. Metadata (count, prefix, token "
            "target, tokenizer) is appended automatically."
        ),
    )
    parser.add_argument(
        "--approx-input-tokens",
        type=int,
        default=0,
        help="Approximate number of tokens each prompt should contain (default: 0 = no adjustment)",
    )
    parser.add_argument(
        "--tokenizer-model",
        default=None,
        help=(
            "Hugging Face tokenizer identifier to use when approximating token counts. "
            "Defaults to 'gpt2' when not specified."
        ),
    )
    parser.add_argument(
        "--token-tolerance",
        type=int,
        default=None,
        help="Acceptable +/- token tolerance when approximating lengths (default: max(5, 5%% of target))",
    )
    parser.add_argument(
        "--huggingface-token",
        default=None,
        help=(
            "Personal access token for Hugging Face Hub (optional). If omitted, the "
            "generator checks HUGGINGFACE_TOKEN and HUGGING_FACE_HUB_TOKEN env vars."
        ),
    )
    return parser.parse_args(argv)


def resolve_tolerance(target_tokens: int, explicit: int | None) -> int:
    if target_tokens <= 0:
        return 0
    if explicit is not None and explicit >= 0:
        return explicit
    return max(5, int(target_tokens * 0.05))


def sanitize_component(value: str | None) -> str:
    if not value:
        return "none"
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "-", value).strip("-._")
    return cleaned or "none"


def format_prefix(prefix_overlap: float) -> str:
    return f"{prefix_overlap:.2f}".replace(".", "p")


def build_output_path(
    base_path: Path,
    *,
    count: int,
    prefix_overlap: float,
    target_tokens: int | None,
    tokenizer_label: str,
) -> Path:
    tokens_label = str(target_tokens) if target_tokens and target_tokens > 0 else "none"
    prefix_label = format_prefix(prefix_overlap)
    tokenizer_component = sanitize_component(tokenizer_label)
    metadata_suffix = (
        f"count-{count}_tokens-{tokens_label}_prefix-{prefix_label}_tokenizer-{tokenizer_component}"
    )

    if base_path.suffix == ".jsonl" and not base_path.is_dir():
        directory = base_path.parent if base_path.parent else Path(".")
        stem = sanitize_component(base_path.stem) or "requests"
        filename = f"{stem}_{metadata_suffix}.jsonl"
    else:
        directory = base_path
        filename = f"requests_{metadata_suffix}.jsonl"

    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / filename
    return output_path


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)

    hf_token = (
        args.huggingface_token
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
    )

    target_tokens = args.approx_input_tokens if args.approx_input_tokens > 0 else None
    tokenizer_label = args.tokenizer_model or "gpt2"
    tokenizer = load_tokenizer(tokenizer_label, hf_token)
    tolerance = (
        resolve_tolerance(target_tokens, args.token_tolerance) if target_tokens else 0
    )

    prompts = assemble_prompts(
        count=args.count,
        prefix_overlap=args.prefix_overlap,
        target_tokens=target_tokens,
        tokenizer=tokenizer,
        tolerance=tolerance,
    )

    output_path = build_output_path(
        args.output,
        count=args.count,
        prefix_overlap=args.prefix_overlap,
        target_tokens=target_tokens,
        tokenizer_label=tokenizer_label,
    )

    if output_path.exists():
        print(
            f"Output file {output_path} already exists; skipping generation.",
            file=sys.stderr,
        )
        print(output_path)
        return 0

    with output_path.open("w", encoding="utf-8") as handle:
        for prompt in prompts:
            json.dump({"text": prompt}, handle)
            handle.write("\n")

    print(
        f"Wrote {len(prompts)} request prompts to {output_path} "
        f"with prefix overlap {args.prefix_overlap:.2f}.",
        file=sys.stderr,
    )

    print(output_path)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
