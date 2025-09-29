"""Generate JSONL prompts for an OpenAI-compatible benchmark.

Each output line contains a JSON object with a single ``text`` field.
Prompts can be padded or trimmed to approximate a target input token count
using Hugging Face tokenizers.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, List

DEFAULT_PREFIX_TEXT = (
    "In this experiment we explore the capability of large language models "
    "to adapt their narrative based on subtle contextual variations. The "
    "following prompt requests creative output across a range of scenarios."
)

FILLER_SNIPPETS = [
    "Provide an illustrative example to deepen the reader's intuition.",
    "Highlight a practical application that underscores the concept's impact.",
    "Mention a subtle caveat that careful practitioners should consider.",
    "Describe a short scenario that grounds the abstract idea in reality.",
    "Emphasize how interdisciplinary perspectives can broaden understanding.",
]

ANGLE_SNIPPETS = [
    "highlighting contrasting viewpoints",
    "capturing the main trade-offs involved",
    "emphasizing practical implications for everyday work",
    "connecting the idea to long-term strategic thinking",
    "framing the explanation for a technical audience",
    "noting where additional research could be valuable",
    "touching on both benefits and risks succinctly",
]


def load_tokenizer(model_name: str, token: str | None = None):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - dependency managed externally
        raise SystemExit(
            "Approximate token sizing requires the 'transformers' package. "
            "Install it with `pip install transformers`."
        ) from exc

    load_kwargs = {"use_fast": True}
    if token:
        load_kwargs["token"] = token

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
    except Exception as exc:  # pragma: no cover - download/config handled externally
        raise SystemExit(f"Failed to load tokenizer '{model_name}': {exc}") from exc

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def build_shared_prefix(prefix_text: str, overlap_ratio: float) -> str:
    """Return the shared prefix text based on the desired overlap ratio."""
    cleaned = prefix_text.strip().split()
    if not cleaned:
        return ""

    overlap_ratio = max(0.0, min(overlap_ratio, 1.0))
    overlap_count = max(0, min(len(cleaned), math.floor(len(cleaned) * overlap_ratio)))
    if overlap_count == 0:
        return ""
    return " ".join(cleaned[:overlap_count])


def build_unique_prompt(index: int) -> str:
    rng = random.Random(index)
    detail = rng.choice(ANGLE_SNIPPETS)
    return f"Write a concise explanation for scenario {index + 1}, {detail}."


def assemble_prompts(
    count: int,
    prefix_overlap: float,
    prefix_text: str,
    *,
    target_tokens: int | None = None,
    tokenizer: Any | None = None,
    tolerance: int = 5,
) -> List[str]:
    shared_prefix = build_shared_prefix(prefix_text, prefix_overlap)
    prompts: List[str] = []

    for i in range(count):
        unique_prompt = build_unique_prompt(i)
        if shared_prefix:
            user_content = f"{shared_prefix} {unique_prompt}"
        else:
            user_content = unique_prompt

        if target_tokens and tokenizer:
            user_content = adjust_prompt_length(
                base_text=user_content,
                target_tokens=target_tokens,
                tokenizer=tokenizer,
                tolerance=tolerance,
                seed=i,
            )

        prompts.append(user_content)

    return prompts


def parse_args() -> argparse.Namespace:
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
        "--prefix-text",
        default=DEFAULT_PREFIX_TEXT,
        help="Text used to build the shared prefix portion (default: predefined paragraph)",
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
        help="Acceptable +/- token tolerance when approximating lengths (default: max(5, 5% of target))",
    )
    parser.add_argument(
        "--huggingface-token",
        default=None,
        help=(
            "Personal access token for Hugging Face Hub (optional). If omitted, the "
            "generator checks HUGGINGFACE_TOKEN and HUGGING_FACE_HUB_TOKEN env vars."
        ),
    )
    return parser.parse_args()


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
    metadata_suffix = f"count-{count}_tokens-{tokens_label}_prefix-{prefix_label}_tokenizer-{tokenizer_component}"

    if base_path.suffix == ".jsonl" and not base_path.is_dir():
        directory = base_path.parent if base_path.parent else Path(".")
        stem = sanitize_component(base_path.stem) or "requests"
        filename = f"{stem}_{metadata_suffix}.jsonl"
    else:
        directory = base_path
        filename = f"requests_{metadata_suffix}.jsonl"

    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / filename
    if output_path.exists():
        raise SystemExit(
            f"Refusing to overwrite existing file {output_path}. Delete it or choose a different output directory."
        )
    return output_path


def adjust_prompt_length(
    *,
    base_text: str,
    target_tokens: int,
    tokenizer: Any,
    tolerance: int,
    seed: int,
) -> str:
    if target_tokens <= 0:
        return base_text

    text = base_text
    encode_kwargs = {"add_special_tokens": False}
    tokens = tokenizer.encode(text, **encode_kwargs)

    if len(tokens) > target_tokens + tolerance:
        trimmed_tokens = tokens[:target_tokens]
        return tokenizer.decode(
            trimmed_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    if len(tokens) >= target_tokens - tolerance:
        return text

    rng = random.Random(7919 * (seed + 1))
    attempts = 0
    while len(tokens) < target_tokens and attempts < 256:
        filler = rng.choice(FILLER_SNIPPETS)
        text = f"{text} {filler}"
        tokens = tokenizer.encode(text, **encode_kwargs)
        attempts += 1

    if len(tokens) > target_tokens + tolerance:
        trimmed_tokens = tokens[:target_tokens]
        text = tokenizer.decode(
            trimmed_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    return text


def main() -> None:
    args = parse_args()

    tokenizer = None
    target_tokens = args.approx_input_tokens if args.approx_input_tokens > 0 else None
    tolerance = 0
    tokenizer_label = args.tokenizer_model or ("gpt2" if target_tokens else "none")
    if target_tokens:
        hf_token = (
            args.huggingface_token
            or os.getenv("HUGGINGFACE_TOKEN")
            or os.getenv("HUGGING_FACE_HUB_TOKEN")
        )
        tokenizer = load_tokenizer(tokenizer_label, hf_token)
        tolerance = resolve_tolerance(target_tokens, args.token_tolerance)

    prompts = assemble_prompts(
        count=args.count,
        prefix_overlap=args.prefix_overlap,
        prefix_text=args.prefix_text,
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

    with output_path.open("w", encoding="utf-8") as f:
        for prompt in prompts:
            json.dump({"text": prompt}, f)
            f.write("\n")

    print(
        f"Wrote {len(prompts)} request prompts to {output_path} "
        f"with prefix overlap {args.prefix_overlap:.2f}."
    )


if __name__ == "__main__":
    main()
