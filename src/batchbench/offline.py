"""Offline vLLM benchmarking CLI entrypoint."""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import shlex
import shutil
import subprocess
from contextlib import contextmanager
from random import randint
from statistics import mean, pstdev
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from vllm import LLM, SamplingParams  # type: ignore

os.environ.setdefault("VLLM_LOG_STATS_INTERVAL", "1")
logger = logging.getLogger(__name__)

class VLLMThroughputCollector(logging.Handler):
    """Logging handler that captures vLLM throughput stats from INFO logs."""

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.prompt_tps: List[float] = []
        self.gen_tps: List[float] = []
        self.TP_LINE = re.compile(
            r"Avg prompt throughput:\s*([0-9.]+)\s*tokens/s,\s*"
            r"Avg generation throughput:\s*([0-9.]+)\s*tokens/s"
        )

    def emit(self, record: logging.LogRecord) -> None:  # type: ignore[override]
        msg = record.getMessage()
        match = self.TP_LINE.search(msg)
        if match:
            self.prompt_tps.append(float(match.group(1)))
            self.gen_tps.append(float(match.group(2)))

    def summary(self) -> Optional[Dict[str, float]]:
        if not self.prompt_tps or not self.gen_tps:
            return None
        return {
            "prompt_avg": mean(self.prompt_tps),
            "prompt_std": pstdev(self.prompt_tps),
            "gen_avg": mean(self.gen_tps),
            "gen_std": pstdev(self.gen_tps),
        }

    def save_csv(self, path: str = "vllm_throughput_history.csv") -> str:
        """Persist the captured throughput history for later inspection."""
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["prompt_tps", "gen_tps"])
            for prompt_tps, gen_tps in zip(self.prompt_tps, self.gen_tps):
                writer.writerow([prompt_tps, gen_tps])
        return path


@contextmanager
def capture_vllm_throughput() -> Iterable[VLLMThroughputCollector]:
    """Attach the collector to the vLLM logger while the workload runs."""
    logger = logging.getLogger("vllm")
    collector = VLLMThroughputCollector()
    logger.addHandler(collector)
    if logger.level > logging.INFO:
        logger.setLevel(logging.INFO)
    try:
        yield collector
    finally:
        logger.removeHandler(collector)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the metrics workload while allowing the model and vLLM options "
            "to be configured from the command line."
        )
    )
    parser.add_argument(
        "--model",
        default="facebook/opt-125m",
        help="Model identifier or path to load with vLLM."
    )
    parser.add_argument(
        "--num_reqs",
        type=int,
        default=2048,
        help="Number of synthetic prompts to generate."
    )
    parser.add_argument(
        "--icl",
        type=int,
        default=1024,
        help="Input context length (tokens per prompt)."
    )
    parser.add_argument(
        "--ocl",
        type=int,
        default=1,
        help="Output context length (max tokens generated per request)."
    )
    parser.add_argument(
        "--throughput_dir",
        default=".",
        help=(
            "Directory where the throughput history CSV will be written. "
            "Filename is derived from the runtime configuration."
        )
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel world size for vLLM initialisation."
    )
    parser.add_argument(
        "--pipeline_parallel_size",
        type=int,
        default=1,
        help="Pipeline parallel world size for vLLM initialisation."
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="Target GPU memory utilization for vLLM."
    )
    parser.add_argument(
        "--max_num_batched_tokens",
        type=int,
        default=512,
        help="Maximum tokens per batch when pre-filling prompts."
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.num_reqs < 1:
        raise ValueError("num_reqs must be >= 1")
    if args.icl < 1:
        raise ValueError("icl must be >= 1")
    if args.ocl < 0:
        raise ValueError("ocl must be >= 0")
    if args.tensor_parallel_size < 1:
        raise ValueError("tensor_parallel_size must be >= 1")
    if args.pipeline_parallel_size < 1:
        raise ValueError("pipeline_parallel_size must be >= 1")
    if args.max_num_batched_tokens < 1:
        raise ValueError("max_num_batched_tokens must be >= 1")


def derive_throughput_csv_name(
    model: str,
    icl: int,
    ocl: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    max_num_batched_tokens: int,
) -> str:
    sanitized_model = re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-._")
    if not sanitized_model:
        sanitized_model = "model"
    return (
        f"throughput_{sanitized_model}_icl{icl}_ocl{ocl}_"
        f"tp{tensor_parallel_size}_pp{pipeline_parallel_size}_"
        f"mbt{max_num_batched_tokens}.csv"
    )


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    llm_kwargs = {
        "model": args.model,
        "disable_log_stats": False,
        "enable_chunked_prefill": True,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization
    }

    sampling_params = SamplingParams(
        temperature=0.8,
        top_k=20,
        max_tokens=args.ocl,
        min_tokens=args.ocl,
    )

    llm = LLM(**llm_kwargs)

    tokenizer = llm.get_tokenizer()

    tokenized_prompts = [
        [randint(1, 10000) for _ in range(args.icl)]
        for _ in range(args.num_reqs)
    ]
    prompts = tokenizer.batch_decode(tokenized_prompts)

    with capture_vllm_throughput() as collector:
        llm.generate(prompts, sampling_params, use_tqdm=True)
        stats = collector.summary()
        throughput_name = derive_throughput_csv_name(
            model=args.model,
            icl=args.icl,
            ocl=args.ocl,
            tensor_parallel_size=args.tensor_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            max_num_batched_tokens=args.max_num_batched_tokens,
        )
        throughput_dir = Path(args.throughput_dir)
        throughput_dir.mkdir(parents=True, exist_ok=True)
        csv_path = collector.save_csv(str(throughput_dir / throughput_name))

    if stats is None:
        print("No throughput lines were captured. Make sure vLLM is emitting stats logs.")
    else:
        print(
            f"Prompt TPS  mean={stats['prompt_avg']:.2f} +/- {stats['prompt_std']:.2f}\n"
            f"Gen TPS     mean={stats['gen_avg']:.2f} +/- {stats['gen_std']:.2f}\n"
            f"History CSV written to: {csv_path}"
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
