"""Locust load test that submits single OpenAI-compatible requests per task."""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

from gevent.lock import Semaphore
from gevent.queue import Empty, Queue
from locust import HttpUser, constant, task
from locust.exception import StopUser

LOGGER = logging.getLogger(__name__)

CSV_LOCK = Semaphore()
INIT_LOCK = Semaphore()
AGG_LOCK = Semaphore()


def load_requests(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Request file {path} does not exist")

    prompts: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Failed to decode JSON on line {line_number}: {exc}"
                ) from exc

            text = payload.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Line {line_number} must contain a non-empty 'text' field"
                )
            prompts.append(text)

    if not prompts:
        raise ValueError("The request file must contain at least one JSON object")

    return prompts


def extract_usage_tokens(usage: Dict) -> Tuple[int, int, int]:
    """Extract prompt/completion/total tokens from a usage payload."""

    def first_available(keys):
        for key in keys:
            value = usage.get(key)
            if isinstance(value, int):
                return value
        return 0

    prompt_tokens = first_available(("prompt_tokens", "input_tokens"))
    completion_tokens = first_available(("completion_tokens", "output_tokens"))
    total_tokens = usage.get("total_tokens")
    if not isinstance(total_tokens, int):
        total_tokens = prompt_tokens + completion_tokens
    return prompt_tokens, completion_tokens, total_tokens


def compute_percentiles(latencies_ms: List[float]) -> Dict[str, float]:
    if not latencies_ms:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0}

    data = sorted(latencies_ms)

    def percentile(p: float) -> float:
        if len(data) == 1:
            return data[0]
        rank = (len(data) - 1) * (p / 100.0)
        low = math.floor(rank)
        high = math.ceil(rank)
        if low == high:
            return data[low]
        weight = rank - low
        return data[low] + (data[high] - data[low]) * weight

    return {
        "p50": percentile(50.0),
        "p90": percentile(90.0),
        "p99": percentile(99.0),
    }


class BatchUser(HttpUser):
    wait_time = constant(0)
    prompt_queue: Queue | None = None
    total_prompts: int = 0
    results_path: Path | None = None
    start_time: float | None = None
    completed_count: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    latencies_ms: List[float] | None = None
    summary_logged: bool = False

    def on_start(self) -> None:
        requests_path = Path(os.getenv("REQUESTS_FILE", "requests.jsonl"))
        self.endpoint_path = os.getenv("OPENAI_ENDPOINT_PATH", "/v1/chat/completions")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.system_prompt = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant.")
        self.results_csv = Path(os.getenv("RESULTS_CSV", "results.csv"))

        cls = type(self)
        with INIT_LOCK:
            if cls.prompt_queue is None:
                prompts = list(load_requests(requests_path))
                queue: Queue = Queue()
                for index, prompt in enumerate(prompts):
                    queue.put((index, prompt))
                cls.prompt_queue = queue
                cls.total_prompts = len(prompts)
                cls.results_path = self.results_csv
                cls.start_time = time.perf_counter()
                cls.completed_count = 0
                cls.total_prompt_tokens = 0
                cls.total_completion_tokens = 0
                cls.latencies_ms = []
                cls.summary_logged = False
                if not self.results_csv.exists():
                    self._initialize_csv(self.results_csv)
                LOGGER.info(
                    "Loaded %s prompts from %s; targeting endpoint %s",
                    len(prompts),
                    requests_path,
                    self.endpoint_path,
                )

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self.headers = headers

    @staticmethod
    def _initialize_csv(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "index",
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "latency_ms",
                    "status",
                    "error",
                ]
            )

    @task
    def run_batch(self) -> None:
        cls = type(self)
        if cls.prompt_queue is None:
            raise StopUser()

        try:
            index, prompt = cls.prompt_queue.get_nowait()
        except Empty:
            raise StopUser()

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }

        name = f"request_{index}"
        start_time = time.perf_counter()
        status = "success"
        error_message = ""
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        with self.client.post(
            self.endpoint_path,
            headers=self.headers,
            json=payload,
            name=name,
            catch_response=True,
        ) as response:
            if response.status_code >= 400:
                status = "http_error"
                error_message = f"HTTP {response.status_code}: {response.text}"
                response.failure(error_message)
            else:
                try:
                    body = response.json()
                except Exception as exc:  # noqa: BLE001
                    status = "decode_error"
                    error_message = f"Failed to decode JSON: {exc}"
                    response.failure(error_message)
                else:
                    usage = body.get("usage") or {}
                    prompt_tokens, completion_tokens, total_tokens = extract_usage_tokens(usage)
                    response.success()

        latency_ms = (time.perf_counter() - start_time) * 1000
        self._write_result(
            index=index,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            status=status,
            error=error_message,
        )
        self._record_metrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            status=status,
        )

        raise StopUser()

    def _write_result(
        self,
        *,
        index: int,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        latency_ms: float,
        status: str,
        error: str,
    ) -> None:
        path = type(self).results_path or self.results_csv
        row = [
            index,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            f"{latency_ms:.3f}",
            status,
            error,
        ]
        with CSV_LOCK:
            with path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)

    def _record_metrics(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        status: str,
    ) -> None:
        cls = type(self)
        with AGG_LOCK:
            if cls.latencies_ms is None:
                cls.latencies_ms = []
            cls.completed_count += 1
            cls.total_prompt_tokens += prompt_tokens
            cls.total_completion_tokens += completion_tokens
            cls.latencies_ms.append(latency_ms)

            finished = (
                cls.completed_count >= cls.total_prompts and not cls.summary_logged
            )

            if finished:
                cls.summary_logged = True
                end_time = time.perf_counter()
                duration_s = 0.0
                if cls.start_time is not None:
                    duration_s = max(end_time - cls.start_time, 0.0)
                latency_percentiles = compute_percentiles(cls.latencies_ms)
                duration_for_rate = duration_s if duration_s > 0 else 1e-9
                input_throughput = cls.total_prompt_tokens / duration_for_rate
                output_throughput = cls.total_completion_tokens / duration_for_rate

                LOGGER.info(
                    (
                        "Benchmark summary: %s requests, total time %.3f s, "
                        "prompt tokens %s, completion tokens %s, "
                        "input throughput %.2f tok/s, output throughput %.2f tok/s, "
                        "latency p50=%.2f ms p90=%.2f ms p99=%.2f ms"
                    ),
                    cls.completed_count,
                    duration_s,
                    cls.total_prompt_tokens,
                    cls.total_completion_tokens,
                    input_throughput,
                    output_throughput,
                    latency_percentiles["p50"],
                    latency_percentiles["p90"],
                    latency_percentiles["p99"],
                )
