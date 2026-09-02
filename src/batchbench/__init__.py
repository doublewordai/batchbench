import json
from typing import Any, Mapping

try:
    from . import _core
except ImportError:  # pragma: no cover - pure-Python checkout without the compiled extension
    _core = None


JSONDict = dict[str, Any]


def _require_core():
    if _core is None:
        raise RuntimeError(
            "batchbench._core is not built; install the batchbench wheel or run maturin develop"
        )
    return _core


def request_entry(body: Mapping[str, Any], line_idx: int = 0, input_tokens: int = 0) -> JSONDict:
    return {
        "body": dict(body),
        "line_idx": line_idx,
        "input_tokens": input_tokens,
    }


def finite_mode(requests_per_user: int) -> JSONDict:
    return {
        "kind": "finite",
        "requests_per_user": requests_per_user,
    }


def long_running_mode(duration_secs: float) -> JSONDict:
    return {
        "kind": "long_running",
        "duration_secs": duration_secs,
    }


def generate_requests(options: Mapping[str, Any], model: str) -> list[JSONDict]:
    payload = _require_core().generate_requests_json(json.dumps(options), model)
    return json.loads(payload)


def run_benchmark(config: Mapping[str, Any]) -> JSONDict:
    payload = _require_core().run_benchmark_json(json.dumps(config))
    return json.loads(payload)


def run_cli(argv: list[str]) -> None:
    _require_core().run_cli(argv)


__all__ = [
    "finite_mode",
    "generate_requests",
    "long_running_mode",
    "request_entry",
    "run_cli",
    "run_benchmark",
]
