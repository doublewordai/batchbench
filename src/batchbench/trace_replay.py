"""Compare a replay of Dynamo request traces with what production recorded for the same requests.

A manifest exported with ``batchbench-export-plans --dynamo-traces`` names each request's source trace row in its
trajectory ``metadata.requests``; ``batchbench-agent --request-events-jsonl`` records the replay's client-side timings
by trajectory and 1-based invocation. Joining the two on the request's source id puts the replay's first-token time,
time per output token and end-to-end time beside production's for the same request. Only requests whose source
arrived inside the scoring window count, so a replay that started with a link window and ran past the window's end
is scored on its steady part.

A production context file, when given, names the production worker set that served each request (by its decode
worker id) and each set's GPUs, so production's figures are split by set and each set's throughput per GPU over the
window is reported beside the replay's::

    {"worker_sets": {"<decode worker id, decimal>": "<set>", ...}, "set_gpus": {"<set>": <GPUs>, ...}}
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional, Sequence

from .dynamo_traces import TRACE_EVENT, _trace_lines, TraceStats, trace_files
from .export_plans import ExportError

UNMAPPED = "unmapped"


@dataclass
class Timing:
    ttft_s: Optional[float]
    tpot_s: Optional[float]
    e2e_s: Optional[float]
    output_tokens: Optional[int]
    ok: bool
    worker_set: str = ""


def parse_time(value: str) -> float:
    """Milliseconds since the epoch of an ISO-8601 time (UTC when it names no zone)."""

    ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.timestamp() * 1000.0


def load_sources(lines: Iterable[str]) -> dict[tuple[str, int], tuple[str, int]]:
    """Each manifest request's source row, ``(trajectory_id, invocation) -> (request_id, received_ms)``."""

    sources = {}
    for line in lines:
        if not line.strip():
            continue
        trajectory = json.loads(line)
        for invocation, request in enumerate((trajectory.get("metadata") or {}).get("requests") or [], start=1):
            if request.get("request_id"):
                sources[(trajectory["trajectory_id"], invocation)] = (request["request_id"], int(request["received_ms"]))
    return sources


def _seconds(ms: Any) -> Optional[float]:
    return None if ms is None else float(ms) / 1000.0


def load_replay(lines: Iterable[str], sources: Mapping[tuple[str, int], tuple[str, int]], lo_ms: float,
                hi_ms: float) -> dict[str, Timing]:
    """The replay's timing of each request whose source arrived in ``[lo_ms, hi_ms)``, by source request id."""

    replay = {}
    for line in lines:
        if not line.strip():
            continue
        event = json.loads(line)
        source = sources.get((event["trajectory_id"], int(event["invocation"])))
        if source is None or not lo_ms <= source[1] < hi_ms:
            continue
        replay[source[0]] = Timing(_seconds(event.get("ttft_ms")), _seconds(event.get("tpot_ms")),
                                   _seconds(event.get("e2e_ms")), event.get("completion_tokens"),
                                   event.get("status") == "ok")
    return replay


@dataclass
class SetTotals:
    requests: int = 0
    output_tokens: int = 0
    within_slo: int = 0


def production_timing(request: Mapping[str, Any], worker_sets: Mapping[str, str]) -> Timing:
    worker = request.get("worker") or {}
    ttft = _seconds(request.get("ttft_ms"))
    return Timing(ttft, _seconds(request.get("avg_itl_ms")), _seconds(request.get("total_time_ms")),
                  request.get("output_tokens"), ttft is not None,
                  worker_sets.get(str(worker.get("decode_worker_id")), UNMAPPED))


def within_slo(timing: Timing, slo_ttft_s: float, slo_tpot_s: float) -> bool:
    """Succeeded, first token within the bound, and time per output token within its bound (or under two tokens)."""

    return (timing.ok and timing.ttft_s is not None and timing.ttft_s <= slo_ttft_s
            and (timing.tpot_s is None or timing.tpot_s <= slo_tpot_s))


def load_production(paths: Sequence[str], wanted: set[str], worker_sets: Mapping[str, str], model: Optional[str],
                    lo_ms: float, hi_ms: float, slo_ttft_s: float, slo_tpot_s: float,
                    stats: TraceStats) -> tuple[dict[str, Timing], dict[str, SetTotals]]:
    """Production's timing of each wanted request; and, with ``model``, every request of the model that arrived in
    ``[lo_ms, hi_ms)``, counted by the worker set that served it."""

    found: dict[str, Timing] = {}
    totals: dict[str, SetTotals] = {}
    for path in trace_files(list(paths)):
        for line in _trace_lines(path, stats):
            if f'"{TRACE_EVENT}"' not in line:
                continue
            request = json.loads(line).get("request") or {}
            request_id = request.get("request_id")
            in_window = model is not None and request.get("model") == model and lo_ms <= (request.get("request_received_ms") or 0) < hi_ms
            if request_id not in wanted and not in_window:
                continue
            timing = production_timing(request, worker_sets)
            if request_id in wanted:
                found[request_id] = timing
            if in_window:
                t = totals.setdefault(timing.worker_set, SetTotals())
                t.requests += 1
                t.output_tokens += timing.output_tokens or 0
                t.within_slo += within_slo(timing, slo_ttft_s, slo_tpot_s)
    return found, totals


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def summarize(timings: Sequence[Timing], slo_ttft_s: float, slo_tpot_s: float, seconds: Optional[float] = None,
              gpus: Optional[float] = None) -> dict[str, Any]:
    done = [t for t in timings if t.ok]
    ttft = [t.ttft_s for t in done if t.ttft_s is not None]
    tpot = [t.tpot_s * 1000 for t in done if t.tpot_s is not None]
    e2e = [t.e2e_s for t in done if t.e2e_s is not None]
    out = {
        "requests": len(timings),
        "failed": len(timings) - len(done),
        "within_slo": sum(within_slo(t, slo_ttft_s, slo_tpot_s) for t in timings) / max(len(timings), 1),
        "ttft_s_p50": percentile(ttft, 0.5), "ttft_s_p90": percentile(ttft, 0.9), "ttft_s_p99": percentile(ttft, 0.99),
        "tpot_ms_p50": percentile(tpot, 0.5), "tpot_ms_p90": percentile(tpot, 0.9), "tpot_ms_p99": percentile(tpot, 0.99),
        "e2e_s_p50": percentile(e2e, 0.5), "e2e_s_p90": percentile(e2e, 0.9),
        "e2e_s_mean": statistics.fmean(e2e) if e2e else math.nan,
    }
    if seconds and gpus:
        out["requests_per_s_per_gpu"] = len(done) / seconds / gpus
        out["output_tokens_per_s_per_gpu"] = sum(t.output_tokens or 0 for t in done) / seconds / gpus
        out["within_slo_per_s_per_gpu"] = sum(within_slo(t, slo_ttft_s, slo_tpot_s) for t in timings) / seconds / gpus
    return out


def compare(replay: Mapping[str, Timing], production: Mapping[str, Timing], totals: Mapping[str, SetTotals],
            set_gpus: Mapping[str, float], seconds: float, replay_gpus: Optional[float], slo_ttft_s: float,
            slo_tpot_s: float) -> dict[str, Any]:
    both = [request_id for request_id in replay if request_id in production]
    report: dict[str, Any] = {
        "replayed": len(replay),
        "matched": len(both),
        "replay": summarize([replay[r] for r in both], slo_ttft_s, slo_tpot_s, seconds, replay_gpus),
        "production": summarize([production[r] for r in both], slo_ttft_s, slo_tpot_s),
        "production_by_set": {},
        "production_set_throughput": {},
    }
    by_set: dict[str, list[Timing]] = {}
    for request_id in both:
        by_set.setdefault(production[request_id].worker_set, []).append(production[request_id])
    for name in sorted(by_set):
        report["production_by_set"][name] = {
            "production": summarize(by_set[name], slo_ttft_s, slo_tpot_s),
            "replay": summarize([replay[r] for r in both if production[r].worker_set == name], slo_ttft_s, slo_tpot_s),
        }
    for name in sorted(totals):
        t, gpus = totals[name], set_gpus.get(name)
        row: dict[str, Any] = {"requests": t.requests, "within_slo": t.within_slo / max(t.requests, 1)}
        if gpus:
            row.update(gpus=gpus, requests_per_s_per_gpu=t.requests / seconds / gpus,
                       output_tokens_per_s_per_gpu=t.output_tokens / seconds / gpus,
                       within_slo_per_s_per_gpu=t.within_slo / seconds / gpus)
        report["production_set_throughput"][name] = row
    return report


def _fmt(value: Any) -> str:
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def render(report: Mapping[str, Any]) -> str:
    lines = [f"replayed {report['replayed']} requests in the window, {report['matched']} matched to production rows"]

    def block(title: str, summary: Mapping[str, Any]) -> None:
        lines.append(f"{title}: " + "  ".join(f"{k}={_fmt(v)}" for k, v in summary.items()))

    block("replay", report["replay"])
    block("production (same requests)", report["production"])
    for name, pair in report["production_by_set"].items():
        block(f"  served by {name}: production", pair["production"])
        block(f"  served by {name}: replay", pair["replay"])
    for name, row in report["production_set_throughput"].items():
        block(f"production set {name}, every request in the window", row)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="batchbench-compare-trace-replay",
        description="Compare a replay of Dynamo request traces with production's record of the same requests.",
    )
    parser.add_argument("--manifest", required=True, help="the replayed manifest (batchbench-export-plans --dynamo-traces)")
    parser.add_argument("--request-events", required=True, help="the replay's batchbench-agent --request-events-jsonl")
    parser.add_argument("--dynamo-traces", action="append", required=True, metavar="PATH",
                        help="production trace files: a directory, file or glob; repeatable")
    parser.add_argument("--score-start", required=True, help="scoring window start (ISO-8601, source time)")
    parser.add_argument("--score-end", required=True, help="scoring window end (ISO-8601, source time)")
    parser.add_argument("--slo-ttft-s", type=float, default=15.0, help="first-token bound in seconds (default 15)")
    parser.add_argument("--slo-tpot-ms", type=float, default=33.0, help="time-per-output-token bound in ms (default 33)")
    parser.add_argument("--replay-gpus", type=float, help="GPUs the replay ran on, for its throughput per GPU")
    parser.add_argument("--production-context", help="JSON naming worker sets and their GPUs (see module docs)")
    parser.add_argument("--model", help="production model name: count every request of it in the window by set")
    parser.add_argument("--json-out", help="also write the report here as JSON")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    lo, hi = parse_time(args.score_start), parse_time(args.score_end)
    if hi <= lo:
        print("error: --score-end must be after --score-start", file=sys.stderr)
        return 2
    context: dict[str, Any] = {}
    if args.production_context:
        with open(args.production_context, encoding="utf-8") as handle:
            context = json.load(handle)
    worker_sets = {str(k): v for k, v in (context.get("worker_sets") or {}).items()}
    set_gpus = {k: float(v) for k, v in (context.get("set_gpus") or {}).items()}
    slo_ttft, slo_tpot = args.slo_ttft_s, args.slo_tpot_ms / 1000.0
    with open(args.manifest, encoding="utf-8") as handle:
        sources = load_sources(handle)
    with open(args.request_events, encoding="utf-8") as handle:
        replay = load_replay(handle, sources, lo, hi)
    try:
        production, totals = load_production(args.dynamo_traces, set(replay), worker_sets, args.model, lo, hi,
                                             slo_ttft, slo_tpot, TraceStats())
    except ExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    report = compare(replay, production, totals, set_gpus, (hi - lo) / 1000.0, args.replay_gpus,
                     slo_ttft, slo_tpot)
    print(render(report))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
