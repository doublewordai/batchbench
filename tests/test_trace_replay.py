"""Tests for batchbench-compare-trace-replay."""

from __future__ import annotations

import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from batchbench import trace_replay

T0 = 1_794_600_000_000  # ms
WINDOW = ("2026-11-13T20:00:00Z", "2026-11-13T20:10:00Z")  # T0 is the window start


def production_row(request_id, received_ms, ttft_ms, itl_ms, output_tokens=100, worker=1, model="m"):
    return {
        "schema": "dynamo.request.trace.v1",
        "event_type": "request_end",
        "request": {
            "request_id": request_id,
            "model": model,
            "request_received_ms": received_ms,
            "ttft_ms": ttft_ms,
            "avg_itl_ms": itl_ms,
            "total_time_ms": ttft_ms + itl_ms * output_tokens,
            "output_tokens": output_tokens,
            "worker": {"decode_worker_id": worker},
        },
    }


def event(trajectory_id, invocation, ttft_ms, tpot_ms, status="ok", completion_tokens=100):
    return {
        "trajectory_id": trajectory_id, "agent_id": 0, "invocation": invocation, "sent_at_ms": 0,
        "ttft_ms": ttft_ms if status == "ok" else None, "last_token_ms": None,
        "e2e_ms": (ttft_ms + tpot_ms * completion_tokens) if status == "ok" else 50.0,
        "tpot_ms": tpot_ms if status == "ok" else None, "prompt_tokens": 10,
        "completion_tokens": completion_tokens if status == "ok" else None, "cached_tokens": None, "attempts": 1,
        "status": status, "error": None if status == "ok" else "HTTP 529", "http_status": 200 if status == "ok" else 529,
    }


class CompareTest(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix="batchbench-compare-"))
        # Trajectory a: a request before the window (link context) and two inside; trajectory b: one inside.
        manifest = [
            {"trajectory_id": "a", "requests": [], "metadata": {"requests": [
                {"request_id": "a0", "x_request_id": None, "received_ms": T0 - 60_000},
                {"request_id": "a1", "x_request_id": None, "received_ms": T0 + 1_000},
                {"request_id": "a2", "x_request_id": None, "received_ms": T0 + 2_000},
            ]}},
            {"trajectory_id": "b", "requests": [], "metadata": {"requests": [
                {"request_id": "b1", "x_request_id": None, "received_ms": T0 + 3_000},
            ]}},
        ]
        events = [event("a", 1, 900, 9), event("a", 2, 1000, 10), event("a", 3, 20_000, 10),
                  event("b", 1, 0, 0, status="error")]
        rows = [production_row("a0", T0 - 60_000, 500, 10),
                production_row("a1", T0 + 1_000, 2_000, 40, worker=1),
                production_row("a2", T0 + 2_000, 3_000, 20, worker=2),
                production_row("b1", T0 + 3_000, 1_000, 20, worker=2),
                # Not replayed, but served in the window: counted for the sets' throughput.
                production_row("c1", T0 + 4_000, 1_000, 20, worker=2),
                production_row("other-model", T0 + 4_000, 1_000, 20, worker=2, model="n")]
        self.manifest = self.write("plans.jsonl", manifest)
        self.events = self.write("events.jsonl", events)
        self.traces = self.write("traces.jsonl", rows)
        self.context = self.directory / "context.json"
        self.context.write_text(json.dumps({"worker_sets": {"1": "gh200", "2": "b300"},
                                            "set_gpus": {"gh200": 4, "b300": 2}}), encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def write(self, name, rows):
        path = self.directory / name
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return str(path)

    def run_cli(self, *extra):
        out = self.directory / "report.json"
        argv = ["--manifest", self.manifest, "--request-events", self.events, "--dynamo-traces", self.traces,
                "--score-start", WINDOW[0], "--score-end", WINDOW[1], "--json-out", str(out), *extra]
        with redirect_stdout(io.StringIO()) as printed:
            self.assertEqual(trace_replay.main(argv), 0)
        return json.loads(out.read_text(encoding="utf-8")), printed.getvalue()

    def test_scores_the_same_requests_on_both_sides(self):
        report, printed = self.run_cli()
        # a1, a2 and b1 arrived in the window; a0 is link context.
        self.assertEqual((report["replayed"], report["matched"]), (3, 3))
        replay, production = report["replay"], report["production"]
        self.assertEqual((replay["requests"], replay["failed"]), (3, 1))
        # The replay: a1 within, a2 over the first-token bound, b1 failed.
        self.assertAlmostEqual(replay["within_slo"], 1 / 3)
        # Production: a1 over the time-per-token bound (40 ms), a2 and b1 within.
        self.assertAlmostEqual(production["within_slo"], 2 / 3)
        self.assertEqual(production["ttft_s_p50"], 2.0)
        self.assertIn("3 matched", printed)

    def test_failed_requests_count_against_the_replay_but_not_its_latencies(self):
        report, _ = self.run_cli()
        self.assertEqual(report["replay"]["ttft_s_p99"], 20.0)
        self.assertEqual(report["replay"]["tpot_ms_p50"], 10.0)

    def test_context_splits_production_by_set_and_prices_each_per_gpu(self):
        report, _ = self.run_cli("--production-context", str(self.context), "--model", "m", "--replay-gpus", "2")
        self.assertEqual(sorted(report["production_by_set"]), ["b300", "gh200"])
        self.assertEqual(report["production_by_set"]["b300"]["production"]["requests"], 2)
        self.assertEqual(report["production_by_set"]["b300"]["replay"]["requests"], 2)
        throughput = report["production_set_throughput"]
        # Every request of model m that arrived in the window, replayed or not.
        self.assertEqual(throughput["b300"]["requests"], 3)
        self.assertAlmostEqual(throughput["b300"]["requests_per_s_per_gpu"], 3 / 600 / 2)
        self.assertAlmostEqual(throughput["gh200"]["requests_per_s_per_gpu"], 1 / 600 / 4)
        self.assertAlmostEqual(report["replay"]["requests_per_s_per_gpu"], 2 / 600 / 2)
        self.assertAlmostEqual(report["replay"]["within_slo_per_s_per_gpu"], 1 / 600 / 2)

    def test_unknown_workers_are_unmapped(self):
        self.context.write_text(json.dumps({"worker_sets": {"1": "gh200"}}), encoding="utf-8")
        report, _ = self.run_cli("--production-context", str(self.context))
        self.assertEqual(sorted(report["production_by_set"]), ["gh200", trace_replay.UNMAPPED])

    def test_rejects_an_empty_window(self):
        argv = ["--manifest", self.manifest, "--request-events", self.events, "--dynamo-traces", self.traces,
                "--score-start", WINDOW[1], "--score-end", WINDOW[0]]
        self.assertEqual(trace_replay.main(argv), 2)


if __name__ == "__main__":
    unittest.main()
