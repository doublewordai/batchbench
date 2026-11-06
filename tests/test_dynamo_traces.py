"""Tests for the Dynamo request-trace source. All traces are synthetic: hashes are made here."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from batchbench import export_plans
from batchbench.dynamo_traces import (
    TraceStats,
    analytics_window,
    apply_analytics,
    build_analytics_query,
    build_trace_plans,
    drop_duplicate_attempts,
    hash_seed,
    load_trace_records,
    normalize_request_id,
    record_from_trace,
    segment_cuts,
)
from batchbench.export_plans import ExportStats, validate_plans

from test_export_plans import find_agent_binary, word_level_tokenizer_json

T0 = datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc)
WINDOW_START = T0
WINDOW_END = T0 + timedelta(hours=1)
MODEL = "example-org/model-a"
OTHER_MODEL = "example-org/model-b"
USER_A = "aaaaaaaa-0000-4000-8000-000000000001"
USER_B = "bbbbbbbb-0000-4000-8000-000000000002"


def chain_of(labels, block_size=64):
    """Cumulative 64-bit hashes over block labels, like Dynamo's sequence-aware hashes."""

    hashes = []
    previous = f"bs{block_size}".encode()
    for label in labels:
        digest = hashlib.blake2b(previous + b"|" + label.encode(), digest_size=8).digest()
        hashes.append(int.from_bytes(digest, "big"))
        previous = digest
    return hashes


_request_counter = [0]


def trace(
    received,
    labels,
    block_size=64,
    tail=10,
    output_tokens=20,
    total_time_ms=1000.0,
    model=MODEL,
    x_request_id=None,
    session_id=None,
    hashes=None,
    finish_reason="stop",
):
    _request_counter[0] += 1
    hashes = chain_of(labels, block_size) if hashes is None else hashes
    request = {
        "request_id": f"req-{_request_counter[0]:04d}",
        "model": model,
        "request_received_ms": int(received.timestamp() * 1000),
        "total_time_ms": total_time_ms,
        "ttft_ms": 50.0,
        "input_tokens": block_size * len(hashes) + tail,
        "output_tokens": output_tokens,
        "cached_tokens": 0,
        "worker": {"decode_worker_id": 1},
        "replay": {
            "trace_block_size": block_size,
            "input_length": block_size * len(hashes) + tail,
            "input_sequence_hashes": hashes,
        },
    }
    if finish_reason:
        request["finish_reason_metadata"] = {"finish_reason": finish_reason}
    if x_request_id is not None:
        request["x_request_id"] = x_request_id
    row = {
        "schema": "dynamo.request.trace.v1",
        "event_type": "request_end",
        "event_time_unix_ms": request["request_received_ms"] + int(total_time_ms),
        "event_source": "dynamo",
        "request": request,
    }
    if session_id is not None:
        row["agent_context"] = {"session_id": session_id, "input_trigger": "user"}
    return row


def records_for(rows, stats=None):
    stats = stats if stats is not None else TraceStats()
    records = [record_from_trace(row, stats) for row in rows]
    return drop_duplicate_attempts([record for record in records if record is not None], stats)


def plans_for(rows, analytics=None, stats=None, trace_stats=None):
    trace_stats = trace_stats if trace_stats is not None else TraceStats()
    records = records_for(rows, trace_stats)
    if analytics is not None:
        apply_analytics(records, analytics, trace_stats)
    return build_trace_plans(records, WINDOW_START, WINDOW_END, stats=stats, trace_stats=trace_stats)


def seeds(labels, block_size=64):
    return [hash_seed(value) for value in chain_of(labels, block_size)]


class BlockTest(unittest.TestCase):
    def test_prompts_are_cut_where_chains_diverge_or_end(self):
        first = trace(T0, ["s1", "s2", "u1"], session_id="sess-1")
        second = trace(T0 + timedelta(seconds=5), ["s1", "s2", "u1", "x1", "x2"], tail=5, session_id="sess-1")
        branch = trace(T0 + timedelta(seconds=9), ["s1", "s2", "v1"], tail=0)
        trace_stats = TraceStats()
        plans = plans_for([first, second, branch], trace_stats=trace_stats)
        self.assertEqual(validate_plans(plans), [])
        self.assertEqual(len(plans), 2)
        session, other = plans
        s2, u1 = seeds(["s1", "s2", "u1"])[1:]
        x2 = seeds(["s1", "s2", "u1", "x1", "x2"])[-1]
        v1 = seeds(["s1", "s2", "v1"])[-1]

        turn1, turn2 = session["requests"]
        # The shared system prefix is one segment ending where the branch diverges.
        self.assertEqual(turn1["blocks"][:2], [
            {"seed": s2, "tokens": 128, "role": "user"},
            {"seed": u1, "tokens": 64, "role": "user"},
        ])
        tail1 = turn1["blocks"][2]
        self.assertEqual((tail1["tokens"], tail1["role"]), (10, "user"))
        self.assertEqual(turn1["prompt_tokens"], 3 * 64 + 10)
        self.assertEqual(turn1["overhead_tokens"], 0)

        # The extension replays the previous tail, then the live reply, then the new content.
        self.assertNotIn("reset_before", turn2)
        self.assertEqual(turn2["blocks"][:3], turn1["blocks"])
        live = turn2["blocks"][3]
        self.assertEqual((live["tokens"], live["role"], live["live"]), (20, "assistant", True))
        rest = turn2["blocks"][4]
        self.assertEqual(rest, {"seed": f"{x2}:rest{128 - 10 - 20}", "tokens": 98, "role": "user"})
        self.assertEqual(turn2["blocks"][5]["tokens"], 5)
        self.assertEqual(sum(block["tokens"] for block in turn2["blocks"]), turn2["prompt_tokens"])
        self.assertEqual(turn2["overhead_tokens"], 0)
        self.assertEqual(trace_stats.live_blocks, 1)

        # The other branch shares the system segment's seed, so its bytes replay identically.
        self.assertEqual([block["seed"] for block in other["requests"][0]["blocks"]], [s2, v1])
        self.assertEqual(session["metadata"], {"source": "dynamo-request-trace", "model": MODEL, "block_size": 64})

    def test_cuts_are_per_model_and_block_size(self):
        records = records_for([
            trace(T0, ["a", "b"]),
            trace(T0, ["a", "c"]),
            trace(T0, ["a", "b"], model=OTHER_MODEL),
        ])
        cuts = segment_cuts(records)
        a, b = chain_of(["a", "b"])
        c = chain_of(["a", "c"])[1]
        self.assertEqual(cuts[(MODEL, 64)], {a, b, c})
        self.assertEqual(cuts[(OTHER_MODEL, 64)], {b})

    def test_later_turns_repeat_the_blocks_sent_before(self):
        rows = [
            trace(T0, ["s1", "u1"], tail=10, session_id="sess-1"),
            trace(T0 + timedelta(seconds=5), ["s1", "u1", "x1", "x2"], tail=20, session_id="sess-1"),
            trace(T0 + timedelta(seconds=9), ["s1", "u1", "x1", "x2", "y1", "y2"], tail=3, session_id="sess-1"),
        ]
        trace_stats = TraceStats()
        plans = plans_for(rows, trace_stats=trace_stats)
        self.assertEqual(validate_plans(plans), [])
        turn1, turn2, turn3 = plans[0]["requests"]
        # Turn three keeps turn two's reply and tails instead of re-cutting its prefix.
        self.assertEqual(turn3["blocks"][: len(turn2["blocks"])], turn2["blocks"])
        self.assertEqual(turn2["blocks"][: len(turn1["blocks"])], turn1["blocks"])
        self.assertEqual([block.get("live", False) for block in turn3["blocks"]].count(True), 2)
        self.assertTrue(turn3["blocks"][len(turn2["blocks"])]["live"])
        for request in (turn1, turn2, turn3):
            self.assertEqual(sum(block["tokens"] for block in request["blocks"]), request["prompt_tokens"])
        self.assertEqual(trace_stats.live_blocks, 2)

    def test_a_reply_longer_than_the_extension_is_generated_not_live(self):
        first = trace(T0, ["s1"], tail=30, output_tokens=500, session_id="sess-1")
        second = trace(T0 + timedelta(seconds=5), ["s1", "u2"], tail=7, session_id="sess-1")
        trace_stats = TraceStats()
        plans = plans_for([first, second], trace_stats=trace_stats)
        blocks = plans[0]["requests"][1]["blocks"]
        # 64 + 7 extension tokens: 30 are the previous tail, and the 500-token reply would not fit
        # in the other 41, so they are generated rather than replaying the real reply.
        self.assertEqual([(block["tokens"], block["role"], block.get("live", False)) for block in blocks], [
            (64, "user", False), (30, "user", False), (41, "assistant", False),
        ])
        self.assertEqual((trace_stats.live_clamped, trace_stats.live_blocks), (1, 0))
        self.assertEqual(validate_plans(plans), [])


class LinkingTest(unittest.TestCase):
    def test_non_default_block_size_and_mixed_block_sizes_never_link(self):
        short = trace(T0, ["p1", "p2"], block_size=1056, tail=100)
        hashes = chain_of(["p1", "p2", "p3"], 1056)
        longer = trace(T0 + timedelta(seconds=3), [], block_size=1056, hashes=hashes, tail=0)
        # Same hash values under another block size: a different pool, never the same session.
        foreign = trace(T0 + timedelta(seconds=6), [], block_size=128, hashes=hashes + [7], tail=0)
        plans = plans_for([short, longer, foreign])
        self.assertEqual(validate_plans(plans), [])
        self.assertEqual(sorted(len(plan["requests"]) for plan in plans), [1, 2])
        linked = next(plan for plan in plans if len(plan["requests"]) == 2)
        self.assertEqual(linked["metadata"]["block_size"], 1056)
        self.assertEqual(linked["requests"][0]["blocks"][0]["tokens"], 2 * 1056)
        alone = next(plan for plan in plans if len(plan["requests"]) == 1)
        self.assertEqual(alone["metadata"]["block_size"], 128)
        self.assertEqual(sum(block["tokens"] for block in alone["requests"][0]["blocks"]), 4 * 128)
        self.assertFalse(any(block.get("live") for block in alone["requests"][0]["blocks"]))

    def test_sessions_and_principals_separate_otherwise_matching_chains(self):
        first = trace(T0, ["s1", "u1"], x_request_id="00000000-0000-7000-8000-00000000000A")
        second = trace(T0 + timedelta(seconds=5), ["s1", "u1", "x1"], x_request_id="00000000-0000-7000-8000-00000000000b")
        # Without the join both rows have no principal and link on their chains alone.
        self.assertEqual([len(plan["requests"]) for plan in plans_for([first, second])], [2])

        analytics = [
            {"x_request_id": "00000000-0000-7000-8000-00000000000a", "instance_id": "i-1", "correlation_id": 1, "model": "alias-a", "user_id": USER_A, "api_key_id": None},
            {"x_request_id": "00000000-0000-7000-8000-00000000000b", "instance_id": "i-1", "correlation_id": 2, "model": "alias-a", "user_id": USER_B, "api_key_id": None},
        ]
        trace_stats = TraceStats()
        plans = plans_for([first, second], analytics=analytics, trace_stats=trace_stats)
        self.assertEqual([len(plan["requests"]) for plan in plans], [1, 1])
        self.assertEqual(trace_stats.analytics_matched, 2)
        self.assertEqual(plans[0]["metadata"]["alias"], "alias-a")
        self.assertEqual(plans[0]["metadata"]["principal"], USER_A)

        # Agent session ids keep two sessions apart even without a principal.
        one = trace(T0, ["s1", "u1"], session_id="sess-1")
        two = trace(T0 + timedelta(seconds=5), ["s1", "u1", "x1"], session_id="sess-2")
        self.assertEqual([len(plan["requests"]) for plan in plans_for([one, two])], [1, 1])

    def test_retried_attempts_keep_the_last_one_that_finished(self):
        request_id = "00000000-0000-7000-8000-0000000000c1"
        failed = trace(T0, ["s1", "u1"], output_tokens=0, finish_reason=None, x_request_id=request_id)
        final = trace(T0 + timedelta(seconds=2), ["s1", "u1"], output_tokens=33, x_request_id=request_id.upper())
        unrelated = trace(T0 + timedelta(seconds=3), ["s1", "w1"])
        trace_stats = TraceStats()
        records = records_for([failed, final, unrelated], trace_stats)
        self.assertEqual(trace_stats.duplicate_attempts, 1)
        self.assertEqual(trace_stats.with_x_request_id, 2)
        self.assertEqual(len(records), 2)
        kept = next(record for record in records if record.x_request_id)
        self.assertEqual((kept.completion_tokens, kept.x_request_id), (33, request_id))
        self.assertEqual([record.index for record in records], [0, 1])


class SourceTest(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix="batchbench-traces-"))

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def write(self, name, rows, gz=True):
        path = self.directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        data = "".join(json.dumps(row) + "\n" for row in rows).encode()
        path.write_bytes(gzip.compress(data) if gz else data)
        return path

    def test_rows_are_windowed_by_receipt_not_file_name(self):
        # Files are named by upload time; the 13:10 upload holds a row received at 11:55.
        self.write("2026/09/01/frontend-a-121000-run1-0.jsonl.gz", [trace(T0 + timedelta(minutes=5), ["s1", "u1", "x1"], session_id="sess-1")])
        self.write("2026/09/01/frontend-a-131000-run1-1.jsonl.gz", [
            trace(T0 - timedelta(minutes=5), ["s1", "u1"], session_id="sess-1"),
            trace(T0 + timedelta(hours=1, minutes=1), ["s1", "late"]),
            trace(T0 - timedelta(hours=30), ["s1", "stale"]),
            trace(T0 + timedelta(minutes=7), ["s1", "u1"], model=OTHER_MODEL),
            {"schema": "dynamo.request.trace.v1", "event_type": "tool_start", "event_time_unix_ms": 0},
        ])
        (self.directory / "2026/09/01/broken.jsonl.gz").write_bytes(gzip.compress(b"{}\n")[:-12])
        trace_stats = TraceStats()
        records = load_trace_records([str(self.directory)], WINDOW_START, WINDOW_END, model=MODEL, stats=trace_stats)
        self.assertEqual(trace_stats.files, 3)
        self.assertEqual(trace_stats.unreadable_files, 1)
        self.assertEqual(trace_stats.loaded, 2)
        self.assertEqual(trace_stats.outside_window, 2)
        self.assertEqual(trace_stats.other_model, 1)
        self.assertEqual(trace_stats.not_request_end, 1)
        self.assertEqual(trace_stats.block_sizes, {64: 2})

        stats = ExportStats()
        plans = build_trace_plans(drop_duplicate_attempts(records, trace_stats), WINDOW_START, WINDOW_END, stats=stats, trace_stats=trace_stats)
        self.assertEqual(len(plans), 1)
        self.assertEqual(stats.prewindow_sessions, 1)
        first = plans[0]["requests"][0]
        self.assertTrue(first["reset_before"])
        self.assertFalse(any(block.get("live") for block in first["blocks"]))
        self.assertEqual(plans[0]["start_after_ms"], 5 * 60_000)

    def test_glob_and_missing_paths(self):
        self.write("a.jsonl", [trace(T0, ["s1"])], gz=False)
        records = load_trace_records([str(self.directory / "*.jsonl")], WINDOW_START, WINDOW_END)
        self.assertEqual(len(records), 1)
        with self.assertRaises(export_plans.ExportError):
            load_trace_records([str(self.directory / "nothing-*.gz")], WINDOW_START, WINDOW_END)

    def run_cli(self, *extra):
        output = self.directory / "plans.jsonl"
        code = export_plans.main([
            "--dynamo-traces", str(self.directory / "traces"),
            "--start", WINDOW_START.isoformat(),
            "--end", WINDOW_END.isoformat(),
            "--model", MODEL,
            "--output", str(output),
            *extra,
        ])
        lines = [json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
        return code, lines

    def session_rows(self):
        return [
            trace(T0, ["s1", "u1"], x_request_id="00000000-0000-7000-8000-000000000001", session_id="sess-1"),
            trace(T0 + timedelta(seconds=4), ["s1", "u1", "x1"], x_request_id="00000000-0000-7000-8000-000000000002", session_id="sess-1"),
            trace(T0 + timedelta(seconds=2), ["s1", "q1"], block_size=256, tail=200),
            trace(T0 + timedelta(seconds=3), ["s1", "q1", "q2"], block_size=256, tail=3),
        ]

    def test_cli_exports_from_traces_alone_and_with_analytics_rows(self):
        self.write("traces/frontend-a-120100-run1-0.jsonl.gz", self.session_rows())
        code, plans = self.run_cli()
        self.assertEqual(code, 0)
        self.assertEqual(validate_plans(plans), [])
        self.assertEqual(sorted(len(plan["requests"]) for plan in plans), [2, 2])
        self.assertTrue(all("principal" not in plan["metadata"] for plan in plans))

        analytics = self.directory / "analytics.jsonl"
        analytics.write_text(json.dumps({"x_request_id": "00000000-0000-7000-8000-000000000001", "instance_id": "i-1", "correlation_id": 9, "model": "alias-a", "user_id": USER_A, "api_key_id": None}) + "\n")
        code, plans = self.run_cli("--analytics-rows-jsonl", str(analytics))
        self.assertEqual(code, 0)
        # One turn matched a principal, the other did not, so the session splits.
        self.assertEqual(sorted(len(plan["requests"]) for plan in plans), [1, 1, 2])
        self.assertEqual(validate_plans(plans), [])
        code, plans = self.run_cli("--analytics-rows-jsonl", str(analytics), "--principal-id", USER_A)
        self.assertEqual((code, len(plans)), (0, 1))
        self.assertEqual(plans[0]["metadata"]["alias"], "alias-a")

    def test_cli_rejects_mixed_sources(self):
        self.write("traces/a.jsonl.gz", self.session_rows())
        code, _ = self.run_cli("--served-by", "upstream-a")
        self.assertEqual(code, 1)
        code, _ = self.run_cli("--analytics-table", "requests")
        self.assertEqual(code, 1)  # no ClickHouse URL

    def test_exported_manifest_dry_runs_in_batchbench_agent(self):
        binary = find_agent_binary()
        if binary is None:
            self.skipTest("batchbench-agent binary not built")
        self.write("traces/a.jsonl.gz", self.session_rows())
        code, _ = self.run_cli()
        self.assertEqual(code, 0)
        tokenizer = self.directory / "tokenizer.json"
        tokenizer.write_text(json.dumps(word_level_tokenizer_json()), encoding="utf-8")
        result = subprocess.run(
            [binary, "--model", "test-model", "--tokenizer-model", str(tokenizer), "--host", "http://127.0.0.1:9",
             "--agent-plans-jsonl", str(self.directory / "plans.jsonl"), "--admission", "open-loop",
             "--time-scale", "1000", "--dry-run"],
            capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn("Agents: 2 completed / 2 total", result.stdout)
        # Each session's second turn replays the first turn's real reply.
        self.assertEqual(result.stdout.count("live_blocks=1"), 2)


class AnalyticsQueryTest(unittest.TestCase):
    def test_query_and_window(self):
        sql = build_analytics_query("requests", "x_request_id", "timestamp")
        self.assertIn("toString(x_request_id) AS x_request_id", sql)
        self.assertIn("FROM requests", sql)
        self.assertIn("timestamp >= {start:DateTime64(3)}", sql)
        self.assertIn("x_request_id IS NOT NULL", sql)
        records = records_for([
            trace(T0, ["a"], x_request_id="00000000-0000-7000-8000-000000000001"),
            trace(T0 + timedelta(minutes=30), ["b"], total_time_ms=60_000.0, x_request_id="00000000-0000-7000-8000-000000000002"),
            trace(T0 + timedelta(hours=5), ["c"]),
        ])
        start, end = analytics_window(records)
        self.assertEqual(start, T0 - timedelta(minutes=15))
        self.assertEqual(end, T0 + timedelta(minutes=46))
        self.assertIsNone(analytics_window(records_for([trace(T0, ["d"])])))
        self.assertEqual(normalize_request_id(" 0000000A-0000-7000-8000-000000000001 "), "0000000a-0000-7000-8000-000000000001")
        self.assertEqual(normalize_request_id("not-a-uuid"), "not-a-uuid")
        self.assertIsNone(normalize_request_id(""))


if __name__ == "__main__":
    unittest.main()
