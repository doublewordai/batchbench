"""Tests for the ClickHouse-to-manifest exporter. No live ClickHouse: rows are fixtures."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from batchbench import export_plans
from batchbench.export_plans import (
    ExportStats,
    build_plans,
    build_query,
    build_trajectories,
    link_parents,
    records_from_rows,
    sample_plans,
    session_length_bucket,
    validate_plan,
    validate_plans,
)

T0 = datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc)
WINDOW_START = T0
WINDOW_END = T0 + timedelta(hours=1)
PRINCIPAL = "11111111-1111-1111-1111-111111111111"
OTHER_PRINCIPAL = "22222222-2222-2222-2222-222222222222"

# Content hashes are cumulative, so a chain is identified by its leaf; use readable labels.
SYS, U1, A1, U2, A2, C1, R1, U1B = "sys", "u1", "a1", "u2", "a2", "c1", "r1", "u1b"

_counter = [0]


def row(
    ts,
    chain,
    roles,
    tokens=None,
    prompt_tokens=None,
    completion_tokens=8,
    duration_ms=1000.0,
    principal=PRINCIPAL,
    stream=None,
    max_tokens=None,
    served_by="dynamo",
    model="glm-5.2",
):
    _counter[0] += 1
    if tokens is None:
        tokens = [10 * (index + 1) for index in range(len(chain))]
    if prompt_tokens is None:
        prompt_tokens = sum(tokens) + 5
    return {
        "ts": ts.isoformat(),
        "instance_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "correlation_id": _counter[0],
        "principal_id": principal,
        "model": model,
        "chain": chain,
        "block_roles": roles,
        "block_tokens": tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "duration_ms": duration_ms,
        "stream": stream,
        "max_tokens": max_tokens,
        "served_by": served_by,
        "finish_reason": "stop",
        "request_origin": "realtime",
        "user_id": None,
        "api_key_id": None,
        "duration_to_first_byte_ms": 100.0,
    }


def two_turn_session(start=T0, principal=PRINCIPAL, **extra):
    """Turn 1: sys+user; turn 2 adds the assistant reply and a new user message."""

    first = row(start, [SYS, U1], ["system", "user"], principal=principal, **extra)
    second = row(
        start + timedelta(seconds=5),
        [SYS, U1, A1, U2],
        ["system", "user", "assistant", "user"],
        principal=principal,
        **extra,
    )
    return [first, second]


def plans_for(rows, time_scale=1.0, stats=None, start=WINDOW_START, end=WINDOW_END):
    return build_plans(records_from_rows(rows), start, end, time_scale, stats=stats)


class SessionReconstructionTest(unittest.TestCase):
    def test_strict_prefix_links_continuations_within_a_principal(self):
        rows = two_turn_session()
        third = row(T0 + timedelta(seconds=20), [SYS, U1, A1, U2, A2, U1B], ["system", "user", "assistant", "user", "assistant", "user"])
        identical_root = row(T0 + timedelta(seconds=30), [SYS, U1], ["system", "user"])
        other_principal = row(T0 + timedelta(seconds=40), [SYS, U1, A1, U2], ["system", "user", "assistant", "user"], principal=OTHER_PRINCIPAL)
        records = records_from_rows(rows + [third, identical_root, other_principal])

        parents = link_parents(records)
        self.assertEqual(parents[0], None)
        self.assertEqual(parents[1], 0)
        self.assertEqual(parents[2], 1)
        # An identical chain is not a strict extension: it is a new root, not a retry turn.
        self.assertEqual(parents[3], None)
        # Chains never link across principals.
        self.assertEqual(parents[4], None)

        plans = plans_for(rows + [third, identical_root, other_principal])
        self.assertEqual([len(plan["requests"]) for plan in plans], [3, 1, 1])
        self.assertEqual(plans[0]["trajectory_id"], f"{U1}-0")
        self.assertEqual(plans[1]["trajectory_id"], f"{U1}-1")
        self.assertEqual(plans[2]["trajectory_id"], f"{U2}-0")

    def test_latest_matching_parent_wins_and_link_window_applies(self):
        early = row(T0, [SYS, U1], ["system", "user"])
        later_same = row(T0 + timedelta(seconds=10), [SYS, U1], ["system", "user"])
        child = row(T0 + timedelta(seconds=20), [SYS, U1, A1, U2], ["system", "user", "assistant", "user"])
        records = records_from_rows([early, later_same, child])
        parents = link_parents(records)
        self.assertEqual(parents[2], 1)

        stale = row(T0 - timedelta(hours=30), [SYS, U1], ["system", "user"])
        records = records_from_rows([stale, child])
        self.assertEqual(link_parents(records)[1], None)
        self.assertEqual(link_parents(records, timedelta(hours=48))[1], 0)

    def test_sessions_that_began_before_the_window_reset_their_first_request(self):
        before = row(T0 - timedelta(minutes=10), [SYS, U1], ["system", "user"])
        inside = row(T0 + timedelta(minutes=1), [SYS, U1, A1, U2], ["system", "user", "assistant", "user"])
        records = records_from_rows([before, inside])
        trajectories = build_trajectories(records, link_parents(records), WINDOW_START, WINDOW_END)
        self.assertEqual(len(trajectories), 1)
        self.assertTrue(trajectories[0].began_before_window)
        self.assertEqual([r.index for r in trajectories[0].requests], [1])

        stats = ExportStats()
        plans = plans_for([before, inside], stats=stats)
        self.assertEqual(len(plans), 1)
        first = plans[0]["requests"][0]
        self.assertTrue(first["reset_before"])
        # The reply to a pre-window request is not live: nothing in the replay produced it.
        self.assertFalse(any(block.get("live") for block in first["blocks"]))
        self.assertEqual(stats.prewindow_sessions, 1)
        self.assertEqual(plans[0]["start_after_ms"], 60_000)

    def test_branching_child_resets_and_carries_no_live_block(self):
        root, child_a = two_turn_session()
        child_b = row(T0 + timedelta(seconds=9), [SYS, U1, A1, "u2-alt"], ["system", "user", "assistant", "user"])
        child_b["block_tokens"] = [12, 20, 30, 40]  # re-tokenized system block across a reset
        stats = ExportStats()
        plans = plans_for([root, child_a, child_b], stats=stats)
        self.assertEqual(validate_plans(plans), [])
        self.assertEqual(stats.normalized_block_tokens, 1)
        self.assertEqual(plans[0]["requests"][2]["blocks"][0]["tokens"], 10)
        self.assertEqual(len(plans), 1)
        requests = plans[0]["requests"]
        self.assertEqual(len(requests), 3)
        self.assertNotIn("reset_before", requests[1])
        self.assertTrue(requests[1]["blocks"][2]["live"])
        self.assertTrue(requests[2]["reset_before"])
        self.assertFalse(any(block.get("live") for block in requests[2]["blocks"]))


class PlanShapeTest(unittest.TestCase):
    def test_live_block_is_the_reply_to_the_previous_request(self):
        plans = plans_for(two_turn_session(stream=True, max_tokens=256))
        requests = plans[0]["requests"]
        first, second = requests
        self.assertEqual(first["blocks"], [
            {"seed": SYS, "tokens": 10, "role": "system"},
            {"seed": U1, "tokens": 20, "role": "user"},
        ])
        self.assertEqual(first["overhead_tokens"], 5)
        self.assertEqual(first["prompt_tokens"], 35)
        self.assertEqual(first["output_tokens"], 8)
        self.assertTrue(first["stream"])
        self.assertEqual(first["max_tokens"], 256)
        self.assertEqual(second["blocks"][2], {"seed": A1, "tokens": 30, "role": "assistant", "live": True})
        self.assertEqual([block.get("live", False) for block in second["blocks"]], [False, False, True, False])
        self.assertNotIn("delay_after_ms", second)
        self.assertNotIn("reset_before", first)

    def test_delays_and_offsets_follow_timestamps_and_time_scale(self):
        rows = two_turn_session(start=T0 + timedelta(seconds=30))
        plans = plans_for(rows)
        self.assertEqual(plans[0]["start_after_ms"], 30_000)
        # next.ts - (this.ts + duration_ms) = 5000 - 1000
        self.assertEqual(plans[0]["requests"][0]["delay_after_ms"], 4_000)

        scaled = plans_for(rows, time_scale=4.0)
        self.assertEqual(scaled[0]["start_after_ms"], 7_500)
        self.assertEqual(scaled[0]["requests"][0]["delay_after_ms"], 1_000)

        overlapping = two_turn_session()
        overlapping[0]["duration_ms"] = 9_000.0
        self.assertEqual(plans_for(overlapping)[0]["requests"][0]["delay_after_ms"], 0)

    def test_requests_without_block_tokens_get_a_single_user_block(self):
        rows = [row(T0, [SYS, U1], ["system", "user"], tokens=[], prompt_tokens=123)]
        request = plans_for(rows)[0]["requests"][0]
        self.assertEqual(request["blocks"], [{"seed": U1, "tokens": 123, "role": "user"}])
        self.assertEqual(request["overhead_tokens"], 0)
        self.assertEqual(request["prompt_tokens"], 123)

    def test_unusable_rows_are_dropped_and_counted(self):
        good = row(T0, [SYS, U1], ["system", "user"])
        no_completion = row(T0 + timedelta(seconds=1), [SYS, "x"], ["system", "user"], completion_tokens=0)
        no_prompt = row(T0 + timedelta(seconds=2), [SYS, "y"], ["system", "user"], tokens=[])
        no_prompt["prompt_tokens"] = None
        malformed = row(T0 + timedelta(seconds=3), [SYS, "z"], ["system"], tokens=[1, 2])
        stats = ExportStats()
        plans = plans_for([good, no_completion, no_prompt, malformed], stats=stats)
        self.assertEqual(len(plans), 1)
        self.assertEqual(stats.rows, 4)
        self.assertEqual(stats.usable, 1)
        self.assertEqual(stats.dropped_no_completion_tokens, 1)
        self.assertEqual(stats.dropped_no_prompt_tokens, 1)
        self.assertEqual(stats.dropped_malformed_blocks, 1)

    def test_block_tokens_are_normalized_per_seed_within_a_trajectory(self):
        first, second = two_turn_session()
        second["block_tokens"] = [11, 20, 30, 40]  # system block re-tokenized differently
        stats = ExportStats()
        plans = plans_for([first, second], stats=stats)
        blocks = plans[0]["requests"][1]["blocks"]
        self.assertEqual(blocks[0]["tokens"], 10)
        self.assertEqual(stats.normalized_block_tokens, 1)
        # prompt_tokens (105) minus the normalized block sum (100)
        self.assertEqual(plans[0]["requests"][1]["overhead_tokens"], 5)
        self.assertEqual(validate_plans(plans), [])

    def test_output_is_ordered_by_offset_and_validates(self):
        rows = two_turn_session(start=T0 + timedelta(minutes=5)) + two_turn_session(start=T0 + timedelta(minutes=2), principal=OTHER_PRINCIPAL)
        plans = plans_for(rows)
        self.assertEqual([plan["start_after_ms"] for plan in plans], [120_000, 300_000])
        self.assertEqual(validate_plans(plans), [])
        self.assertTrue(all(plan["schema_version"] == 2 for plan in plans))


class SamplingTest(unittest.TestCase):
    def make_plans(self, count):
        plans = []
        for index in range(count):
            requests = [{"prompt_tokens": 10, "output_tokens": 1, "blocks": [{"seed": f"s{index}", "tokens": 10, "role": "user"}]}] * (1 + index % 6)
            plans.append({"schema_version": 2, "trajectory_id": f"t{index}", "start_after_ms": index, "requests": requests})
        return plans

    def test_sampling_is_deterministic_and_seed_dependent(self):
        plans = self.make_plans(40)
        first = sample_plans(plans, 0.25, seed=7)
        second = sample_plans(list(reversed(plans)), 0.25, seed=7)
        self.assertEqual(len(first), 10)
        self.assertEqual([p["trajectory_id"] for p in first], [p["trajectory_id"] for p in second])
        self.assertEqual([p["start_after_ms"] for p in first], sorted(p["start_after_ms"] for p in first))
        other_seed = sample_plans(plans, 0.25, seed=8)
        self.assertNotEqual([p["trajectory_id"] for p in first], [p["trajectory_id"] for p in other_seed])
        self.assertEqual(sample_plans(plans, None), plans)
        self.assertEqual(sample_plans(plans, 1.0), plans)
        self.assertEqual(sample_plans(plans, 0.0), [])
        for bad in (1.5, -0.1, float("nan"), float("inf")):
            with self.assertRaises(export_plans.ExportError):
                sample_plans(plans, bad)

    def test_stratified_sampling_keeps_every_length_bucket(self):
        plans = self.make_plans(40)
        kept = sample_plans(plans, 0.1, seed=1, stratify_by_session_length=True)
        buckets = {session_length_bucket(len(p["requests"])) for p in plans}
        kept_buckets = {session_length_bucket(len(p["requests"])) for p in kept}
        self.assertEqual(kept_buckets, buckets)
        self.assertEqual(kept, sample_plans(plans, 0.1, seed=1, stratify_by_session_length=True))
        self.assertEqual([session_length_bucket(n) for n in (1, 2, 3, 4, 7, 8, 16)], [0, 1, 1, 2, 2, 3, 4])


class ValidatorTest(unittest.TestCase):
    def valid(self):
        return json.loads(json.dumps(plans_for(two_turn_session())[0]))

    def test_valid_plan_has_no_errors(self):
        self.assertEqual(validate_plan(self.valid()), [])

    def test_validator_mirrors_the_rust_rules(self):
        cases = {
            "unknown trajectory fields": lambda p: p.update(extra=1),
            "schema_version must be 2": lambda p: p.update(schema_version=1),
            "must be a positive integer": lambda p: p["requests"][0].update(output_tokens=0),
            "unknown fields": lambda p: p["requests"][0].update(typo=True),
            "is not one of": lambda p: p["requests"][0]["blocks"][0].update(role="narrator"),
            "is live but has role": lambda p: p["requests"][0]["blocks"][0].update(live=True),
            "redefines seed": lambda p: p["requests"][1]["blocks"][0].update(tokens=99),
            "omit blocks": lambda p: p["requests"][0].pop("blocks"),
            "non-empty list": lambda p: p["requests"][1].update(blocks=[]),
            "must be a boolean": lambda p: p["requests"][0].update(stream="yes"),
        }
        for expected, mutate in cases.items():
            plan = self.valid()
            mutate(plan)
            errors = validate_plan(plan)
            self.assertTrue(any(expected in error for error in errors), f"{expected}: {errors}")
        duplicate = [self.valid(), self.valid()]
        self.assertTrue(any("duplicate trajectory_id" in e for e in validate_plans(duplicate)))


class QueryTest(unittest.TestCase):
    def test_query_applies_filters_as_parameters(self):
        sql, parameters = build_query("clay.prompt_chains", "clay.http_analytics", PRINCIPAL, "glm-5.2", "dynamo", chains_final=True)
        self.assertIn("FROM clay.prompt_chains FINAL", sql)
        self.assertIn("principal_id = {principal_id:UUID}", sql)
        self.assertIn("model = {model:String}", sql)
        self.assertIn("h.served_by = {served_by:String}", sql)
        # Both join sides are bounded by the window before joining.
        self.assertIn("timestamp >= {lookback_start:DateTime64(3)} - INTERVAL 1 DAY", sql)
        self.assertEqual(parameters, {"principal_id": PRINCIPAL, "model": "glm-5.2", "served_by": "dynamo"})
        sql, parameters = build_query(export_plans.DEFAULT_CHAINS_TABLE, "clay.http_analytics", None, None, None, analytics_ts_column="ts")
        self.assertIn("FROM clay.prompt_chains_current\n", sql)
        self.assertNotIn("FINAL", sql)
        self.assertNotIn("WHERE h.", sql)
        self.assertIn("ts >= {lookback_start:DateTime64(3)} - INTERVAL 1 DAY", sql)
        self.assertEqual(parameters, {})


def find_agent_binary():
    explicit = os.environ.get("BATCHBENCH_AGENT_BIN")
    if explicit and Path(explicit).is_file():
        return explicit
    root = Path(__file__).resolve().parents[1]
    for profile in ("release", "debug"):
        candidate = root / "rust" / "target" / profile / "batchbench-agent"
        if candidate.is_file():
            return str(candidate)
    return shutil.which("batchbench-agent")


def word_level_tokenizer_json():
    vocab = {"[UNK]": 0, "alpha": 1, "beta": 2, "gamma": 3}
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"},
    }


class CliTest(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix="batchbench-export-"))

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def write_rows(self, rows):
        path = self.directory / "rows.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    def test_cli_exports_from_rows_and_applies_filters(self):
        rows = two_turn_session() + two_turn_session(start=T0 + timedelta(minutes=3), principal=OTHER_PRINCIPAL, served_by="openrouter")
        rows_path = self.write_rows(rows)
        output = self.directory / "plans.jsonl"
        code = export_plans.main([
            "--rows-jsonl", str(rows_path),
            "--start", WINDOW_START.isoformat(),
            "--end", WINDOW_END.isoformat(),
            "--served-by", "dynamo",
            "--time-scale", "2",
            "--output", str(output),
        ])
        self.assertEqual(code, 0)
        lines = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["requests"][0]["delay_after_ms"], 2_000)
        self.assertEqual(validate_plans(lines), [])

    def test_cli_rejects_an_empty_export(self):
        rows_path = self.write_rows(two_turn_session(principal=OTHER_PRINCIPAL))
        output = self.directory / "plans.jsonl"
        code = export_plans.main([
            "--rows-jsonl", str(rows_path),
            "--start", WINDOW_START.isoformat(),
            "--end", WINDOW_END.isoformat(),
            "--principal-id", PRINCIPAL,
            "--output", str(output),
        ])
        self.assertEqual(code, 1)
        self.assertFalse(output.exists())

    def test_cli_rejects_a_bad_window(self):
        rows_path = self.write_rows(two_turn_session())
        code = export_plans.main([
            "--rows-jsonl", str(rows_path),
            "--start", WINDOW_END.isoformat(),
            "--end", WINDOW_START.isoformat(),
            "--output", str(self.directory / "plans.jsonl"),
        ])
        self.assertEqual(code, 1)

    def test_exported_manifest_dry_runs_in_batchbench_agent(self):
        binary = find_agent_binary()
        if binary is None:
            self.skipTest("batchbench-agent binary not built")
        rows = two_turn_session(stream=True, max_tokens=64) + two_turn_session(start=T0 + timedelta(seconds=1), principal=OTHER_PRINCIPAL)
        rows.append(row(T0 - timedelta(minutes=5), ["pre", "pu"], ["system", "user"], principal="33333333-3333-3333-3333-333333333333"))
        rows.append(row(T0 + timedelta(minutes=2), ["pre", "pu", "pa", "pu2"], ["system", "user", "assistant", "user"], principal="33333333-3333-3333-3333-333333333333"))
        output = self.directory / "plans.jsonl"
        code = export_plans.main([
            "--rows-jsonl", str(self.write_rows(rows)),
            "--start", WINDOW_START.isoformat(),
            "--end", WINDOW_END.isoformat(),
            "--output", str(output),
        ])
        self.assertEqual(code, 0)
        tokenizer = self.directory / "tokenizer.json"
        tokenizer.write_text(json.dumps(word_level_tokenizer_json()), encoding="utf-8")
        for admission in ("closed-loop", "open-loop"):
            result = subprocess.run(
                [
                    binary,
                    "--model", "test-model",
                    "--tokenizer-model", str(tokenizer),
                    "--host", "http://127.0.0.1:9",
                    "--agent-plans-jsonl", str(output),
                    "--admission", admission,
                    "--time-scale", "1000",
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertIn("Agents: 3 completed / 3 total", result.stdout)
            self.assertIn("live_blocks=1", result.stdout)


if __name__ == "__main__":
    unittest.main()
