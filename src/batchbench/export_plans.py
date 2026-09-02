"""Export batchbench trajectory manifests (schema version 2) from ClickHouse prompt-chain records.

One ``clay.prompt_chains`` row describes one chat-completions request as a chain of keyed
content hashes, one per prompt block, joined with ``clay.http_analytics`` for its token counts
and request scalars. This module turns a time window of those rows into trajectories that
``batchbench-agent --agent-plans-jsonl`` can replay:

* Within a principal, request B continues request A when A is the latest earlier request
  (within the link window) whose full chain is a strict prefix of B's chain. Cumulative hashing
  makes that a lookup of A's leaf hash inside B's chain. Each root and its descendants form one
  trajectory ordered by timestamp.
* Every block becomes ``{seed, tokens, role}`` with the chain hash as the seed, so requests that
  share a prefix replay with identical bytes. The block that is the reply to the previous
  request in the trajectory is marked ``live`` and replays with the model's real reply.
* ``start_after_ms`` and ``delay_after_ms`` reproduce the recorded arrival pattern, divided by
  ``--time-scale``.

The ClickHouse client is confined to :func:`fetch_rows`; everything else works on plain row
dictionaries so it can be tested with fixtures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Optional, Sequence

SCHEMA_VERSION = 2
DEFAULT_LINK_WINDOW = timedelta(hours=24)
DEFAULT_CHAINS_TABLE = "clay.prompt_chains_current"
DEFAULT_ANALYTICS_TABLE = "clay.http_analytics"
DEFAULT_ANALYTICS_TS_COLUMN = "timestamp"
BLOCK_ROLES = ("tool_definition", "system", "user", "assistant", "tool", "tool_call")

TRAJECTORY_FIELDS = {"schema_version", "trajectory_id", "start_after_ms", "requests", "metadata"}
REQUEST_FIELDS = {
    "prompt_tokens",
    "output_tokens",
    "delay_after_ms",
    "reset_before",
    "overhead_tokens",
    "stream",
    "max_tokens",
    "blocks",
}
BLOCK_FIELDS = {"seed", "tokens", "role", "live"}


class ExportError(Exception):
    """Raised for invalid inputs or an export that would not replay."""


# ---------------------------------------------------------------------------
# Records


@dataclass
class ChainRecord:
    """One prompt-chain row joined with its analytics row."""

    ts: datetime
    instance_id: str
    correlation_id: int
    principal_id: str
    model: str
    chain: list[str]
    block_roles: list[str]
    block_tokens: list[int]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    duration_ms: Optional[float]
    stream: Optional[bool]
    max_tokens: Optional[int]
    served_by: Optional[str] = None
    finish_reason: Optional[str] = None
    request_origin: Optional[str] = None
    user_id: Optional[str] = None
    api_key_id: Optional[str] = None
    duration_to_first_byte_ms: Optional[float] = None
    index: int = field(default=-1, compare=False)

    @property
    def leaf(self) -> Optional[str]:
        return self.chain[-1] if self.chain else None

    @property
    def end_ts(self) -> datetime:
        return self.ts + timedelta(milliseconds=self.duration_ms or 0.0)


def parse_timestamp(value: Any) -> datetime:
    """Parse an RFC 3339 string or datetime into an aware UTC datetime."""

    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith("Z") or text.endswith("z"):
            text = text[:-1] + "+00:00"
        if " " in text and "T" not in text:
            text = text.replace(" ", "T", 1)
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as error:
            raise ExportError(f"invalid timestamp {value!r}: {error}") from error
    else:
        raise ExportError(f"invalid timestamp {value!r}")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes"}
    return bool(value)


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _hex_seed(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    return str(value).lower()


def record_from_row(row: Mapping[str, Any], index: int = -1) -> ChainRecord:
    """Build a :class:`ChainRecord` from a plain row dictionary (ClickHouse column names)."""

    chain = [_hex_seed(hash_) for hash_ in (row.get("chain") or [])]
    block_roles = [str(role) for role in (row.get("block_roles") or [])]
    block_tokens = [int(tokens) for tokens in (row.get("block_tokens") or [])]
    return ChainRecord(
        ts=parse_timestamp(row["ts"]),
        instance_id=str(row.get("instance_id", "")),
        correlation_id=int(row.get("correlation_id", 0)),
        principal_id=str(row.get("principal_id", "")),
        model=str(row.get("model", "")),
        chain=chain,
        block_roles=block_roles,
        block_tokens=block_tokens,
        prompt_tokens=_optional_int(row.get("prompt_tokens")),
        completion_tokens=_optional_int(row.get("completion_tokens")),
        duration_ms=_optional_float(row.get("duration_ms")),
        stream=_optional_bool(row.get("stream")),
        max_tokens=_optional_int(row.get("max_tokens")),
        served_by=_optional_str(row.get("served_by")),
        finish_reason=_optional_str(row.get("finish_reason")),
        request_origin=_optional_str(row.get("request_origin")),
        user_id=_optional_str(row.get("user_id")),
        api_key_id=_optional_str(row.get("api_key_id")),
        duration_to_first_byte_ms=_optional_float(row.get("duration_to_first_byte_ms")),
        index=index,
    )


def records_from_rows(rows: Iterable[Mapping[str, Any]]) -> list[ChainRecord]:
    return [record_from_row(row, index) for index, row in enumerate(rows)]


@dataclass
class ExportStats:
    rows: int = 0
    usable: int = 0
    dropped_no_prompt_tokens: int = 0
    dropped_no_completion_tokens: int = 0
    dropped_malformed_blocks: int = 0
    in_window: int = 0
    trajectories: int = 0
    sampled: int = 0
    normalized_block_tokens: int = 0
    prewindow_sessions: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


def usable_records(records: Iterable[ChainRecord], stats: ExportStats) -> list[ChainRecord]:
    """Keep records that can be replayed: positive prompt and completion token counts and
    block arrays that agree with the chain."""

    usable = []
    for record in records:
        stats.rows += 1
        if record.block_tokens and (
            len(record.block_tokens) != len(record.chain)
            or len(record.block_roles) != len(record.chain)
        ):
            stats.dropped_malformed_blocks += 1
            continue
        if not record.prompt_tokens or record.prompt_tokens <= 0:
            block_sum = sum(record.block_tokens)
            if block_sum > 0:
                record.prompt_tokens = block_sum
            else:
                stats.dropped_no_prompt_tokens += 1
                continue
        if not record.completion_tokens or record.completion_tokens <= 0:
            stats.dropped_no_completion_tokens += 1
            continue
        usable.append(record)
    stats.usable = len(usable)
    return usable


# ---------------------------------------------------------------------------
# Session reconstruction


def _record_order(record: ChainRecord) -> tuple:
    return (record.ts, record.instance_id, record.correlation_id, record.index)


def link_parents(
    records: Sequence[ChainRecord], link_window: timedelta = DEFAULT_LINK_WINDOW
) -> dict[int, Optional[int]]:
    """Map each record index to the index of the request it continues, or ``None`` for a root.

    Within a principal, the parent of B is the latest earlier request (within ``link_window``)
    whose full chain is a strict prefix of B's chain. Because chain hashes are cumulative, A's
    chain is a prefix of B's exactly when A's leaf hash appears in B's chain before B's leaf.
    """

    parents: dict[int, Optional[int]] = {}
    by_principal: dict[str, list[ChainRecord]] = {}
    for record in records:
        by_principal.setdefault(record.principal_id, []).append(record)

    for principal_records in by_principal.values():
        principal_records.sort(key=_record_order)
        latest_by_leaf: dict[str, ChainRecord] = {}
        for record in principal_records:
            parent: Optional[ChainRecord] = None
            for hash_ in record.chain[:-1]:
                candidate = latest_by_leaf.get(hash_)
                if candidate is None:
                    continue
                if record.ts - candidate.ts > link_window:
                    continue
                if parent is None or _record_order(candidate) > _record_order(parent):
                    parent = candidate
            parents[record.index] = None if parent is None else parent.index
            if record.leaf is not None:
                latest_by_leaf[record.leaf] = record
    return parents


@dataclass
class Trajectory:
    root: ChainRecord
    requests: list[ChainRecord]
    began_before_window: bool


def build_trajectories(
    records: Sequence[ChainRecord],
    parents: Mapping[int, Optional[int]],
    window_start: datetime,
    window_end: datetime,
) -> list[Trajectory]:
    """Group in-window records into trajectories: each root and its descendants, by timestamp.

    Records before ``window_start`` only serve as link targets. An in-window request whose
    parent lies before the window starts a trajectory that is flagged as having begun earlier.
    """

    by_index = {record.index: record for record in records}

    def in_window(record: ChainRecord) -> bool:
        return window_start <= record.ts < window_end

    root_of: dict[int, int] = {}
    began_before: dict[int, bool] = {}

    def resolve_root(record: ChainRecord) -> int:
        path = []
        current = record
        while current.index not in root_of:
            parent_index = parents.get(current.index)
            parent = by_index.get(parent_index) if parent_index is not None else None
            if parent is None or not in_window(parent):
                root_of[current.index] = current.index
                began_before[current.index] = parent is not None
                break
            path.append(current.index)
            current = parent
        root = root_of[current.index]
        for index in path:
            root_of[index] = root
        return root

    groups: dict[int, list[ChainRecord]] = {}
    for record in sorted(records, key=_record_order):
        if not in_window(record):
            continue
        groups.setdefault(resolve_root(record), []).append(record)

    trajectories = []
    for root_index, members in groups.items():
        members.sort(key=_record_order)
        trajectories.append(
            Trajectory(
                root=by_index[root_index],
                requests=members,
                began_before_window=began_before.get(root_index, False),
            )
        )
    trajectories.sort(key=lambda trajectory: _record_order(trajectory.requests[0]))
    return trajectories


# ---------------------------------------------------------------------------
# Plans


def extends(previous: Sequence[str], current: Sequence[str]) -> bool:
    """True when ``current`` strictly extends ``previous`` (same prefix, at least one more block)."""

    return len(current) > len(previous) and list(current[: len(previous)]) == list(previous)


def scale_ms(milliseconds: float, time_scale: float) -> int:
    return int(round(max(milliseconds, 0.0) / time_scale))


def trajectory_id_for(trajectory: Trajectory, counter: int) -> str:
    root = trajectory.root
    stem = root.leaf or f"{root.principal_id}-{root.correlation_id}"
    return f"{stem}-{counter}"


def request_blocks(
    record: ChainRecord,
    previous: Optional[ChainRecord],
    seed_tokens: dict[str, int],
    stats: ExportStats,
) -> tuple[list[dict[str, Any]], Optional[int]]:
    """Blocks for one request plus its overhead, marking the reply to ``previous`` as live.

    Token counts for a seed already seen in the trajectory are normalized to the first value so
    the replay's per-seed cache stays consistent.
    """

    if not record.block_tokens:
        seed = record.leaf or f"{record.principal_id}-{record.correlation_id}"
        return [{"seed": seed, "tokens": int(record.prompt_tokens or 0), "role": "user"}], 0

    live_index = None
    if previous is not None and extends(previous.chain, record.chain):
        candidate = len(previous.chain)
        if record.block_roles[candidate] == "assistant":
            live_index = candidate

    blocks = []
    for position, (seed, role, tokens) in enumerate(
        zip(record.chain, record.block_roles, record.block_tokens)
    ):
        known = seed_tokens.get(seed)
        if known is None:
            seed_tokens[seed] = tokens
        elif known != tokens:
            stats.normalized_block_tokens += 1
            tokens = known
        block: dict[str, Any] = {"seed": seed, "tokens": int(tokens), "role": role}
        if position == live_index:
            block["live"] = True
        blocks.append(block)
    overhead = max(int(record.prompt_tokens or 0) - sum(block["tokens"] for block in blocks), 0)
    return blocks, overhead


def trajectory_plan(
    trajectory: Trajectory,
    counter: int,
    window_start: datetime,
    time_scale: float,
    stats: ExportStats,
) -> dict[str, Any]:
    """Convert one trajectory into a schema version 2 manifest line."""

    requests = trajectory.requests
    first_offset_ms = (requests[0].ts - window_start).total_seconds() * 1000.0
    plan_requests = []
    seed_tokens: dict[str, int] = {}
    previous: Optional[ChainRecord] = None
    for position, record in enumerate(requests):
        if position == 0:
            reset_before = trajectory.began_before_window
        else:
            reset_before = not extends(previous.chain, record.chain)
        # Seed definitions persist across resets: the replay validates them per trajectory.
        blocks, overhead = request_blocks(
            record, None if reset_before else previous, seed_tokens, stats
        )
        request: dict[str, Any] = {
            "prompt_tokens": int(record.prompt_tokens or 0),
            "output_tokens": int(record.completion_tokens or 0),
        }
        if position + 1 < len(requests):
            gap_ms = (requests[position + 1].ts - record.end_ts).total_seconds() * 1000.0
            request["delay_after_ms"] = scale_ms(gap_ms, time_scale)
        if reset_before:
            request["reset_before"] = True
        if overhead is not None:
            request["overhead_tokens"] = overhead
        if record.stream is not None:
            request["stream"] = record.stream
        if record.max_tokens is not None and record.max_tokens > 0:
            request["max_tokens"] = int(record.max_tokens)
        request["blocks"] = blocks
        plan_requests.append(request)
        previous = record

    return {
        "schema_version": SCHEMA_VERSION,
        "trajectory_id": trajectory_id_for(trajectory, counter),
        "start_after_ms": scale_ms(first_offset_ms, time_scale),
        "requests": plan_requests,
    }


def build_plans(
    records: Sequence[ChainRecord],
    window_start: datetime,
    window_end: datetime,
    time_scale: float = 1.0,
    link_window: timedelta = DEFAULT_LINK_WINDOW,
    stats: Optional[ExportStats] = None,
) -> list[dict[str, Any]]:
    """Reconstruct sessions from usable records and convert them to manifest lines."""

    stats = stats if stats is not None else ExportStats()
    if time_scale <= 0 or not math.isfinite(time_scale):
        raise ExportError("time scale must be a positive finite number")
    usable = usable_records(records, stats)
    parents = link_parents(usable, link_window)
    trajectories = build_trajectories(usable, parents, window_start, window_end)
    stats.in_window = sum(len(trajectory.requests) for trajectory in trajectories)
    stats.trajectories = len(trajectories)
    stats.prewindow_sessions = sum(1 for t in trajectories if t.began_before_window)

    counters: dict[str, int] = {}
    plans = []
    for trajectory in trajectories:
        stem = trajectory.root.leaf or ""
        counter = counters.get(stem, 0)
        counters[stem] = counter + 1
        plans.append(trajectory_plan(trajectory, counter, window_start, time_scale, stats))
    plans.sort(key=lambda plan: (plan["start_after_ms"], plan["trajectory_id"]))
    return plans


# ---------------------------------------------------------------------------
# Sampling


def sample_key(seed: int, trajectory_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{trajectory_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def session_length_bucket(request_count: int) -> int:
    """Power-of-two bucket: 1, 2-3, 4-7, 8-15, ..."""

    return max(request_count, 1).bit_length() - 1


def sample_plans(
    plans: Sequence[dict[str, Any]],
    fraction: Optional[float],
    seed: int = 0,
    stratify_by_session_length: bool = False,
) -> list[dict[str, Any]]:
    """Deterministically keep ``fraction`` of the plans.

    Plans are ordered by a seeded hash of their trajectory id and the first ``round(fraction *
    n)`` are kept. With stratification the same rule is applied per session-length bucket with
    ``ceil`` instead of ``round``, so every non-empty bucket keeps at least one trajectory.
    """

    if fraction is None:
        return list(plans)
    if not math.isfinite(fraction) or fraction < 0.0 or fraction > 1.0:
        raise ExportError("sample fraction must be between 0 and 1")
    if fraction == 1.0:
        return list(plans)

    def take(group: list[dict[str, Any]], rounding) -> list[dict[str, Any]]:
        ordered = sorted(group, key=lambda plan: (sample_key(seed, plan["trajectory_id"]), plan["trajectory_id"]))
        return ordered[: int(rounding(fraction * len(ordered)))]

    if stratify_by_session_length:
        buckets: dict[int, list[dict[str, Any]]] = {}
        for plan in plans:
            buckets.setdefault(session_length_bucket(len(plan["requests"])), []).append(plan)
        kept = [plan for bucket in buckets.values() for plan in take(bucket, math.ceil)]
    else:
        kept = take(list(plans), round)
    kept.sort(key=lambda plan: (plan["start_after_ms"], plan["trajectory_id"]))
    return kept


# ---------------------------------------------------------------------------
# Validation (mirrors the checks batchbench-agent applies to schema version 2)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_plan(plan: Mapping[str, Any]) -> list[str]:
    """Return the problems that would make ``batchbench-agent`` reject this manifest line."""

    errors: list[str] = []
    unknown = set(plan) - TRAJECTORY_FIELDS
    if unknown:
        errors.append(f"unknown trajectory fields {sorted(unknown)}")
    if plan.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    trajectory_id = plan.get("trajectory_id")
    if not isinstance(trajectory_id, str) or not trajectory_id.strip():
        errors.append("trajectory_id must be a non-empty string")
    start_after = plan.get("start_after_ms")
    if start_after is not None and (not _is_int(start_after) or start_after < 0):
        errors.append("start_after_ms must be a non-negative integer")
    requests = plan.get("requests")
    if not isinstance(requests, list) or not requests:
        return errors + ["requests must be a non-empty list"]

    uses_blocks = "blocks" in requests[0]
    seed_definitions: dict[str, tuple[int, str]] = {}
    for number, request in enumerate(requests, start=1):
        prefix = f"request {number}"
        if not isinstance(request, Mapping):
            errors.append(f"{prefix} must be an object")
            continue
        unknown = set(request) - REQUEST_FIELDS
        if unknown:
            errors.append(f"{prefix} has unknown fields {sorted(unknown)}")
        for key in ("prompt_tokens", "output_tokens"):
            value = request.get(key)
            if not _is_int(value) or value <= 0:
                errors.append(f"{prefix} {key} must be a positive integer")
        for key in ("delay_after_ms", "overhead_tokens"):
            value = request.get(key)
            if value is not None and (not _is_int(value) or value < 0):
                errors.append(f"{prefix} {key} must be a non-negative integer")
        max_tokens = request.get("max_tokens")
        if max_tokens is not None and (not _is_int(max_tokens) or max_tokens <= 0):
            errors.append(f"{prefix} max_tokens must be a positive integer")
        for key in ("reset_before", "stream"):
            value = request.get(key)
            if value is not None and not isinstance(value, bool):
                errors.append(f"{prefix} {key} must be a boolean")
        if ("blocks" in request) != uses_blocks:
            errors.append(f"{prefix} must {'define' if uses_blocks else 'omit'} blocks like the first request")
            continue
        if not uses_blocks:
            continue
        blocks = request.get("blocks")
        if not isinstance(blocks, list) or not blocks:
            errors.append(f"{prefix} blocks must be a non-empty list")
            continue
        for position, block in enumerate(blocks, start=1):
            block_prefix = f"{prefix} block {position}"
            if not isinstance(block, Mapping):
                errors.append(f"{block_prefix} must be an object")
                continue
            unknown = set(block) - BLOCK_FIELDS
            if unknown:
                errors.append(f"{block_prefix} has unknown fields {sorted(unknown)}")
            seed = block.get("seed")
            if not isinstance(seed, str) or not seed:
                errors.append(f"{block_prefix} seed must be a non-empty string")
                continue
            tokens = block.get("tokens")
            if not _is_int(tokens) or tokens < 0:
                errors.append(f"{block_prefix} tokens must be a non-negative integer")
                continue
            role = block.get("role")
            if role not in BLOCK_ROLES:
                errors.append(f"{block_prefix} role {role!r} is not one of {list(BLOCK_ROLES)}")
                continue
            live = block.get("live")
            if live is not None and not isinstance(live, bool):
                errors.append(f"{block_prefix} live must be a boolean")
            if live and role != "assistant":
                errors.append(f"{block_prefix} is live but has role {role}")
            definition = (tokens, role)
            known = seed_definitions.setdefault(seed, definition)
            if known != definition:
                errors.append(f"{block_prefix} redefines seed {seed} as {definition}; first seen as {known}")
    return errors


def validate_plans(plans: Iterable[Mapping[str, Any]]) -> list[str]:
    errors = []
    seen: set[str] = set()
    for line_number, plan in enumerate(plans, start=1):
        for problem in validate_plan(plan):
            errors.append(f"line {line_number}: {problem}")
        trajectory_id = plan.get("trajectory_id")
        if isinstance(trajectory_id, str):
            if trajectory_id in seen:
                errors.append(f"line {line_number}: duplicate trajectory_id {trajectory_id!r}")
            seen.add(trajectory_id)
    return errors


# ---------------------------------------------------------------------------
# I/O


def write_manifest(plans: Iterable[Mapping[str, Any]], path: str) -> int:
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for plan in plans:
            handle.write(json.dumps(plan, separators=(",", ":"), ensure_ascii=False))
            handle.write("\n")
            count += 1
    return count


def read_rows_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_query(
    chains_table: str,
    analytics_table: str,
    principal_id: Optional[str],
    model: Optional[str],
    served_by: Optional[str],
    chains_final: bool = False,
    analytics_ts_column: str = DEFAULT_ANALYTICS_TS_COLUMN,
) -> tuple[str, dict[str, Any]]:
    """The ClickHouse query and its named parameters (window bounds are filled in later).

    Both sides of the join are restricted to the lookback window before joining; the analytics
    side is widened by a day on each end because its timestamp is the request start while the
    chain timestamp is taken at capture.
    """

    chain_filters = ["ts >= {lookback_start:DateTime64(3)}", "ts < {window_end:DateTime64(3)}"]
    analytics_filters = [
        f"{analytics_ts_column} >= {{lookback_start:DateTime64(3)}} - INTERVAL 1 DAY",
        f"{analytics_ts_column} < {{window_end:DateTime64(3)}} + INTERVAL 1 DAY",
    ]
    parameters: dict[str, Any] = {}
    if principal_id:
        chain_filters.append("principal_id = {principal_id:UUID}")
        parameters["principal_id"] = principal_id
    if model:
        chain_filters.append("model = {model:String}")
        parameters["model"] = model
    outer_filters = []
    if served_by:
        outer_filters.append("h.served_by = {served_by:String}")
        parameters["served_by"] = served_by
    final = " FINAL" if chains_final else ""
    sql = f"""
SELECT
    c.ts AS ts,
    toString(c.instance_id) AS instance_id,
    c.correlation_id AS correlation_id,
    toString(c.principal_id) AS principal_id,
    c.model AS model,
    arrayMap(x -> lower(hex(x)), c.chain) AS chain,
    c.block_roles AS block_roles,
    c.block_tokens AS block_tokens,
    h.prompt_tokens AS prompt_tokens,
    h.completion_tokens AS completion_tokens,
    h.stream AS stream,
    h.max_tokens AS max_tokens,
    h.finish_reason AS finish_reason,
    h.served_by AS served_by,
    h.request_origin AS request_origin,
    h.user_id AS user_id,
    h.api_key_id AS api_key_id,
    h.duration_ms AS duration_ms,
    h.duration_to_first_byte_ms AS duration_to_first_byte_ms
FROM (
    SELECT * FROM {chains_table}{final}
    WHERE {' AND '.join(chain_filters)}
) AS c
LEFT JOIN (
    SELECT instance_id, correlation_id, prompt_tokens, completion_tokens, stream, max_tokens,
        finish_reason, served_by, request_origin, user_id, api_key_id, duration_ms,
        duration_to_first_byte_ms
    FROM {analytics_table}
    WHERE {' AND '.join(analytics_filters)}
) AS h
    ON h.instance_id = c.instance_id AND h.correlation_id = c.correlation_id
{('WHERE ' + ' AND '.join(outer_filters)) if outer_filters else ''}
ORDER BY c.principal_id, c.ts, c.instance_id, c.correlation_id
""".strip()
    return sql, parameters


def fetch_rows(
    url: str,
    user: Optional[str],
    password: Optional[str],
    sql: str,
    parameters: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Run the export query through ``clickhouse-connect`` and return plain row dictionaries."""

    try:
        import clickhouse_connect  # type: ignore[import-not-found]
    except ImportError as error:  # pragma: no cover - exercised only without the extra
        raise ExportError(
            "clickhouse-connect is required to read ClickHouse; install batchbench[export]"
        ) from error

    client_args: dict[str, Any] = {"dsn": url}
    if user:
        client_args["username"] = user
    if password:
        client_args["password"] = password
    client = clickhouse_connect.get_client(**client_args)
    try:
        result = client.query(sql, parameters=dict(parameters))
        return [dict(row) for row in result.named_results()]
    finally:
        client.close()


# ---------------------------------------------------------------------------
# CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="batchbench export-plans",
        description="Export a batchbench schema v2 trajectory manifest from ClickHouse prompt-chain records.",
    )
    source = parser.add_argument_group("source")
    source.add_argument("--clickhouse-url", default=os.environ.get("CLICKHOUSE_URL"), help="ClickHouse HTTP(S) URL (env CLICKHOUSE_URL)")
    source.add_argument("--clickhouse-user", default=os.environ.get("CLICKHOUSE_USER"), help="ClickHouse user (env CLICKHOUSE_USER)")
    source.add_argument("--clickhouse-password", default=os.environ.get("CLICKHOUSE_PASSWORD"), help="ClickHouse password (env CLICKHOUSE_PASSWORD)")
    source.add_argument("--chains-table", default=DEFAULT_CHAINS_TABLE, help=f"prompt-chain table or view (default {DEFAULT_CHAINS_TABLE})")
    source.add_argument("--chains-final", action="store_true", help="read the chains table with FINAL (when reading the ReplacingMergeTree base table instead of its deduplicating view)")
    source.add_argument("--analytics-table", default=DEFAULT_ANALYTICS_TABLE, help=f"request analytics table (default {DEFAULT_ANALYTICS_TABLE})")
    source.add_argument("--analytics-ts-column", default=DEFAULT_ANALYTICS_TS_COLUMN, help=f"timestamp column used to bound the analytics side of the join (default {DEFAULT_ANALYTICS_TS_COLUMN})")
    source.add_argument("--rows-jsonl", help="read previously fetched rows from this JSONL file instead of ClickHouse")
    source.add_argument("--dump-rows-jsonl", help="also write the fetched rows to this JSONL file")

    window = parser.add_argument_group("window and filters")
    window.add_argument("--start", required=True, help="window start (RFC 3339)")
    window.add_argument("--end", required=True, help="window end (RFC 3339, exclusive)")
    window.add_argument("--principal-id", help="restrict to one principal")
    window.add_argument("--model", help="restrict to one model alias")
    window.add_argument("--served-by", help="restrict to requests served by this upstream")
    window.add_argument("--link-window-hours", type=float, default=DEFAULT_LINK_WINDOW.total_seconds() / 3600.0, help="how far back a continuation may look for its parent (default 24)")

    shaping = parser.add_argument_group("shaping")
    shaping.add_argument("--sample", type=float, help="fraction of trajectories to keep (0-1)")
    shaping.add_argument("--seed", type=int, default=0, help="seed for deterministic sampling")
    shaping.add_argument("--stratify-by-session-length", action="store_true", help="sample within power-of-two session-length buckets")
    shaping.add_argument("--time-scale", type=float, default=1.0, help="divide start offsets and delays by this factor")

    parser.add_argument("--output", required=True, help="manifest JSONL path")
    return parser


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    return str(value)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        window_start = parse_timestamp(args.start)
        window_end = parse_timestamp(args.end)
        if window_end <= window_start:
            raise ExportError("--end must be after --start")
        if args.link_window_hours < 0:
            raise ExportError("--link-window-hours must be non-negative")
        link_window = timedelta(hours=args.link_window_hours)

        if args.rows_jsonl:
            rows = read_rows_jsonl(args.rows_jsonl)
        else:
            if not args.clickhouse_url:
                raise ExportError("--clickhouse-url (or CLICKHOUSE_URL) is required without --rows-jsonl")
            sql, parameters = build_query(
                args.chains_table,
                args.analytics_table,
                args.principal_id,
                args.model,
                args.served_by,
                chains_final=args.chains_final,
                analytics_ts_column=args.analytics_ts_column,
            )
            parameters["lookback_start"] = (window_start - link_window).replace(tzinfo=None)
            parameters["window_end"] = window_end.replace(tzinfo=None)
            rows = fetch_rows(args.clickhouse_url, args.clickhouse_user, args.clickhouse_password, sql, parameters)
            if args.dump_rows_jsonl:
                with open(args.dump_rows_jsonl, "w", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row, default=_json_default) + "\n")

        stats = ExportStats()
        records = records_from_rows(rows)
        if args.rows_jsonl:
            records = [
                record
                for record in records
                if (not args.principal_id or record.principal_id == args.principal_id)
                and (not args.model or record.model == args.model)
                and (not args.served_by or record.served_by == args.served_by)
            ]
        plans = build_plans(records, window_start, window_end, args.time_scale, link_window, stats)
        plans = sample_plans(plans, args.sample, args.seed, args.stratify_by_session_length)
        stats.sampled = len(plans)
        if not plans:
            summary = ", ".join(f"{key}={value}" for key, value in stats.as_dict().items())
            raise ExportError(f"no trajectories to export ({summary})")
        problems = validate_plans(plans)
        if problems:
            preview = "\n".join(problems[:10])
            raise ExportError(f"export produced {len(problems)} invalid manifest line(s):\n{preview}")
        written = write_manifest(plans, args.output)
    except ExportError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    summary = ", ".join(f"{key}={value}" for key, value in stats.as_dict().items())
    print(f"wrote {written} trajectories to {args.output} ({summary})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
