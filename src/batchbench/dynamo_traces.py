"""Build schema version 2 manifests from Dynamo frontend request traces.

A Dynamo frontend with request tracing enabled writes one ``request_end`` row
(``dynamo.request.trace.v1``) per request: timings, token counts, and the KV router's
cumulative block hashes of the prompt (``replay.input_sequence_hashes``, one per full block of
``replay.trace_block_size`` tokens). The hashes play the role of a prompt chain, at block rather
than message granularity, so sessions are rebuilt with the same strict-prefix rule as
:mod:`batchbench.export_plans`:

* Requests link only within the same model, block size, principal, and agent session.
* Every prompt is cut wherever its chain diverges from another chain of the same model and block
  size, or where another chain ends. Each segment becomes one ``user`` block whose seed is the
  segment's last hash, so prompts that share a prefix replay with identical bytes.
* The tokens past the last full block become a final ``user`` block of their own.
* A request that extends the previous one repeats that request's blocks, and the first
  ``output_tokens`` tokens after them become the ``live`` assistant block.

An optional analytics join on ``x_request_id`` fills the principal and the model alias. The
expected input is documented in ``docs/export-plans.md``.
"""

from __future__ import annotations

import glob
import gzip
import json
import os
import uuid
import zlib
from array import array
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

from .export_plans import (
    DEFAULT_LINK_WINDOW,
    SCHEMA_VERSION,
    ChainRecord,
    ExportError,
    ExportStats,
    Trajectory,
    build_trajectories,
    extends,
    link_parents,
    scale_ms,
    usable_records,
)

TRACE_SCHEMA = "dynamo.request.trace.v1"
TRACE_EVENT = "request_end"
TRACE_SUFFIXES = (".jsonl.gz", ".jsonl")
DEFAULT_ANALYTICS_ID_COLUMN = "x_request_id"
# The analytics timestamp is when the gateway received the request, the trace's when Dynamo did;
# retries and failover land later still.
ANALYTICS_PAD = timedelta(minutes=15)
ANALYTICS_COLUMNS = ("x_request_id", "instance_id", "correlation_id", "model", "user_id", "api_key_id")


# ---------------------------------------------------------------------------
# Records


@dataclass
class TraceRecord(ChainRecord):
    """One ``request_end`` row. ``chain`` holds the block hashes as unsigned 64-bit integers."""

    block_size: int = 0
    tail_tokens: int = 0
    request_id: str = ""
    x_request_id: Optional[str] = None
    session_id: Optional[str] = None
    alias: Optional[str] = None
    partial_hash: Optional[int] = None

    @property
    def link_key(self) -> Any:
        return (self.model, self.block_size, self.principal_id, self.session_id or "")

    @property
    def end_ms(self) -> float:
        return self.ts.timestamp() * 1000.0 + (self.duration_ms or 0.0)


@dataclass
class TraceStats:
    files: int = 0
    unreadable_files: int = 0
    lines: int = 0
    bad_lines: int = 0
    not_request_end: int = 0
    no_replay: int = 0
    other_model: int = 0
    outside_window: int = 0
    malformed_hashes: int = 0
    loaded: int = 0
    with_x_request_id: int = 0
    duplicate_attempts: int = 0
    analytics_rows: int = 0
    analytics_matched: int = 0
    with_session_id: int = 0
    live_blocks: int = 0
    live_clamped: int = 0
    block_sizes: dict[int, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        values: dict[str, Any] = {key: value for key, value in self.__dict__.items() if key != "block_sizes"}
        values["block_sizes"] = ",".join(f"{size}:{count}" for size, count in sorted(self.block_sizes.items())) or "none"
        return values


def normalize_request_id(value: Any) -> Optional[str]:
    """Canonical lower-case form of an ``x-request-id``; UUIDs in their hyphenated form."""

    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    try:
        return str(uuid.UUID(text))
    except ValueError:
        return text


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def record_from_trace(
    row: Mapping[str, Any],
    stats: TraceStats,
    model: Optional[str] = None,
    window_ms: Optional[tuple[float, float]] = None,
) -> Optional[TraceRecord]:
    """Build a :class:`TraceRecord` from one parsed trace line, or count why it was skipped."""

    if row.get("schema") != TRACE_SCHEMA or row.get("event_type") != TRACE_EVENT:
        stats.not_request_end += 1
        return None
    request = row.get("request")
    replay = request.get("replay") if isinstance(request, Mapping) else None
    received_ms = request.get("request_received_ms") if isinstance(request, Mapping) else None
    if not isinstance(replay, Mapping) or not _is_count(received_ms):
        stats.no_replay += 1
        return None
    row_model = str(request.get("model") or "")
    if model is not None and row_model != model:
        stats.other_model += 1
        return None
    if window_ms is not None and not (window_ms[0] <= received_ms < window_ms[1]):
        stats.outside_window += 1
        return None

    block_size = replay.get("trace_block_size")
    hashes = replay.get("input_sequence_hashes") or []
    input_length = replay.get("input_length")
    if not _is_count(block_size) or block_size == 0 or not isinstance(hashes, list):
        stats.malformed_hashes += 1
        return None
    try:
        chain = array("Q", hashes)
    except (TypeError, OverflowError):
        stats.malformed_hashes += 1
        return None
    # Dynamo hashes every full block and then, when tokens remain, the partial last block. The
    # partial hash changes as soon as a later turn fills that block, so it never links or cuts.
    hashed_tokens = input_length if _is_count(input_length) else request.get("input_tokens")
    partial_hash = None
    if _is_count(hashed_tokens):
        if len(chain) > -(-hashed_tokens // block_size):
            stats.malformed_hashes += 1
            return None
        if hashed_tokens % block_size and len(chain) == hashed_tokens // block_size + 1:
            partial_hash = chain.pop()
    full_tokens = block_size * len(chain)
    prompt_tokens = request.get("input_tokens")
    if not _is_count(prompt_tokens) or prompt_tokens < full_tokens:
        prompt_tokens = input_length
    if not _is_count(prompt_tokens) or prompt_tokens < full_tokens:
        stats.malformed_hashes += 1
        return None

    agent = row.get("agent_context") if isinstance(row.get("agent_context"), Mapping) else {}
    session_id = agent.get("session_id")
    finish = request.get("finish_reason_metadata")
    finish_reason = finish.get("finish_reason") if isinstance(finish, Mapping) else None
    total_time_ms = request.get("total_time_ms")
    output_tokens = request.get("output_tokens")
    record = TraceRecord(
        ts=datetime.fromtimestamp(received_ms / 1000.0, tz=timezone.utc),
        instance_id="",
        correlation_id=0,
        principal_id="",
        model=row_model,
        chain=chain,  # type: ignore[arg-type]
        block_roles=[],
        block_tokens=[],
        prompt_tokens=int(prompt_tokens),
        completion_tokens=int(output_tokens) if _is_count(output_tokens) else None,
        duration_ms=float(total_time_ms) if isinstance(total_time_ms, (int, float)) and not isinstance(total_time_ms, bool) else None,
        stream=None,
        max_tokens=None,
        finish_reason=None if finish_reason is None else str(finish_reason),
        block_size=int(block_size),
        tail_tokens=int(prompt_tokens) - full_tokens,
        partial_hash=partial_hash,
        request_id=str(request.get("request_id") or ""),
        x_request_id=normalize_request_id(request.get("x_request_id")),
        session_id=None if session_id in (None, "") else str(session_id),
    )
    stats.loaded += 1
    stats.block_sizes[record.block_size] = stats.block_sizes.get(record.block_size, 0) + 1
    if record.x_request_id:
        stats.with_x_request_id += 1
    if record.session_id:
        stats.with_session_id += 1
    return record


def trace_files(paths: Sequence[str]) -> list[str]:
    """Expand directories (recursively) and glob patterns into trace files, sorted."""

    found: set[str] = set()
    for path in paths:
        if os.path.isdir(path):
            for directory, _, names in os.walk(path):
                found.update(os.path.join(directory, name) for name in names if name.endswith(TRACE_SUFFIXES))
        elif os.path.isfile(path):
            found.add(path)
        else:
            matches = [match for match in glob.glob(path, recursive=True) if os.path.isfile(match)]
            if not matches:
                raise ExportError(f"no trace files match {path!r}")
            found.update(matches)
    if not found:
        raise ExportError("no trace files found (expected *.jsonl.gz or *.jsonl)")
    return sorted(found)


def _trace_lines(path: str, stats: TraceStats) -> Iterator[str]:
    stats.files += 1
    try:
        if path.endswith(".gz"):
            handle = gzip.open(path, "rt", encoding="utf-8")
        else:
            handle = open(path, "r", encoding="utf-8")
        with handle:
            for line in handle:
                yield line
    except (OSError, EOFError, zlib.error, UnicodeDecodeError):
        # A truncated upload keeps the rows read before the damage.
        stats.unreadable_files += 1


def load_trace_records(
    paths: Sequence[str],
    window_start: datetime,
    window_end: datetime,
    link_window: timedelta = DEFAULT_LINK_WINDOW,
    model: Optional[str] = None,
    stats: Optional[TraceStats] = None,
) -> list[TraceRecord]:
    """Read ``request_end`` rows received in ``[window_start - link_window, window_end)``.

    Files are named by upload time and can hold rows received long before it, so every file is
    read and rows are windowed by ``request_received_ms``.
    """

    stats = stats if stats is not None else TraceStats()
    window_ms = (
        (window_start - link_window).timestamp() * 1000.0,
        window_end.timestamp() * 1000.0,
    )
    # A cheap substring test before parsing. Only for ASCII names: a writer may escape others.
    model_needle = json.dumps(model) if model is not None and model.isascii() else None
    records: list[TraceRecord] = []
    for path in trace_files(paths):
        for line in _trace_lines(path, stats):
            if not line.strip():
                continue
            stats.lines += 1
            if f'"{TRACE_EVENT}"' not in line:
                stats.not_request_end += 1
                continue
            if model_needle is not None and model_needle not in line:
                stats.other_model += 1
                continue
            try:
                row = json.loads(line)
            except ValueError:
                stats.bad_lines += 1
                continue
            if not isinstance(row, Mapping):
                stats.bad_lines += 1
                continue
            record = record_from_trace(row, stats, model, window_ms)
            if record is not None:
                records.append(record)
    return records


def _attempt_rank(record: TraceRecord) -> tuple:
    return (
        bool(record.completion_tokens),
        record.finish_reason is not None,
        record.end_ms,
        record.request_id,
    )


def drop_duplicate_attempts(records: Iterable[TraceRecord], stats: TraceStats) -> list[TraceRecord]:
    """Keep one row per ``x_request_id``: the last attempt that finished.

    A gateway retry or failover inside one client request reuses its ``x-request-id``, so the
    failed attempts and the final one all have rows. Rows are then ordered and re-indexed.
    """

    kept: list[TraceRecord] = []
    best: dict[str, TraceRecord] = {}
    with_id = 0
    for record in records:
        if not record.x_request_id:
            kept.append(record)
            continue
        with_id += 1
        current = best.get(record.x_request_id)
        if current is None or _attempt_rank(record) > _attempt_rank(current):
            best[record.x_request_id] = record
    stats.duplicate_attempts += with_id - len(best)
    kept.extend(best.values())
    kept.sort(key=lambda record: (record.ts, record.request_id, record.x_request_id or ""))
    for index, record in enumerate(kept):
        record.index = index
    return kept


# ---------------------------------------------------------------------------
# Analytics join


def build_analytics_query(table: str, id_column: str = DEFAULT_ANALYTICS_ID_COLUMN, ts_column: str = "timestamp") -> str:
    """Analytics rows with an ``x-request-id`` in ``[{start}, {end})``; columns as ``ANALYTICS_COLUMNS``."""

    return f"""
SELECT
    toString({id_column}) AS x_request_id,
    toString(instance_id) AS instance_id,
    correlation_id,
    model,
    toString(user_id) AS user_id,
    toString(api_key_id) AS api_key_id
FROM {table}
WHERE {ts_column} >= {{start:DateTime64(3)}}
    AND {ts_column} < {{end:DateTime64(3)}}
    AND {id_column} IS NOT NULL
""".strip()


def analytics_window(records: Iterable[TraceRecord]) -> Optional[tuple[datetime, datetime]]:
    """The analytics time range that covers every record with an ``x_request_id``."""

    starts = []
    ends = []
    for record in records:
        if record.x_request_id:
            starts.append(record.ts)
            ends.append(record.end_ts)
    if not starts:
        return None
    return min(starts) - ANALYTICS_PAD, max(ends) + ANALYTICS_PAD


def fetch_analytics_rows(
    url: str,
    user: Optional[str],
    password: Optional[str],
    sql: str,
    parameters: Mapping[str, Any],
    wanted: set[str],
) -> list[dict[str, Any]]:
    """Stream the analytics query and keep the rows whose id is in ``wanted``."""

    try:
        import clickhouse_connect  # type: ignore[import-not-found]
    except ImportError as error:  # pragma: no cover - exercised only without the extra
        raise ExportError("clickhouse-connect is required to read ClickHouse; install batchbench[export]") from error

    client_args: dict[str, Any] = {"dsn": url}
    if user:
        client_args["username"] = user
    if password:
        client_args["password"] = password
    client = clickhouse_connect.get_client(**client_args)
    rows = []
    try:
        with client.query_rows_stream(sql, parameters=dict(parameters)) as stream:
            for values in stream:
                row = dict(zip(ANALYTICS_COLUMNS, values))
                if normalize_request_id(row["x_request_id"]) in wanted:
                    rows.append(row)
    finally:
        client.close()
    return rows


def apply_analytics(records: Iterable[TraceRecord], rows: Iterable[Mapping[str, Any]], stats: TraceStats) -> None:
    """Fill principal, alias and gateway ids from analytics rows matched on ``x_request_id``.

    The principal is the analytics ``user_id``, else its ``api_key_id``.
    """

    by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        request_id = normalize_request_id(row.get("x_request_id"))
        if request_id:
            by_id[request_id] = row
    stats.analytics_rows += len(by_id)
    for record in records:
        row = by_id.get(record.x_request_id) if record.x_request_id else None
        if row is None:
            continue
        stats.analytics_matched += 1
        user_id = row.get("user_id")
        api_key_id = row.get("api_key_id")
        record.principal_id = str(user_id or api_key_id or "")
        record.user_id = None if user_id is None else str(user_id)
        record.api_key_id = None if api_key_id is None else str(api_key_id)
        record.alias = None if row.get("model") is None else str(row["model"])
        if row.get("instance_id") is not None:
            record.instance_id = str(row["instance_id"])
        if row.get("correlation_id") is not None:
            record.correlation_id = int(row["correlation_id"])


# ---------------------------------------------------------------------------
# Blocks


def _common_prefix_length(left: array, right: array) -> int:
    low, high = 0, min(len(left), len(right))
    while low < high:
        middle = (low + high + 1) // 2
        if left[:middle] == right[:middle]:
            low = middle
        else:
            high = middle - 1
    return low


def segment_cuts(records: Iterable[TraceRecord]) -> dict[tuple[str, int], set[int]]:
    """Per (model, block size), the hashes after which a prompt is cut into a new segment.

    A prompt is cut after a hash where some chain ends or where two chains continue differently.
    In lexicographic order, chains through one hash are contiguous, so both cases show up as the
    common prefix of two neighbouring chains.
    """

    groups: dict[tuple[str, int], list[array]] = {}
    for record in records:
        if len(record.chain):
            groups.setdefault((record.model, record.block_size), []).append(record.chain)  # type: ignore[arg-type]
    cuts: dict[tuple[str, int], set[int]] = {}
    for key, chains in groups.items():
        chains.sort()
        group_cuts = {chain[-1] for chain in chains}
        for left, right in zip(chains, chains[1:]):
            shared = _common_prefix_length(left, right)
            if shared:
                group_cuts.add(left[shared - 1])
        cuts[key] = group_cuts
    return cuts


def hash_seed(value: int) -> str:
    return f"{value:016x}"


def record_segments(record: TraceRecord, cuts: set[int]) -> list[tuple[int, str, int]]:
    """``(end position, seed, tokens)`` for each segment of the record's full blocks."""

    chain = record.chain
    positions = [position for position, hash_ in enumerate(chain) if hash_ in cuts]
    segments = []
    start = 0
    for position in positions:
        end = position + 1
        segments.append((end, hash_seed(chain[position]), record.block_size * (end - start)))  # type: ignore[arg-type]
        start = end
    if start < len(chain):  # unreachable when cuts include every leaf; keep the blocks anyway
        segments.append((len(chain), hash_seed(chain[-1]), record.block_size * (len(chain) - start)))  # type: ignore[arg-type]
    return segments


def _request_anchor(record: TraceRecord) -> str:
    leaf = hash_seed(record.chain[-1]) if len(record.chain) else "none"  # type: ignore[arg-type]
    return f"{leaf}:{record.request_id or record.index}"


def tail_seed(record: TraceRecord) -> str:
    """The tokens past the last full block: seeded by Dynamo's partial-block hash when the row
    has one (identical prompts share it), otherwise by the request itself."""

    if record.partial_hash is not None:
        return f"{hash_seed(record.partial_hash)}:tail{record.tail_tokens}"
    return f"{_request_anchor(record)}:tail{record.tail_tokens}"


def _carve(pieces: list[tuple[str, int, int]], tokens: int) -> list[tuple[str, int, int]]:
    """Remove ``tokens`` from the front of ``(seed, full tokens, tokens left)`` pieces."""

    remaining = []
    for seed, full, left in pieces:
        if tokens >= left:
            tokens -= left
            continue
        remaining.append((seed, full, left - tokens))
        tokens = 0
    return remaining


def _piece_block(seed: str, full: int, left: int) -> dict[str, Any]:
    # A piece cut short is a different block from the whole piece, so it gets its own seed.
    return _user(seed if left == full else f"{seed}:rest{left}", left)


def _user(seed: str, tokens: int) -> dict[str, Any]:
    return {"seed": seed, "tokens": int(tokens), "role": "user"}


# Per trajectory: for a segment seed, the blocks already sent that render the prompt exactly up
# to the end of that segment, as (blocks, count).
Renderings = dict[str, tuple[list[dict[str, Any]], int]]


def trace_request_blocks(
    record: TraceRecord,
    previous: Optional[TraceRecord],
    previous_blocks: Sequence[Mapping[str, Any]],
    cuts: set[int],
    stats: TraceStats,
    renderings: Optional[Renderings] = None,
) -> list[dict[str, Any]]:
    """Blocks for one request; ``previous`` is the request it extends, if any.

    An extension repeats ``previous_blocks``, the blocks emitted for the previous request (so
    earlier replies and tails keep their seeds and roles). The tokens after them are the previous
    reply, as a ``live`` block of the previous output, and then new user content. The previous
    tail is already in ``previous_blocks``, so it is carved off the extension first.

    Any other request starts from the longest prefix it shares with a request already sent in the
    trajectory, rendered with the blocks sent then (``renderings``), so a branch replays the same
    bytes as the turn it branches from. ``renderings`` is updated with this request's blocks.
    """

    segments = record_segments(record, cuts)
    tail = [(tail_seed(record), record.tail_tokens)] if record.tail_tokens > 0 else []
    renderings = renderings if renderings is not None else {}
    if previous is None:
        blocks: list[dict[str, Any]] = []
        shared = 0
        for end, seed, _ in reversed(segments):
            rendering = renderings.get(seed)
            if rendering is not None:
                blocks = [dict(block) for block in rendering[0][: rendering[1]]]
                shared = end
                break
        blocks += [_user(seed, tokens) for end, seed, tokens in segments if end > shared]
        blocks += [_user(seed, tokens) for seed, tokens in tail]
        _record_renderings(record, segments, blocks, renderings)
        return blocks

    shared = len(previous.chain)
    blocks = [dict(block) for block in previous_blocks]
    extension = [(seed, tokens, tokens) for end, seed, tokens in segments if end > shared]
    extension += [(seed, tokens, tokens) for seed, tokens in tail]
    extension = _carve(extension, previous.tail_tokens)
    reply = int(previous.completion_tokens or 0)
    available = sum(left for _, _, left in extension)
    if reply > available:
        # The live block replays the whole real reply, which would not fit: generate it instead.
        stats.live_clamped += 1
        if available > 0:
            blocks.append({"seed": f"{_request_anchor(previous)}:reply{available}", "tokens": available, "role": "assistant"})
        extension = []
    elif reply > 0:
        blocks.append({"seed": f"{_request_anchor(previous)}:reply{reply}", "tokens": reply, "role": "assistant", "live": True})
        extension = _carve(extension, reply)
        stats.live_blocks += 1
    blocks.extend(_piece_block(*piece) for piece in extension)
    _record_renderings(record, segments, blocks, renderings)
    return blocks


def _record_renderings(
    record: TraceRecord,
    segments: Sequence[tuple[int, str, int]],
    blocks: list[dict[str, Any]],
    renderings: Renderings,
) -> None:
    """Remember, for each segment end that falls on a block boundary, the blocks up to it."""

    ends = {record.block_size * end: seed for end, seed, _ in segments}
    total = 0
    for count, block in enumerate(blocks, start=1):
        total += block["tokens"]
        seed = ends.get(total)
        if seed is not None:
            renderings.setdefault(seed, (blocks, count))


# ---------------------------------------------------------------------------
# Plans


def _trajectory_stem(record: TraceRecord) -> str:
    if len(record.chain):
        return hash_seed(record.chain[-1])  # type: ignore[arg-type]
    return f"req-{record.request_id or record.index}"


def trace_trajectory_plan(
    trajectory: Trajectory,
    trajectory_id: str,
    window_start: datetime,
    time_scale: float,
    cuts: Mapping[tuple[str, int], set[int]],
    stats: TraceStats,
) -> dict[str, Any]:
    requests: list[TraceRecord] = trajectory.requests  # type: ignore[assignment]
    root: TraceRecord = trajectory.root  # type: ignore[assignment]
    group_cuts = cuts.get((root.model, root.block_size), set())
    first_offset_ms = (requests[0].ts - window_start).total_seconds() * 1000.0
    plan_requests = []
    previous: Optional[TraceRecord] = None
    previous_blocks: list[dict[str, Any]] = []
    renderings: Renderings = {}
    for position, record in enumerate(requests):
        if position == 0:
            reset_before = trajectory.began_before_window
        else:
            reset_before = not extends(previous.chain, record.chain)  # type: ignore[union-attr]
        blocks = trace_request_blocks(
            record, None if reset_before else previous, previous_blocks, group_cuts, stats, renderings
        )
        prompt_tokens = int(record.prompt_tokens or 0)
        request: dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "output_tokens": int(record.completion_tokens or 0),
        }
        if position + 1 < len(requests):
            gap_ms = (requests[position + 1].ts - record.end_ts).total_seconds() * 1000.0
            request["delay_after_ms"] = scale_ms(gap_ms, time_scale)
        if reset_before:
            request["reset_before"] = True
        request["overhead_tokens"] = max(prompt_tokens - sum(block["tokens"] for block in blocks), 0)
        request["blocks"] = blocks
        plan_requests.append(request)
        previous = record
        previous_blocks = blocks

    metadata: dict[str, Any] = {"source": "dynamo-request-trace", "model": root.model, "block_size": root.block_size}
    if root.alias:
        metadata["alias"] = root.alias
    if root.principal_id:
        metadata["principal"] = root.principal_id
    return {
        "schema_version": SCHEMA_VERSION,
        "trajectory_id": trajectory_id,
        "start_after_ms": scale_ms(first_offset_ms, time_scale),
        "requests": plan_requests,
        "metadata": metadata,
    }


def build_trace_plans(
    records: Sequence[TraceRecord],
    window_start: datetime,
    window_end: datetime,
    time_scale: float = 1.0,
    link_window: timedelta = DEFAULT_LINK_WINDOW,
    stats: Optional[ExportStats] = None,
    trace_stats: Optional[TraceStats] = None,
) -> list[dict[str, Any]]:
    """Reconstruct sessions from trace records and convert them to manifest lines."""

    stats = stats if stats is not None else ExportStats()
    trace_stats = trace_stats if trace_stats is not None else TraceStats()
    if time_scale <= 0 or time_scale != time_scale or time_scale == float("inf"):
        raise ExportError("time scale must be a positive finite number")
    usable = usable_records(records, stats)
    cuts = segment_cuts(usable)  # type: ignore[arg-type]
    parents = link_parents(usable, link_window)
    trajectories = build_trajectories(usable, parents, window_start, window_end)
    stats.in_window = sum(len(trajectory.requests) for trajectory in trajectories)
    stats.trajectories = len(trajectories)
    stats.prewindow_sessions = sum(1 for trajectory in trajectories if trajectory.began_before_window)

    counters: dict[str, int] = {}
    plans = []
    for trajectory in trajectories:
        stem = _trajectory_stem(trajectory.root)  # type: ignore[arg-type]
        counter = counters.get(stem, 0)
        counters[stem] = counter + 1
        plans.append(
            trace_trajectory_plan(trajectory, f"{stem}-{counter}", window_start, time_scale, cuts, trace_stats)
        )
    plans.sort(key=lambda plan: (plan["start_after_ms"], plan["trajectory_id"]))
    return plans
