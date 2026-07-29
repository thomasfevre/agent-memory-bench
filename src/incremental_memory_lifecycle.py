#!/usr/bin/env python3
"""Compare deterministic memory-update policies on one incremental event stream."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

from graph_benchmark_common import write_result


PROTOCOL_VERSION = "incremental-memory-lifecycle-v1"
MINIMUM_AUTHORITY_CONFIDENCE = 0.8


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_is_authoritative(record: dict[str, Any]) -> bool:
    return (
        float(record.get("confidence", 0.0))
        >= MINIMUM_AUTHORITY_CONFIDENCE
        and record.get("kind") != "rumor"
    )


def record_is_valid(record: dict[str, Any], query_date: str) -> bool:
    when = date.fromisoformat(query_date)
    valid_from = date.fromisoformat(record["valid_from"])
    valid_to = (
        date.fromisoformat(record["valid_to"])
        if record.get("valid_to")
        else None
    )
    return valid_from <= when and (valid_to is None or when <= valid_to)


def resolve_temporal_records(
    records: list[dict[str, Any]],
    key: str,
    query_date: str,
    retractions: dict[str, str] | None = None,
    scope: str | None = None,
) -> list[dict[str, Any]]:
    resolved, _ = resolve_temporal_records_with_cost(
        records,
        key,
        query_date,
        retractions,
        scope,
    )
    return resolved


def resolve_temporal_records_with_cost(
    records: list[dict[str, Any]],
    key: str,
    query_date: str,
    retractions: dict[str, str] | None = None,
    scope: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    retractions = retractions or {}
    inspections = len(records)
    authoritative = [record for record in records if record_is_authoritative(record)]
    inspections += len(authoritative)
    candidates = [
        record
        for record in authoritative
        if record["key"] == key and record_is_valid(record, query_date)
    ]
    if scope is not None:
        exact_scope = [
            record for record in candidates if record.get("scope", "*") == scope
        ]
        candidates = exact_scope or [
            record for record in candidates if record.get("scope", "*") == "*"
        ]
    result = []
    for candidate in candidates:
        inspections += 1
        retracted_at = retractions.get(candidate["id"])
        if (
            retracted_at is not None
            and date.fromisoformat(retracted_at)
            <= date.fromisoformat(query_date)
        ):
            continue
        superseded = False
        for other in authoritative:
            inspections += 1
            if (
                candidate["id"] in other.get("supersedes", [])
                and date.fromisoformat(other["valid_from"])
                <= date.fromisoformat(query_date)
            ):
                superseded = True
                break
        if not superseded:
            result.append(candidate)
    return (
        sorted(result, key=lambda record: (record["value"], record["id"])),
        inspections,
    )


def render_records(
    records: list[dict[str, Any]],
    *,
    events_scanned: int,
    records_inspected: int,
    index_lookups: int,
) -> dict[str, Any]:
    return {
        "values": sorted({str(record["value"]) for record in records}),
        "source_ids": sorted(
            {str(record["source_id"]) for record in records}
        ),
        "record_ids": sorted(str(record["id"]) for record in records),
        "events_scanned": events_scanned,
        "records_inspected": records_inspected,
        "index_lookups": index_lookups,
    }


def observe_shard(
    shards: dict[str, dict[str, Any]],
    event: dict[str, Any],
    *,
    auto_promote: bool,
) -> None:
    state = shards.setdefault(
        event["shard_id"],
        {
            "id": event["shard_id"],
            "key": event["key"],
            "value": event["value"],
            "source_ids": [],
            "occurrences": 0,
            "review": "pending",
            "active": False,
        },
    )
    if (state["key"], state["value"]) != (event["key"], event["value"]):
        raise ValueError(
            "shard identity collision: "
            f"{event['shard_id']} maps to multiple key/value pairs"
        )
    state["occurrences"] += 1
    if event["source_id"] not in state["source_ids"]:
        state["source_ids"].append(event["source_id"])
    if auto_promote and state["occurrences"] >= 2:
        state["active"] = True


def review_shard(
    shards: dict[str, dict[str, Any]],
    event: dict[str, Any],
) -> None:
    state = shards[event["shard_id"]]
    state["review"] = event["decision"]
    state["active"] = event["decision"] == "approved"


def resolve_shard_events(
    events: list[dict[str, Any]],
    key: str,
    query_date: str,
    *,
    auto_promote: bool,
    honor_reviews: bool,
) -> list[dict[str, Any]]:
    shards: dict[str, dict[str, Any]] = {}
    query_day = date.fromisoformat(query_date)
    for event in events:
        if date.fromisoformat(event["timestamp"]) > query_day:
            continue
        if event["event"] == "observe_shard":
            observe_shard(shards, event, auto_promote=auto_promote)
        elif event["event"] == "review_shard" and honor_reviews:
            review_shard(shards, event)
    return [
        {
            "id": shard["id"],
            "value": shard["value"],
            "source_id": source_id,
        }
        for shard in shards.values()
        if shard["key"] == key and shard["active"]
        for source_id in shard["source_ids"]
    ]


class MemoryPolicy:
    name = "base"

    def apply(self, event: dict[str, Any]) -> None:
        raise NotImplementedError

    def query(
        self,
        kind: str,
        key: str,
        query_date: str,
        scope: str | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def snapshot(self) -> dict[str, Any]:
        raise NotImplementedError

    def state_components(self) -> dict[str, int]:
        raise NotImplementedError

    def state_digest(self) -> str:
        encoded = json.dumps(
            self.snapshot(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


class TemporalPolicyBase(MemoryPolicy):
    review_required = True

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.records_by_key: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        self.retractions: dict[str, str] = {}
        self.shards: dict[str, dict[str, Any]] = {}
        self.shard_events: list[dict[str, Any]] = []

    def apply(self, event: dict[str, Any]) -> None:
        event_type = event["event"]
        if event_type == "upsert":
            record = dict(event["record"])
            self.records.append(record)
            self.records_by_key[record["key"]].append(record)
            return
        if event_type == "retract":
            self.retractions[event["target_id"]] = event["effective_at"]
            return
        if event_type == "observe_shard":
            self.shard_events.append(event)
            observe_shard(
                self.shards,
                event,
                auto_promote=not self.review_required,
            )
            return
        if event_type == "review_shard":
            self.shard_events.append(event)
            if self.review_required:
                review_shard(self.shards, event)
            return
        raise ValueError(f"Unknown event type: {event_type}")

    def query(
        self,
        kind: str,
        key: str,
        query_date: str,
        scope: str | None = None,
    ) -> dict[str, Any]:
        if kind == "fact":
            candidates = self.records_by_key.get(key, [])
            records, inspected = resolve_temporal_records_with_cost(
                candidates,
                key,
                query_date,
                self.retractions,
                scope,
            )
            return render_records(
                records,
                events_scanned=0,
                records_inspected=inspected,
                index_lookups=1,
            )
        if kind == "shard":
            matches = resolve_shard_events(
                self.shard_events,
                key,
                query_date,
                auto_promote=not self.review_required,
                honor_reviews=self.review_required,
            )
            return render_records(
                matches,
                events_scanned=0,
                records_inspected=len(self.shard_events),
                index_lookups=1,
            )
        raise ValueError(f"Unknown query kind: {kind}")

    def snapshot(self) -> dict[str, Any]:
        return {
            "records": self.records,
            "retractions": self.retractions,
            "shards": self.shards,
            "shard_events": self.shard_events,
        }

    def state_components(self) -> dict[str, int]:
        return {
            "fact_versions": len(self.records),
            "retractions": len(self.retractions),
            "shard_events": len(self.shard_events),
            "shard_projections": len(self.shards),
        }


class TemporalAutoPromoteIndex(TemporalPolicyBase):
    name = "temporal_auto_promote"
    review_required = False


class ReviewedTemporalIndex(TemporalPolicyBase):
    name = "reviewed_temporal_index"
    review_required = True


class LatestWriteSnapshot(MemoryPolicy):
    name = "latest_write_snapshot"

    def __init__(self) -> None:
        self.current: dict[str, dict[str, Any]] = {}
        self.shards: dict[str, dict[str, Any]] = {}

    @staticmethod
    def slot(key: str, scope: str) -> str:
        return f"{key}\x1f{scope}"

    def apply(self, event: dict[str, Any]) -> None:
        event_type = event["event"]
        if event_type == "upsert":
            record = dict(event["record"])
            self.current[
                self.slot(record["key"], record.get("scope", "*"))
            ] = record
            return
        if event_type == "retract":
            target_id = event["target_id"]
            for slot, record in list(self.current.items()):
                if record["id"] == target_id:
                    del self.current[slot]
            return
        if event_type == "observe_shard":
            observe_shard(
                self.shards,
                event,
                auto_promote=True,
            )
            return
        if event_type == "review_shard":
            # This baseline promotes by repetition only and ignores reviews.
            return
        raise ValueError(f"Unknown event type: {event_type}")

    def query(
        self,
        kind: str,
        key: str,
        query_date: str,
        scope: str | None = None,
    ) -> dict[str, Any]:
        if kind == "fact":
            record = (
                self.current.get(self.slot(key, scope))
                if scope is not None
                else None
            )
            if record is None:
                record = self.current.get(self.slot(key, "*"))
            records = (
                [record]
                if record is not None and record_is_valid(record, query_date)
                else []
            )
            return render_records(
                records,
                events_scanned=0,
                records_inspected=1 if record is not None else 0,
                index_lookups=1,
            )
        if kind == "shard":
            matches = [
                {
                    "id": shard["id"],
                    "value": shard["value"],
                    "source_id": source_id,
                }
                for shard in self.shards.values()
                if shard["key"] == key and shard["active"]
                for source_id in shard["source_ids"]
            ]
            return render_records(
                matches,
                events_scanned=0,
                records_inspected=1 if matches else 0,
                index_lookups=1,
            )
        raise ValueError(f"Unknown query kind: {kind}")

    def snapshot(self) -> dict[str, Any]:
        return {"current": self.current, "shards": self.shards}

    def state_components(self) -> dict[str, int]:
        return {
            "current_records": len(self.current),
            "shard_projections": len(self.shards),
        }


class RawLogScan(MemoryPolicy):
    name = "raw_log_scan"

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def apply(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    def query(
        self,
        kind: str,
        key: str,
        query_date: str,
        scope: str | None = None,
    ) -> dict[str, Any]:
        if kind == "fact":
            records = [
                event["record"]
                for event in self.events
                if event["event"] == "upsert"
            ]
            retractions = {
                event["target_id"]: event["effective_at"]
                for event in self.events
                if event["event"] == "retract"
            }
            resolved, inspections = resolve_temporal_records_with_cost(
                    records,
                    key,
                    query_date,
                    retractions,
                    scope,
                )
            return render_records(
                resolved,
                events_scanned=len(self.events),
                records_inspected=inspections,
                index_lookups=0,
            )
        if kind == "shard":
            matches = resolve_shard_events(
                self.events,
                key,
                query_date,
                auto_promote=False,
                honor_reviews=True,
            )
            return render_records(
                matches,
                events_scanned=len(self.events),
                records_inspected=len(self.events),
                index_lookups=0,
            )
        raise ValueError(f"Unknown query kind: {kind}")

    def snapshot(self) -> dict[str, Any]:
        return {"events": self.events}

    def state_components(self) -> dict[str, int]:
        return {"events": len(self.events)}


DEFAULT_ENGINE_FACTORIES: dict[str, Callable[[], MemoryPolicy]] = {
    RawLogScan.name: RawLogScan,
    LatestWriteSnapshot.name: LatestWriteSnapshot,
    TemporalAutoPromoteIndex.name: TemporalAutoPromoteIndex,
    ReviewedTemporalIndex.name: ReviewedTemporalIndex,
}


def validate_workload(
    events: list[dict[str, Any]],
    questions: list[dict[str, Any]],
) -> None:
    event_ids = [event["id"] for event in events]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("event ids must be unique")
    question_ids = [question["id"] for question in questions]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("question ids must be unique")
    sequences = [event.get("sequence") for event in events]
    if sequences != list(range(1, len(events) + 1)):
        raise ValueError("event sequence must be contiguous and ordered from one")
    try:
        timestamps = [date.fromisoformat(event["timestamp"]) for event in events]
        for question in questions:
            date.fromisoformat(question["query_date"])
    except (KeyError, ValueError) as error:
        raise ValueError(f"invalid workload date: {error}") from error
    if list(zip(timestamps, sequences)) != sorted(zip(timestamps, sequences)):
        raise ValueError("events must be ordered by timestamp and sequence")
    record_ids: set[str] = set()
    observed_shards: dict[str, tuple[str, str]] = {}
    for event in events:
        event_type = event.get("event")
        if event_type == "upsert":
            record = event["record"]
            if record["id"] in record_ids:
                raise ValueError(f"duplicate record id: {record['id']}")
            missing_superseded = [
                target
                for target in record.get("supersedes", [])
                if target not in record_ids
            ]
            if missing_superseded:
                raise ValueError(
                    f"supersedes unknown records: {missing_superseded}"
                )
            record_ids.add(record["id"])
            confidence = float(record["confidence"])
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"invalid confidence: {record['id']}")
            valid_from = date.fromisoformat(record["valid_from"])
            valid_to = (
                date.fromisoformat(record["valid_to"])
                if record.get("valid_to")
                else None
            )
            if valid_to is not None and valid_to < valid_from:
                raise ValueError(f"invalid validity window: {record['id']}")
        elif event_type == "observe_shard":
            identity = (event["key"], event["value"])
            previous = observed_shards.setdefault(event["shard_id"], identity)
            if previous != identity:
                raise ValueError(
                    f"shard identity collision: {event['shard_id']}"
                )
        elif event_type == "review_shard":
            if event["shard_id"] not in observed_shards:
                raise ValueError(
                    f"review references unknown shard: {event['shard_id']}"
                )
            if event.get("decision") not in {"approved", "rejected", "deferred"}:
                raise ValueError(f"invalid review decision: {event.get('decision')}")
        elif event_type == "retract":
            if event["target_id"] not in record_ids:
                raise ValueError(
                    f"retraction references unknown record: {event['target_id']}"
                )
            effective_at = date.fromisoformat(event["effective_at"])
            if effective_at < date.fromisoformat(event["timestamp"]):
                raise ValueError(
                    f"retraction predates its event: {event['target_id']}"
                )
        else:
            raise ValueError(f"unknown event type: {event_type}")
    missing = sorted(
        {
            question["after_event"]
            for question in questions
            if question["after_event"] not in set(event_ids)
        }
    )
    if missing:
        raise ValueError(f"questions reference missing events: {missing}")
    inconsistent = [
        question["id"]
        for question in questions
        if bool(question["should_abstain"])
        != (not bool(question["expected_values"]))
    ]
    if inconsistent:
        raise ValueError(
            f"abstention labels disagree with expected values: {inconsistent}"
        )


def score_question(
    question: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    expected_values = set(map(str, question["expected_values"]))
    actual_values = set(map(str, result["values"]))
    expected_sources = set(map(str, question["expected_source_ids"]))
    actual_sources = set(map(str, result["source_ids"]))
    exact = actual_values == expected_values
    return {
        "exact_value_set": exact,
        "source_id_set_exact": (
            exact and actual_sources == expected_sources
            if expected_values
            else None
        ),
        "contradiction_leakage": bool(actual_values - expected_values),
        "false_refusal": bool(expected_values) and not actual_values,
        "unsafe_answer": not expected_values and bool(actual_values),
    }


def summarize(
    rows: list[dict[str, Any]],
    engines: dict[str, MemoryPolicy],
    reconstruction: dict[str, bool],
) -> list[dict[str, Any]]:
    by_architecture: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_architecture[row["architecture"]].append(row)
    summaries = []
    for architecture, selected in sorted(by_architecture.items()):
        by_dimension: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            by_dimension[row["dimension"]].append(row)
        answerable = [row for row in selected if not row["should_abstain"]]
        unanswerable = [row for row in selected if row["should_abstain"]]
        summaries.append(
            {
                "architecture": architecture,
                "questions": len(selected),
                "exact_value_accuracy": statistics.fmean(
                    float(row["exact_value_set"]) for row in selected
                ),
                "source_id_set_accuracy_answerable": statistics.fmean(
                    float(row["source_id_set_exact"]) for row in selected
                    if row["source_id_set_exact"] is not None
                ),
                "source_id_questions": len(answerable),
                "correct_abstention_rate": statistics.fmean(
                    float(row["exact_value_set"]) for row in unanswerable
                )
                if unanswerable
                else 0.0,
                "contradiction_leakage_rate": statistics.fmean(
                    float(row["contradiction_leakage"]) for row in selected
                ),
                "false_refusal_rate": statistics.fmean(
                    float(row["false_refusal"]) for row in answerable
                )
                if answerable
                else 0.0,
                "unsafe_answer_rate": statistics.fmean(
                    float(row["unsafe_answer"]) for row in unanswerable
                )
                if unanswerable
                else 0.0,
                "mean_events_scanned": statistics.fmean(
                    row["events_scanned"] for row in selected
                ),
                "mean_records_inspected": statistics.fmean(
                    row["records_inspected"] for row in selected
                ),
                "mean_index_lookups": statistics.fmean(
                    row["index_lookups"] for row in selected
                ),
                "mean_query_latency_ms": statistics.fmean(
                    row["query_latency_ms"] for row in selected
                ),
                "final_state_components": engines[architecture].state_components(),
                "json_roundtrip_reconstruction_digest_match": reconstruction[
                    architecture
                ],
                "dimension_accuracy": {
                    dimension: statistics.fmean(
                        float(row["exact_value_set"]) for row in dimension_rows
                    )
                    for dimension, dimension_rows in sorted(by_dimension.items())
                },
                "dimension_counts": {
                    dimension: len(dimension_rows)
                    for dimension, dimension_rows in sorted(by_dimension.items())
                },
            }
        )
    return summaries


def evaluate(
    events: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    *,
    engine_factories: dict[
        str, Callable[[], MemoryPolicy]
    ] = DEFAULT_ENGINE_FACTORIES,
) -> dict[str, Any]:
    validate_workload(events, questions)
    engines = {name: factory() for name, factory in engine_factories.items()}
    questions_by_event: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for question in questions:
        questions_by_event[question["after_event"]].append(question)
    rows = []
    for event in events:
        for engine in engines.values():
            engine.apply(event)
        for question in questions_by_event[event["id"]]:
            for architecture, engine in engines.items():
                started = time.perf_counter()
                result = engine.query(
                    question["kind"],
                    question["memory_key"],
                    question["query_date"],
                    question.get("scope"),
                )
                latency_ms = (time.perf_counter() - started) * 1000
                rows.append(
                    {
                        "question_id": question["id"],
                        "after_event": question["after_event"],
                        "query_date": question["query_date"],
                        "kind": question["kind"],
                        "memory_key": question["memory_key"],
                        "scope": question.get("scope"),
                        "dimension": question["dimension"],
                        "should_abstain": question["should_abstain"],
                        "expected_values": question["expected_values"],
                        "expected_source_ids": question["expected_source_ids"],
                        "architecture": architecture,
                        "retrieved_values": result["values"],
                        "retrieved_source_ids": result["source_ids"],
                        "retrieved_record_ids": result["record_ids"],
                        "events_scanned": result["events_scanned"],
                        "records_inspected": result["records_inspected"],
                        "index_lookups": result["index_lookups"],
                        "query_latency_ms": latency_ms,
                        **score_question(question, result),
                    }
                )
    reconstruction = {}
    for name, factory in engine_factories.items():
        replay = factory()
        reloaded_events = json.loads(json.dumps(events))
        for event in reloaded_events:
            replay.apply(event)
        reconstruction[name] = replay.state_digest() == engines[name].state_digest()
    return {
        "rows": rows,
        "summaries": summarize(rows, engines, reconstruction),
        "final_state_digests": {
            name: engine.state_digest() for name, engine in engines.items()
        },
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--events",
        type=Path,
        default=root / "data" / "incremental-memory-events.jsonl",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=root / "data" / "incremental-memory-questions.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = load_jsonl(args.events)
    questions = load_jsonl(args.questions)
    payload = evaluate(events, questions)
    payload["manifest"] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_version": PROTOCOL_VERSION,
        "scope": (
            "Deterministic incremental memory-policy isolation on one public "
            "synthetic event stream"
        ),
        "events": str(args.events.resolve()),
        "events_sha256": sha256_file(args.events),
        "questions": str(args.questions.resolve()),
        "questions_sha256": sha256_file(args.questions),
        "event_count": len(events),
        "question_count": len(questions),
        "architectures": list(DEFAULT_ENGINE_FACTORIES),
        "architecture_contracts": {
            "raw_log_scan": (
                "Retain the immutable event log and reconstruct temporally valid, "
                "reviewed state at query time."
            ),
            "latest_write_snapshot": (
                "Keep only the last arrived value per key and auto-activate a "
                "shard after two observations; human review events are ignored."
            ),
            "temporal_auto_promote": (
                "Preserve authoritative temporal versions and retractions, but "
                "auto-activate a shard after two observations and ignore human "
                "review events."
            ),
            "reviewed_temporal_index": (
                "Preserve authoritative temporal versions and retractions, and "
                "activate shards only after explicit approval."
            ),
        },
        "authority_confidence_threshold": MINIMUM_AUTHORITY_CONFIDENCE,
        "limitations": [
            "Structured events are gold inputs, so extraction quality is isolated out.",
            "The workload is synthetic and intentionally small.",
            "Source-id set accuracy checks identifiers, not semantic entailment.",
            "Events scanned, records inspected, and index lookups are separate counters and must not be collapsed into one cost score.",
            "Microsecond timings are implementation diagnostics, not production latency claims.",
            "Human review decisions are synthetic labels, not measured reviewer behavior.",
            "Expiry and retraction test non-retrieval, not physical deletion or storage compaction.",
            "Reconstruction uses a JSON round-trip in one process; crash recovery and persisted-state restoration remain untested.",
        ],
    }
    write_result(args.output, payload)
    print(args.output)


if __name__ == "__main__":
    main()
