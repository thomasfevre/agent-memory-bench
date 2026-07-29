#!/usr/bin/env python3
"""Score Priority 3 extraction quality and temporal-state preservation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from priority3_temporal_model import EXTRACTION_FIELDS, load_jsonl


CANDIDATE_TYPES = {
    "assertion",
    "correction",
    "late_arrival",
    "shard_approval",
}
TEMPORAL_FIELDS = ("asserted_at", "effective_from", "effective_until")
POSTHOC_UNOBSERVABLE_FIELDS = {
    "expiration": {"confidence", "effective_from"},
    "retraction": {"confidence"},
}


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
        timezone.utc
    )


class TemporalMemory:
    """Order-independent event store with deterministic temporal queries."""

    def __init__(self) -> None:
        self.events: dict[str, dict[str, Any]] = {}
        self.duplicate_delivery_count = 0
        self.conflicting_delivery_count = 0

    @property
    def unique_event_count(self) -> int:
        return len(self.events)

    def ingest(self, event_id: str, extraction: dict[str, Any]) -> None:
        previous = self.events.get(event_id)
        if previous is not None:
            self.duplicate_delivery_count += 1
            if previous != extraction:
                self.conflicting_delivery_count += 1
            return
        self.events[event_id] = extraction

    def _invalid_after(
        self,
        event_id: str,
        candidate: dict[str, Any],
    ) -> list[datetime]:
        invalidations = []
        shard_id = candidate.get("target_event_id")
        for other in self.events.values():
            target = other.get("target_event_id")
            if other["event_type"] == "correction" and target == event_id:
                invalidations.append(parse_time(other["effective_from"]))
            elif other["event_type"] == "retraction" and target == event_id:
                invalidations.append(parse_time(other["effective_from"]))
            elif (
                other["event_type"] == "expiration"
                and target == event_id
                and other.get("effective_until")
            ):
                invalidations.append(parse_time(other["effective_until"]))
            elif (
                other["event_type"] == "shard_rejection"
                and shard_id
                and target == shard_id
            ):
                invalidations.append(parse_time(other["effective_from"]))
        if candidate.get("effective_until"):
            invalidations.append(parse_time(candidate["effective_until"]))
        return invalidations

    def query_record(
        self,
        entity_key: str,
        query_at: str,
    ) -> dict[str, Any] | None:
        when = parse_time(query_at)
        candidates = []
        for event_id, event in self.events.items():
            if event["entity_key"] != entity_key:
                continue
            if event["event_type"] not in CANDIDATE_TYPES:
                continue
            if event["confidence"] < 0.8:
                continue
            effective_from = parse_time(event["effective_from"])
            if effective_from > when:
                continue
            invalidations = self._invalid_after(event_id, event)
            if any(when >= invalid_at for invalid_at in invalidations):
                continue
            candidates.append(
                (
                    float(event["confidence"]),
                    effective_from,
                    parse_time(event["asserted_at"]),
                    event_id,
                    event,
                )
            )
        if not candidates:
            return None
        return max(candidates, key=lambda row: row[:4])[4]

    def query(self, entity_key: str, query_at: str) -> str | None:
        record = self.query_record(entity_key, query_at)
        return record["value"] if record else None

    def signature(self) -> str:
        canonical = json.dumps(
            self.events,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def score_extraction_rows(
    observations: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_by_id = {
        observation["id"]: observation["expected"]
        for observation in observations
    }
    field_matches = 0
    observable_field_matches = 0
    observable_field_total = 0
    temporal_matches = 0
    event_type_matches = 0
    provenance_matches = 0
    successful = 0
    extractions_by_event: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        expected = expected_by_id[row["event_id"]]
        observable_fields = EXTRACTION_FIELDS - POSTHOC_UNOBSERVABLE_FIELDS.get(
            expected["event_type"],
            set(),
        )
        observable_field_total += len(observable_fields)
        if row["status"] != "success":
            extractions_by_event[row["event_id"]].append("<error>")
            continue
        successful += 1
        extracted = row["extraction"]
        field_matches += sum(
            extracted.get(field) == expected[field]
            for field in EXTRACTION_FIELDS
        )
        observable_field_matches += sum(
            extracted.get(field) == expected[field]
            for field in observable_fields
        )
        temporal_matches += all(
            extracted.get(field) == expected[field]
            for field in TEMPORAL_FIELDS
        )
        event_type_matches += (
            extracted.get("event_type") == expected["event_type"]
        )
        provenance_matches += (
            extracted.get("source_id") == expected["source_id"]
        )
        extractions_by_event[row["event_id"]].append(
            json.dumps(extracted, sort_keys=True, separators=(",", ":"))
        )
    attempts = len(rows)
    stable = sum(
        len(values) >= 2 and len(set(values)) == 1
        for values in extractions_by_event.values()
    )
    return {
        "attempts": attempts,
        "successful_attempts": successful,
        "field_accuracy": field_matches / (attempts * len(EXTRACTION_FIELDS)),
        "text_observable_field_accuracy": (
            observable_field_matches / observable_field_total
        ),
        "text_observable_score_is_posthoc": True,
        "temporal_window_exact": temporal_matches / attempts,
        "event_type_accuracy": event_type_matches / attempts,
        "provenance_exact": provenance_matches / attempts,
        "stable_event_fraction": stable / len(expected_by_id),
    }


def build_memory(
    deliveries: list[dict[str, Any]],
    events_by_id: dict[str, dict[str, Any]],
) -> TemporalMemory:
    memory = TemporalMemory()
    for delivery in deliveries:
        event = events_by_id.get(delivery["event_id"])
        if event is not None:
            memory.ingest(delivery["event_id"], event)
    return memory


def entity_query_points(
    observations: list[dict[str, Any]],
) -> dict[str, list[str]]:
    points: dict[str, set[datetime]] = defaultdict(set)
    latest = datetime.min.replace(tzinfo=timezone.utc)
    for observation in observations:
        event = observation["expected"]
        entity = event["entity_key"]
        for field in ("effective_from", "effective_until"):
            value = event.get(field)
            if not value:
                continue
            boundary = parse_time(value)
            points[entity].add(boundary)
            points[entity].add(boundary - timedelta(seconds=1))
            latest = max(latest, boundary)
    final = latest + timedelta(days=1)
    for entity in points:
        points[entity].add(final)
    return {
        entity: [
            timestamp.isoformat().replace("+00:00", "Z")
            for timestamp in sorted(timestamps)
        ]
        for entity, timestamps in points.items()
    }


def evaluate_schedule_matrix(
    observations: list[dict[str, Any]],
    schedules: dict[str, dict[str, Any]],
    extraction_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_by_id = {
        observation["id"]: observation["expected"]
        for observation in observations
    }
    query_points = entity_query_points(observations)
    final_at = max(
        timestamp
        for timestamps in query_points.values()
        for timestamp in timestamps
    )
    rows_by_repetition: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in extraction_rows:
        if row["status"] == "success":
            rows_by_repetition[row["repetition"]][row["event_id"]] = row[
                "extraction"
            ]

    schedule_rows = []
    final_signatures: dict[int, set[str]] = defaultdict(set)
    for repetition, extracted_by_id in sorted(rows_by_repetition.items()):
        for schedule_name, schedule in schedules.items():
            deliveries = schedule["deliveries"]
            expected_memory = build_memory(deliveries, expected_by_id)
            extracted_memory = build_memory(deliveries, extracted_by_id)
            entity_keys = sorted(query_points)
            final_correct = sum(
                extracted_memory.query(entity, final_at)
                == expected_memory.query(entity, final_at)
                for entity in entity_keys
            )
            historical_correct = 0
            historical_total = 0
            for entity, timestamps in query_points.items():
                for timestamp in timestamps:
                    historical_total += 1
                    historical_correct += (
                        extracted_memory.query(entity, timestamp)
                        == expected_memory.query(entity, timestamp)
                    )

            abstention_correct = 0
            abstention_total = 0
            for observation in observations:
                event = observation["expected"]
                if event["event_type"] not in {"expiration", "retraction"}:
                    continue
                boundary = (
                    event["effective_until"]
                    if event["event_type"] == "expiration"
                    else event["effective_from"]
                )
                if expected_memory.query(event["entity_key"], boundary) is None:
                    abstention_total += 1
                    abstention_correct += (
                        extracted_memory.query(event["entity_key"], boundary)
                        is None
                    )

            final_values = {
                entity: extracted_memory.query(entity, final_at)
                for entity in entity_keys
            }
            final_signature = hashlib.sha256(
                json.dumps(
                    final_values,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            final_signatures[repetition].add(final_signature)
            retry_deliveries = sum(
                delivery["attempt"] > 1 for delivery in deliveries
            )
            duplicate_amplification = (
                max(
                    0,
                    extracted_memory.unique_event_count
                    - len(set(extracted_by_id)),
                )
                / retry_deliveries
                if retry_deliveries
                else 0.0
            )
            schedule_rows.append(
                {
                    "repetition": repetition,
                    "schedule": schedule_name,
                    "final_state_exact": final_correct / len(entity_keys),
                    "historical_query_accuracy": (
                        historical_correct / historical_total
                    ),
                    "abstention_after_invalidation": (
                        abstention_correct / abstention_total
                        if abstention_total
                        else None
                    ),
                    "abstention_cases": abstention_total,
                    "duplicate_amplification": duplicate_amplification,
                    "unique_events": extracted_memory.unique_event_count,
                    "duplicate_deliveries": (
                        extracted_memory.duplicate_delivery_count
                    ),
                    "conflicting_deliveries": (
                        extracted_memory.conflicting_delivery_count
                    ),
                    "final_state_signature": final_signature,
                }
            )
    return {
        "rows": schedule_rows,
        "schedule_stability_by_repetition": {
            str(repetition): len(signatures) == 1
            for repetition, signatures in final_signatures.items()
        },
        "query_count_per_repetition_schedule": sum(
            len(timestamps) for timestamps in query_points.values()
        ),
        "final_query_at": final_at,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--schedules", type=Path, required=True)
    parser.add_argument("--extractions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    observations = load_jsonl(args.observations)
    schedules = json.loads(args.schedules.read_text(encoding="utf-8"))
    extractions = json.loads(args.extractions.read_text(encoding="utf-8"))
    payload = {
        "schema_version": 1,
        "campaign": "priority3_temporal_scoring",
        "source_extraction": str(args.extractions),
        "source_extraction_sha256": hashlib.sha256(
            args.extractions.read_bytes()
        ).hexdigest(),
        "extraction_metrics": score_extraction_rows(
            observations,
            extractions["rows"],
        ),
        "schedule_metrics": evaluate_schedule_matrix(
            observations,
            schedules,
            extractions["rows"],
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
