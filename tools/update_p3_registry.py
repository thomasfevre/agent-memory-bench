#!/usr/bin/env python3
"""Publish compact Priority 3 records without exposing model prompts."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REGISTRY = RESULTS / "published" / "registry.json"
TEMPORAL = RESULTS / "P3-TEMPORAL-SCORING-QWEN25-14B-20260729.json"
CRASH = RESULTS / "P3-DERIVED-INDEX-CRASH-MATRIX-20260729.json"
DELETION = RESULTS / "P3-DELETION-COMPACTION-20260729.json"


def rounded(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def upsert(registry: dict[str, Any], record: dict[str, Any]) -> None:
    for index, existing in enumerate(registry["runs"]):
        if existing["id"] == record["id"]:
            registry["runs"][index] = record
            return
    registry["runs"].append(record)


def temporal_record(payload: dict[str, Any]) -> dict[str, Any]:
    extraction = payload["extraction_metrics"]
    rows = payload["schedule_metrics"]["rows"]

    def average(key: str) -> float:
        return rounded(mean(float(row[key]) for row in rows))

    return {
        "id": "priority3-temporal-qwen25-14b-20260729",
        "date": "2026-07-30",
        "phase": "durability",
        "dataset": "Priority 3 public temporal observations",
        "task": "Extract and replay corrections, expirations, retractions and duplicates",
        "method": "Qwen 2.5 14B extraction followed by deterministic event replay",
        "reader": "qwen2.5:14b",
        "evidence_level": "controlled",
        "sample": "60 observations, 3 repetitions and 5 delivery schedules",
        "repetitions": 3,
        "metrics": {
            "extraction": {
                "attempts": extraction["attempts"],
                "successful_attempts": extraction["successful_attempts"],
                "event_type_accuracy": rounded(
                    extraction["event_type_accuracy"]
                ),
                "strict_field_accuracy": rounded(
                    extraction["field_accuracy"]
                ),
                "temporal_window_exact": rounded(
                    extraction["temporal_window_exact"]
                ),
                "provenance_exact": rounded(
                    extraction["provenance_exact"]
                ),
                "posthoc_observable_field_accuracy": rounded(
                    extraction["text_observable_field_accuracy"]
                ),
                "posthoc_metric": bool(
                    extraction["text_observable_score_is_posthoc"]
                ),
                "stable_event_fraction": rounded(
                    extraction["stable_event_fraction"]
                ),
            },
            "schedule": {
                "schedule_count": len(
                    {str(row["schedule"]) for row in rows}
                ),
                "repetition_schedule_rows": len(rows),
                "queries_per_repetition_schedule": payload[
                    "schedule_metrics"
                ]["query_count_per_repetition_schedule"],
                "selected_final_value_exact": average(
                    "selected_final_value_exact"
                ),
                "complete_active_state_exact": average(
                    "final_state_exact"
                ),
                "historical_active_state_accuracy": average(
                    "historical_active_state_accuracy"
                ),
                "stale_record_leakage_rate": average(
                    "stale_record_leakage_rate"
                ),
                "abstention_after_invalidation": average(
                    "abstention_after_invalidation"
                ),
                "duplicate_amplification": average(
                    "duplicate_amplification"
                ),
                "stable_across_schedules": all(
                    payload["schedule_metrics"][
                        "schedule_stability_by_repetition"
                    ].values()
                ),
            },
        },
        "budget": {
            "model": "qwen2.5:14b",
            "observations": 60,
            "repetitions": 3,
            "delivery_schedules": 5,
        },
        "conclusion": (
            "The selected final answer was always correct, but complete active "
            "state exactness was only 85%; the broader state metric exposed "
            "stale records hidden by answer-only scoring."
        ),
        "limitation": (
            "This controlled corpus is synthetic and bounded. The observable-"
            "field score is explicitly posthoc; strict field accuracy remains "
            "the primary extraction metric."
        ),
        "evidence_files": [
            "results/P3-TEMPORAL-SCORING-QWEN25-14B-20260729.json"
        ],
    }


def crash_record(payload: dict[str, Any]) -> dict[str, Any]:
    scenarios = payload["scenarios"]
    passed = [
        row
        for row in scenarios
        if row["crash_observed"]
        and row["semantic_signature_matches_uninterrupted"]
        and row["source_ids_match_uninterrupted"]
        and not row["orphan_generations"]
    ]
    return {
        "id": "priority3-derived-index-crash-matrix-20260729",
        "date": "2026-07-30",
        "phase": "durability",
        "dataset": "Priority 3 deterministic derived-index fixture",
        "task": "Recover temporal, full-text, vector and graph views after injected crashes",
        "method": "Crash injection around one transactional generation manifest",
        "reader": None,
        "evidence_level": "controlled",
        "sample": "Eight declared crash boundaries",
        "repetitions": 1,
        "metrics": {
            "declared_boundaries": len(scenarios),
            "passed_boundaries": len(passed),
            "all_scenarios_pass": bool(payload["all_scenarios_pass"]),
            "orphan_generations": sum(
                len(row["orphan_generations"]) for row in scenarios
            ),
            "recovery_actions": {
                action: sum(
                    row["recovery_action"] == action for row in scenarios
                )
                for action in sorted(
                    {str(row["recovery_action"]) for row in scenarios}
                )
            },
            "wall_time_seconds": rounded(
                payload["wall_time_seconds"], 6
            ),
        },
        "budget": {
            "crash_boundaries": len(scenarios),
            "derived_views": 4,
            "execution": "local deterministic fixture",
        },
        "conclusion": (
            "All eight injected boundaries recovered the same semantic state "
            "and source IDs as the uninterrupted build, with no orphan "
            "generation."
        ),
        "limitation": (
            "This validates one local SQLite-backed design. It does not prove "
            "distributed durability, arbitrary kernel failure behavior or "
            "hardware flush guarantees."
        ),
        "evidence_files": [
            "results/P3-DERIVED-INDEX-CRASH-MATRIX-20260729.json"
        ],
    }


def deletion_record(payload: dict[str, Any]) -> dict[str, Any]:
    verification = payload["verification"]
    return {
        "id": "priority3-deletion-compaction-20260729",
        "date": "2026-07-30",
        "phase": "durability",
        "dataset": "Priority 3 deterministic deletion fixture",
        "task": "Remove retracted, expired and tombstoned data from every active surface",
        "method": "Logical deletion, index rebuild, SQLite VACUUM and backup verification",
        "reader": None,
        "evidence_level": "controlled",
        "sample": "Seven active storage and retrieval surfaces",
        "repetitions": 1,
        "metrics": {
            "clean_surfaces": sum(bool(value) for value in verification.values()),
            "declared_surfaces": len(verification),
            "all_active_surfaces_clean": bool(
                payload["all_active_surfaces_clean"]
            ),
            "signed_generation_valid": bool(
                payload["signed_generation_valid"]
            ),
            "audit_events_use_hashed_targets": bool(
                payload["audit_events_use_hashed_targets"]
            ),
            "immutable_audit_events_retained": payload[
                "immutable_audit_events_retained"
            ],
            "old_backup_contains_deleted_payloads": bool(
                payload["old_backup_contains_deleted_payloads"]
            ),
            "secure_flash_erasure_claimed": bool(
                payload["secure_flash_erasure_claimed"]
            ),
            "wall_time_seconds": rounded(
                payload["wall_time_seconds"], 6
            ),
        },
        "budget": {
            "active_surfaces": len(verification),
            "execution": "local deterministic fixture",
        },
        "conclusion": (
            "All seven active surfaces and the new backup were clean after "
            "compaction while hashed audit events remained verifiable."
        ),
        "limitation": (
            "The old backup intentionally retains deleted payloads and needs "
            "its own retention policy. SQLite VACUUM does not prove physical "
            "SSD flash erasure."
        ),
        "evidence_files": [
            "results/P3-DELETION-COMPACTION-20260729.json"
        ],
    }


def main() -> int:
    required = [REGISTRY, TEMPORAL, CRASH, DELETION]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {', '.join(missing)}")
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    temporal = json.loads(TEMPORAL.read_text(encoding="utf-8"))
    crash = json.loads(CRASH.read_text(encoding="utf-8"))
    deletion = json.loads(DELETION.read_text(encoding="utf-8"))
    upsert(registry, temporal_record(temporal))
    upsert(registry, crash_record(crash))
    upsert(registry, deletion_record(deletion))
    registry["updated_at"] = "2026-07-30"
    for finding in (
        "Answer-only temporal scoring hid stale active records: selected values were 100% correct while complete active-state exactness was 85%.",
        "A deterministic four-view derived index recovered the uninterrupted semantic state at all eight injected crash boundaries.",
        "Deletion and compaction removed payloads from all seven active surfaces, but old backups still require an explicit retention policy.",
    ):
        if finding not in registry["findings"]:
            registry["findings"].append(finding)
    for limitation in (
        "Priority 3 temporal, crash and deletion results use controlled local fixtures and do not establish distributed-system guarantees.",
        "SQLite VACUUM verifies the rebuilt database file but does not prove physical SSD flash erasure; old backups retain data until separately expired.",
    ):
        if limitation not in registry["limitations"]:
            registry["limitations"].append(limitation)
    REGISTRY.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    print(REGISTRY.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
