#!/usr/bin/env python3
"""Verify logical deletion across active local memory surfaces."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from priority3_derived_recovery import canonical_json, deterministic_vector


AS_OF = date(2026, 10, 1)
RETENTION_DAYS = 30


def fixture_journal() -> list[dict[str, Any]]:
    return [
        {
            "id": "event_source_delete_alpha",
            "timestamp": "2026-09-20",
            "kind": "record",
            "key": "deployment_note",
            "value": "source_sensitive_payload_needle",
            "source_id": "source_delete_alpha",
        },
        {
            "id": "event_fact_delete_beta",
            "timestamp": "2026-09-22",
            "kind": "record",
            "key": "temporary_secret",
            "value": "fact_sensitive_payload_needle",
            "source_id": "source_keep_beta",
        },
        {
            "id": "event_shard_delete_gamma",
            "timestamp": "2026-09-23",
            "kind": "shard",
            "key": "team_rule",
            "value": "shard_sensitive_payload_needle",
            "source_id": "source_keep_gamma",
            "shard_id": "shard_delete_gamma",
        },
        {
            "id": "event_retention_delete_delta",
            "timestamp": "2026-07-01",
            "kind": "record",
            "key": "old_note",
            "value": "retention_sensitive_payload_needle",
            "source_id": "source_retention_delta",
        },
        {
            "id": "event_keep_epsilon",
            "timestamp": "2026-09-25",
            "kind": "record",
            "key": "safe_note",
            "value": "safe_active_value",
            "source_id": "source_keep_epsilon",
        },
    ]


def deleted_needles() -> list[str]:
    return [
        "event_source_delete_alpha",
        "source_delete_alpha",
        "source_sensitive_payload_needle",
        "event_fact_delete_beta",
        "fact_sensitive_payload_needle",
        "event_shard_delete_gamma",
        "shard_delete_gamma",
        "shard_sensitive_payload_needle",
        "event_retention_delete_delta",
        "source_retention_delta",
        "retention_sensitive_payload_needle",
    ]


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def create_original_database(path: Path, events: list[dict[str, Any]]) -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE canonical_journal(event_id TEXT PRIMARY KEY, payload TEXT)"
    )
    connection.executemany(
        "INSERT INTO canonical_journal(event_id, payload) VALUES (?, ?)",
        [(event["id"], canonical_json(event)) for event in events],
    )
    connection.commit()
    connection.execute("VACUUM")
    connection.close()


def backup_database(source: Path, destination: Path) -> None:
    source_connection = sqlite3.connect(source)
    destination_connection = sqlite3.connect(destination)
    source_connection.backup(destination_connection)
    destination_connection.close()
    source_connection.close()


def compact_journal(
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    source_tombstones = {"source_delete_alpha"}
    fact_retractions = {"event_fact_delete_beta"}
    shard_expirations = {"shard_delete_gamma"}
    cutoff = AS_OF - timedelta(days=RETENTION_DAYS)
    active = []
    audit = []
    operation_counts = {
        "source_tombstones": 0,
        "fact_retractions": 0,
        "shard_expirations": 0,
        "retention_deletions": 0,
        "retention_days": RETENTION_DAYS,
    }
    for event in events:
        reason = None
        target = event["id"]
        if event["source_id"] in source_tombstones:
            reason = "source_tombstone"
            target = event["source_id"]
            operation_counts["source_tombstones"] += 1
        elif event["id"] in fact_retractions:
            reason = "fact_retraction"
            operation_counts["fact_retractions"] += 1
        elif event.get("shard_id") in shard_expirations:
            reason = "shard_expiration"
            target = event["shard_id"]
            operation_counts["shard_expirations"] += 1
        elif date.fromisoformat(event["timestamp"]) < cutoff:
            reason = "retention"
            operation_counts["retention_deletions"] += 1
        if reason:
            audit.append(
                {
                    "kind": reason,
                    "recorded_at": AS_OF.isoformat(),
                    "target_sha256": sha256(target),
                    "payload_sha256": sha256(canonical_json(event)),
                }
            )
        else:
            active.append(event)
    return active, audit, operation_counts


def create_compacted_database(
    path: Path,
    *,
    source_events: list[dict[str, Any]],
    active_events: list[dict[str, Any]],
    audit_events: list[dict[str, Any]],
) -> dict[str, str]:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE canonical_events (
            event_id TEXT PRIMARY KEY,
            payload TEXT NOT NULL
        );
        CREATE TABLE immutable_audit (
            audit_id INTEGER PRIMARY KEY,
            payload TEXT NOT NULL
        );
        CREATE TABLE temporal_current (
            record_id TEXT PRIMARY KEY,
            memory_key TEXT NOT NULL,
            value TEXT NOT NULL,
            source_id TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE full_text USING fts5(
            record_id UNINDEXED,
            content
        );
        CREATE TABLE dense_vectors (
            record_id TEXT PRIMARY KEY,
            vector_json TEXT NOT NULL
        );
        CREATE TABLE graph_edges (
            edge_id TEXT PRIMARY KEY,
            source_node TEXT NOT NULL,
            target_node TEXT NOT NULL,
            relation TEXT NOT NULL
        );
        CREATE TABLE signed_generation (
            generation_id INTEGER PRIMARY KEY,
            source_journal_sha256 TEXT NOT NULL,
            active_sha256 TEXT NOT NULL,
            audit_sha256 TEXT NOT NULL,
            signature TEXT NOT NULL
        );
        """
    )
    connection.executemany(
        "INSERT INTO canonical_events(event_id, payload) VALUES (?, ?)",
        [(event["id"], canonical_json(event)) for event in active_events],
    )
    connection.executemany(
        "INSERT INTO immutable_audit(payload) VALUES (?)",
        [(canonical_json(event),) for event in audit_events],
    )
    for event in active_events:
        content = (
            f"{event['kind']} {event['key']} {event['value']} "
            f"source {event['source_id']}"
        )
        connection.execute(
            """
            INSERT INTO temporal_current(
                record_id, memory_key, value, source_id
            ) VALUES (?, ?, ?, ?)
            """,
            (
                event["id"],
                event["key"],
                event["value"],
                event["source_id"],
            ),
        )
        connection.execute(
            "INSERT INTO full_text(record_id, content) VALUES (?, ?)",
            (event["id"], content),
        )
        connection.execute(
            """
            INSERT INTO dense_vectors(record_id, vector_json)
            VALUES (?, ?)
            """,
            (event["id"], canonical_json(deterministic_vector(content))),
        )
        connection.executemany(
            """
            INSERT INTO graph_edges(
                edge_id, source_node, target_node, relation
            ) VALUES (?, ?, ?, ?)
            """,
            [
                (
                    f"{event['id']}:value",
                    f"key:{event['key']}",
                    f"value:{event['value']}",
                    "HAS_VALUE",
                ),
                (
                    f"{event['id']}:source",
                    f"value:{event['value']}",
                    f"source:{event['source_id']}",
                    "SUPPORTED_BY",
                ),
            ],
        )
    source_hash = sha256(
        "\n".join(canonical_json(event) for event in source_events)
    )
    active_hash = sha256(
        "\n".join(canonical_json(event) for event in active_events)
    )
    audit_hash = sha256(
        "\n".join(canonical_json(event) for event in audit_events)
    )
    signature = sha256(f"{source_hash}\0{active_hash}\0{audit_hash}")
    connection.execute(
        """
        INSERT INTO signed_generation(
            generation_id, source_journal_sha256, active_sha256,
            audit_sha256, signature
        ) VALUES (1, ?, ?, ?, ?)
        """,
        (source_hash, active_hash, audit_hash, signature),
    )
    connection.commit()
    connection.execute("VACUUM")
    connection.close()
    return {
        "source_journal_sha256": source_hash,
        "active_sha256": active_hash,
        "audit_sha256": audit_hash,
        "signature": signature,
    }


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm)


def database_surface_payloads(path: Path, needles: list[str]) -> dict[str, Any]:
    connection = sqlite3.connect(path)
    current = connection.execute(
        """
        SELECT record_id, memory_key, value, source_id
        FROM temporal_current ORDER BY record_id
        """
    ).fetchall()
    fts_hits = []
    for needle in needles:
        quoted = '"' + needle.replace('"', '""') + '"'
        fts_hits.extend(
            connection.execute(
                """
                SELECT record_id, content, bm25(full_text)
                FROM full_text WHERE full_text MATCH ?
                ORDER BY bm25(full_text)
                """,
                (quoted,),
            ).fetchall()
        )
    vectors = [
        (row[0], json.loads(row[1]))
        for row in connection.execute(
            """
            SELECT record_id, vector_json FROM dense_vectors
            ORDER BY record_id
            """
        ).fetchall()
    ]
    dense_neighbors = []
    for needle in needles:
        query_vector = deterministic_vector(needle)
        ranked = sorted(
            (
                (record_id, cosine_similarity(query_vector, vector))
                for record_id, vector in vectors
            ),
            key=lambda row: row[1],
            reverse=True,
        )
        dense_neighbors.extend(ranked[:3])
    graph = connection.execute(
        """
        SELECT edge_id, source_node, target_node, relation
        FROM graph_edges ORDER BY edge_id
        """
    ).fetchall()
    canonical = [
        json.loads(row[0])
        for row in connection.execute(
            "SELECT payload FROM canonical_events ORDER BY event_id"
        ).fetchall()
    ]
    audits = [
        json.loads(row[0])
        for row in connection.execute(
            "SELECT payload FROM immutable_audit ORDER BY audit_id"
        ).fetchall()
    ]
    manifest = connection.execute(
        """
        SELECT source_journal_sha256, active_sha256, audit_sha256, signature
        FROM signed_generation WHERE generation_id = 1
        """
    ).fetchone()
    connection.close()
    return {
        "current": current,
        "fts_hits": fts_hits,
        "dense_neighbors": dense_neighbors,
        "graph": graph,
        "canonical": canonical,
        "audits": audits,
        "manifest": manifest,
    }


def contains_any(payload: Any, needles: list[str]) -> bool:
    serialized = canonical_json(payload)
    return any(needle in serialized for needle in needles)


def run_deletion_verification(output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    events = fixture_journal()
    needles = deleted_needles()
    original_database = output_dir / "before-deletion.sqlite3"
    old_backup = output_dir / "old-backup.sqlite3"
    compacted_database = output_dir / "compacted.sqlite3"
    new_backup = output_dir / "new-backup.sqlite3"
    active_export = output_dir / "active-export.json"

    create_original_database(original_database, events)
    backup_database(original_database, old_backup)
    active, audits, operations = compact_journal(events)
    manifest = create_compacted_database(
        compacted_database,
        source_events=events,
        active_events=active,
        audit_events=audits,
    )
    active_export.write_text(
        json.dumps(active, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    backup_database(compacted_database, new_backup)
    surfaces = database_surface_payloads(compacted_database, needles)

    active_hash = sha256(
        "\n".join(canonical_json(event) for event in surfaces["canonical"])
    )
    audit_hash = sha256(
        "\n".join(canonical_json(event) for event in surfaces["audits"])
    )
    expected_signature = sha256(
        f"{manifest['source_journal_sha256']}\0{active_hash}\0{audit_hash}"
    )
    stored_manifest = surfaces["manifest"]
    signed_generation_valid = (
        active_hash == stored_manifest[1]
        and audit_hash == stored_manifest[2]
        and expected_signature == stored_manifest[3]
    )
    verification = {
        "direct_current_state_clean": not contains_any(
            surfaces["current"], needles
        ),
        "full_text_bm25_clean": not surfaces["fts_hits"],
        "dense_neighbors_clean": not contains_any(
            surfaces["dense_neighbors"], needles
        ),
        "graph_traversal_clean": not contains_any(surfaces["graph"], needles),
        "active_export_clean": not any(
            needle in active_export.read_text(encoding="utf-8")
            for needle in needles
        ),
        "vacuumed_database_pages_clean": not any(
            needle.encode("utf-8") in compacted_database.read_bytes()
            for needle in needles
        ),
        "new_backup_clean": not any(
            needle.encode("utf-8") in new_backup.read_bytes()
            for needle in needles
        ),
    }
    return {
        "protocol": "priority3-deletion-compaction-v1",
        "wall_time_seconds": time.perf_counter() - started,
        "operations": operations,
        "verification": verification,
        "all_active_surfaces_clean": all(verification.values()),
        "signed_generation_valid": signed_generation_valid,
        "old_backup_contains_deleted_payloads": all(
            needle.encode("utf-8") in old_backup.read_bytes()
            for needle in needles
        ),
        "immutable_audit_events_retained": len(surfaces["audits"]),
        "audit_events_use_hashed_targets": all(
            set(audit)
            == {
                "kind",
                "recorded_at",
                "target_sha256",
                "payload_sha256",
            }
            for audit in surfaces["audits"]
        ),
        "secure_flash_erasure_claimed": False,
        "limits": [
            "Old backups intentionally retain the pre-deletion payloads.",
            "Immutable audit events retain only hashes, operation kind and date.",
            "VACUUM proves absence from the rebuilt SQLite file, not SSD flash erasure.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_deletion_verification(args.work_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0 if payload["all_active_surfaces_clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
