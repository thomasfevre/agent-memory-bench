#!/usr/bin/env python3
"""Exercise crash-safe local persistence for the incremental memory event log."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph_benchmark_common import write_result
from incremental_memory_lifecycle import (
    evaluate,
    load_jsonl,
    sha256_file,
    validate_workload,
)


PROTOCOL_VERSION = "incremental-memory-persistence-v1"
CHAIN_GENESIS = "0" * 64


def canonical_event(event: dict[str, Any]) -> str:
    return json.dumps(event, sort_keys=True, separators=(",", ":"))


def events_digest(events: list[dict[str, Any]]) -> str:
    encoded = "\n".join(canonical_event(event) for event in events).encode()
    return hashlib.sha256(encoded).hexdigest()


def chain_hash(
    sequence: int,
    event_id: str,
    payload_sha256: str,
    previous_chain_hash: str,
) -> str:
    encoded = "\0".join(
        [
            str(sequence),
            event_id,
            payload_sha256,
            previous_chain_hash,
        ]
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def expected_terminal_chain_hash(events: list[dict[str, Any]]) -> str:
    previous = CHAIN_GENESIS
    for event in events:
        payload_digest = hashlib.sha256(
            canonical_event(event).encode()
        ).hexdigest()
        previous = chain_hash(
            event["sequence"],
            event["id"],
            payload_digest,
            previous,
        )
    return previous


class SqliteEventStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.connection = sqlite3.connect(path)
        self.journal_mode = str(
            self.connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        ).lower()
        self.connection.execute("PRAGMA synchronous=FULL")
        self.synchronous = int(
            self.connection.execute("PRAGMA synchronous").fetchone()[0]
        )
        if self.journal_mode != "wal" or self.synchronous != 2:
            raise RuntimeError(
                "SQLite durability pragmas not applied: "
                f"journal_mode={self.journal_mode} "
                f"synchronous={self.synchronous}"
            )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                sequence INTEGER PRIMARY KEY,
                event_id TEXT NOT NULL UNIQUE,
                payload TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL,
                previous_chain_hash TEXT NOT NULL,
                chain_hash TEXT NOT NULL
            )
            """
        )
        self.connection.execute(
            """
            CREATE TRIGGER IF NOT EXISTS events_no_update
            BEFORE UPDATE ON events
            BEGIN
                SELECT RAISE(ABORT, 'events are append-only');
            END
            """
        )
        self.connection.execute(
            """
            CREATE TRIGGER IF NOT EXISTS events_no_delete
            BEFORE DELETE ON events
            BEGIN
                SELECT RAISE(ABORT, 'events are append-only');
            END
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def append(self, event: dict[str, Any]) -> bool:
        payload = canonical_event(event)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        existing = self.connection.execute(
            """
            SELECT sequence, event_id, payload
            FROM events WHERE sequence = ? OR event_id = ?
            """,
            (event["sequence"], event["id"]),
        ).fetchone()
        if existing is not None:
            if existing == (event["sequence"], event["id"], payload):
                return False
            raise ValueError(
                "event identity collision: "
                f"sequence={event['sequence']} id={event['id']}"
            )
        last = self.connection.execute(
            """
            SELECT sequence, chain_hash FROM events
            ORDER BY sequence DESC LIMIT 1
            """
        ).fetchone()
        expected_sequence = 1 if last is None else int(last[0]) + 1
        if event["sequence"] != expected_sequence:
            raise ValueError(
                "non-contiguous event sequence: "
                f"expected={expected_sequence} actual={event['sequence']}"
            )
        previous_chain_hash = CHAIN_GENESIS if last is None else str(last[1])
        event_chain_hash = chain_hash(
            event["sequence"],
            event["id"],
            digest,
            previous_chain_hash,
        )
        with self.connection:
            cursor = self.connection.execute(
                """
                INSERT INTO events(
                    sequence, event_id, payload, payload_sha256,
                    previous_chain_hash, chain_hash
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event["sequence"],
                    event["id"],
                    payload,
                    digest,
                    previous_chain_hash,
                    event_chain_hash,
                ),
            )
        return cursor.rowcount == 1

    def load(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT sequence, event_id, payload, payload_sha256,
                   previous_chain_hash, chain_hash
            FROM events ORDER BY sequence
            """
        ).fetchall()
        events = []
        expected_previous = CHAIN_GENESIS
        for (
            sequence,
            event_id,
            payload,
            expected_digest,
            stored_previous,
            stored_chain_hash,
        ) in rows:
            actual_digest = hashlib.sha256(payload.encode()).hexdigest()
            if actual_digest != expected_digest:
                raise ValueError(f"event payload checksum mismatch: {event_id}")
            actual_chain_hash = chain_hash(
                sequence,
                event_id,
                expected_digest,
                stored_previous,
            )
            if (
                stored_previous != expected_previous
                or stored_chain_hash != actual_chain_hash
            ):
                raise ValueError(f"event hash-chain mismatch: {event_id}")
            event = json.loads(payload)
            if event["sequence"] != sequence or event["id"] != event_id:
                raise ValueError(f"event identity mismatch: {event_id}")
            events.append(event)
            expected_previous = stored_chain_hash
        return events

    def terminal_chain_hash(self) -> str:
        row = self.connection.execute(
            "SELECT chain_hash FROM events ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        return CHAIN_GENESIS if row is None else str(row[0])

    def integrity_check(self) -> str:
        return str(
            self.connection.execute("PRAGMA integrity_check").fetchone()[0]
        )


def insert_event_in_open_transaction(
    connection: sqlite3.Connection,
    event: dict[str, Any],
) -> None:
    last = connection.execute(
        "SELECT sequence, chain_hash FROM events ORDER BY sequence DESC LIMIT 1"
    ).fetchone()
    previous_chain_hash = CHAIN_GENESIS if last is None else str(last[1])
    payload = canonical_event(event)
    payload_digest = hashlib.sha256(payload.encode()).hexdigest()
    connection.execute(
        """
        INSERT INTO events(
            sequence, event_id, payload, payload_sha256,
            previous_chain_hash, chain_hash
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            event["sequence"],
            event["id"],
            payload,
            payload_digest,
            previous_chain_hash,
            chain_hash(
                event["sequence"],
                event["id"],
                payload_digest,
                previous_chain_hash,
            ),
        ),
    )


def crash_at_transaction_boundary(
    path: Path,
    event: dict[str, Any],
    *,
    commit: bool,
) -> None:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("BEGIN IMMEDIATE")
    insert_event_in_open_transaction(connection, event)
    if commit:
        connection.commit()
        os._exit(74)
    os._exit(73)


def semantic_signature(payload: dict[str, Any]) -> str:
    row_fields = (
        "question_id",
        "after_event",
        "query_date",
        "kind",
        "memory_key",
        "scope",
        "dimension",
        "should_abstain",
        "expected_values",
        "expected_source_ids",
        "architecture",
        "retrieved_values",
        "retrieved_source_ids",
        "retrieved_record_ids",
        "events_scanned",
        "records_inspected",
        "index_lookups",
        "exact_value_set",
        "source_id_set_exact",
        "contradiction_leakage",
        "false_refusal",
        "unsafe_answer",
    )
    summary_fields = (
        "architecture",
        "questions",
        "exact_value_accuracy",
        "source_id_set_accuracy_answerable",
        "source_id_questions",
        "correct_abstention_rate",
        "contradiction_leakage_rate",
        "false_refusal_rate",
        "unsafe_answer_rate",
        "mean_events_scanned",
        "mean_records_inspected",
        "mean_index_lookups",
        "final_state_components",
        "json_roundtrip_reconstruction_digest_match",
        "dimension_accuracy",
        "dimension_counts",
    )
    rows = [{key: row[key] for key in row_fields} for row in payload["rows"]]
    summaries = [
        {key: row[key] for key in summary_fields}
        for row in payload["summaries"]
    ]
    encoded = json.dumps(
        {
            "rows": rows,
            "summaries": summaries,
            "final_state_digests": payload["final_state_digests"],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def child_environment() -> dict[str, str]:
    return {
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(Path(__file__).resolve().parent),
    }


def initialize_store(
    database: Path,
    prefix: list[dict[str, Any]],
) -> tuple[str, str, int]:
    store = SqliteEventStore(database)
    for event in prefix:
        if not store.append(event):
            raise ValueError(f"unexpected duplicate event: {event['id']}")
    digest = events_digest(store.load())
    journal_mode = store.journal_mode
    synchronous = store.synchronous
    store.close()
    return digest, journal_mode, synchronous


def run_crash_child(
    database: Path,
    event: dict[str, Any],
    *,
    commit: bool,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--crash-db",
            str(database),
            "--crash-mode",
            "postcommit" if commit else "precommit",
        ],
        input=canonical_event(event),
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
        env=child_environment(),
    )


def run_crash_scenario(
    events: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    crash_after: int,
    baseline_signature: str,
) -> dict[str, Any]:
    if not 0 <= crash_after < len(events):
        raise ValueError("crash point must leave at least one event uncommitted")
    with tempfile.TemporaryDirectory(prefix="memory-persistence-") as temporary:
        temporary_path = Path(temporary)
        rollback_database = temporary_path / "rollback.sqlite3"
        prefix_digest, journal_mode, synchronous = initialize_store(
            rollback_database,
            events[:crash_after],
        )
        crashed = run_crash_child(
            rollback_database,
            events[crash_after],
            commit=False,
        )

        recovered = SqliteEventStore(rollback_database)
        after_crash = recovered.load()
        rollback_preserved_prefix = (
            crashed.returncode == 73
            and len(after_crash) == crash_after
            and events_digest(after_crash) == prefix_digest
        )
        for event in events[crash_after:]:
            recovered.append(event)
        final_events = recovered.load()
        digest_before_duplicate = events_digest(final_events)
        duplicate_inserted = recovered.append(events[-1])
        final_after_duplicate = recovered.load()
        integrity = recovered.integrity_check()
        final_chain_hash = recovered.terminal_chain_hash()
        recovered.close()

        postcommit_database = temporary_path / "postcommit.sqlite3"
        initialize_store(postcommit_database, events[:crash_after])
        postcommit_crash = run_crash_child(
            postcommit_database,
            events[crash_after],
            commit=True,
        )
        postcommit_store = SqliteEventStore(postcommit_database)
        postcommit_prefix = postcommit_store.load()
        postcommit_retained = (
            postcommit_crash.returncode == 74
            and postcommit_prefix == events[: crash_after + 1]
        )
        for event in events[crash_after + 1 :]:
            postcommit_store.append(event)
        postcommit_final = postcommit_store.load()
        postcommit_integrity = postcommit_store.integrity_check()
        postcommit_chain_hash = postcommit_store.terminal_chain_hash()
        postcommit_store.close()

        questions_path = temporary_path / "questions.jsonl"
        questions_path.write_text(
            "\n".join(canonical_event(question) for question in questions)
            + "\n"
        )
        independent = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--signature-db",
                str(rollback_database),
                "--questions",
                str(questions_path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
            env=child_environment(),
        )
        independent_signature = independent.stdout.strip()
        return {
            "crash_after_committed_events": crash_after,
            "child_exit_code": crashed.returncode,
            "rollback_preserved_prefix": rollback_preserved_prefix,
            "postcommit_child_exit_code": postcommit_crash.returncode,
            "postcommit_event_retained": postcommit_retained,
            "final_event_count": len(final_after_duplicate),
            "final_event_structural_digest": events_digest(final_after_duplicate),
            "final_structurally_matches_source_events": (
                final_after_duplicate == events
            ),
            "duplicate_inserted": duplicate_inserted,
            "duplicate_preserved_digest": (
                events_digest(final_after_duplicate)
                == digest_before_duplicate
            ),
            "sqlite_integrity_check": integrity,
            "postcommit_final_structurally_matches_source_events": (
                postcommit_final == events
            ),
            "postcommit_sqlite_integrity_check": postcommit_integrity,
            "journal_mode_observed": journal_mode,
            "synchronous_observed": synchronous,
            "terminal_chain_hash": final_chain_hash,
            "postcommit_terminal_chain_hash": postcommit_chain_hash,
            "terminal_chain_matches_source_anchor": (
                final_chain_hash == expected_terminal_chain_hash(events)
                and postcommit_chain_hash == expected_terminal_chain_hash(events)
            ),
            "independent_process_exit_code": independent.returncode,
            "independent_process_semantic_signature_matches_uninterrupted": (
                independent.returncode == 0
                and independent_signature == baseline_signature
            ),
        }


def run(
    events: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    crash_points: list[int],
) -> dict[str, Any]:
    validate_workload(events, questions)
    baseline = evaluate(events, questions)
    baseline_signature = semantic_signature(baseline)
    scenarios = [
        run_crash_scenario(
            events,
            questions,
            crash_after,
            baseline_signature,
        )
        for crash_after in crash_points
    ]
    return {
        "baseline_semantic_signature": baseline_signature,
        "source_event_structural_digest": events_digest(events),
        "source_terminal_chain_hash": expected_terminal_chain_hash(events),
        "scenarios": scenarios,
        "all_scenarios_pass": all(
            scenario["rollback_preserved_prefix"]
            and scenario["postcommit_event_retained"]
            and scenario["final_structurally_matches_source_events"]
            and scenario[
                "postcommit_final_structurally_matches_source_events"
            ]
            and not scenario["duplicate_inserted"]
            and scenario["duplicate_preserved_digest"]
            and scenario["sqlite_integrity_check"] == "ok"
            and scenario["postcommit_sqlite_integrity_check"] == "ok"
            and scenario["journal_mode_observed"] == "wal"
            and scenario["synchronous_observed"] == 2
            and scenario["terminal_chain_matches_source_anchor"]
            and scenario[
                "independent_process_semantic_signature_matches_uninterrupted"
            ]
            for scenario in scenarios
        ),
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
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--crash-points",
        nargs="+",
        type=int,
        default=[0, 5, 10, 15, 19, 20],
    )
    parser.add_argument("--crash-db", type=Path)
    parser.add_argument(
        "--crash-mode",
        choices=("precommit", "postcommit"),
    )
    parser.add_argument("--signature-db", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.crash_db is not None:
        if args.crash_mode is None:
            raise ValueError("--crash-mode is required with --crash-db")
        event = json.loads(sys.stdin.read())
        crash_at_transaction_boundary(
            args.crash_db,
            event,
            commit=args.crash_mode == "postcommit",
        )
    if args.signature_db is not None:
        questions = load_jsonl(args.questions)
        store = SqliteEventStore(args.signature_db)
        events = store.load()
        store.close()
        print(semantic_signature(evaluate(events, questions)))
        return
    if args.output is None:
        raise ValueError("--output is required")
    events = load_jsonl(args.events)
    questions = load_jsonl(args.questions)
    payload = run(events, questions, args.crash_points)
    payload["manifest"] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_version": PROTOCOL_VERSION,
        "scope": (
            "Local SQLite append-only event-log precommit rollback, acknowledged "
            "postcommit retention, idempotent replay, hash-chain verification, "
            "and deterministic derived-state reconstruction"
        ),
        "events": str(args.events.resolve()),
        "events_sha256": sha256_file(args.events),
        "questions": str(args.questions.resolve()),
        "questions_sha256": sha256_file(args.questions),
        "crash_points": args.crash_points,
        "sqlite_journal_mode": "WAL",
        "sqlite_synchronous": "FULL",
        "limitations": [
            "Only the append-only event log is persisted; derived indexes are rebuilt.",
            "Recovery is structurally exact after canonical JSON serialization; it is not a byte-for-byte reconstruction of the source JSONL.",
            "Crashes are injected immediately before or after one transaction commit; multi-event transactions, WAL checkpoints, and arbitrary filesystem or kernel failures are not tested.",
            "The terminal hash-chain anchor is stored in the external result artifact; it does not provide signatures or remote attestation.",
            "Temporary local databases are used; no multi-host replication is tested.",
            "Physical deletion and secure erasure are not tested.",
        ],
    }
    write_result(args.output, payload)
    print(args.output)


if __name__ == "__main__":
    main()
