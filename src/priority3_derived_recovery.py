#!/usr/bin/env python3
"""Crash-test four persisted memory views and their generation manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from incremental_memory_lifecycle import ReviewedTemporalIndex, load_jsonl


CRASH_BOUNDARIES = (
    "before_transaction_begin",
    "after_temporal_update",
    "after_full_text_update",
    "after_vector_update",
    "after_graph_update",
    "before_manifest_commit",
    "immediately_after_commit",
    "during_full_rebuild",
)
EXIT_CODES = {
    boundary: 80 + index for index, boundary in enumerate(CRASH_BOUNDARIES)
}


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def journal_digest(events: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        "\n".join(canonical_json(event) for event in events).encode("utf-8")
    ).hexdigest()


def deterministic_vector(text: str, dimensions: int = 8) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [
        round(int.from_bytes(digest[index * 4 : index * 4 + 4], "big") / 2**32, 9)
        for index in range(dimensions)
    ]


def materialize_current_rows(
    events: list[dict[str, Any]],
) -> list[dict[str, str]]:
    engine = ReviewedTemporalIndex()
    for event in events:
        engine.apply(event)
    query_date = max(event["timestamp"] for event in events)
    fact_keys = sorted(
        {
            event["record"]["key"]
            for event in events
            if event["event"] == "upsert"
        }
    )
    shard_keys = sorted(
        {
            event["key"]
            for event in events
            if event["event"] == "observe_shard"
        }
    )
    rows = []
    for kind, keys in (("fact", fact_keys), ("shard", shard_keys)):
        for key in keys:
            result = engine.query(kind, key, query_date)
            for value in result["values"]:
                for source_id in result["source_ids"]:
                    record_id = hashlib.sha256(
                        f"{kind}\0{key}\0{value}\0{source_id}".encode("utf-8")
                    ).hexdigest()[:24]
                    rows.append(
                        {
                            "record_id": record_id,
                            "kind": kind,
                            "memory_key": key,
                            "value": value,
                            "source_id": source_id,
                            "content": f"{kind} {key} {value} source {source_id}",
                        }
                    )
    return sorted(rows, key=lambda row: row["record_id"])


class DerivedIndexStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS temporal_current (
                generation_id INTEGER NOT NULL,
                record_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                memory_key TEXT NOT NULL,
                value TEXT NOT NULL,
                source_id TEXT NOT NULL,
                PRIMARY KEY(generation_id, record_id)
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS full_text USING fts5(
                generation_id UNINDEXED,
                record_id UNINDEXED,
                content
            );
            CREATE TABLE IF NOT EXISTS dense_vectors (
                generation_id INTEGER NOT NULL,
                record_id TEXT NOT NULL,
                vector_json TEXT NOT NULL,
                PRIMARY KEY(generation_id, record_id)
            );
            CREATE TABLE IF NOT EXISTS graph_edges (
                generation_id INTEGER NOT NULL,
                edge_id TEXT NOT NULL,
                source_node TEXT NOT NULL,
                target_node TEXT NOT NULL,
                relation TEXT NOT NULL,
                source_id TEXT NOT NULL,
                PRIMARY KEY(generation_id, edge_id)
            );
            CREATE TABLE IF NOT EXISTS generation_manifest (
                generation_id INTEGER PRIMARY KEY,
                journal_sequence INTEGER NOT NULL,
                journal_sha256 TEXT NOT NULL,
                temporal_count INTEGER NOT NULL,
                full_text_count INTEGER NOT NULL,
                vector_count INTEGER NOT NULL,
                graph_count INTEGER NOT NULL,
                semantic_signature TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def next_generation(self) -> int:
        row = self.connection.execute(
            "SELECT COALESCE(MAX(generation_id), 0) + 1 FROM generation_manifest"
        ).fetchone()
        return int(row[0])

    def _insert_temporal(
        self,
        generation: int,
        rows: list[dict[str, str]],
    ) -> None:
        self.connection.executemany(
            """
            INSERT INTO temporal_current(
                generation_id, record_id, kind, memory_key, value, source_id
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    generation,
                    row["record_id"],
                    row["kind"],
                    row["memory_key"],
                    row["value"],
                    row["source_id"],
                )
                for row in rows
            ],
        )

    def _insert_full_text(
        self,
        generation: int,
        rows: list[dict[str, str]],
    ) -> None:
        self.connection.executemany(
            """
            INSERT INTO full_text(generation_id, record_id, content)
            VALUES (?, ?, ?)
            """,
            [
                (generation, row["record_id"], row["content"])
                for row in rows
            ],
        )

    def _insert_vectors(
        self,
        generation: int,
        rows: list[dict[str, str]],
    ) -> None:
        self.connection.executemany(
            """
            INSERT INTO dense_vectors(
                generation_id, record_id, vector_json
            ) VALUES (?, ?, ?)
            """,
            [
                (
                    generation,
                    row["record_id"],
                    canonical_json(deterministic_vector(row["content"])),
                )
                for row in rows
            ],
        )

    def _insert_graph(
        self,
        generation: int,
        rows: list[dict[str, str]],
    ) -> None:
        edges = []
        for row in rows:
            key_node = f"key:{row['memory_key']}"
            value_node = f"value:{row['value']}"
            source_node = f"source:{row['source_id']}"
            edges.extend(
                [
                    (
                        generation,
                        f"{row['record_id']}:value",
                        key_node,
                        value_node,
                        "HAS_VALUE",
                        row["source_id"],
                    ),
                    (
                        generation,
                        f"{row['record_id']}:source",
                        value_node,
                        source_node,
                        "SUPPORTED_BY",
                        row["source_id"],
                    ),
                ]
            )
        self.connection.executemany(
            """
            INSERT INTO graph_edges(
                generation_id, edge_id, source_node, target_node,
                relation, source_id
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            edges,
        )

    def view_payload(self, generation: int) -> dict[str, Any]:
        temporal = self.connection.execute(
            """
            SELECT record_id, kind, memory_key, value, source_id
            FROM temporal_current WHERE generation_id = ?
            ORDER BY record_id
            """,
            (generation,),
        ).fetchall()
        full_text = self.connection.execute(
            """
            SELECT record_id, content FROM full_text
            WHERE generation_id = ? ORDER BY record_id
            """,
            (generation,),
        ).fetchall()
        vectors = self.connection.execute(
            """
            SELECT record_id, vector_json FROM dense_vectors
            WHERE generation_id = ? ORDER BY record_id
            """,
            (generation,),
        ).fetchall()
        graph = self.connection.execute(
            """
            SELECT edge_id, source_node, target_node, relation, source_id
            FROM graph_edges WHERE generation_id = ? ORDER BY edge_id
            """,
            (generation,),
        ).fetchall()
        return {
            "temporal": temporal,
            "full_text": full_text,
            "vectors": vectors,
            "graph": graph,
        }

    def semantic_signature(self, generation: int) -> str:
        return hashlib.sha256(
            canonical_json(self.view_payload(generation)).encode("utf-8")
        ).hexdigest()

    def source_ids(self, generation: int) -> list[str]:
        return [
            str(row[0])
            for row in self.connection.execute(
                """
                SELECT DISTINCT source_id FROM temporal_current
                WHERE generation_id = ? ORDER BY source_id
                """,
                (generation,),
            ).fetchall()
        ]

    def rebuild(
        self,
        events: list[dict[str, Any]],
        *,
        crash_boundary: str | None = None,
    ) -> int:
        if (
            crash_boundary is not None
            and crash_boundary not in CRASH_BOUNDARIES
        ):
            raise ValueError(f"unknown crash boundary: {crash_boundary}")
        if crash_boundary == "before_transaction_begin":
            os._exit(EXIT_CODES[crash_boundary])

        generation = self.next_generation()
        rows = materialize_current_rows(events)
        self.connection.execute("BEGIN IMMEDIATE")
        if crash_boundary == "during_full_rebuild":
            self.connection.execute("DELETE FROM temporal_current")
            self.connection.execute("DELETE FROM full_text")
            self.connection.execute("DELETE FROM dense_vectors")
            self.connection.execute("DELETE FROM graph_edges")
            self._insert_temporal(generation, rows[: max(1, len(rows) // 2)])
            os._exit(EXIT_CODES[crash_boundary])

        self._insert_temporal(generation, rows)
        if crash_boundary == "after_temporal_update":
            os._exit(EXIT_CODES[crash_boundary])
        self._insert_full_text(generation, rows)
        if crash_boundary == "after_full_text_update":
            os._exit(EXIT_CODES[crash_boundary])
        self._insert_vectors(generation, rows)
        if crash_boundary == "after_vector_update":
            os._exit(EXIT_CODES[crash_boundary])
        self._insert_graph(generation, rows)
        if crash_boundary == "after_graph_update":
            os._exit(EXIT_CODES[crash_boundary])
        if crash_boundary == "before_manifest_commit":
            os._exit(EXIT_CODES[crash_boundary])

        signature = self.semantic_signature(generation)
        self.connection.execute(
            """
            INSERT INTO generation_manifest(
                generation_id, journal_sequence, journal_sha256,
                temporal_count, full_text_count, vector_count, graph_count,
                semantic_signature
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                generation,
                max(event["sequence"] for event in events),
                journal_digest(events),
                len(rows),
                len(rows),
                len(rows),
                len(rows) * 2,
                signature,
            ),
        )
        self.connection.commit()
        if crash_boundary == "immediately_after_commit":
            os._exit(EXIT_CODES[crash_boundary])
        return generation

    def validate_generation(
        self,
        generation: int,
        events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        manifest = self.connection.execute(
            """
            SELECT journal_sequence, journal_sha256, temporal_count,
                   full_text_count, vector_count, graph_count,
                   semantic_signature
            FROM generation_manifest WHERE generation_id = ?
            """,
            (generation,),
        ).fetchone()
        if manifest is None:
            return {
                "generation_id": generation,
                "valid": False,
                "reason": "manifest_missing",
            }
        payload = self.view_payload(generation)
        counts = {
            "temporal_count": len(payload["temporal"]),
            "full_text_count": len(payload["full_text"]),
            "vector_count": len(payload["vectors"]),
            "graph_count": len(payload["graph"]),
        }
        manifest_counts = {
            "temporal_count": int(manifest[2]),
            "full_text_count": int(manifest[3]),
            "vector_count": int(manifest[4]),
            "graph_count": int(manifest[5]),
        }
        signature = self.semantic_signature(generation)
        manifest_matches_journal = (
            int(manifest[0]) == max(event["sequence"] for event in events)
            and str(manifest[1]) == journal_digest(events)
        )
        valid = (
            manifest_matches_journal
            and counts == manifest_counts
            and signature == str(manifest[6])
        )
        return {
            "generation_id": generation,
            "valid": valid,
            "manifest_matches_journal": manifest_matches_journal,
            "manifest_counts_match_views": counts == manifest_counts,
            "signature_matches_manifest": signature == str(manifest[6]),
            "semantic_signature": signature,
            **counts,
        }

    def recover(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        generations = [
            int(row[0])
            for row in self.connection.execute(
                """
                SELECT generation_id FROM generation_manifest
                ORDER BY generation_id DESC
                """
            ).fetchall()
        ]
        for generation in generations:
            validation = self.validate_generation(generation, events)
            if validation["valid"]:
                return {
                    "action": "accepted_committed",
                    "generation_id": generation,
                    "validation": validation,
                }
        generation = self.rebuild(events)
        return {
            "action": "rebuilt",
            "generation_id": generation,
            "validation": self.validate_generation(generation, events),
        }


def child_environment() -> dict[str, str]:
    return {
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(Path(__file__).resolve().parent),
    }


def run_crash_child(
    database: Path,
    events_path: Path,
    boundary: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child-database",
            str(database),
            "--events",
            str(events_path),
            "--crash-boundary",
            boundary,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
        env=child_environment(),
    )


def run_crash_matrix(events: list[dict[str, Any]]) -> dict[str, Any]:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(
        prefix="priority3-derived-recovery-"
    ) as temporary:
        root = Path(temporary)
        events_path = root / "events.jsonl"
        events_path.write_text(
            "".join(canonical_json(event) + "\n" for event in events),
            encoding="utf-8",
        )
        baseline_store = DerivedIndexStore(root / "baseline.sqlite3")
        baseline_generation = baseline_store.rebuild(events)
        baseline_signature = baseline_store.semantic_signature(
            baseline_generation
        )
        baseline_source_ids = baseline_store.source_ids(baseline_generation)
        baseline_store.close()

        scenarios = []
        prefix = events[: max(1, len(events) // 2)]
        for boundary in CRASH_BOUNDARIES:
            database = root / f"{boundary}.sqlite3"
            store = DerivedIndexStore(database)
            store.rebuild(prefix)
            store.close()
            child = run_crash_child(database, events_path, boundary)
            recovered_store = DerivedIndexStore(database)
            recovery = recovered_store.recover(events)
            generation = recovery["generation_id"]
            validation = recovery["validation"]
            signature = recovered_store.semantic_signature(generation)
            source_ids = recovered_store.source_ids(generation)
            orphan_generations = [
                int(row[0])
                for row in recovered_store.connection.execute(
                    """
                    SELECT DISTINCT generation_id FROM (
                        SELECT generation_id FROM temporal_current
                        UNION SELECT generation_id FROM full_text
                        UNION SELECT generation_id FROM dense_vectors
                        UNION SELECT generation_id FROM graph_edges
                    )
                    WHERE generation_id NOT IN (
                        SELECT generation_id FROM generation_manifest
                    )
                    """
                ).fetchall()
            ]
            recovered_store.close()
            scenarios.append(
                {
                    "boundary": boundary,
                    "child_exit_code": child.returncode,
                    "expected_exit_code": EXIT_CODES[boundary],
                    "crash_observed": child.returncode
                    == EXIT_CODES[boundary],
                    "recovery_action": recovery["action"],
                    "recovered_generation": generation,
                    "recovered_generation_valid": validation["valid"],
                    "semantic_signature_matches_uninterrupted": (
                        signature == baseline_signature
                    ),
                    "source_ids_match_uninterrupted": (
                        source_ids == baseline_source_ids
                    ),
                    "orphan_generations": orphan_generations,
                }
            )
        return {
            "protocol": "priority3-derived-recovery-v1",
            "wall_time_seconds": time.perf_counter() - started,
            "boundaries": list(CRASH_BOUNDARIES),
            "baseline_semantic_signature": baseline_signature,
            "baseline_source_ids": baseline_source_ids,
            "scenarios": scenarios,
            "all_scenarios_pass": all(
                scenario["crash_observed"]
                and scenario["recovered_generation_valid"]
                and scenario[
                    "semantic_signature_matches_uninterrupted"
                ]
                and scenario["source_ids_match_uninterrupted"]
                and not scenario["orphan_generations"]
                for scenario in scenarios
            ),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child-database", type=Path)
    parser.add_argument(
        "--crash-boundary",
        choices=CRASH_BOUNDARIES,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    events = load_jsonl(args.events)
    if args.child_database:
        if not args.crash_boundary:
            raise ValueError("--crash-boundary is required in child mode")
        store = DerivedIndexStore(args.child_database)
        store.rebuild(events, crash_boundary=args.crash_boundary)
        store.close()
        return 0
    if not args.output:
        raise ValueError("--output is required")
    payload = run_crash_matrix(events)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0 if payload["all_scenarios_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
