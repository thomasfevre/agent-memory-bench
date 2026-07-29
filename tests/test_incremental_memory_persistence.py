from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from incremental_memory_lifecycle import load_jsonl
from incremental_memory_persistence import (
    SqliteEventStore,
    canonical_event,
    events_digest,
    expected_terminal_chain_hash,
    run,
)


class IncrementalMemoryPersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.events = load_jsonl(
            cls.root / "data" / "incremental-memory-events.jsonl"
        )
        cls.questions = load_jsonl(
            cls.root / "data" / "incremental-memory-questions.jsonl"
        )

    def test_sqlite_store_roundtrips_and_deduplicates_events(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = SqliteEventStore(Path(temporary) / "events.sqlite3")
            self.assertTrue(store.append(self.events[0]))
            self.assertEqual("wal", store.journal_mode)
            self.assertEqual(2, store.synchronous)
            self.assertFalse(store.append(self.events[0]))
            loaded = store.load()
            self.assertEqual([self.events[0]], loaded)
            self.assertEqual(events_digest([self.events[0]]), events_digest(loaded))
            self.assertEqual("ok", store.integrity_check())
            collision = {**self.events[0], "id": "other"}
            with self.assertRaisesRegex(ValueError, "identity collision"):
                store.append(collision)
            same_id_other_sequence = {**self.events[0], "sequence": 2}
            with self.assertRaisesRegex(ValueError, "identity collision"):
                store.append(same_id_other_sequence)
            self.assertEqual(
                expected_terminal_chain_hash([self.events[0]]),
                store.terminal_chain_hash(),
            )
            with self.assertRaisesRegex(
                Exception, "events are append-only"
            ):
                store.connection.execute(
                    "UPDATE events SET event_id = 'changed'"
                )
            store.connection.rollback()
            with self.assertRaisesRegex(
                Exception, "events are append-only"
            ):
                store.connection.execute("DELETE FROM events")
            store.connection.rollback()
            store.close()

    def test_hash_chain_detects_payload_and_checksum_rewrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = SqliteEventStore(Path(temporary) / "events.sqlite3")
            store.append(self.events[0])
            changed = {
                **self.events[0],
                "record": {**self.events[0]["record"], "value": "changed"},
            }
            payload = canonical_event(changed)
            store.connection.execute("DROP TRIGGER events_no_update")
            store.connection.execute(
                """
                UPDATE events SET payload = ?, payload_sha256 = ?
                WHERE sequence = 1
                """,
                (payload, hashlib.sha256(payload.encode()).hexdigest()),
            )
            store.connection.commit()
            with self.assertRaisesRegex(ValueError, "hash-chain mismatch"):
                store.load()
            store.close()

    def test_child_process_crash_rolls_back_and_replay_matches(self):
        payload = run(self.events, self.questions, [0, 5, 10, 15, 19, 20])
        self.assertTrue(payload["all_scenarios_pass"])
        for scenario in payload["scenarios"]:
            self.assertEqual(73, scenario["child_exit_code"])
            self.assertTrue(scenario["rollback_preserved_prefix"])
            self.assertTrue(
                scenario[
                    "independent_process_semantic_signature_matches_uninterrupted"
                ]
            )


if __name__ == "__main__":
    unittest.main()
