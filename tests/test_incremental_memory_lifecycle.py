from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from incremental_memory_lifecycle import (
    LatestWriteSnapshot,
    RawLogScan,
    ReviewedTemporalIndex,
    TemporalAutoPromoteIndex,
    evaluate,
    load_jsonl,
    validate_workload,
)


EVENTS = [
    {
        "id": "e01",
        "sequence": 1,
        "timestamp": "2026-01-05",
        "event": "upsert",
        "record": {
            "id": "r-canary",
            "key": "atlas_branch",
            "value": "canary",
            "source_id": "d01",
            "valid_from": "2026-01-05",
            "valid_to": None,
            "confidence": 1.0,
            "kind": "authoritative",
            "supersedes": [],
        },
    },
    {
        "id": "e02",
        "sequence": 2,
        "timestamp": "2026-06-30",
        "event": "upsert",
        "record": {
            "id": "r-rumor",
            "key": "atlas_branch",
            "value": "sapphire",
            "source_id": "d20",
            "valid_from": "2026-06-30",
            "valid_to": None,
            "confidence": 0.2,
            "kind": "rumor",
            "supersedes": [],
        },
    },
    {
        "id": "e03",
        "sequence": 3,
        "timestamp": "2026-07-15",
        "event": "upsert",
        "record": {
            "id": "r-ember",
            "key": "atlas_branch",
            "value": "ember",
            "source_id": "d02",
            "valid_from": "2026-07-15",
            "valid_to": None,
            "confidence": 1.0,
            "kind": "authoritative",
            "supersedes": ["r-canary"],
        },
    },
]


class IncrementalMemoryLifecycleTests(unittest.TestCase):
    def test_latest_write_snapshot_loses_history_and_accepts_rumor(self):
        engine = LatestWriteSnapshot()
        engine.apply(EVENTS[0])
        engine.apply(EVENTS[1])
        self.assertEqual(
            ["sapphire"],
            engine.query("fact", "atlas_branch", "2026-07-01")["values"],
        )
        engine.apply(EVENTS[2])
        self.assertEqual(
            [],
            engine.query("fact", "atlas_branch", "2026-07-10")["values"],
        )

    def test_temporal_index_preserves_history_and_filters_low_confidence(self):
        engine = ReviewedTemporalIndex()
        for event in EVENTS:
            engine.apply(event)
        self.assertEqual(
            ["canary"],
            engine.query("fact", "atlas_branch", "2026-07-10")["values"],
        )
        self.assertEqual(
            ["ember"],
            engine.query("fact", "atlas_branch", "2026-07-28")["values"],
        )

    def test_raw_log_scan_reconstructs_temporal_state(self):
        engine = RawLogScan()
        for event in EVENTS:
            engine.apply(event)
        result = engine.query("fact", "atlas_branch", "2026-07-10")
        self.assertEqual(["canary"], result["values"])
        self.assertEqual(len(EVENTS), result["events_scanned"])

    def test_review_gate_prevents_repetition_from_becoming_approval(self):
        observations = [
            {
                "id": "s1",
                "timestamp": "2026-04-02",
                "event": "observe_shard",
                "shard_id": "utc",
                "key": "incident_timestamp_rule",
                "value": "UTC",
                "source_id": "d09",
            },
            {
                "id": "s2",
                "timestamp": "2026-04-18",
                "event": "observe_shard",
                "shard_id": "utc",
                "key": "incident_timestamp_rule",
                "value": "UTC",
                "source_id": "d10",
            },
        ]
        auto = TemporalAutoPromoteIndex()
        reviewed = ReviewedTemporalIndex()
        for event in observations:
            auto.apply(event)
            reviewed.apply(event)
        self.assertEqual(
            ["UTC"],
            auto.query(
                "shard", "incident_timestamp_rule", "2026-04-19"
            )["values"],
        )
        self.assertEqual(
            [],
            reviewed.query(
                "shard", "incident_timestamp_rule", "2026-04-19"
            )["values"],
        )

    def test_retraction_preserves_prior_state_only_in_temporal_index(self):
        upsert = {
            "id": "r1",
            "timestamp": "2026-09-01",
            "event": "upsert",
            "record": {
                "id": "waiver",
                "key": "waiver",
                "value": "allowed",
                "source_id": "source",
                "valid_from": "2026-09-01",
                "valid_to": None,
                "confidence": 1.0,
                "kind": "ephemeral",
                "supersedes": [],
            },
        }
        retract = {
            "id": "r2",
            "timestamp": "2026-09-02",
            "event": "retract",
            "target_id": "waiver",
            "effective_at": "2026-09-02",
        }
        temporal = ReviewedTemporalIndex()
        snapshot = LatestWriteSnapshot()
        for engine in (temporal, snapshot):
            engine.apply(upsert)
            engine.apply(retract)
        self.assertEqual(
            ["allowed"], temporal.query("fact", "waiver", "2026-09-01")["values"]
        )
        self.assertEqual(
            [], temporal.query("fact", "waiver", "2026-09-03")["values"]
        )
        self.assertEqual(
            [], snapshot.query("fact", "waiver", "2026-09-01")["values"]
        )

    def test_evaluate_scores_checkpoints_and_dimensions(self):
        questions = [
            {
                "id": "q1",
                "after_event": "e01",
                "query_date": "2026-01-10",
                "kind": "fact",
                "memory_key": "atlas_branch",
                "expected_values": ["canary"],
                "expected_source_ids": ["d01"],
                "should_abstain": False,
                "dimension": "current_state",
            },
            {
                "id": "q2",
                "after_event": "e02",
                "query_date": "2026-07-01",
                "kind": "fact",
                "memory_key": "atlas_branch",
                "expected_values": ["canary"],
                "expected_source_ids": ["d01"],
                "should_abstain": False,
                "dimension": "source_quality",
            },
        ]
        payload = evaluate(
            EVENTS[:2],
            questions,
            engine_factories={"latest_write_snapshot": LatestWriteSnapshot},
        )
        summary = payload["summaries"][0]
        self.assertEqual(2, summary["questions"])
        self.assertEqual(0.5, summary["exact_value_accuracy"])
        self.assertEqual(0.0, summary["dimension_accuracy"]["source_quality"])

    def test_workload_rejects_inconsistent_abstention_label(self):
        questions = [
            {
                "id": "q",
                "after_event": "e01",
                "query_date": "2026-01-10",
                "expected_values": ["canary"],
                "should_abstain": True,
            }
        ]
        with self.assertRaisesRegex(ValueError, "abstention labels"):
            validate_workload(EVENTS, questions)

    def test_shard_identity_collision_is_rejected(self):
        engine = ReviewedTemporalIndex()
        engine.apply(
            {
                "event": "observe_shard",
                "shard_id": "same",
                "key": "rule",
                "value": "UTC",
                "source_id": "one",
            }
        )
        with self.assertRaisesRegex(ValueError, "shard identity collision"):
            engine.apply(
                {
                    "event": "observe_shard",
                    "shard_id": "same",
                    "key": "rule",
                    "value": "local time",
                    "source_id": "two",
                }
            )

    def test_complete_fixture_has_preregistered_policy_outcomes(self):
        root = Path(__file__).resolve().parents[1]
        events = load_jsonl(root / "data" / "incremental-memory-events.jsonl")
        questions = load_jsonl(
            root / "data" / "incremental-memory-questions.jsonl"
        )
        payload = evaluate(events, questions)
        summaries = {
            row["architecture"]: row for row in payload["summaries"]
        }
        self.assertEqual(
            13 / 25,
            summaries["latest_write_snapshot"]["exact_value_accuracy"],
        )
        self.assertEqual(
            19 / 25,
            summaries["temporal_auto_promote"]["exact_value_accuracy"],
        )
        self.assertEqual(
            1.0,
            summaries["reviewed_temporal_index"]["exact_value_accuracy"],
        )
        self.assertEqual(
            1.0,
            summaries["raw_log_scan"]["exact_value_accuracy"],
        )
        self.assertEqual(
            16,
            summaries["raw_log_scan"]["source_id_questions"],
        )
        self.assertTrue(
            all(
                row["json_roundtrip_reconstruction_digest_match"]
                for row in summaries.values()
            )
        )


if __name__ == "__main__":
    unittest.main()
