from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from context_shard_policy_benchmark import (
    evaluate_static_policy,
    run_lifecycle,
)


class ContextShardPolicyBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.shards = [
            {"id": "good", "occurrences": 2, "review": "approved"},
            {"id": "bad", "occurrences": 2, "review": "rejected"},
            {"id": "rare", "occurrences": 1, "review": "deferred"},
        ]

    def test_auto_promotion_activates_rejected_pattern(self) -> None:
        result = evaluate_static_policy(
            self.shards,
            "auto_promote_repeated",
            2,
        )
        self.assertEqual(result["strict_decision_accuracy"], 0.5)
        self.assertEqual(result["active_precision"], 0.5)
        self.assertEqual(result["rejected_activation_rate"], 1.0)

    def test_review_registry_preserves_negative_decision(self) -> None:
        result = evaluate_static_policy(
            self.shards,
            "review_registry",
            2,
        )
        self.assertEqual(result["decision_coverage"], 1.0)
        self.assertEqual(result["strict_decision_accuracy"], 1.0)
        self.assertEqual(result["rejected_activation_rate"], 0.0)

    def test_rejected_recurrence_requeues_without_activation(self) -> None:
        events = [
            {
                "timestamp": "2026-01-01",
                "shard_id": "bad",
                "event": "observe",
                "synthetic": False,
            },
            {
                "timestamp": "2026-01-02",
                "shard_id": "bad",
                "event": "observe",
                "synthetic": False,
            },
            {
                "timestamp": "2026-01-03",
                "shard_id": "bad",
                "event": "review",
                "decision": "rejected",
                "synthetic": True,
            },
            {
                "timestamp": "2026-02-05",
                "shard_id": "bad",
                "event": "observe",
                "synthetic": True,
            },
        ]
        result = run_lifecycle(events, 2, 30)
        self.assertEqual(result["requeues_after_cooldown"], 1)
        self.assertEqual(result["unsafe_activation_events"], 0)
        self.assertEqual(
            result["final_states"]["bad"]["status"],
            "pending_review",
        )
        self.assertFalse(result["final_states"]["bad"]["active"])


if __name__ == "__main__":
    unittest.main()
