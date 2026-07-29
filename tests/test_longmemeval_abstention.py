from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from longmemeval_abstention import (
    apply_threshold,
    choose_threshold,
    classification_metrics,
    fold_for,
)


class LongMemEvalAbstentionTests(unittest.TestCase):
    def test_classification_metrics_penalize_never_abstain(self) -> None:
        metrics = classification_metrics(
            [True, False, False],
            [False, False, False],
        )
        self.assertEqual(metrics["accuracy"], 2 / 3)
        self.assertEqual(metrics["balanced_accuracy"], 0.5)
        self.assertEqual(metrics["recall"], 0.0)

    def test_threshold_can_learn_low_score_abstention(self) -> None:
        rows = [
            {"should_abstain": True, "score": 0.1},
            {"should_abstain": True, "score": 0.2},
            {"should_abstain": False, "score": 0.8},
            {"should_abstain": False, "score": 0.9},
        ]
        threshold, direction = choose_threshold(rows, "score")
        self.assertEqual(direction, "low")
        self.assertTrue(apply_threshold(0.2, threshold, direction))
        self.assertFalse(apply_threshold(0.8, threshold, direction))

    def test_fold_assignment_is_stable(self) -> None:
        self.assertEqual(fold_for("question-1", 5), fold_for("question-1", 5))


if __name__ == "__main__":
    unittest.main()
