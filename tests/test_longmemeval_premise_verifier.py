from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from longmemeval_premise_verifier import (
    bootstrap_balanced_accuracy_difference,
    fit_logistic,
    pair_group_id,
    predict_logistic,
    stable_group_folds,
)


class LongMemEvalPremiseVerifierTests(unittest.TestCase):
    def test_pair_group_id_keeps_near_miss_with_base(self) -> None:
        self.assertEqual(pair_group_id("abc_abs"), "abc")
        self.assertEqual(pair_group_id("abc_abs_2"), "abc")
        self.assertEqual(pair_group_id("abc"), "abc")

    def test_grouped_folds_never_split_pair(self) -> None:
        rows = [
            {
                "question_id": "a",
                "group_id": "a",
                "should_abstain": False,
            },
            {
                "question_id": "a_abs",
                "group_id": "a",
                "should_abstain": True,
            },
            {
                "question_id": "b",
                "group_id": "b",
                "should_abstain": False,
            },
        ]
        assignments = stable_group_folds(rows, 2, "test")
        self.assertEqual(assignments["a"], assignments["a"])
        self.assertIn(assignments["b"], {0, 1})

    def test_logistic_separates_simple_training_data(self) -> None:
        matrix = np.asarray(
            [[-2.0], [-1.0], [1.0], [2.0]],
            dtype=np.float64,
        )
        labels = np.asarray([0, 0, 1, 1], dtype=np.float64)
        model = fit_logistic(
            matrix,
            labels,
            l2=0.01,
            iterations=1_500,
            learning_rate=0.1,
        )
        probabilities = predict_logistic(model, matrix)
        self.assertTrue(all(probabilities[:2] < 0.5))
        self.assertTrue(all(probabilities[2:] > 0.5))

    def test_bootstrap_comparison_preserves_pair_groups(self) -> None:
        rows = [
            {
                "question_id": "a",
                "group_id": "a",
                "should_abstain": False,
            },
            {
                "question_id": "a_abs",
                "group_id": "a",
                "should_abstain": True,
            },
            {
                "question_id": "b",
                "group_id": "b",
                "should_abstain": False,
            },
            {
                "question_id": "c_abs",
                "group_id": "c",
                "should_abstain": True,
            },
        ]
        challenger = [
            {
                **row,
                "predicted_abstention": row["should_abstain"],
            }
            for row in rows
        ]
        baseline = [
            {**row, "predicted_abstention": False} for row in rows
        ]
        result = bootstrap_balanced_accuracy_difference(
            rows,
            challenger,
            baseline,
            resamples=100,
            seed=2,
        )
        self.assertEqual(result["groups"], 3)
        self.assertEqual(result["observed_difference"], 0.5)


if __name__ == "__main__":
    unittest.main()
