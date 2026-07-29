from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from locomo_predicted_router import (
    classification_summary,
    predict_from_ranking,
    question_records,
)


class LoCoMoPredictedRouterTests(unittest.TestCase):
    def test_question_records_skip_unscored_questions(self) -> None:
        records = question_records(
            [
                {
                    "sample_id": "conv",
                    "qa": [
                        {
                            "question": "kept",
                            "category": 2,
                            "evidence": ["d1"],
                        },
                        {
                            "question": "skipped",
                            "category": 5,
                            "evidence": [],
                        },
                    ],
                }
            ]
        )
        self.assertEqual(
            records,
            [
                {
                    "id": "conv:0",
                    "sample_id": "conv",
                    "text": "kept",
                    "category": "2",
                }
            ],
        )

    def test_prediction_uses_score_rank_vote(self) -> None:
        ranking = [
            {"category": "1", "_score": 2.0},
            {"category": "2", "_score": 5.0},
            {"category": "2", "_score": 1.0},
        ]
        self.assertEqual(
            predict_from_ranking(
                ranking,
                3,
                Counter({"1": 10, "2": 10}),
            ),
            "2",
        )

    def test_empty_ranking_falls_back_to_training_prior(self) -> None:
        self.assertEqual(
            predict_from_ranking(
                [],
                5,
                Counter({"4": 8, "1": 2}),
            ),
            "4",
        )

    def test_classification_summary_builds_gold_row_confusion(self) -> None:
        summary = classification_summary(
            [
                {
                    "gold_category": "1",
                    "predicted_category": "1",
                    "category_correct": True,
                },
                {
                    "gold_category": "1",
                    "predicted_category": "2",
                    "category_correct": False,
                },
                {
                    "gold_category": "2",
                    "predicted_category": "2",
                    "category_correct": True,
                },
            ]
        )
        self.assertAlmostEqual(summary["accuracy"], 2 / 3)
        self.assertEqual(
            summary["confusion_matrix_gold_rows"]["1"],
            {"1": 1, "2": 1},
        )


if __name__ == "__main__":
    unittest.main()
