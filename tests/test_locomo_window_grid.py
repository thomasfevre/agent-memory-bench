from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from locomo_window_grid import build_grid_views, compare_top_configurations


class LoCoMoWindowGridTests(unittest.TestCase):
    def test_grid_labels_and_window_sizes_are_distinct(self) -> None:
        sample = {
            "conversation": {
                "session_1": [
                    {
                        "dia_id": f"d{index}",
                        "speaker": "A",
                        "text": str(index),
                    }
                    for index in range(8)
                ]
            }
        }
        views = build_grid_views(sample, [2, 4, 8])
        self.assertEqual(
            set(views),
            {"turn", "session", "window2", "window4", "window8"},
        )
        self.assertEqual(len(views["window2"][0]["source_ids"]), 2)
        self.assertEqual(len(views["window4"][0]["source_ids"]), 4)
        self.assertEqual(len(views["window8"][0]["source_ids"]), 8)

    def test_top_comparison_uses_conversations_as_sampling_units(self) -> None:
        summaries = [
            {
                "representation": "window2",
                "strategy": "hybrid",
                "mean_evidence_recall": 0.8,
            },
            {
                "representation": "window4",
                "strategy": "bm25",
                "mean_evidence_recall": 0.7,
            },
        ]
        rows = [
            {
                "sample_id": "a",
                "representation": "window2",
                "strategy": "hybrid",
                "evidence_recall": 1.0,
            },
            {
                "sample_id": "a",
                "representation": "window4",
                "strategy": "bm25",
                "evidence_recall": 0.0,
            },
            {
                "sample_id": "b",
                "representation": "window2",
                "strategy": "hybrid",
                "evidence_recall": 0.0,
            },
            {
                "sample_id": "b",
                "representation": "window4",
                "strategy": "bm25",
                "evidence_recall": 1.0,
            },
        ]
        result = compare_top_configurations(
            summaries,
            rows,
            ["a", "b"],
            resamples=100,
        )
        assert result is not None
        self.assertEqual(result["best_wins"], 1)
        self.assertEqual(result["runner_up_wins"], 1)
        self.assertEqual(result["conversation_weighted_mean_difference"], 0.0)


if __name__ == "__main__":
    unittest.main()
