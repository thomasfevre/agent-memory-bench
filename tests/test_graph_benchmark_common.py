from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graph_benchmark_common import score_retrieval, source_ids_from_text


class GraphBenchmarkCommonTests(unittest.TestCase):
    def test_source_ids_are_unique_and_ordered(self) -> None:
        self.assertEqual(
            source_ids_from_text("[SOURCE d02] then d01, and d02 again"),
            ["d02", "d01"],
        )

    def test_temporal_recall_does_not_hide_leakage(self) -> None:
        questions = [
            {
                "id": "q1",
                "category": "temporal",
                "gold_source_ids": ["d02"],
            }
        ]
        rows = [
            {
                "question_id": "q1",
                "retrieved_source_ids": ["d01", "d02"],
                "latency_ms": 10,
            }
        ]

        metrics = score_retrieval(questions, rows)

        self.assertEqual(metrics["temporal_correctness"], 1.0)
        self.assertEqual(metrics["temporal_context_precision"], 0.5)
        self.assertEqual(metrics["temporal_exact_source_set"], 0.0)

    def test_abstention_requires_no_source(self) -> None:
        questions = [
            {
                "id": "q1",
                "category": "abstention",
                "gold_source_ids": [],
            }
        ]
        rows = [
            {
                "question_id": "q1",
                "retrieved_source_ids": [],
                "latency_ms": 10,
            }
        ]

        self.assertEqual(score_retrieval(questions, rows)["abstention_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
