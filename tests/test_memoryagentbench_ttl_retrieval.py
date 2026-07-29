from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memoryagentbench_ttl_retrieval import (
    parse_labeled_examples,
    weighted_label_vote,
)


class MemoryAgentBenchTtlRetrievalTests(unittest.TestCase):
    def test_parse_labeled_examples(self) -> None:
        examples = parse_labeled_examples(
            "first request\nlabel: 2\n\nsecond request\nlabel: 7\n"
        )
        self.assertEqual(
            [(item["text"], item["label"]) for item in examples],
            [("first request", "2"), ("second request", "7")],
        )

    def test_weighted_vote_uses_scores_not_raw_count(self) -> None:
        ranked = [
            {"label": "a", "_score": 3.0},
            {"label": "b", "_score": 1.0},
            {"label": "b", "_score": 1.0},
        ]
        self.assertEqual(weighted_label_vote(ranked), "a")

    def test_weighted_vote_tie_prefers_best_rank(self) -> None:
        ranked = [
            {"label": "b", "_score": 1.0},
            {"label": "a", "_score": 1.0},
        ]
        self.assertEqual(weighted_label_vote(ranked), "b")


if __name__ == "__main__":
    unittest.main()
