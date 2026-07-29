from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from longmemeval_hybrid_fusion import (
    score,
    select_top_items,
    select_word_budget,
)


class LongMemEvalHybridFusionTests(unittest.TestCase):
    def test_top_items_reports_actual_context_words(self) -> None:
        selected, words = select_top_items(
            [
                {"id": "a", "text": "one two"},
                {"id": "b", "text": "three"},
                {"id": "c", "text": "four five six"},
            ],
            2,
        )
        self.assertEqual(selected, {"a", "b"})
        self.assertEqual(words, 3)

    def test_word_budget_skips_oversized_session(self) -> None:
        selected, words = select_word_budget(
            [
                {"id": "large", "text": "one two three four five"},
                {"id": "small", "text": "one two"},
                {"id": "also-small", "text": "three four"},
            ],
            4,
        )
        self.assertEqual(selected, {"small", "also-small"})
        self.assertEqual(words, 4)

    def test_score_uses_selected_set_for_recall_and_ranking_for_mrr(
        self,
    ) -> None:
        ranking = [
            {"id": "noise"},
            {"id": "gold"},
        ]
        recall, mrr = score({"gold"}, set(), ranking)
        self.assertEqual(recall, 0.0)
        self.assertEqual(mrr, 0.5)


if __name__ == "__main__":
    unittest.main()
