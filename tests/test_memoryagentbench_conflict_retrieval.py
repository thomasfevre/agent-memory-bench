from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memoryagentbench_conflict_retrieval import (
    contains_answer,
    first_answer_rank,
    greedy_expand,
    summarize_rows,
)
from benchmark import Bm25


class MemoryAgentBenchConflictRetrievalTests(unittest.TestCase):
    def test_contains_answer_uses_benchmark_normalization(self) -> None:
        self.assertTrue(contains_answer("Citizenship: the Belgium.", ["Belgium"]))
        self.assertFalse(contains_answer("Citizenship: France.", ["Belgium"]))

    def test_first_answer_rank_returns_first_matching_fact(self) -> None:
        ranked = [
            {"text": "No answer here."},
            {"text": "The target is Belgium."},
            {"text": "Belgium appears again."},
        ]
        self.assertEqual(first_answer_rank(ranked, ["Belgium"]), 2)
        self.assertIsNone(first_answer_rank(ranked, ["Taipei"]))

    def test_summary_counts_missing_evidence_as_zero_reciprocal_rank(self) -> None:
        rows = [
            {
                "first_answer_rank": 2,
                "greedy_first_answer_rank": 1,
                "answer_present_in_full_context": True,
                "retrieval_latency_ms": 1.0,
                "greedy_latency_ms": 4.0,
                "top20_words": 20,
            },
            {
                "first_answer_rank": None,
                "greedy_first_answer_rank": None,
                "answer_present_in_full_context": True,
                "retrieval_latency_ms": 3.0,
                "greedy_latency_ms": 6.0,
                "top20_words": 10,
            },
        ]
        summary = summarize_rows(
            rows,
            cutoffs=[1, 5],
            iterative_cutoffs=[1, 5],
        )
        self.assertEqual(summary["bm25_hit_at_k"], {"1": 0.0, "5": 0.5})
        self.assertEqual(summary["greedy_hit_at_k"], {"1": 0.5, "5": 0.5})
        self.assertEqual(summary["bm25_mean_reciprocal_rank"], 0.25)
        self.assertEqual(summary["bm25_no_answer_evidence_count"], 1)

    def test_greedy_expansion_follows_entity_introduced_by_first_fact(self) -> None:
        index = Bm25(
            [
                {"id": "f1", "text": "The author of Book Z is Alice Example."},
                {"id": "f2", "text": "Alice Example is married to Bob Example."},
                {"id": "f3", "text": "Bob Example is a citizen of Belgium."},
            ]
        )
        selected = greedy_expand(
            index,
            "What is the citizenship of the spouse of the author of Book Z?",
            budget=3,
        )

        self.assertEqual([item["id"] for item in selected], ["f1", "f2", "f3"])


if __name__ == "__main__":
    unittest.main()
