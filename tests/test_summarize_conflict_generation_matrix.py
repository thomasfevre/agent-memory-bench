from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from summarize_conflict_generation_matrix import (
    evidence_available,
    summarize_group,
)


class SummarizeConflictGenerationMatrixTests(unittest.TestCase):
    def test_evidence_availability_depends_on_strategy(self) -> None:
        retrieval = {
            "first_answer_rank": 7,
            "answer_present_in_full_context": True,
        }
        self.assertFalse(evidence_available("bm25", retrieval, 5))
        self.assertTrue(evidence_available("bm25", retrieval, 10))
        self.assertTrue(evidence_available("long_context", retrieval, 5))

    def test_summary_separates_retrieval_and_reader_failure(self) -> None:
        rows = [
            {
                "question_index": 0,
                "substring_exact_match": True,
                "exact_match": True,
                "token_f1": 1.0,
                "latency_ms": 10,
                "prompt_tokens": 100,
                "output_tokens": 2,
            },
            {
                "question_index": 1,
                "substring_exact_match": False,
                "exact_match": False,
                "token_f1": 0.0,
                "latency_ms": 20,
                "prompt_tokens": 100,
                "output_tokens": 2,
            },
            {
                "question_index": 2,
                "substring_exact_match": False,
                "exact_match": False,
                "token_f1": 0.0,
                "latency_ms": 30,
                "prompt_tokens": 100,
                "output_tokens": 2,
            },
        ]
        retrieval = {
            ("source", 0): {
                "first_answer_rank": 1,
                "answer_present_in_full_context": True,
            },
            ("source", 1): {
                "first_answer_rank": 2,
                "answer_present_in_full_context": True,
            },
            ("source", 2): {
                "first_answer_rank": None,
                "answer_present_in_full_context": True,
            },
        }
        result = summarize_group(
            "source",
            "bm25",
            rows,
            retrieval,
            expected_questions=3,
            top_k=5,
        )
        self.assertEqual(result["literal_evidence_rate"], 2 / 3)
        self.assertEqual(result["reader_success_given_literal_evidence"], 0.5)
        self.assertEqual(result["outcomes"]["evidence_correct"], 1)
        self.assertEqual(result["outcomes"]["evidence_incorrect"], 1)
        self.assertEqual(result["outcomes"]["no_evidence_incorrect"], 1)


if __name__ == "__main__":
    unittest.main()
