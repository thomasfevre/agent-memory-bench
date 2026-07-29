from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark import Bm25
from memoryagentbench_summarization_coverage import (
    keypoint_support,
    select_oracle_rrf,
    select_uniform,
)


class MemoryAgentBenchSummarizationCoverageTests(unittest.TestCase):
    def test_keypoint_support_ignores_common_words(self) -> None:
        self.assertEqual(
            keypoint_support(
                "Alice marries Bob in Paris.",
                "Later Bob and Alice marries in Paris.",
            ),
            1.0,
        )

    def test_uniform_selection_keeps_ends_and_budget(self) -> None:
        chunks = [{"id": str(index), "text": str(index)} for index in range(100)]
        selected = select_uniform(chunks, 5)
        self.assertEqual(len(selected), 5)
        self.assertEqual(selected[0]["id"], "0")
        self.assertEqual(selected[-1]["id"], "99")

    def test_uniform_selection_supports_zero_and_one_budget(self) -> None:
        chunks = [{"id": str(index), "text": str(index)} for index in range(3)]
        self.assertEqual(select_uniform(chunks, 0), [])
        self.assertEqual(select_uniform(chunks, 1), [chunks[0]])

    def test_oracle_rrf_returns_unique_chunks(self) -> None:
        chunks = [
            {"id": "a", "text": "Alice marries Bob."},
            {"id": "b", "text": "Carol travels to Rome."},
        ]
        selected = select_oracle_rrf(
            Bm25(chunks),
            ["Alice and Bob marry.", "Carol visits Rome."],
            2,
        )
        self.assertEqual({item["id"] for item in selected}, {"a", "b"})


if __name__ == "__main__":
    unittest.main()
