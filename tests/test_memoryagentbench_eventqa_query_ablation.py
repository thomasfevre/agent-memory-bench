from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark import Bm25
from memoryagentbench_eventqa_query_ablation import (
    anchor_neighborhood,
    chunk_context,
    parse_options,
    predict_option,
)


class MemoryAgentBenchEventQaTests(unittest.TestCase):
    def test_parse_options(self) -> None:
        question = """
Below is a list of possible subsequent events:
['first event', 'second event']

Your task is to choose one.
"""
        self.assertEqual(parse_options(question), ["first event", "second event"])

    def test_predict_option_uses_retrieved_evidence(self) -> None:
        self.assertEqual(
            predict_option(
                ["Alice went home.", "Bob stayed outside."],
                "Later, Alice went home before dark.",
            ),
            "Alice went home.",
        )

    def test_anchor_neighborhood_preserves_sequence(self) -> None:
        chunks = chunk_context(
            "zero one two three four five six seven",
            window_words=2,
            stride_words=2,
        )
        index = Bm25(chunks)
        selected = anchor_neighborhood(
            index,
            chunks,
            "two",
            anchors=1,
            following_chunks=2,
            stride_words=2,
        )
        self.assertEqual(
            [item["position"] for item in selected],
            [2, 4],
        )


if __name__ == "__main__":
    unittest.main()
