from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mem0_common_benchmark import (
    get_all_memories,
    normalize_results,
    source_ids_from_result,
)


class Mem0CommonBenchmarkTests(unittest.TestCase):
    def test_normalize_results_accepts_current_envelope(self) -> None:
        payload = {"results": [{"id": "m1", "memory": "hello"}]}
        self.assertEqual(normalize_results(payload), payload["results"])

    def test_source_ids_prefer_metadata_and_fall_back_to_text(self) -> None:
        self.assertEqual(
            source_ids_from_result(
                {
                    "memory": "[SOURCE d02] Atlas uses ember.",
                    "metadata": {"source_id": "d01"},
                }
            ),
            ["d01", "d02"],
        )

    def test_source_ids_are_unique(self) -> None:
        self.assertEqual(
            source_ids_from_result(
                {
                    "memory": "[SOURCE d02] and d02",
                    "metadata": {"source_ids": ["d02", "d03"]},
                }
            ),
            ["d02", "d03"],
        )

    def test_get_all_memories_avoids_mem0_default_twenty_item_limit(self) -> None:
        class FakeMemory:
            def __init__(self) -> None:
                self.top_k = None

            def get_all(self, *, filters, top_k):
                self.top_k = top_k
                return {"results": [{"id": f"m{index}"} for index in range(28)]}

        memory = FakeMemory()
        results = get_all_memories(
            memory,
            user_id="common-raw-1",
            expected_documents=28,
        )

        self.assertEqual(len(results), 28)
        self.assertEqual(memory.top_k, 112)


if __name__ == "__main__":
    unittest.main()
