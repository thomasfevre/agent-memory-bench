from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from merge_memoryagentbench_slices import merge_slices


class MergeMemoryAgentBenchSlicesTests(unittest.TestCase):
    def test_merge_deduplicates_question_and_strategy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for name, indices in (("a", (0, 1)), ("b", (1, 2))):
                rows = []
                for index in indices:
                    rows.append(
                        {
                            "question_index": index,
                            "strategy": "bm25",
                            "exact_match": index == 2,
                            "substring_exact_match": index == 2,
                            "token_f1": float(index == 2),
                            "latency_ms": 1.0,
                            "prompt_tokens": 1,
                            "output_tokens": 1,
                        }
                    )
                path = Path(directory) / f"{name}.json"
                path.write_text(
                    json.dumps(
                        {
                            "manifest": {"dataset": "test"},
                            "rows": rows,
                            "summaries": [],
                        }
                    )
                )
                paths.append(path)

            merged = merge_slices(paths)

        self.assertEqual(merged["manifest"]["questions"], 3)
        self.assertEqual(len(merged["rows"]), 3)
        self.assertAlmostEqual(
            merged["summaries"][0]["substring_exact_match"], 1 / 3
        )


if __name__ == "__main__":
    unittest.main()
