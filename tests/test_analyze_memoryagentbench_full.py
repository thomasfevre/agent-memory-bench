from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyze_memoryagentbench_full import (
    analyze,
    exact_mcnemar_p_value,
    wilson_interval,
)


def row(index: int, strategy: str, correct: bool) -> dict:
    return {
        "question_index": index,
        "strategy": strategy,
        "substring_exact_match": correct,
        "latency_ms": 10 if strategy == "bm25" else 100,
        "prompt_tokens": 10 if strategy == "bm25" else 100,
        "output_tokens": 1,
    }


class AnalyzeMemoryAgentBenchFullTests(unittest.TestCase):
    def test_paired_outcomes_and_ratios(self) -> None:
        payload = {
            "rows": [
                row(0, "bm25", True),
                row(0, "long_context", True),
                row(1, "bm25", True),
                row(1, "long_context", False),
                row(2, "bm25", False),
                row(2, "long_context", True),
                row(3, "bm25", False),
                row(3, "long_context", False),
            ]
        }

        result = analyze(payload)

        self.assertEqual(
            result["paired_outcomes"],
            {
                "both_correct": 1,
                "bm25_only": 1,
                "long_context_only": 1,
                "neither_correct": 1,
            },
        )
        self.assertEqual(
            result["efficiency_ratios_long_context_over_bm25"][
                "mean_latency"
            ],
            10,
        )

    def test_mcnemar_exact_extremes(self) -> None:
        self.assertEqual(exact_mcnemar_p_value(0, 0), 1.0)
        self.assertAlmostEqual(exact_mcnemar_p_value(8, 0), 0.0078125)

    def test_wilson_interval_contains_observed_rate(self) -> None:
        low, high = wilson_interval(10, 100)
        self.assertLess(low, 0.1)
        self.assertGreater(high, 0.1)


if __name__ == "__main__":
    unittest.main()
