from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jcode_common_benchmark import (
    aggregate_repetitions,
    build_memory_entries,
    parse_search_output,
)


class JcodeCommonBenchmarkTests(unittest.TestCase):
    def test_build_memory_entries_preserves_source_identity_and_time(self) -> None:
        entries = build_memory_entries(
            [
                {
                    "id": "d02",
                    "timestamp": "2026-07-15",
                    "text": "Atlas uses ember.",
                    "kind": "runbook",
                    "confidence": 1.0,
                },
                {
                    "id": "d20",
                    "timestamp": "2026-06-30",
                    "text": "A rumor.",
                    "kind": "rumor",
                    "confidence": 0.2,
                },
            ]
        )

        self.assertEqual(entries[0]["id"], "d02")
        self.assertEqual(entries[0]["source"], "benchmark:d02")
        self.assertEqual(entries[0]["created_at"], "2026-07-15T00:00:00Z")
        self.assertEqual(entries[0]["trust"], "high")
        self.assertEqual(entries[1]["trust"], "low")
        self.assertEqual(entries[1]["confidence"], 0.2)

    def test_parse_search_output_keeps_rank_order_and_deduplicates(self) -> None:
        output = """
Found 3 memories:

- [fact] First
  id: d02 (score: 78%)

- [fact] Second
  id: d01 (score: 70%)

- [fact] Duplicate
  id: d02 (score: 60%)
"""
        self.assertEqual(parse_search_output(output, top_k=5), ["d02", "d01"])
        self.assertEqual(parse_search_output(output, top_k=1), ["d02"])

    def test_aggregate_repetitions_reports_mean_and_sample_stddev(self) -> None:
        repeats = [
            {
                "strategies": {
                    "prod_hybrid": {
                        "metrics": {"mean_recall": 0.8, "mean_latency_ms": 10.0}
                    }
                }
            },
            {
                "strategies": {
                    "prod_hybrid": {
                        "metrics": {"mean_recall": 1.0, "mean_latency_ms": 14.0}
                    }
                }
            },
        ]

        summary = aggregate_repetitions(repeats)

        self.assertAlmostEqual(summary["prod_hybrid"]["mean_recall"]["mean"], 0.9)
        self.assertAlmostEqual(
            summary["prod_hybrid"]["mean_latency_ms"]["sample_stddev"],
            2.8284271247461903,
        )


if __name__ == "__main__":
    unittest.main()
