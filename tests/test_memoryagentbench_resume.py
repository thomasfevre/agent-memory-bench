from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memoryagentbench_slice import build_payload, validate_resume_manifest


def args() -> argparse.Namespace:
    return argparse.Namespace(
        parquet=Path("/tmp/data.parquet"),
        model="qwen2.5:14b",
        source="sample",
        offset=20,
        questions=10,
        top_k=20,
        seed=11,
        strategies=["long_context", "bm25"],
    )


class MemoryAgentBenchResumeTests(unittest.TestCase):
    def test_partial_payload_is_marked_incomplete(self) -> None:
        payload = build_payload(
            args=args(),
            facts=[{"id": "f1", "text": "fact"}],
            selected_questions=[("q1", ["a1"]), ("q2", ["a2"])],
            rows=[
                {
                    "question_index": 20,
                    "strategy": "bm25",
                    "exact_match": False,
                    "substring_exact_match": False,
                    "token_f1": 0.0,
                    "latency_ms": 1.0,
                    "prompt_tokens": 1,
                    "output_tokens": 1,
                }
            ],
        )

        self.assertEqual(payload["manifest"]["expected_rows"], 4)
        self.assertEqual(payload["manifest"]["completed_rows"], 1)
        self.assertFalse(payload["manifest"]["complete"])

    def test_resume_rejects_changed_protocol(self) -> None:
        manifest = {
            "source": "sample",
            "model": "other-model",
            "questions": 10,
            "question_offset": 20,
            "top_k": 20,
            "seed": 11,
            "strategies": ["long_context", "bm25"],
        }

        with self.assertRaisesRegex(ValueError, "model"):
            validate_resume_manifest(
                manifest,
                args(),
                [("question", ["answer"])] * 10,
            )

    def test_single_strategy_changes_expected_rows(self) -> None:
        arguments = args()
        arguments.strategies = ["bm25"]
        payload = build_payload(
            args=arguments,
            facts=[{"id": "f1", "text": "fact"}],
            selected_questions=[("q1", ["a1"]), ("q2", ["a2"])],
            rows=[],
        )

        self.assertEqual(payload["manifest"]["expected_rows"], 2)


if __name__ == "__main__":
    unittest.main()
