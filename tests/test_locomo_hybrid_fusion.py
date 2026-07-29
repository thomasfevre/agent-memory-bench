from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from locomo_hybrid_fusion import (
    choose_training_config,
    cross_validate,
    cross_validate_oracle_category_router,
    pareto_frontier,
    weighted_rrf,
)


def ranked(ids: list[str]) -> list[dict[str, object]]:
    return [{"id": item_id, "text": item_id} for item_id in ids]


def row(
    sample: str,
    representation: str,
    alpha: float,
    recall: float,
    latency: float = 1.0,
    question: str = "q",
) -> dict[str, object]:
    return {
        "sample_id": sample,
        "question_id": f"{sample}:{question}",
        "question_type": "1",
        "representation": representation,
        "alpha": alpha,
        "evidence_recall": recall,
        "context_precision": recall,
        "context_words": 100,
        "context_turns": 4,
        "latency_ms": latency,
    }


class LoCoMoHybridFusionTests(unittest.TestCase):
    def test_weight_extremes_preserve_source_rankings(self) -> None:
        bm25 = ranked(["a", "b", "c"])
        dense = ranked(["c", "b", "a"])
        self.assertEqual(
            [item["id"] for item in weighted_rrf(bm25, dense, 1.0, 3)],
            ["a", "b", "c"],
        )
        self.assertEqual(
            [item["id"] for item in weighted_rrf(bm25, dense, 0.0, 3)],
            ["c", "b", "a"],
        )

    def test_equal_weight_rewards_agreement(self) -> None:
        fused = weighted_rrf(
            ranked(["a", "b", "c"]),
            ranked(["c", "b", "d"]),
            0.5,
            4,
        )
        self.assertEqual(fused[0]["id"], "c")
        self.assertEqual(fused[1]["id"], "b")

    def test_holdout_is_not_used_to_choose_configuration(self) -> None:
        rows = [
            row("train-a", "window2", 0.5, 1.0),
            row("train-a", "window4", 1.0, 0.0),
            row("train-b", "window2", 0.5, 1.0),
            row("train-b", "window4", 1.0, 0.0),
            row("holdout", "window2", 0.5, 0.0),
            row("holdout", "window4", 1.0, 1.0),
        ]
        representation, alpha, score = choose_training_config(
            rows,
            "holdout",
        )
        self.assertEqual((representation, alpha), ("window2", 0.5))
        self.assertEqual(score, 1.0)

    def test_cross_validation_selects_once_per_held_out_group(self) -> None:
        rows = []
        for sample in ("a", "b", "c"):
            rows.extend(
                [
                    row(sample, "window2", 0.5, 0.8),
                    row(sample, "window4", 1.0, 0.7),
                ]
            )
        result = cross_validate(rows, ["a", "b", "c"], 100, 3)
        self.assertEqual(len(result["folds"]), 3)
        self.assertEqual(
            result["selection_counts"],
            {"window2:alpha=0.50": 3},
        )
        self.assertEqual(
            result["out_of_fold_selected_metrics"]["questions"],
            3,
        )

    def test_oracle_router_selects_separately_by_category(self) -> None:
        rows = []
        for sample in ("a", "b", "c"):
            rows.extend(
                [
                    row(
                        sample,
                        "window2",
                        0.5,
                        0.9,
                        question="cat-1-a",
                    ),
                    row(
                        sample,
                        "window4",
                        1.0,
                        0.1,
                        question="cat-1-b",
                    ),
                    {
                        **row(
                            sample,
                            "window2",
                            0.5,
                            0.2,
                            question="cat-2-a",
                        ),
                        "question_type": "2",
                    },
                    {
                        **row(
                            sample,
                            "window4",
                            1.0,
                            0.8,
                            question="cat-2-b",
                        ),
                        "question_type": "2",
                    },
                ]
            )
        result = cross_validate_oracle_category_router(
            rows,
            ["a", "b", "c"],
            100,
            3,
        )
        self.assertEqual(
            result["selection_counts_by_category"]["1"],
            {"window2:alpha=0.50": 3},
        )
        self.assertEqual(
            result["selection_counts_by_category"]["2"],
            {"window4:alpha=1.00": 3},
        )

    def test_pareto_frontier_removes_slower_worse_config(self) -> None:
        summaries = [
            {
                "config": "fast",
                "conversation_weighted_evidence_recall": 0.7,
                "mean_query_latency_ms": 1.0,
            },
            {
                "config": "slow-worse",
                "conversation_weighted_evidence_recall": 0.6,
                "mean_query_latency_ms": 2.0,
            },
            {
                "config": "slow-better",
                "conversation_weighted_evidence_recall": 0.8,
                "mean_query_latency_ms": 2.0,
            },
        ]
        self.assertEqual(
            [item["config"] for item in pareto_frontier(summaries)],
            ["fast", "slow-better"],
        )


if __name__ == "__main__":
    unittest.main()
