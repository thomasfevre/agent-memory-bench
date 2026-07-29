#!/usr/bin/env python3
"""Evaluate a predicted LoCoMo category router with nested group validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark import Bm25
from graph_benchmark_common import write_result
from locomo_hybrid_fusion import (
    choose_training_config,
    compare_paired_by_conversation,
    config_id,
    metrics_for_rows,
)


DEFAULT_K_VALUES = [1, 5, 20, 100, 200]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--fusion-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=DEFAULT_K_VALUES,
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=23)
    return parser.parse_args()


def question_records(dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for sample in dataset:
        for index, question in enumerate(sample["qa"]):
            if not question.get("evidence"):
                continue
            records.append(
                {
                    "id": f"{sample['sample_id']}:{index}",
                    "sample_id": sample["sample_id"],
                    "text": question["question"],
                    "category": str(question["category"]),
                }
            )
    return records


def predict_from_ranking(
    ranking: list[dict[str, Any]],
    k: int,
    prior: Counter[str],
) -> str:
    votes: defaultdict[str, float] = defaultdict(float)
    for rank, item in enumerate(ranking[:k], start=1):
        votes[item["category"]] += item["_score"] / rank
    if not votes:
        return prior.most_common(1)[0][0]
    return max(
        votes,
        key=lambda category: (
            votes[category],
            prior[category],
            category,
        ),
    )


def choose_k_nested(
    questions: list[dict[str, Any]],
    outer_holdout: str,
    k_values: list[int],
) -> tuple[int, dict[str, Any]]:
    training_samples = sorted(
        {
            question["sample_id"]
            for question in questions
            if question["sample_id"] != outer_holdout
        }
    )
    correct = Counter({k: 0 for k in k_values})
    total = 0
    fold_scores = []
    maximum_k = max(k_values)
    for inner_holdout in training_samples:
        train = [
            question
            for question in questions
            if question["sample_id"] not in {outer_holdout, inner_holdout}
        ]
        test = [
            question
            for question in questions
            if question["sample_id"] == inner_holdout
        ]
        index = Bm25(train)
        prior = Counter(question["category"] for question in train)
        fold_correct = Counter({k: 0 for k in k_values})
        for question in test:
            ranking = index.search(question["text"], maximum_k)
            for k in k_values:
                predicted = predict_from_ranking(ranking, k, prior)
                if predicted == question["category"]:
                    correct[k] += 1
                    fold_correct[k] += 1
            total += 1
        fold_scores.append(
            {
                "inner_holdout": inner_holdout,
                "questions": len(test),
                "accuracy_by_k": {
                    str(k): fold_correct[k] / len(test)
                    for k in k_values
                },
            }
        )
    chosen = min(
        k_values,
        key=lambda k: (-correct[k] / total, k),
    )
    return (
        chosen,
        {
            "outer_holdout": outer_holdout,
            "training_questions_scored": total,
            "accuracy_by_k": {
                str(k): correct[k] / total for k in k_values
            },
            "selected_k": chosen,
            "inner_folds": fold_scores,
        },
    )


def category_configurations(
    fusion_rows: list[dict[str, Any]],
    holdout: str,
) -> dict[str, tuple[str, float]]:
    categories = sorted({row["question_type"] for row in fusion_rows})
    return {
        category: choose_training_config(
            [
                row
                for row in fusion_rows
                if row["question_type"] == category
            ],
            holdout,
        )[:2]
        for category in categories
    }


def classification_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    categories = sorted({row["gold_category"] for row in rows})
    confusion = {
        gold: {
            predicted: sum(
                row["gold_category"] == gold
                and row["predicted_category"] == predicted
                for row in rows
            )
            for predicted in categories
        }
        for gold in categories
    }
    by_category = []
    for category in categories:
        selected = [
            row for row in rows if row["gold_category"] == category
        ]
        by_category.append(
            {
                "category": category,
                "questions": len(selected),
                "accuracy": statistics.fmean(
                    row["category_correct"] for row in selected
                ),
            }
        )
    return {
        "questions": len(rows),
        "accuracy": statistics.fmean(
            row["category_correct"] for row in rows
        ),
        "by_category": by_category,
        "confusion_matrix_gold_rows": confusion,
    }


def make_payload(
    args: argparse.Namespace,
    dataset_sha256: str,
    fusion_sha256: str,
    rows: list[dict[str, Any]],
    k_selection: list[dict[str, Any]],
    fusion: dict[str, Any],
) -> dict[str, Any]:
    sample_ids = list(fusion["manifest"]["completed_samples"])
    routing_rows = [
        {
            **row,
            "sample_id": row["sample_id"],
            "evidence_recall": row["evidence_recall"],
            "context_precision": row["context_precision"],
            "context_words": row["context_words"],
            "context_turns": row["context_turns"],
            "latency_ms": row["total_query_latency_ms"],
        }
        for row in rows
    ]
    baselines = {}
    for name, (representation, alpha) in {
        "equal_rrf_window2": ("window2", 0.5),
        "bm25_window4": ("window4", 1.0),
    }.items():
        baseline_rows = [
            row
            for row in fusion["rows"]
            if row["representation"] == representation
            and row["alpha"] == alpha
        ]
        baselines[name] = {
            "config": config_id(representation, alpha),
            "metrics": metrics_for_rows(baseline_rows),
            "predicted_router_minus_baseline": (
                compare_paired_by_conversation(
                    routing_rows,
                    baseline_rows,
                    sample_ids,
                    args.bootstrap_resamples,
                    args.bootstrap_seed,
                )
            ),
        }
    return {
        "manifest": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": "LoCoMo",
            "dataset_source_file": str(args.dataset),
            "dataset_source_sha256": dataset_sha256,
            "fusion_results_file": str(args.fusion_results),
            "fusion_results_sha256": fusion_sha256,
            "classifier": "BM25 k-nearest questions with score/rank vote",
            "k_values": args.k_values,
            "outer_grouping": "leave one complete conversation out",
            "inner_grouping": (
                "leave one of the remaining conversations out to select k"
            ),
            "complete": len(rows)
            == sum(
                1
                for row in fusion["rows"]
                if row["representation"] == "window2"
                and row["alpha"] == 0.5
            ),
            "metric_scope": (
                "Predicted category routing over the already computed weighted "
                "RRF configurations. Gold category labels are used only in "
                "outer-training conversations to select category-specific "
                "retrieval configs and in inner validation to select k. "
                "Classifier latency is added to retrieval latency. Index build "
                "time and answer generation are excluded."
            ),
        },
        "classification": classification_summary(rows),
        "selected_k_counts": dict(
            sorted(
                Counter(
                    str(item["selected_k"]) for item in k_selection
                ).items()
            )
        ),
        "routing_config_counts": dict(
            sorted(Counter(row["selected_config"] for row in rows).items())
        ),
        "predicted_router_metrics": metrics_for_rows(routing_rows),
        "mean_classifier_latency_ms": statistics.fmean(
            row["classifier_latency_ms"] for row in rows
        ),
        "mean_retrieval_latency_ms": statistics.fmean(
            row["retrieval_latency_ms"] for row in rows
        ),
        "baselines": baselines,
        "oracle_category_router_reference": {
            "metrics": fusion["oracle_category_router"][
                "out_of_fold_selected_metrics"
            ],
            "note": (
                "Copied from the fusion result for context; unlike this run, "
                "the oracle sees the gold category of held-out questions."
            ),
        },
        "k_selection": k_selection,
        "rows": rows,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.k_values or any(k <= 0 for k in args.k_values):
        raise ValueError("k values must be positive")
    dataset = json.loads(args.dataset.read_text())
    fusion = json.loads(args.fusion_results.read_text())
    questions = question_records(dataset)
    sample_ids = sorted({question["sample_id"] for question in questions})
    if sample_ids != sorted(fusion["manifest"]["completed_samples"]):
        raise ValueError("Dataset groups do not match fusion result groups")
    row_map = {
        (
            row["question_id"],
            row["representation"],
            row["alpha"],
        ): row
        for row in fusion["rows"]
    }
    rows = []
    k_selection = []
    maximum_k = max(args.k_values)
    for holdout in sample_ids:
        selected_k, nested = choose_k_nested(
            questions,
            holdout,
            args.k_values,
        )
        k_selection.append(nested)
        training = [
            question
            for question in questions
            if question["sample_id"] != holdout
        ]
        test = [
            question
            for question in questions
            if question["sample_id"] == holdout
        ]
        started = time.perf_counter()
        classifier = Bm25(training)
        classifier_build_ms = (time.perf_counter() - started) * 1000
        prior = Counter(question["category"] for question in training)
        configurations = category_configurations(
            fusion["rows"],
            holdout,
        )
        for question in test:
            started = time.perf_counter()
            ranking = classifier.search(question["text"], maximum_k)
            classifier_latency_ms = (
                time.perf_counter() - started
            ) * 1000
            predicted = predict_from_ranking(
                ranking,
                selected_k,
                prior,
            )
            representation, alpha = configurations[predicted]
            retrieval = row_map[
                (question["id"], representation, alpha)
            ]
            rows.append(
                {
                    "sample_id": holdout,
                    "question_id": question["id"],
                    "gold_category": question["category"],
                    "predicted_category": predicted,
                    "category_correct": (
                        predicted == question["category"]
                    ),
                    "selected_k": selected_k,
                    "selected_config": config_id(
                        representation,
                        alpha,
                    ),
                    "evidence_recall": retrieval["evidence_recall"],
                    "context_precision": retrieval["context_precision"],
                    "context_words": retrieval["context_words"],
                    "context_turns": retrieval["context_turns"],
                    "classifier_build_ms_for_fold": classifier_build_ms,
                    "classifier_latency_ms": classifier_latency_ms,
                    "retrieval_latency_ms": retrieval["latency_ms"],
                    "total_query_latency_ms": (
                        classifier_latency_ms + retrieval["latency_ms"]
                    ),
                }
            )
        print(f"outer folds {len(k_selection)}/{len(sample_ids)}", flush=True)
    return make_payload(
        args,
        hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        hashlib.sha256(args.fusion_results.read_bytes()).hexdigest(),
        rows,
        k_selection,
        fusion,
    )


if __name__ == "__main__":
    arguments = parse_args()
    result = run(arguments)
    write_result(arguments.output, result)
    print(arguments.output)
