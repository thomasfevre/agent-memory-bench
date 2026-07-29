#!/usr/bin/env python3
"""Calibrate retrieval-score abstention on LongMemEval-S near-miss questions."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark import Bm25, DenseIndex, MiniLm, MODEL_DIR, tokenize
from external_retrieval import session_text
from graph_benchmark_common import write_result


FEATURES = (
    "bm25_top1",
    "bm25_margin",
    "bm25_per_query_term",
    "bm25_top1_z",
    "dense_top1",
    "dense_margin",
    "dense_top1_z",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    return parser.parse_args()


def safe_z(top: float, values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    deviation = statistics.pstdev(values)
    if deviation == 0:
        return 0.0
    return (top - statistics.fmean(values)) / deviation


def extract_features(
    item: dict[str, Any],
    encoder: MiniLm,
) -> dict[str, Any]:
    candidates = [
        {"id": session_id, "text": session_text(session)}
        for session_id, session in zip(
            item["haystack_session_ids"],
            item["haystack_sessions"],
            strict=True,
        )
    ]
    bm25_rows = Bm25(candidates).search(item["question"], len(candidates))
    dense_rows = DenseIndex(candidates, encoder).search(
        item["question"],
        len(candidates),
    )
    bm25_scores = [float(row["_score"]) for row in bm25_rows]
    dense_scores = [float(row["_score"]) for row in dense_rows]
    bm25_top1 = bm25_scores[0] if bm25_scores else 0.0
    bm25_top2 = bm25_scores[1] if len(bm25_scores) > 1 else 0.0
    dense_top1 = dense_scores[0] if dense_scores else 0.0
    dense_top2 = dense_scores[1] if len(dense_scores) > 1 else 0.0
    query_terms = len(set(tokenize(item["question"]))) or 1
    return {
        "question_id": item["question_id"],
        "question_type": item["question_type"],
        "should_abstain": "_abs" in item["question_id"],
        "sessions": len(candidates),
        "bm25_top1": bm25_top1,
        "bm25_margin": bm25_top1 - bm25_top2,
        "bm25_per_query_term": bm25_top1 / query_terms,
        "bm25_top1_z": safe_z(bm25_top1, bm25_scores),
        "dense_top1": dense_top1,
        "dense_margin": dense_top1 - dense_top2,
        "dense_top1_z": safe_z(dense_top1, dense_scores),
    }


def classification_metrics(
    labels: list[bool],
    predictions: list[bool],
) -> dict[str, float | int]:
    tp = sum(label and prediction for label, prediction in zip(labels, predictions))
    tn = sum(
        not label and not prediction
        for label, prediction in zip(labels, predictions)
    )
    fp = sum(
        not label and prediction
        for label, prediction in zip(labels, predictions)
    )
    fn = sum(
        label and not prediction
        for label, prediction in zip(labels, predictions)
    )
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": (tp + tn) / len(labels),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_abstention_rate": fp / (tn + fp) if tn + fp else 0.0,
    }


def threshold_candidates(values: list[float]) -> list[float]:
    unique = sorted(set(values))
    if len(unique) == 1:
        return [unique[0]]
    epsilon = max(abs(unique[0]), abs(unique[-1]), 1.0) * 1e-9
    return [
        unique[0] - epsilon,
        *[
            (left + right) / 2
            for left, right in zip(unique, unique[1:])
        ],
        unique[-1] + epsilon,
    ]


def apply_threshold(value: float, threshold: float, direction: str) -> bool:
    return value <= threshold if direction == "low" else value >= threshold


def choose_threshold(
    rows: list[dict[str, Any]],
    feature: str,
) -> tuple[float, str]:
    labels = [bool(row["should_abstain"]) for row in rows]
    values = [float(row[feature]) for row in rows]
    best: tuple[tuple[float, float, float], float, str] | None = None
    for direction in ("low", "high"):
        for threshold in threshold_candidates(values):
            predictions = [
                apply_threshold(value, threshold, direction) for value in values
            ]
            metrics = classification_metrics(labels, predictions)
            rank = (
                float(metrics["balanced_accuracy"]),
                float(metrics["f1"]),
                -float(metrics["false_abstention_rate"]),
            )
            candidate = (rank, threshold, direction)
            if best is None or candidate > best:
                best = candidate
    assert best is not None
    return best[1], best[2]


def fold_for(question_id: str, folds: int) -> int:
    digest = hashlib.sha256(question_id.encode()).hexdigest()
    return int(digest[:8], 16) % folds


def cross_validated_feature(
    rows: list[dict[str, Any]],
    feature: str,
    folds: int,
) -> dict[str, Any]:
    predictions: dict[str, bool] = {}
    calibration = []
    for fold in range(folds):
        train = [row for row in rows if fold_for(row["question_id"], folds) != fold]
        test = [row for row in rows if fold_for(row["question_id"], folds) == fold]
        threshold, direction = choose_threshold(train, feature)
        calibration.append(
            {
                "fold": fold,
                "train_questions": len(train),
                "test_questions": len(test),
                "threshold": threshold,
                "direction": direction,
            }
        )
        for row in test:
            predictions[row["question_id"]] = apply_threshold(
                float(row[feature]),
                threshold,
                direction,
            )
    ordered_predictions = [predictions[row["question_id"]] for row in rows]
    metrics = classification_metrics(
        [bool(row["should_abstain"]) for row in rows],
        ordered_predictions,
    )
    abstention_values = [
        float(row[feature]) for row in rows if row["should_abstain"]
    ]
    answerable_values = [
        float(row[feature]) for row in rows if not row["should_abstain"]
    ]
    return {
        "feature": feature,
        **metrics,
        "abstention_median": statistics.median(abstention_values),
        "answerable_median": statistics.median(answerable_values),
        "calibration": calibration,
    }


def feature_cache_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}.features.json")


def extract_all_features(
    dataset: list[dict[str, Any]],
    args: argparse.Namespace,
    source_sha256: str,
) -> list[dict[str, Any]]:
    cache_path = feature_cache_path(args.output)
    cached_rows: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        manifest = cached["manifest"]
        if (
            manifest["source_file_sha256"] != source_sha256
            or manifest["model_dir"] != str(args.model_dir)
        ):
            raise ValueError("Feature cache does not match dataset or model")
        cached_rows = {
            row["question_id"]: row for row in cached.get("rows", [])
        }

    missing = [
        item for item in dataset if item["question_id"] not in cached_rows
    ]
    encoder = MiniLm(args.model_dir) if missing else None
    for index, item in enumerate(missing, 1):
        assert encoder is not None
        row = extract_features(item, encoder)
        cached_rows[row["question_id"]] = row
        if index % 10 == 0 or index == len(missing):
            ordered = [
                cached_rows[source["question_id"]]
                for source in dataset
                if source["question_id"] in cached_rows
            ]
            write_result(
                cache_path,
                {
                    "manifest": {
                        "source_file_sha256": source_sha256,
                        "model_dir": str(args.model_dir),
                        "completed_questions": len(ordered),
                        "expected_questions": len(dataset),
                    },
                    "rows": ordered,
                },
            )
            print(
                f"features {len(ordered)}/{len(dataset)}",
                flush=True,
            )
    return [cached_rows[item["question_id"]] for item in dataset]


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset = json.loads(args.dataset.read_text())
    source_sha256 = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    rows = extract_all_features(dataset, args, source_sha256)
    labels = [bool(row["should_abstain"]) for row in rows]
    never_abstain = classification_metrics(labels, [False] * len(rows))
    return {
        "manifest": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": "LongMemEval-S cleaned",
            "source_file": str(args.dataset),
            "source_file_sha256": source_sha256,
            "feature_cache": str(feature_cache_path(args.output)),
            "questions": len(rows),
            "abstention_questions": sum(labels),
            "answerable_questions": len(rows) - sum(labels),
            "folds": args.folds,
            "embedding_model": "all-MiniLM-L6-v2 local ONNX",
            "features": list(FEATURES),
            "metric_scope": (
                "Out-of-fold threshold calibration on retrieval score features. "
                "The 30 _abs questions are deliberate semantic near-misses. This "
                "tests whether retrieval confidence alone supports abstention, "
                "not final answer generation."
            ),
        },
        "baseline_never_abstain": never_abstain,
        "features": [
            cross_validated_feature(rows, feature, args.folds)
            for feature in FEATURES
        ],
        "rows": rows,
    }


if __name__ == "__main__":
    arguments = parse_args()
    write_result(arguments.output, run(arguments))
    print(arguments.output)
