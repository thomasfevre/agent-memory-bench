#!/usr/bin/env python3
"""Evaluate premise-aware abstention with pair-grouped nested validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from benchmark import Bm25, tokenize
from external_retrieval import session_text
from graph_benchmark_common import write_result
from longmemeval_abstention import (
    FEATURES as RETRIEVAL_FEATURES,
    choose_threshold,
    classification_metrics,
    safe_z,
)


STOPWORDS = {
    "a",
    "an",
    "the",
    "my",
    "your",
    "our",
    "their",
    "his",
    "her",
    "its",
    "i",
    "you",
    "we",
    "they",
    "he",
    "she",
    "it",
    "me",
    "us",
    "them",
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "when",
    "where",
    "why",
    "how",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "can",
    "could",
    "would",
    "should",
    "will",
    "may",
    "might",
    "much",
    "many",
    "long",
    "often",
    "time",
    "times",
    "ago",
    "before",
    "after",
    "first",
    "now",
    "current",
    "currently",
    "in",
    "on",
    "at",
    "to",
    "from",
    "for",
    "of",
    "with",
    "and",
    "or",
    "but",
    "as",
    "into",
    "about",
    "than",
    "then",
    "every",
    "total",
    "both",
    "different",
    "past",
    "previously",
    "instead",
    "just",
    "new",
    "name",
}

PREMISE_FEATURES = (
    "history_content_coverage",
    "history_idf_coverage",
    "history_missing_fraction",
    "history_min_session_df",
    "message_bm25_top1",
    "message_bm25_margin",
    "message_bm25_per_query_term",
    "message_bm25_top1_z",
    "message_top1_coverage",
    "message_top5_coverage",
)

FEATURE_SETS = {
    "retrieval_scores": tuple(RETRIEVAL_FEATURES),
    "premise_support": PREMISE_FEATURES,
    "combined": tuple(RETRIEVAL_FEATURES) + PREMISE_FEATURES,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--retrieval-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument(
        "--l2-values",
        type=float,
        nargs="+",
        default=[0.01, 0.1, 1.0, 10.0],
    )
    parser.add_argument("--iterations", type=int, default=1_000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=31)
    return parser.parse_args()


def pair_group_id(question_id: str) -> str:
    return re.sub(r"_abs(?:_.*)?$", "", question_id)


def stable_group_folds(
    rows: list[dict[str, Any]],
    folds: int,
    salt: str,
) -> dict[str, int]:
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    positive = [
        group
        for group, selected in groups.items()
        if any(row["should_abstain"] for row in selected)
    ]
    negative = [
        group
        for group, selected in groups.items()
        if not any(row["should_abstain"] for row in selected)
    ]

    def ordered(values: list[str]) -> list[str]:
        return sorted(
            values,
            key=lambda value: hashlib.sha256(
                f"{salt}:{value}".encode()
            ).hexdigest(),
        )

    assignments = {}
    loads = [0] * folds
    for group in ordered(positive):
        fold = min(range(folds), key=lambda item: (loads[item], item))
        assignments[group] = fold
        loads[fold] += len(groups[group])
    for group in ordered(negative):
        fold = min(range(folds), key=lambda item: (loads[item], item))
        assignments[group] = fold
        loads[fold] += len(groups[group])
    return assignments


def stem(term: str) -> str:
    if len(term) > 5 and term.endswith("ing"):
        return term[:-3]
    if len(term) > 4 and term.endswith("ed"):
        return term[:-2]
    if len(term) > 4 and term.endswith("es"):
        return term[:-2]
    if len(term) > 3 and term.endswith("s"):
        return term[:-1]
    return term


def content_terms(text: str) -> list[str]:
    return list(
        dict.fromkeys(
            stem(term)
            for term in tokenize(text)
            if term not in STOPWORDS and len(term) > 1
        )
    )


def premise_features(item: dict[str, Any]) -> dict[str, float | int]:
    terms = content_terms(item["question"])
    session_token_sets = []
    messages = []
    for session_index, session in enumerate(item["haystack_sessions"]):
        text = session_text(session)
        session_token_sets.append(
            {stem(term) for term in tokenize(text)}
        )
        for message_index, message in enumerate(session):
            content = message.get("content", "")
            if not content:
                continue
            messages.append(
                {
                    "id": f"{session_index}:{message_index}",
                    "text": content,
                }
            )
    history = (
        set().union(*session_token_sets) if session_token_sets else set()
    )
    session_count = len(session_token_sets) or 1
    dfs = {
        term: sum(term in session for session in session_token_sets)
        for term in terms
    }
    weights = {
        term: math.log((session_count + 1) / (dfs[term] + 1)) + 1
        for term in terms
    }
    term_count = len(terms) or 1
    supported = sum(term in history for term in terms)
    total_weight = sum(weights.values()) or 1.0
    supported_weight = sum(
        weights[term] for term in terms if term in history
    )

    message_rows = Bm25(messages).search(
        item["question"],
        len(messages),
    )
    scores = [float(row["_score"]) for row in message_rows]
    top1 = scores[0] if scores else 0.0
    top2 = scores[1] if len(scores) > 1 else 0.0
    message_sets = [
        {stem(term) for term in tokenize(row["text"])}
        for row in message_rows[:5]
    ]
    top1_terms = message_sets[0] if message_sets else set()
    top5_terms = set().union(*message_sets) if message_sets else set()
    return {
        "question_content_terms": len(terms),
        "history_content_coverage": supported / term_count,
        "history_idf_coverage": supported_weight / total_weight,
        "history_missing_fraction": 1.0 - supported / term_count,
        "history_min_session_df": (
            min(dfs.values()) / session_count if terms else 1.0
        ),
        "message_bm25_top1": top1,
        "message_bm25_margin": top1 - top2,
        "message_bm25_per_query_term": top1 / term_count,
        "message_bm25_top1_z": safe_z(top1, scores),
        "message_top1_coverage": (
            sum(term in top1_terms for term in terms) / term_count
        ),
        "message_top5_coverage": (
            sum(term in top5_terms for term in terms) / term_count
        ),
        "messages": len(messages),
    }


def feature_cache_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}.features.json")


def extract_features(
    dataset: list[dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    source_sha256: str,
    retrieval_sha256: str,
) -> list[dict[str, Any]]:
    retrieval_by_id = {
        row["question_id"]: row for row in retrieval_rows
    }
    cache_path = feature_cache_path(args.output)
    cached_by_id: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        manifest = cached["manifest"]
        if (
            manifest["source_file_sha256"] != source_sha256
            or manifest["retrieval_features_sha256"] != retrieval_sha256
        ):
            raise ValueError("Feature cache does not match input files")
        cached_by_id = {
            row["question_id"]: row for row in cached["rows"]
        }
    missing = [
        item
        for item in dataset
        if item["question_id"] not in cached_by_id
    ]
    for index, item in enumerate(missing, start=1):
        retrieval = retrieval_by_id[item["question_id"]]
        row = {
            "question_id": item["question_id"],
            "question_type": item["question_type"],
            "group_id": pair_group_id(item["question_id"]),
            "should_abstain": "_abs" in item["question_id"],
            **{
                feature: float(retrieval[feature])
                for feature in RETRIEVAL_FEATURES
            },
            **premise_features(item),
        }
        cached_by_id[item["question_id"]] = row
        if index % 25 == 0 or index == len(missing):
            ordered = [
                cached_by_id[source["question_id"]]
                for source in dataset
                if source["question_id"] in cached_by_id
            ]
            write_result(
                cache_path,
                {
                    "manifest": {
                        "source_file_sha256": source_sha256,
                        "retrieval_features_sha256": retrieval_sha256,
                        "completed_questions": len(ordered),
                        "expected_questions": len(dataset),
                    },
                    "rows": ordered,
                },
            )
            print(
                f"premise features {len(ordered)}/{len(dataset)}",
                flush=True,
            )
    return [cached_by_id[item["question_id"]] for item in dataset]


def fit_logistic(
    matrix: np.ndarray,
    labels: np.ndarray,
    l2: float,
    iterations: int,
    learning_rate: float,
) -> dict[str, np.ndarray]:
    mean = matrix.mean(axis=0)
    deviation = matrix.std(axis=0)
    deviation = np.where(deviation == 0, 1.0, deviation)
    standardized = (matrix - mean) / deviation
    design = np.column_stack([np.ones(len(matrix)), standardized])
    weights = np.zeros(design.shape[1], dtype=np.float64)
    positives = max(float(labels.sum()), 1.0)
    negatives = max(float(len(labels) - labels.sum()), 1.0)
    sample_weights = np.where(
        labels == 1,
        len(labels) / (2.0 * positives),
        len(labels) / (2.0 * negatives),
    )
    normalization = sample_weights.sum()
    for _ in range(iterations):
        logits = np.clip(design @ weights, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = (
            design.T
            @ ((probabilities - labels) * sample_weights)
            / normalization
        )
        gradient[1:] += l2 * weights[1:] / normalization
        weights -= learning_rate * gradient
    return {
        "mean": mean,
        "deviation": deviation,
        "weights": weights,
    }


def predict_logistic(
    model: dict[str, np.ndarray],
    matrix: np.ndarray,
) -> np.ndarray:
    standardized = (
        matrix - model["mean"]
    ) / model["deviation"]
    design = np.column_stack([np.ones(len(matrix)), standardized])
    logits = np.clip(design @ model["weights"], -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-logits))


def matrix_for(
    rows: list[dict[str, Any]],
    features: tuple[str, ...],
) -> np.ndarray:
    return np.asarray(
        [
            [float(row[feature]) for feature in features]
            for row in rows
        ],
        dtype=np.float64,
    )


def grouped_threshold_feature(
    rows: list[dict[str, Any]],
    feature: str,
    assignments: dict[str, int],
    folds: int,
) -> dict[str, Any]:
    predictions: dict[str, bool] = {}
    calibration = []
    for fold in range(folds):
        train = [
            row
            for row in rows
            if assignments[row["group_id"]] != fold
        ]
        test = [
            row
            for row in rows
            if assignments[row["group_id"]] == fold
        ]
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
            value = float(row[feature])
            predictions[row["question_id"]] = (
                value <= threshold
                if direction == "low"
                else value >= threshold
            )
    ordered = [predictions[row["question_id"]] for row in rows]
    return {
        "feature": feature,
        **classification_metrics(
            [bool(row["should_abstain"]) for row in rows],
            ordered,
        ),
        "calibration": calibration,
        "rows": [
            {
                "question_id": row["question_id"],
                "group_id": row["group_id"],
                "should_abstain": row["should_abstain"],
                "predicted_abstention": predictions[row["question_id"]],
            }
            for row in rows
        ],
    }


def nested_logistic(
    rows: list[dict[str, Any]],
    features: tuple[str, ...],
    assignments: dict[str, int],
    args: argparse.Namespace,
    label: str,
) -> dict[str, Any]:
    predictions: dict[str, bool] = {}
    probabilities: dict[str, float] = {}
    calibration = []
    for outer_fold in range(args.folds):
        outer_train = [
            row
            for row in rows
            if assignments[row["group_id"]] != outer_fold
        ]
        outer_test = [
            row
            for row in rows
            if assignments[row["group_id"]] == outer_fold
        ]
        inner_assignments = stable_group_folds(
            outer_train,
            args.inner_folds,
            f"inner:{label}:{outer_fold}",
        )
        candidates = []
        for l2 in args.l2_values:
            inner_probabilities: dict[str, float] = {}
            for inner_fold in range(args.inner_folds):
                inner_train = [
                    row
                    for row in outer_train
                    if inner_assignments[row["group_id"]] != inner_fold
                ]
                inner_test = [
                    row
                    for row in outer_train
                    if inner_assignments[row["group_id"]] == inner_fold
                ]
                model = fit_logistic(
                    matrix_for(inner_train, features),
                    np.asarray(
                        [
                            int(row["should_abstain"])
                            for row in inner_train
                        ],
                        dtype=np.float64,
                    ),
                    l2,
                    args.iterations,
                    args.learning_rate,
                )
                values = predict_logistic(
                    model,
                    matrix_for(inner_test, features),
                )
                inner_probabilities.update(
                    {
                        row["question_id"]: float(value)
                        for row, value in zip(
                            inner_test,
                            values,
                            strict=True,
                        )
                    }
                )
            probability_rows = [
                {
                    "probability": inner_probabilities[row["question_id"]],
                    "should_abstain": row["should_abstain"],
                }
                for row in outer_train
            ]
            threshold, direction = choose_threshold(
                probability_rows,
                "probability",
            )
            inner_predictions = [
                bool(
                    row["probability"] <= threshold
                    if direction == "low"
                    else row["probability"] >= threshold
                )
                for row in probability_rows
            ]
            metrics = classification_metrics(
                [
                    bool(row["should_abstain"])
                    for row in probability_rows
                ],
                inner_predictions,
            )
            candidates.append(
                (
                    -float(metrics["balanced_accuracy"]),
                    -float(metrics["f1"]),
                    float(metrics["false_abstention_rate"]),
                    l2,
                    threshold,
                    direction,
                    metrics,
                )
            )
        best = min(candidates)
        l2, threshold, direction = best[3], best[4], best[5]
        model = fit_logistic(
            matrix_for(outer_train, features),
            np.asarray(
                [
                    int(row["should_abstain"])
                    for row in outer_train
                ],
                dtype=np.float64,
            ),
            l2,
            args.iterations,
            args.learning_rate,
        )
        values = predict_logistic(
            model,
            matrix_for(outer_test, features),
        )
        for row, value in zip(outer_test, values, strict=True):
            probabilities[row["question_id"]] = float(value)
            predictions[row["question_id"]] = bool(
                value <= threshold
                if direction == "low"
                else value >= threshold
            )
        calibration.append(
            {
                "outer_fold": outer_fold,
                "train_questions": len(outer_train),
                "test_questions": len(outer_test),
                "selected_l2": l2,
                "threshold": threshold,
                "direction": direction,
                "inner_metrics": best[6],
            }
        )
    ordered_predictions = [
        predictions[row["question_id"]] for row in rows
    ]
    result = {
        "feature_set": label,
        "features": list(features),
        **classification_metrics(
            [bool(row["should_abstain"]) for row in rows],
            ordered_predictions,
        ),
        "calibration": calibration,
        "rows": [
            {
                "question_id": row["question_id"],
                "group_id": row["group_id"],
                "should_abstain": row["should_abstain"],
                "probability": probabilities[row["question_id"]],
                "predicted_abstention": predictions[row["question_id"]],
            }
            for row in rows
        ],
    }
    return result


def paired_groups(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    paired = [
        selected
        for selected in groups.values()
        if any(row["should_abstain"] for row in selected)
        and any(not row["should_abstain"] for row in selected)
    ]
    return {
        "groups": len(groups),
        "abstention_groups": sum(
            any(row["should_abstain"] for row in selected)
            for selected in groups.values()
        ),
        "paired_positive_negative_groups": len(paired),
        "unpaired_abstention_groups": sum(
            any(row["should_abstain"] for row in selected)
            and not any(not row["should_abstain"] for row in selected)
            for selected in groups.values()
        ),
    }


def bootstrap_balanced_accuracy_difference(
    rows: list[dict[str, Any]],
    challenger_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    challenger = {
        row["question_id"]: bool(row["predicted_abstention"])
        for row in challenger_rows
    }
    baseline = {
        row["question_id"]: bool(row["predicted_abstention"])
        for row in baseline_rows
    }
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    units = list(groups.values())

    def score(selected: list[list[dict[str, Any]]]) -> tuple[float, float]:
        sampled = [row for group in selected for row in group]
        labels = [bool(row["should_abstain"]) for row in sampled]
        challenger_predictions = [
            challenger[row["question_id"]] for row in sampled
        ]
        baseline_predictions = [
            baseline[row["question_id"]] for row in sampled
        ]
        return (
            float(
                classification_metrics(
                    labels,
                    challenger_predictions,
                )["balanced_accuracy"]
            ),
            float(
                classification_metrics(
                    labels,
                    baseline_predictions,
                )["balanced_accuracy"]
            ),
        )

    observed_challenger, observed_baseline = score(units)
    generator = random.Random(seed)
    differences = sorted(
        (
            lambda values: values[0] - values[1]
        )(
            score(
                [
                    generator.choice(units)
                    for _ in range(len(units))
                ]
            )
        )
        for _ in range(resamples)
    )
    return {
        "bootstrap_unit": "pair group",
        "groups": len(units),
        "resamples": resamples,
        "seed": seed,
        "challenger_balanced_accuracy": observed_challenger,
        "baseline_balanced_accuracy": observed_baseline,
        "observed_difference": observed_challenger - observed_baseline,
        "bootstrap_95_interval": [
            differences[int(0.025 * resamples)],
            differences[int(0.975 * resamples) - 1],
        ],
        "bootstrap_probability_difference_positive": (
            sum(value > 0 for value in differences) / resamples
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset = json.loads(args.dataset.read_text())
    retrieval_payload = json.loads(args.retrieval_features.read_text())
    retrieval_rows = retrieval_payload["rows"]
    source_sha256 = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    retrieval_sha256 = hashlib.sha256(
        args.retrieval_features.read_bytes()
    ).hexdigest()
    rows = extract_features(
        dataset,
        retrieval_rows,
        args,
        source_sha256,
        retrieval_sha256,
    )
    assignments = stable_group_folds(rows, args.folds, "outer")
    single_features = [
        grouped_threshold_feature(
            rows,
            feature,
            assignments,
            args.folds,
        )
        for feature in tuple(RETRIEVAL_FEATURES) + PREMISE_FEATURES
    ]
    logistic = [
        nested_logistic(
            rows,
            features,
            assignments,
            args,
            label,
        )
        for label, features in FEATURE_SETS.items()
    ]
    bm25_top1 = next(
        result
        for result in single_features
        if result["feature"] == "bm25_top1"
    )
    bootstrap_comparisons = [
        {
            "challenger": result["feature_set"],
            "baseline": "bm25_top1_grouped_threshold",
            **bootstrap_balanced_accuracy_difference(
                rows,
                result["rows"],
                bm25_top1["rows"],
                args.bootstrap_resamples,
                args.bootstrap_seed,
            ),
        }
        for result in logistic
        if result["feature_set"] in {"premise_support", "combined"}
    ]
    labels = [bool(row["should_abstain"]) for row in rows]
    return {
        "manifest": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": "LongMemEval-S cleaned",
            "source_file": str(args.dataset),
            "source_file_sha256": source_sha256,
            "retrieval_features_file": str(args.retrieval_features),
            "retrieval_features_sha256": retrieval_sha256,
            "feature_cache": str(feature_cache_path(args.output)),
            "questions": len(rows),
            "abstention_questions": sum(labels),
            "answerable_questions": len(rows) - sum(labels),
            "outer_folds": args.folds,
            "inner_folds": args.inner_folds,
            "l2_values": args.l2_values,
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "bootstrap_resamples": args.bootstrap_resamples,
            "bootstrap_seed": args.bootstrap_seed,
            "complete": len(rows) == len(dataset),
            "metric_scope": (
                "Pair-grouped out-of-fold premise verification. A near-miss "
                "and its answerable base question remain in the same fold. "
                "Nested grouped validation selects logistic regularization and "
                "decision threshold. This is a deterministic local classifier, "
                "not the official LLM answer judge."
            ),
        },
        "pair_structure": paired_groups(rows),
        "fold_summary": [
            {
                "fold": fold,
                "questions": sum(
                    assignments[row["group_id"]] == fold for row in rows
                ),
                "abstention_questions": sum(
                    assignments[row["group_id"]] == fold
                    and row["should_abstain"]
                    for row in rows
                ),
            }
            for fold in range(args.folds)
        ],
        "baseline_never_abstain": classification_metrics(
            labels,
            [False] * len(rows),
        ),
        "single_feature_thresholds": single_features,
        "logistic_models": logistic,
        "bootstrap_comparisons": bootstrap_comparisons,
        "rows": rows,
    }


if __name__ == "__main__":
    arguments = parse_args()
    write_result(arguments.output, run(arguments))
    print(arguments.output)
