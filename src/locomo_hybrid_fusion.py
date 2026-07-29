#!/usr/bin/env python3
"""Evaluate weighted lexical/dense fusion on LoCoMo without group leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark import Bm25, DenseIndex, MiniLm, MODEL_DIR
from graph_benchmark_common import write_result
from locomo_granularity import materialize_word_budget, score_selection
from locomo_window_grid import build_grid_views


DEFAULT_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
DEFAULT_WINDOWS = [2, 4]
RRF_K = 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument(
        "--window-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_WINDOWS,
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=DEFAULT_ALPHAS,
    )
    parser.add_argument("--word-budget", type=int, default=500)
    parser.add_argument("--ranking-depth", type=int, default=20)
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=17)
    return parser.parse_args()


def weighted_rrf(
    bm25_rows: list[dict[str, Any]],
    dense_rows: list[dict[str, Any]],
    alpha: float,
    limit: int,
    rrf_k: int = RRF_K,
) -> list[dict[str, Any]]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between zero and one")
    scores: defaultdict[str, float] = defaultdict(float)
    records: dict[str, dict[str, Any]] = {}
    for weight, ranking in (
        (alpha, bm25_rows),
        (1.0 - alpha, dense_rows),
    ):
        if weight <= 0:
            continue
        for rank, item in enumerate(ranking, start=1):
            scores[item["id"]] += weight / (rrf_k + rank)
            records[item["id"]] = item
    ordered = sorted(
        scores,
        key=lambda item_id: (-scores[item_id], item_id),
    )[:limit]
    return [
        {
            **records[item_id],
            "_score": scores[item_id],
            "_retriever": f"weighted_rrf:{alpha:.2f}",
        }
        for item_id in ordered
    ]


def config_id(representation: str, alpha: float) -> str:
    return f"{representation}:alpha={alpha:.2f}"


def config_rows(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, float], list[dict[str, Any]]]:
    grouped: defaultdict[
        tuple[str, float],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in rows:
        grouped[(row["representation"], row["alpha"])].append(row)
    return dict(grouped)


def sample_mean(
    rows: list[dict[str, Any]],
    metric: str,
) -> float:
    return statistics.fmean(row[metric] for row in rows)


def conversation_weighted_mean(
    rows: list[dict[str, Any]],
    metric: str,
) -> float:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["sample_id"]].append(row)
    return statistics.fmean(
        sample_mean(selected, metric) for selected in grouped.values()
    )


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for (representation, alpha), selected in sorted(config_rows(rows).items()):
        summaries.append(
            {
                "config": config_id(representation, alpha),
                "representation": representation,
                "alpha_bm25": alpha,
                "questions": len(selected),
                "question_weighted_evidence_recall": sample_mean(
                    selected,
                    "evidence_recall",
                ),
                "conversation_weighted_evidence_recall": (
                    conversation_weighted_mean(
                        selected,
                        "evidence_recall",
                    )
                ),
                "mean_context_precision": sample_mean(
                    selected,
                    "context_precision",
                ),
                "mean_context_words": sample_mean(
                    selected,
                    "context_words",
                ),
                "mean_context_turns": sample_mean(
                    selected,
                    "context_turns",
                ),
                "mean_query_latency_ms": sample_mean(
                    selected,
                    "latency_ms",
                ),
            }
        )
    return summaries


def summarize_categories(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: defaultdict[
        tuple[str, float, str],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in rows:
        groups[
            (
                row["representation"],
                row["alpha"],
                row["question_type"],
            )
        ].append(row)
    return [
        {
            "config": config_id(representation, alpha),
            "representation": representation,
            "alpha_bm25": alpha,
            "question_type": question_type,
            "questions": len(selected),
            "mean_evidence_recall": sample_mean(
                selected,
                "evidence_recall",
            ),
            "mean_context_precision": sample_mean(
                selected,
                "context_precision",
            ),
        }
        for (
            representation,
            alpha,
            question_type,
        ), selected in sorted(groups.items())
    ]


def bootstrap_interval(
    differences: list[float],
    resamples: int,
    seed: int,
) -> list[float]:
    generator = random.Random(seed)
    values = sorted(
        statistics.fmean(
            generator.choice(differences) for _ in differences
        )
        for _ in range(resamples)
    )
    return [
        values[int(0.025 * resamples)],
        values[int(0.975 * resamples) - 1],
    ]


def metrics_for_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "questions": len(rows),
        "question_weighted_evidence_recall": sample_mean(
            rows,
            "evidence_recall",
        ),
        "conversation_weighted_evidence_recall": (
            conversation_weighted_mean(rows, "evidence_recall")
        ),
        "mean_context_precision": sample_mean(
            rows,
            "context_precision",
        ),
        "mean_context_words": sample_mean(rows, "context_words"),
        "mean_context_turns": sample_mean(rows, "context_turns"),
        "mean_query_latency_ms": sample_mean(rows, "latency_ms"),
    }


def choose_training_config(
    rows: list[dict[str, Any]],
    holdout: str,
) -> tuple[str, float, float]:
    training = [row for row in rows if row["sample_id"] != holdout]
    candidates = []
    for (representation, alpha), selected in config_rows(training).items():
        candidates.append(
            (
                -conversation_weighted_mean(
                    selected,
                    "evidence_recall",
                ),
                sample_mean(selected, "latency_ms"),
                representation,
                alpha,
            )
        )
    if not candidates:
        raise ValueError(f"No training rows for holdout {holdout}")
    best = min(candidates)
    return best[2], best[3], -best[0]


def compare_paired_by_conversation(
    selected_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    sample_ids: list[str],
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    differences = []
    for sample_id in sample_ids:
        selected = [
            row
            for row in selected_rows
            if row["sample_id"] == sample_id
        ]
        baseline = [
            row
            for row in baseline_rows
            if row["sample_id"] == sample_id
        ]
        differences.append(
            sample_mean(selected, "evidence_recall")
            - sample_mean(baseline, "evidence_recall")
        )
    return {
        "conversation_weighted_mean_difference": statistics.fmean(
            differences
        ),
        "selected_wins": sum(value > 0 for value in differences),
        "baseline_wins": sum(value < 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        "bootstrap_95_interval": bootstrap_interval(
            differences,
            resamples,
            seed,
        ),
    }


def cross_validate(
    rows: list[dict[str, Any]],
    sample_ids: list[str],
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    selected_rows: list[dict[str, Any]] = []
    folds = []
    choices: Counter[str] = Counter()
    for holdout in sample_ids:
        representation, alpha, training_recall = choose_training_config(
            rows,
            holdout,
        )
        chosen = [
            row
            for row in rows
            if row["sample_id"] == holdout
            and row["representation"] == representation
            and row["alpha"] == alpha
        ]
        selected_rows.extend(chosen)
        identifier = config_id(representation, alpha)
        choices[identifier] += 1
        folds.append(
            {
                "holdout_sample": holdout,
                "selected_config": identifier,
                "training_conversation_weighted_evidence_recall": (
                    training_recall
                ),
                "holdout_metrics": metrics_for_rows(chosen),
            }
        )

    baselines = {
        "equal_rrf_window2": ("window2", 0.5),
        "bm25_window4": ("window4", 1.0),
    }
    baseline_results = {}
    for name, (representation, alpha) in baselines.items():
        baseline_rows = [
            row
            for row in rows
            if row["representation"] == representation
            and row["alpha"] == alpha
        ]
        baseline_results[name] = {
            "config": config_id(representation, alpha),
            "metrics": metrics_for_rows(baseline_rows),
            "selected_minus_baseline": compare_paired_by_conversation(
                selected_rows,
                baseline_rows,
                sample_ids,
                resamples,
                seed,
            ),
        }

    category_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        category_groups[row["question_type"]].append(row)
    return {
        "protocol": (
            "Leave one complete LoCoMo conversation out. Select the "
            "representation and alpha on the other nine conversations using "
            "conversation-weighted evidence recall, then score every eligible "
            "question in the held-out conversation. Latency breaks exact "
            "training-score ties only."
        ),
        "folds": folds,
        "selection_counts": dict(sorted(choices.items())),
        "out_of_fold_selected_metrics": metrics_for_rows(selected_rows),
        "out_of_fold_selected_categories": [
            {
                "question_type": category,
                **metrics_for_rows(selected),
            }
            for category, selected in sorted(category_groups.items())
        ],
        "baselines": baseline_results,
    }


def cross_validate_oracle_category_router(
    rows: list[dict[str, Any]],
    sample_ids: list[str],
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    selected_rows: list[dict[str, Any]] = []
    folds = []
    choices: defaultdict[str, Counter[str]] = defaultdict(Counter)
    categories = sorted({row["question_type"] for row in rows})
    for holdout in sample_ids:
        for category in categories:
            category_rows = [
                row
                for row in rows
                if row["question_type"] == category
            ]
            if not any(
                row["sample_id"] == holdout for row in category_rows
            ):
                continue
            representation, alpha, training_recall = (
                choose_training_config(category_rows, holdout)
            )
            chosen = [
                row
                for row in category_rows
                if row["sample_id"] == holdout
                and row["representation"] == representation
                and row["alpha"] == alpha
            ]
            selected_rows.extend(chosen)
            identifier = config_id(representation, alpha)
            choices[category][identifier] += 1
            folds.append(
                {
                    "holdout_sample": holdout,
                    "question_type": category,
                    "selected_config": identifier,
                    "training_conversation_weighted_evidence_recall": (
                        training_recall
                    ),
                    "holdout_metrics": metrics_for_rows(chosen),
                }
            )

    baselines = {
        "equal_rrf_window2": ("window2", 0.5),
        "bm25_window4": ("window4", 1.0),
    }
    baseline_results = {}
    for name, (representation, alpha) in baselines.items():
        baseline_rows = [
            row
            for row in rows
            if row["representation"] == representation
            and row["alpha"] == alpha
        ]
        baseline_results[name] = {
            "config": config_id(representation, alpha),
            "metrics": metrics_for_rows(baseline_rows),
            "selected_minus_baseline": compare_paired_by_conversation(
                selected_rows,
                baseline_rows,
                sample_ids,
                resamples,
                seed,
            ),
        }
    category_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        category_groups[row["question_type"]].append(row)
    return {
        "protocol": (
            "Oracle upper bound only. For each held-out conversation and each "
            "gold LoCoMo question category, select the representation and alpha "
            "on the other conversations, then evaluate that category on the "
            "holdout. A deployable router would first need to predict the "
            "category and may score lower."
        ),
        "folds": folds,
        "selection_counts_by_category": {
            category: dict(sorted(counts.items()))
            for category, counts in sorted(choices.items())
        },
        "out_of_fold_selected_metrics": metrics_for_rows(selected_rows),
        "out_of_fold_selected_categories": [
            {
                "question_type": category,
                **metrics_for_rows(selected),
            }
            for category, selected in sorted(category_groups.items())
        ],
        "baselines": baseline_results,
    }


def pareto_frontier(
    summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    frontier = []
    for candidate in summaries:
        dominated = any(
            other["conversation_weighted_evidence_recall"]
            >= candidate["conversation_weighted_evidence_recall"]
            and other["mean_query_latency_ms"]
            <= candidate["mean_query_latency_ms"]
            and (
                other["conversation_weighted_evidence_recall"]
                > candidate["conversation_weighted_evidence_recall"]
                or other["mean_query_latency_ms"]
                < candidate["mean_query_latency_ms"]
            )
            for other in summaries
        )
        if not dominated:
            frontier.append(candidate)
    return sorted(
        frontier,
        key=lambda item: item["mean_query_latency_ms"],
    )


def summarize_index_build(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["representation"]].append(row)
    return [
        {
            "representation": representation,
            "samples": len(selected),
            "mean_candidates": sample_mean(selected, "candidates"),
            "mean_bm25_build_ms": sample_mean(
                selected,
                "bm25_build_ms",
            ),
            "mean_dense_build_ms": sample_mean(
                selected,
                "dense_build_ms",
            ),
        }
        for representation, selected in sorted(groups.items())
    ]


def make_payload(
    args: argparse.Namespace,
    source_sha256: str,
    rows: list[dict[str, Any]],
    index_build_rows: list[dict[str, Any]],
    completed_samples: list[str],
    expected_samples: int,
) -> dict[str, Any]:
    summaries = summarize(rows) if rows else []
    analysis = (
        cross_validate(
            rows,
            completed_samples,
            args.bootstrap_resamples,
            args.bootstrap_seed,
        )
        if len(completed_samples) >= 2
        else None
    )
    core_rows = [
        row for row in rows if row["question_type"] != "5"
    ]
    return {
        "manifest": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": "LoCoMo",
            "source_file": str(args.dataset),
            "source_file_sha256": source_sha256,
            "embedding_model": "all-MiniLM-L6-v2 local ONNX",
            "window_sizes": args.window_sizes,
            "alphas_bm25": args.alphas,
            "rrf_k": RRF_K,
            "word_budget": args.word_budget,
            "ranking_depth": args.ranking_depth,
            "completed_samples": completed_samples,
            "expected_samples": expected_samples,
            "complete": len(completed_samples) == expected_samples,
            "metric_scope": (
                "LoCoMo evidence retrieval under a shared word budget. Alpha "
                "weights BM25 rank contribution; one minus alpha weights dense "
                "rank contribution. Leave-one-conversation-out selection "
                "prevents tuning on the evaluated conversation. Query latency "
                "excludes index construction. This is not answer generation."
            ),
        },
        "summaries": summaries,
        "category_summaries": summarize_categories(rows) if rows else [],
        "index_build_summaries": summarize_index_build(index_build_rows),
        "pareto_frontier_in_sample": pareto_frontier(summaries),
        "grouped_cross_validation": analysis,
        "grouped_cross_validation_categories_1_to_4": (
            cross_validate(
                core_rows,
                completed_samples,
                args.bootstrap_resamples,
                args.bootstrap_seed,
            )
            if len(completed_samples) >= 2
            else None
        ),
        "oracle_category_router": (
            cross_validate_oracle_category_router(
                rows,
                completed_samples,
                args.bootstrap_resamples,
                args.bootstrap_seed,
            )
            if len(completed_samples) >= 2
            else None
        ),
        "index_build_rows": index_build_rows,
        "rows": rows,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.window_sizes:
        raise ValueError("At least one window size is required")
    if not args.alphas or any(
        alpha < 0.0 or alpha > 1.0 for alpha in args.alphas
    ):
        raise ValueError("Alphas must be between zero and one")
    dataset = json.loads(args.dataset.read_text())
    source_sha256 = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    rows: list[dict[str, Any]] = []
    index_build_rows: list[dict[str, Any]] = []
    completed_samples: list[str] = []
    if args.output.exists():
        existing = json.loads(args.output.read_text())
        manifest = existing["manifest"]
        expected = {
            "source_file_sha256": source_sha256,
            "window_sizes": args.window_sizes,
            "alphas_bm25": args.alphas,
            "word_budget": args.word_budget,
            "ranking_depth": args.ranking_depth,
        }
        if any(manifest[key] != value for key, value in expected.items()):
            raise ValueError("Output checkpoint does not match current protocol")
        rows = existing["rows"]
        index_build_rows = existing["index_build_rows"]
        completed_samples = list(manifest["completed_samples"])

    encoder = MiniLm(args.model_dir)
    for sample in dataset:
        sample_id = sample["sample_id"]
        if sample_id in completed_samples:
            continue
        all_views = build_grid_views(sample, args.window_sizes)
        views = {
            f"window{size}": all_views[f"window{size}"]
            for size in args.window_sizes
        }
        indexes = {}
        for representation, candidates in views.items():
            started = time.perf_counter()
            bm25 = Bm25(candidates)
            bm25_build_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            dense = DenseIndex(candidates, encoder)
            dense_build_ms = (time.perf_counter() - started) * 1000
            indexes[representation] = (bm25, dense)
            index_build_rows.append(
                {
                    "sample_id": sample_id,
                    "representation": representation,
                    "candidates": len(candidates),
                    "bm25_build_ms": bm25_build_ms,
                    "dense_build_ms": dense_build_ms,
                }
            )

        for question_index, question in enumerate(sample["qa"]):
            gold = set(question.get("evidence", []))
            if not gold:
                continue
            for representation, (bm25, dense) in indexes.items():
                started = time.perf_counter()
                bm25_rows = bm25.search(
                    question["question"],
                    args.ranking_depth * 2,
                )
                bm25_ms = (time.perf_counter() - started) * 1000
                started = time.perf_counter()
                dense_rows = dense.search(
                    question["question"],
                    args.ranking_depth * 2,
                )
                dense_ms = (time.perf_counter() - started) * 1000
                for alpha in args.alphas:
                    started = time.perf_counter()
                    ranked = weighted_rrf(
                        bm25_rows,
                        dense_rows,
                        alpha,
                        args.ranking_depth,
                    )
                    fusion_ms = (time.perf_counter() - started) * 1000
                    if alpha == 1.0:
                        latency_ms = bm25_ms + fusion_ms
                    elif alpha == 0.0:
                        latency_ms = dense_ms + fusion_ms
                    else:
                        latency_ms = bm25_ms + dense_ms + fusion_ms
                    selected, words = materialize_word_budget(
                        ranked,
                        args.word_budget,
                    )
                    recall, precision = score_selection(gold, selected)
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "question_id": f"{sample_id}:{question_index}",
                            "question_type": str(question["category"]),
                            "representation": representation,
                            "alpha": alpha,
                            "evidence_recall": recall,
                            "context_precision": precision,
                            "context_words": words,
                            "context_turns": len(selected),
                            "latency_ms": latency_ms,
                        }
                    )
        completed_samples.append(sample_id)
        write_result(
            args.output,
            make_payload(
                args,
                source_sha256,
                rows,
                index_build_rows,
                completed_samples,
                len(dataset),
            ),
        )
        print(
            f"samples {len(completed_samples)}/{len(dataset)}",
            flush=True,
        )
    return make_payload(
        args,
        source_sha256,
        rows,
        index_build_rows,
        completed_samples,
        len(dataset),
    )


if __name__ == "__main__":
    arguments = parse_args()
    result = run(arguments)
    write_result(arguments.output, result)
    print(arguments.output)
