#!/usr/bin/env python3
"""Evaluate weighted hybrid retrieval on answerable LongMemEval-S questions."""

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
from external_retrieval import session_text
from graph_benchmark_common import write_result
from locomo_hybrid_fusion import weighted_rrf
from longmemeval_abstention import fold_for


DEFAULT_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
MODES = ("top5_sessions", "word_budget")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=DEFAULT_ALPHAS,
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--ranking-depth", type=int, default=20)
    parser.add_argument("--word-budget", type=int, default=8_000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=29)
    return parser.parse_args()


def select_top_items(
    ranking: list[dict[str, Any]],
    top_k: int,
) -> tuple[set[str], int]:
    selected = ranking[:top_k]
    return (
        {row["id"] for row in selected},
        sum(len(row["text"].split()) for row in selected),
    )


def select_word_budget(
    ranking: list[dict[str, Any]],
    word_budget: int,
) -> tuple[set[str], int]:
    selected: set[str] = set()
    words = 0
    for row in ranking:
        item_words = len(row["text"].split())
        if item_words > word_budget - words:
            continue
        selected.add(row["id"])
        words += item_words
        if words >= word_budget:
            break
    return selected, words


def score(
    gold: set[str],
    selected: set[str],
    ranking: list[dict[str, Any]],
) -> tuple[float, float]:
    recall = len(gold & selected) / len(gold) if gold else 0.0
    first = next(
        (
            rank
            for rank, row in enumerate(ranking, start=1)
            if row["id"] in gold
        ),
        None,
    )
    return recall, 1.0 / first if first else 0.0


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: defaultdict[
        tuple[str, float],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in rows:
        groups[(row["mode"], row["alpha"])].append(row)
    return [
        {
            "mode": mode,
            "alpha_bm25": alpha,
            "questions": len(selected),
            "mean_evidence_recall": statistics.fmean(
                row["evidence_recall"] for row in selected
            ),
            "mean_mrr": statistics.fmean(row["mrr"] for row in selected),
            "mean_context_words": statistics.fmean(
                row["context_words"] for row in selected
            ),
            "mean_selected_sessions": statistics.fmean(
                row["selected_sessions"] for row in selected
            ),
            "mean_query_latency_ms": statistics.fmean(
                row["latency_ms"] for row in selected
            ),
        }
        for (mode, alpha), selected in sorted(groups.items())
    ]


def summarize_types(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: defaultdict[
        tuple[str, float, str],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in rows:
        groups[(row["mode"], row["alpha"], row["question_type"])].append(
            row
        )
    return [
        {
            "mode": mode,
            "alpha_bm25": alpha,
            "question_type": question_type,
            "questions": len(selected),
            "mean_evidence_recall": statistics.fmean(
                row["evidence_recall"] for row in selected
            ),
        }
        for (
            mode,
            alpha,
            question_type,
        ), selected in sorted(groups.items())
    ]


def row_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "questions": len(rows),
        "mean_evidence_recall": statistics.fmean(
            row["evidence_recall"] for row in rows
        ),
        "mean_mrr": statistics.fmean(row["mrr"] for row in rows),
        "mean_context_words": statistics.fmean(
            row["context_words"] for row in rows
        ),
        "mean_selected_sessions": statistics.fmean(
            row["selected_sessions"] for row in rows
        ),
        "mean_query_latency_ms": statistics.fmean(
            row["latency_ms"] for row in rows
        ),
    }


def paired_bootstrap(
    selected: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    baseline_by_id = {
        row["question_id"]: row["evidence_recall"] for row in baseline
    }
    differences = [
        row["evidence_recall"] - baseline_by_id[row["question_id"]]
        for row in selected
    ]
    generator = random.Random(seed)
    values = sorted(
        statistics.fmean(
            generator.choice(differences) for _ in differences
        )
        for _ in range(resamples)
    )
    return {
        "mean_difference": statistics.fmean(differences),
        "selected_wins": sum(value > 0 for value in differences),
        "baseline_wins": sum(value < 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "bootstrap_unit": "question with independent history",
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        "bootstrap_95_interval": [
            values[int(0.025 * resamples)],
            values[int(0.975 * resamples) - 1],
        ],
    }


def cross_validate(
    rows: list[dict[str, Any]],
    mode: str,
    folds: int,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    mode_rows = [row for row in rows if row["mode"] == mode]
    selected = []
    calibration = []
    alpha_counts: Counter[str] = Counter()
    for fold in range(folds):
        train = [
            row
            for row in mode_rows
            if fold_for(row["question_id"], folds) != fold
        ]
        test = [
            row
            for row in mode_rows
            if fold_for(row["question_id"], folds) == fold
        ]
        candidates = []
        for alpha in sorted({row["alpha"] for row in train}):
            chosen = [row for row in train if row["alpha"] == alpha]
            candidates.append(
                (
                    -statistics.fmean(
                        row["evidence_recall"] for row in chosen
                    ),
                    statistics.fmean(row["latency_ms"] for row in chosen),
                    alpha,
                )
            )
        best = min(candidates)
        alpha = best[2]
        alpha_counts[f"{alpha:.2f}"] += 1
        held_out = [row for row in test if row["alpha"] == alpha]
        selected.extend(held_out)
        calibration.append(
            {
                "fold": fold,
                "train_questions": len(train)
                // len({row["alpha"] for row in train}),
                "test_questions": len(held_out),
                "selected_alpha": alpha,
                "training_mean_evidence_recall": -best[0],
                "held_out_mean_evidence_recall": statistics.fmean(
                    row["evidence_recall"] for row in held_out
                ),
            }
        )
    baselines = {}
    for name, alpha in {"equal_rrf": 0.5, "bm25": 1.0}.items():
        baseline = [
            row for row in mode_rows if row["alpha"] == alpha
        ]
        baselines[name] = {
            "alpha_bm25": alpha,
            "metrics": row_metrics(baseline),
            "selected_minus_baseline": paired_bootstrap(
                selected,
                baseline,
                resamples,
                seed,
            ),
        }
    type_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        type_groups[row["question_type"]].append(row)
    return {
        "mode": mode,
        "protocol": (
            "Five deterministic question-id folds. Alpha is selected on four "
            "folds by mean evidence recall and evaluated on the fifth. Each "
            "question has its own LongMemEval history."
        ),
        "selected_alpha_counts": dict(sorted(alpha_counts.items())),
        "calibration": calibration,
        "out_of_fold_metrics": row_metrics(selected),
        "out_of_fold_question_types": [
            {
                "question_type": question_type,
                **row_metrics(chosen),
            }
            for question_type, chosen in sorted(type_groups.items())
        ],
        "baselines": baselines,
    }


def summarize_index_build(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "questions": len(rows),
        "mean_sessions": statistics.fmean(
            row["sessions"] for row in rows
        ),
        "mean_bm25_build_ms": statistics.fmean(
            row["bm25_build_ms"] for row in rows
        ),
        "mean_dense_build_ms": statistics.fmean(
            row["dense_build_ms"] for row in rows
        ),
    }


def make_payload(
    args: argparse.Namespace,
    source_sha256: str,
    rows: list[dict[str, Any]],
    index_build_rows: list[dict[str, Any]],
    completed: list[str],
    expected: int,
) -> dict[str, Any]:
    return {
        "manifest": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": "LongMemEval-S cleaned",
            "source_file": str(args.dataset),
            "source_file_sha256": source_sha256,
            "embedding_model": "all-MiniLM-L6-v2 local ONNX",
            "alphas_bm25": args.alphas,
            "top_k": args.top_k,
            "ranking_depth": args.ranking_depth,
            "word_budget": args.word_budget,
            "folds": args.folds,
            "completed_questions": completed,
            "expected_answerable_questions": expected,
            "complete": len(completed) == expected,
            "skipped_abstention_questions": 30,
            "metric_scope": (
                "Evidence-session retrieval on answerable questions only. "
                "Weighted RRF compares fixed top-5 sessions and a shared "
                "8,000-word budget. MiniLM truncates candidate sessions to 256 "
                "tokens. This is not answer generation or abstention quality."
            ),
        },
        "summaries": summarize(rows) if rows else [],
        "question_type_summaries": summarize_types(rows) if rows else [],
        "cross_validation": {
            mode: cross_validate(
                rows,
                mode,
                args.folds,
                args.bootstrap_resamples,
                args.bootstrap_seed,
            )
            for mode in MODES
        }
        if len(completed) == expected
        else None,
        "index_build_summary": (
            summarize_index_build(index_build_rows)
            if index_build_rows
            else None
        ),
        "index_build_rows": index_build_rows,
        "rows": rows,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.alphas or any(
        alpha < 0 or alpha > 1 for alpha in args.alphas
    ):
        raise ValueError("Alphas must be between zero and one")
    dataset = json.loads(args.dataset.read_text())
    answerable = [
        item for item in dataset if "_abs" not in item["question_id"]
    ]
    source_sha256 = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    rows: list[dict[str, Any]] = []
    index_build_rows: list[dict[str, Any]] = []
    completed: list[str] = []
    if args.output.exists():
        existing = json.loads(args.output.read_text())
        manifest = existing["manifest"]
        expected_protocol = {
            "source_file_sha256": source_sha256,
            "alphas_bm25": args.alphas,
            "top_k": args.top_k,
            "ranking_depth": args.ranking_depth,
            "word_budget": args.word_budget,
            "folds": args.folds,
        }
        if any(
            manifest[key] != value
            for key, value in expected_protocol.items()
        ):
            raise ValueError("Output checkpoint does not match protocol")
        rows = existing["rows"]
        index_build_rows = existing["index_build_rows"]
        completed = list(manifest["completed_questions"])

    encoder = MiniLm(args.model_dir)
    for index, item in enumerate(answerable, start=1):
        if item["question_id"] in completed:
            continue
        candidates = [
            {"id": session_id, "text": session_text(session)}
            for session_id, session in zip(
                item["haystack_session_ids"],
                item["haystack_sessions"],
                strict=True,
            )
        ]
        started = time.perf_counter()
        bm25 = Bm25(candidates)
        bm25_build_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        dense = DenseIndex(candidates, encoder)
        dense_build_ms = (time.perf_counter() - started) * 1000
        index_build_rows.append(
            {
                "question_id": item["question_id"],
                "sessions": len(candidates),
                "bm25_build_ms": bm25_build_ms,
                "dense_build_ms": dense_build_ms,
            }
        )
        started = time.perf_counter()
        bm25_rows = bm25.search(
            item["question"],
            args.ranking_depth * 2,
        )
        bm25_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        dense_rows = dense.search(
            item["question"],
            args.ranking_depth * 2,
        )
        dense_ms = (time.perf_counter() - started) * 1000
        gold = set(item["answer_session_ids"])
        for alpha in args.alphas:
            started = time.perf_counter()
            ranking = weighted_rrf(
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
            for mode in MODES:
                if mode == "top5_sessions":
                    selected, words = select_top_items(
                        ranking,
                        args.top_k,
                    )
                else:
                    selected, words = select_word_budget(
                        ranking,
                        args.word_budget,
                    )
                recall, mrr = score(gold, selected, ranking)
                rows.append(
                    {
                        "question_id": item["question_id"],
                        "question_type": item["question_type"],
                        "mode": mode,
                        "alpha": alpha,
                        "evidence_recall": recall,
                        "mrr": mrr,
                        "context_words": words,
                        "selected_sessions": len(selected),
                        "latency_ms": latency_ms,
                    }
                )
        completed.append(item["question_id"])
        if index % 10 == 0 or index == len(answerable):
            write_result(
                args.output,
                make_payload(
                    args,
                    source_sha256,
                    rows,
                    index_build_rows,
                    completed,
                    len(answerable),
                ),
            )
            print(
                f"questions {len(completed)}/{len(answerable)}",
                flush=True,
            )
    return make_payload(
        args,
        source_sha256,
        rows,
        index_build_rows,
        completed,
        len(answerable),
    )


if __name__ == "__main__":
    arguments = parse_args()
    result = run(arguments)
    write_result(arguments.output, result)
    print(arguments.output)
