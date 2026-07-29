#!/usr/bin/env python3
"""Evaluate LoCoMo window-size sensitivity under one context budget."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark import Bm25, DenseIndex, MiniLm, MODEL_DIR
from graph_benchmark_common import write_result
from locomo_granularity import (
    STRATEGIES,
    build_representations,
    materialize_word_budget,
    rankings,
    score_selection,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument(
        "--window-sizes",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16],
    )
    parser.add_argument("--word-budget", type=int, default=500)
    parser.add_argument("--ranking-depth", type=int, default=20)
    return parser.parse_args()


def build_grid_views(
    sample: dict[str, Any],
    window_sizes: list[int],
) -> dict[str, list[dict[str, Any]]]:
    base = build_representations(sample, 4, 2)
    views = {
        "turn": base["turn"],
        "session": base["session"],
    }
    for size in window_sizes:
        generated = build_representations(
            sample,
            size,
            max(1, size // 2),
        )
        views[f"window{size}"] = generated["window4"]
    return views


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in rows:
        groups[(row["representation"], row["strategy"])].append(row)
    return [
        {
            "representation": representation,
            "strategy": strategy,
            "questions": len(selected),
            "mean_evidence_recall": statistics.fmean(
                row["evidence_recall"] for row in selected
            ),
            "mean_context_precision": statistics.fmean(
                row["context_precision"] for row in selected
            ),
            "mean_context_words": statistics.fmean(
                row["context_words"] for row in selected
            ),
            "mean_context_turns": statistics.fmean(
                row["context_turns"] for row in selected
            ),
            "mean_latency_ms": statistics.fmean(
                row["latency_ms"] for row in selected
            ),
        }
        for (representation, strategy), selected in sorted(groups.items())
    ]


def compare_top_configurations(
    summaries: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    samples: list[str],
    resamples: int = 20_000,
    seed: int = 11,
) -> dict[str, Any] | None:
    if len(summaries) < 2 or not samples:
        return None
    ranked = sorted(
        summaries,
        key=lambda item: (
            -item["mean_evidence_recall"],
            item["representation"],
            item["strategy"],
        ),
    )
    best, runner_up = ranked[:2]

    def matches(row: dict[str, Any], config: dict[str, Any]) -> bool:
        return (
            row["representation"] == config["representation"]
            and row["strategy"] == config["strategy"]
        )

    differences = []
    for sample_id in samples:
        best_values = [
            row["evidence_recall"]
            for row in rows
            if row["sample_id"] == sample_id and matches(row, best)
        ]
        runner_values = [
            row["evidence_recall"]
            for row in rows
            if row["sample_id"] == sample_id and matches(row, runner_up)
        ]
        if not best_values or not runner_values:
            continue
        differences.append(
            statistics.fmean(best_values) - statistics.fmean(runner_values)
        )
    if not differences:
        return None
    generator = random.Random(seed)
    bootstrap = sorted(
        statistics.fmean(
            generator.choice(differences) for _ in differences
        )
        for _ in range(resamples)
    )
    return {
        "best": {
            "representation": best["representation"],
            "strategy": best["strategy"],
            "mean_evidence_recall": best["mean_evidence_recall"],
        },
        "runner_up": {
            "representation": runner_up["representation"],
            "strategy": runner_up["strategy"],
            "mean_evidence_recall": runner_up["mean_evidence_recall"],
        },
        "conversation_weighted_mean_difference": statistics.fmean(differences),
        "best_wins": sum(value > 0 for value in differences),
        "runner_up_wins": sum(value < 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        "bootstrap_95_interval": [
            bootstrap[int(0.025 * resamples)],
            bootstrap[int(0.975 * resamples) - 1],
        ],
    }


def make_payload(
    args: argparse.Namespace,
    source_sha256: str,
    rows: list[dict[str, Any]],
    completed_samples: list[str],
    expected_samples: int,
) -> dict[str, Any]:
    summaries = summarize(rows)
    return {
        "manifest": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": "LoCoMo",
            "source_file": str(args.dataset),
            "source_file_sha256": source_sha256,
            "embedding_model": "all-MiniLM-L6-v2 local ONNX",
            "window_sizes": args.window_sizes,
            "window_stride_rule": "half window, rounded down with minimum one",
            "representations": [
                "turn",
                *[f"window{size}" for size in args.window_sizes],
                "session",
            ],
            "strategies": list(STRATEGIES),
            "word_budget": args.word_budget,
            "ranking_depth": args.ranking_depth,
            "completed_samples": completed_samples,
            "expected_samples": expected_samples,
            "complete": len(completed_samples) == expected_samples,
            "metric_scope": (
                "LoCoMo evidence retrieval under a shared 500-word budget. "
                "Windows use half-window overlap and MiniLM truncates each "
                "candidate to its configured 256-token encoder limit. This is "
                "not answer-generation accuracy."
            ),
        },
        "summaries": summaries,
        "top_configuration_comparison": compare_top_configurations(
            summaries,
            rows,
            completed_samples,
        ),
        "rows": rows,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset = json.loads(args.dataset.read_text())
    source_sha256 = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    rows: list[dict[str, Any]] = []
    completed_samples: list[str] = []
    if args.output.exists():
        existing = json.loads(args.output.read_text())
        manifest = existing["manifest"]
        expected = {
            "source_file_sha256": source_sha256,
            "window_sizes": args.window_sizes,
            "word_budget": args.word_budget,
            "ranking_depth": args.ranking_depth,
        }
        if any(manifest[key] != value for key, value in expected.items()):
            raise ValueError("Output checkpoint does not match current protocol")
        rows = existing["rows"]
        completed_samples = list(manifest["completed_samples"])

    encoder = MiniLm(args.model_dir)
    for sample in dataset:
        sample_id = sample["sample_id"]
        if sample_id in completed_samples:
            continue
        views = build_grid_views(sample, args.window_sizes)
        indexes = {
            name: (
                Bm25(candidates),
                DenseIndex(candidates, encoder),
            )
            for name, candidates in views.items()
        }
        for question_index, question in enumerate(sample["qa"]):
            gold = set(question.get("evidence", []))
            if not gold:
                continue
            for representation, (bm25, dense) in indexes.items():
                ranked, latencies = rankings(
                    question["question"],
                    bm25,
                    dense,
                    args.ranking_depth,
                )
                for strategy in STRATEGIES:
                    selected, words = materialize_word_budget(
                        ranked[strategy],
                        args.word_budget,
                    )
                    recall, precision = score_selection(gold, selected)
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "question_id": f"{sample_id}:{question_index}",
                            "question_type": str(question["category"]),
                            "representation": representation,
                            "strategy": strategy,
                            "evidence_recall": recall,
                            "context_precision": precision,
                            "context_words": words,
                            "context_turns": len(selected),
                            "latency_ms": latencies[strategy],
                        }
                    )
        completed_samples.append(sample_id)
        write_result(
            args.output,
            make_payload(
                args,
                source_sha256,
                rows,
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
        completed_samples,
        len(dataset),
    )


if __name__ == "__main__":
    arguments = parse_args()
    result = run(arguments)
    write_result(arguments.output, result)
    print(arguments.output)
