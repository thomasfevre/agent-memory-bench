#!/usr/bin/env python3
"""Compare LoCoMo memory granularity under item and word budgets."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark import (
    Bm25,
    DenseIndex,
    MiniLm,
    MODEL_DIR,
    reciprocal_rank_fusion,
)
from graph_benchmark_common import write_result


REPRESENTATIONS = ("turn", "window4", "session")
STRATEGIES = ("bm25", "dense", "hybrid")
MODES = ("top5_items", "word_budget")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--window-turns", type=int, default=4)
    parser.add_argument("--window-stride", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--word-budget", type=int, default=500)
    parser.add_argument("--ranking-depth", type=int, default=20)
    return parser.parse_args()


def turn_text(turn: dict[str, Any]) -> str:
    return f"{turn['speaker']}: {turn['text']}"


def candidate(
    candidate_id: str,
    segments: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "id": candidate_id,
        "text": "\n".join(segment["text"] for segment in segments),
        "segments": segments,
        "source_ids": [segment["id"] for segment in segments],
    }


def build_representations(
    sample: dict[str, Any],
    window_turns: int,
    window_stride: int,
) -> dict[str, list[dict[str, Any]]]:
    representations: dict[str, list[dict[str, Any]]] = {
        name: [] for name in REPRESENTATIONS
    }
    session_number = 0
    for key, turns in sample["conversation"].items():
        if not key.startswith("session_") or key.endswith("_date_time"):
            continue
        session_number += 1
        segments = [
            {"id": turn["dia_id"], "text": turn_text(turn)}
            for turn in turns
        ]
        representations["turn"].extend(
            candidate(segment["id"], [segment]) for segment in segments
        )
        representations["session"].append(
            candidate(f"session-{session_number}", segments)
        )
        for start in range(0, len(segments), window_stride):
            selected = segments[start : start + window_turns]
            if not selected:
                continue
            representations["window4"].append(
                candidate(
                    f"session-{session_number}-window-{start}",
                    selected,
                )
            )
            if start + window_turns >= len(segments):
                break
    return representations


def rankings(
    query: str,
    bm25: Bm25,
    dense: DenseIndex,
    depth: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, float]]:
    started = time.perf_counter()
    bm25_rows = bm25.search(query, depth * 2)
    bm25_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    dense_rows = dense.search(query, depth * 2)
    dense_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    hybrid_rows = reciprocal_rank_fusion(
        [bm25_rows, dense_rows],
        depth,
        "hybrid",
    )
    hybrid_ms = (time.perf_counter() - started) * 1000
    return (
        {
            "bm25": bm25_rows[:depth],
            "dense": dense_rows[:depth],
            "hybrid": hybrid_rows,
        },
        {
            "bm25": bm25_ms,
            "dense": dense_ms,
            "hybrid": bm25_ms + dense_ms + hybrid_ms,
        },
    )


def materialize_top_items(
    ranked: list[dict[str, Any]],
    top_k: int,
) -> tuple[set[str], int]:
    source_ids: set[str] = set()
    words = 0
    for item in ranked[:top_k]:
        for segment in item["segments"]:
            if segment["id"] in source_ids:
                continue
            source_ids.add(segment["id"])
            words += len(segment["text"].split())
    return source_ids, words


def materialize_word_budget(
    ranked: list[dict[str, Any]],
    word_budget: int,
) -> tuple[set[str], int]:
    source_ids: set[str] = set()
    words = 0
    for item in ranked:
        for segment in item["segments"]:
            if segment["id"] in source_ids:
                continue
            segment_words = len(segment["text"].split())
            if words + segment_words > word_budget:
                continue
            source_ids.add(segment["id"])
            words += segment_words
        if words >= word_budget:
            break
    return source_ids, words


def score_selection(
    gold: set[str],
    selected: set[str],
) -> tuple[float, float]:
    overlap = len(gold & selected)
    return (
        overlap / len(gold) if gold else 0.0,
        overlap / len(selected) if selected else 0.0,
    )


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: defaultdict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in rows:
        groups[(row["representation"], row["strategy"], row["mode"])].append(
            row
        )
    return [
        {
            "representation": representation,
            "strategy": strategy,
            "mode": mode,
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
        for (representation, strategy, mode), selected in sorted(groups.items())
    ]


def payload(
    args: argparse.Namespace,
    source_sha256: str,
    rows: list[dict[str, Any]],
    completed_samples: list[str],
    expected_samples: int,
) -> dict[str, Any]:
    return {
        "manifest": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": "LoCoMo",
            "source_file": str(args.dataset),
            "source_file_sha256": source_sha256,
            "embedding_model": "all-MiniLM-L6-v2 local ONNX",
            "representations": list(REPRESENTATIONS),
            "strategies": list(STRATEGIES),
            "modes": list(MODES),
            "window_turns": args.window_turns,
            "window_stride": args.window_stride,
            "top_k": args.top_k,
            "word_budget": args.word_budget,
            "ranking_depth": args.ranking_depth,
            "completed_samples": completed_samples,
            "expected_samples": expected_samples,
            "complete": len(completed_samples) == expected_samples,
            "metric_scope": (
                "Within-dataset evidence retrieval at turn, four-turn-window, "
                "and session granularity. Fixed top-k exposes context inflation; "
                "the fixed word budget materializes whole dialogue turns up to "
                "500 words. This is not answer-generation accuracy."
            ),
        },
        "summaries": summarize(rows),
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
            "window_turns": args.window_turns,
            "window_stride": args.window_stride,
            "top_k": args.top_k,
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
        representations = build_representations(
            sample,
            args.window_turns,
            args.window_stride,
        )
        indexes = {
            name: (
                Bm25(candidates),
                DenseIndex(candidates, encoder),
            )
            for name, candidates in representations.items()
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
                    for mode in MODES:
                        if mode == "top5_items":
                            selected, words = materialize_top_items(
                                ranked[strategy],
                                args.top_k,
                            )
                        else:
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
                                "mode": mode,
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
            payload(
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
    return payload(
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
