#!/usr/bin/env python3
"""Measure fixed-budget context coverage for MemoryAgentBench summarization."""

from __future__ import annotations

import argparse
import hashlib
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from benchmark import Bm25, tokenize
from graph_benchmark_common import write_result
from memoryagentbench_eventqa_query_ablation import chunk_context


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "her",
    "his",
    "in",
    "is",
    "it",
    "of",
    "on",
    "she",
    "that",
    "the",
    "their",
    "they",
    "to",
    "was",
    "with",
}
STRATEGIES = (
    "head",
    "uniform",
    "prompt_bm25",
    "oracle_keypoint_rrf",
    "full_context",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-words", type=int, default=512)
    parser.add_argument("--budget-chunks", type=int, default=20)
    return parser.parse_args()


def content_terms(text: str) -> set[str]:
    return {term for term in tokenize(text) if term not in STOPWORDS}


def keypoint_support(keypoint: str, evidence: str) -> float:
    keypoint_terms = content_terms(keypoint)
    if not keypoint_terms:
        return 0.0
    return len(keypoint_terms & content_terms(evidence)) / len(keypoint_terms)


def select_uniform(
    chunks: list[dict[str, Any]],
    budget: int,
) -> list[dict[str, Any]]:
    if budget <= 0:
        return []
    if len(chunks) <= budget:
        return chunks
    if budget == 1:
        return [chunks[0]]
    indices = {
        round(index * (len(chunks) - 1) / (budget - 1))
        for index in range(budget)
    }
    return [chunks[index] for index in sorted(indices)]


def select_oracle_rrf(
    index: Bm25,
    keypoints: list[str],
    budget: int,
) -> list[dict[str, Any]]:
    scores: defaultdict[str, float] = defaultdict(float)
    records = {}
    for keypoint in keypoints:
        for rank, item in enumerate(index.search(keypoint, 3), start=1):
            scores[item["id"]] += 1.0 / (60 + rank)
            records[item["id"]] = item
    selected_ids = sorted(scores, key=lambda item_id: (-scores[item_id], item_id))[
        :budget
    ]
    return [records[item_id] for item_id in selected_ids]


def summarize_selection(
    keypoints: list[str],
    selected: list[dict[str, Any]],
) -> dict[str, Any]:
    evidence = " ".join(item["text"] for item in selected)
    scores = [keypoint_support(keypoint, evidence) for keypoint in keypoints]
    if not scores:
        return {
            "selected_chunks": len(selected),
            "selected_words": sum(len(item["text"].split()) for item in selected),
            "mean_keypoint_token_recall": 0.0,
            "keypoints_supported_at_0_5": 0.0,
            "keypoints_supported_at_0_8": 0.0,
        }
    return {
        "selected_chunks": len(selected),
        "selected_words": sum(len(item["text"].split()) for item in selected),
        "mean_keypoint_token_recall": statistics.fmean(scores),
        "keypoints_supported_at_0_5": statistics.fmean(
            float(score >= 0.5) for score in scores
        ),
        "keypoints_supported_at_0_8": statistics.fmean(
            float(score >= 0.8) for score in scores
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_rows = pq.read_table(args.parquet).to_pylist()
    rows = []

    for sample_index, sample in enumerate(dataset_rows):
        if sample.get("metadata", {}).get("source") != "infbench_sum_eng_shots2":
            continue
        keypoints = sample.get("metadata", {}).get("keypoints") or []
        chunks = chunk_context(
            sample["context"],
            window_words=args.chunk_words,
            stride_words=args.chunk_words,
        )
        index = Bm25(chunks)
        started = time.perf_counter()
        selections = {
            "head": chunks[: args.budget_chunks],
            "uniform": select_uniform(chunks, args.budget_chunks),
            "prompt_bm25": index.search(
                sample["questions"][0],
                args.budget_chunks,
            ),
            "oracle_keypoint_rrf": select_oracle_rrf(
                index,
                keypoints,
                args.budget_chunks,
            ),
            "full_context": chunks,
        }
        strategies = {
            strategy: summarize_selection(keypoints, selected)
            for strategy, selected in selections.items()
        }
        rows.append(
            {
                "sample_index": sample_index,
                "context_characters": len(sample["context"]),
                "context_words": len(sample["context"].split()),
                "chunks": len(chunks),
                "keypoints": len(keypoints),
                "strategies": strategies,
                "selection_latency_ms": (time.perf_counter() - started) * 1000,
            }
        )

    summary = {
        strategy: {
            metric: statistics.fmean(
                row["strategies"][strategy][metric] for row in rows
            )
            for metric in (
                "selected_chunks",
                "selected_words",
                "mean_keypoint_token_recall",
                "keypoints_supported_at_0_5",
                "keypoints_supported_at_0_8",
            )
        }
        for strategy in STRATEGIES
    }
    return {
        "manifest": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": "ai-hyz/MemoryAgentBench",
            "split": "Long_Range_Understanding",
            "subset": "infbench_sum_eng_shots2",
            "source_file": str(args.parquet),
            "source_file_sha256": hashlib.sha256(args.parquet.read_bytes()).hexdigest(),
            "documents": len(rows),
            "chunk_words": args.chunk_words,
            "budget_chunks": args.budget_chunks,
            "budget_words_nominal": args.chunk_words * args.budget_chunks,
            "strategies": list(STRATEGIES),
            "metric_scope": (
                "Lexical keypoint coverage under a fixed context budget before "
                "summary generation. Oracle keypoint RRF and full context are "
                "upper bounds, not valid production strategies or official scores."
            ),
        },
        "summary": summary,
        "rows": rows,
    }


if __name__ == "__main__":
    arguments = parse_args()
    write_result(arguments.output, run(arguments))
    print(arguments.output)
