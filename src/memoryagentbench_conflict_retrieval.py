#!/usr/bin/env python3
"""Measure deterministic evidence coverage across MemoryAgentBench Conflict variants."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from benchmark import Bm25
from graph_benchmark_common import write_result
from memoryagentbench_slice import normalize_answer, split_facts


DEFAULT_CUTOFFS = (1, 5, 10, 20, 50, 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=DEFAULT_CUTOFFS)
    parser.add_argument("--iterative-cutoffs", type=int, nargs="+", default=[5, 20])
    return parser.parse_args()


def contains_answer(text: str, answers: list[str]) -> bool:
    normalized_text = normalize_answer(text)
    return any(
        normalized_answer and normalized_answer in normalized_text
        for answer in answers
        if (normalized_answer := normalize_answer(answer))
    )


def first_answer_rank(
    ranked: list[dict[str, Any]],
    answers: list[str],
) -> int | None:
    for rank, item in enumerate(ranked, start=1):
        if contains_answer(item["text"], answers):
            return rank
    return None


def greedy_expand(
    index: Bm25,
    question: str,
    *,
    budget: int,
    candidate_pool: int = 100,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    query = question
    while len(selected) < budget:
        ranked = index.search(query, min(len(index.candidates), candidate_pool))
        next_item = next((item for item in ranked if item["id"] not in seen), None)
        if next_item is None:
            break
        selected.append(next_item)
        seen.add(next_item["id"])
        query = f"{question} {next_item['text']}"
    return selected


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    cutoffs: list[int],
    iterative_cutoffs: list[int],
) -> dict[str, Any]:
    found_ranks = [
        int(row["first_answer_rank"])
        for row in rows
        if row["first_answer_rank"] is not None
    ]
    return {
        "questions": len(rows),
        "answer_present_in_full_context": statistics.fmean(
            float(row["answer_present_in_full_context"]) for row in rows
        ),
        "bm25_hit_at_k": {
            str(cutoff): statistics.fmean(
                float(
                    row["first_answer_rank"] is not None
                    and int(row["first_answer_rank"]) <= cutoff
                )
                for row in rows
            )
            for cutoff in cutoffs
        },
        "bm25_mean_reciprocal_rank": statistics.fmean(
            1.0 / int(row["first_answer_rank"])
            if row["first_answer_rank"] is not None
            else 0.0
            for row in rows
        ),
        "bm25_answer_rank_median_when_found": (
            statistics.median(found_ranks) if found_ranks else None
        ),
        "bm25_no_answer_evidence_count": sum(
            row["first_answer_rank"] is None for row in rows
        ),
        "greedy_hit_at_k": {
            str(cutoff): statistics.fmean(
                float(
                    row["greedy_first_answer_rank"] is not None
                    and int(row["greedy_first_answer_rank"]) <= cutoff
                )
                for row in rows
            )
            for cutoff in iterative_cutoffs
        },
        "greedy_mean_reciprocal_rank": statistics.fmean(
            1.0 / int(row["greedy_first_answer_rank"])
            if row["greedy_first_answer_rank"] is not None
            else 0.0
            for row in rows
        ),
        "mean_retrieval_latency_ms": statistics.fmean(
            row["retrieval_latency_ms"] for row in rows
        ),
        "mean_greedy_latency_ms": statistics.fmean(
            row["greedy_latency_ms"] for row in rows
        ),
        "mean_top20_words": statistics.fmean(row["top20_words"] for row in rows),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_rows = pq.read_table(args.parquet).to_pylist()
    cutoffs = sorted(set(args.cutoffs))
    iterative_cutoffs = sorted(set(args.iterative_cutoffs))
    max_cutoff = max(cutoffs)
    max_iterative_cutoff = max(iterative_cutoffs)
    result_rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    indexes: dict[str, tuple[list[dict[str, str]], Bm25]] = {}

    for sample_index, sample in enumerate(dataset_rows):
        source = sample.get("metadata", {}).get("source", f"row-{sample_index}")
        context = sample["context"]
        context_hash = hashlib.sha256(context.encode()).hexdigest()
        if context_hash not in indexes:
            facts = split_facts(context)
            indexes[context_hash] = (facts, Bm25(facts))
        facts, bm25 = indexes[context_hash]
        source_rows = []

        for question_index, (question, answers) in enumerate(
            zip(sample["questions"], sample["answers"], strict=True)
        ):
            started = time.perf_counter()
            ranked = bm25.search(question, max_cutoff)
            latency_ms = (time.perf_counter() - started) * 1000
            rank = first_answer_rank(ranked, answers)
            greedy_started = time.perf_counter()
            greedy_ranked = greedy_expand(
                bm25,
                question,
                budget=max_iterative_cutoff,
            )
            greedy_latency_ms = (time.perf_counter() - greedy_started) * 1000
            row = {
                "source": source,
                "question_index": question_index,
                "question": question,
                "answers": answers,
                "answer_present_in_full_context": contains_answer(context, answers),
                "first_answer_rank": rank,
                "greedy_first_answer_rank": first_answer_rank(
                    greedy_ranked,
                    answers,
                ),
                "retrieved_nonzero_candidates": len(ranked),
                "retrieval_latency_ms": latency_ms,
                "greedy_latency_ms": greedy_latency_ms,
                "top20_words": sum(
                    len(item["text"].split()) for item in ranked[:20]
                ),
            }
            result_rows.append(row)
            source_rows.append(row)

        summary = summarize_rows(
            source_rows,
            cutoffs=cutoffs,
            iterative_cutoffs=iterative_cutoffs,
        )
        summary.update(
            {
                "source": source,
                "sample_index": sample_index,
                "context_sha256": context_hash,
                "context_characters": len(context),
                "context_words": len(context.split()),
                "facts": len(facts),
                "top20_compression_ratio_words": (
                    summary["mean_top20_words"] / max(len(context.split()), 1)
                ),
            }
        )
        sources.append(summary)

    return {
        "manifest": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": "ai-hyz/MemoryAgentBench",
            "split": "Conflict_Resolution",
            "source_file": str(args.parquet),
            "source_file_sha256": hashlib.sha256(args.parquet.read_bytes()).hexdigest(),
            "sources": len(sources),
            "questions": len(result_rows),
            "cutoffs": cutoffs,
            "iterative_cutoffs": iterative_cutoffs,
            "metric_scope": (
                "Deterministic answer-string evidence coverage before generation; "
                "not the benchmark's official end-to-end score."
            ),
        },
        "sources": sources,
        "rows": result_rows,
    }


if __name__ == "__main__":
    arguments = parse_args()
    write_result(arguments.output, run(arguments))
    print(arguments.output)
