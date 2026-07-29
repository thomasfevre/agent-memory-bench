#!/usr/bin/env python3
"""Evaluate retrieval-based in-context classification on MemoryAgentBench TTL."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from benchmark import Bm25
from graph_benchmark_common import write_result


EXAMPLE_PATTERN = re.compile(r"(?s)\s*(.*?)\nlabel:\s*([^\s]+)\s*$")
DEFAULT_CUTOFFS = (1, 3, 5, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=DEFAULT_CUTOFFS)
    return parser.parse_args()


def parse_labeled_examples(context: str) -> list[dict[str, Any]]:
    examples = []
    for index, block in enumerate(context.split("\n\n")):
        match = EXAMPLE_PATTERN.fullmatch(block)
        if not match:
            continue
        examples.append(
            {
                "id": f"example-{index}",
                "text": match.group(1).strip(),
                "label": match.group(2).strip(),
            }
        )
    return examples


def weighted_label_vote(ranked: list[dict[str, Any]]) -> str | None:
    if not ranked:
        return None
    scores: defaultdict[str, float] = defaultdict(float)
    best_rank: dict[str, int] = {}
    for rank, item in enumerate(ranked, start=1):
        label = str(item["label"])
        scores[label] += float(item.get("_score", 0.0))
        best_rank.setdefault(label, rank)
    return min(scores, key=lambda label: (-scores[label], best_rank[label], label))


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_rows = pq.read_table(args.parquet).to_pylist()
    cutoffs = sorted(set(args.cutoffs))
    max_cutoff = max(cutoffs)
    source_summaries = []
    result_rows = []
    skipped_sources = []

    for sample_index, sample in enumerate(dataset_rows):
        source = sample.get("metadata", {}).get("source", f"row-{sample_index}")
        examples = parse_labeled_examples(sample["context"])
        if not examples:
            skipped_sources.append(
                {
                    "source": source,
                    "reason": "context is not a label-per-example ICL corpus",
                }
            )
            continue

        index = Bm25(examples)
        source_rows = []
        for question_index, (question, answers) in enumerate(
            zip(sample["questions"], sample["answers"], strict=True)
        ):
            gold_labels = {str(answer) for answer in answers}
            started = time.perf_counter()
            ranked = index.search(question, max_cutoff)
            latency_ms = (time.perf_counter() - started) * 1000
            predictions = {
                str(cutoff): weighted_label_vote(ranked[:cutoff])
                for cutoff in cutoffs
            }
            row = {
                "source": source,
                "question_index": question_index,
                "question": question,
                "gold_labels": sorted(gold_labels),
                "predictions": predictions,
                "gold_label_retrieved": {
                    str(cutoff): any(
                        str(item["label"]) in gold_labels
                        for item in ranked[:cutoff]
                    )
                    for cutoff in cutoffs
                },
                "latency_ms": latency_ms,
            }
            source_rows.append(row)
            result_rows.append(row)

        source_summaries.append(
            {
                "source": source,
                "sample_index": sample_index,
                "examples": len(examples),
                "questions": len(source_rows),
                "accuracy": {
                    str(cutoff): statistics.fmean(
                        float(
                            row["predictions"][str(cutoff)]
                            in row["gold_labels"]
                        )
                        for row in source_rows
                    )
                    for cutoff in cutoffs
                },
                "gold_label_retrieval": {
                    str(cutoff): statistics.fmean(
                        float(row["gold_label_retrieved"][str(cutoff)])
                        for row in source_rows
                    )
                    for cutoff in cutoffs
                },
                "mean_latency_ms": statistics.fmean(
                    row["latency_ms"] for row in source_rows
                ),
            }
        )

    return {
        "manifest": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": "ai-hyz/MemoryAgentBench",
            "split": "Test_Time_Learning",
            "source_file": str(args.parquet),
            "source_file_sha256": hashlib.sha256(args.parquet.read_bytes()).hexdigest(),
            "cutoffs": cutoffs,
            "evaluated_sources": len(source_summaries),
            "evaluated_questions": len(result_rows),
            "metric_scope": (
                "Deterministic BM25 nearest-example classification with weighted "
                "label voting; not the benchmark's official LLM ICL score."
            ),
        },
        "sources": source_summaries,
        "skipped_sources": skipped_sources,
        "rows": result_rows,
    }


if __name__ == "__main__":
    arguments = parse_args()
    write_result(arguments.output, run(arguments))
    print(arguments.output)
