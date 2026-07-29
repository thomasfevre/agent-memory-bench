#!/usr/bin/env python3
"""Evaluate document retrieval on MemoryAgentBench RULER QA samples."""

from __future__ import annotations

import argparse
import hashlib
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from benchmark import Bm25
from graph_benchmark_common import write_result
from memoryagentbench_conflict_retrieval import contains_answer, first_answer_rank


DOCUMENT_SPLIT = re.compile(r"(?m)^Document\s+(\d+):\s*$")
DEFAULT_CUTOFFS = (1, 5, 10, 20, 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=DEFAULT_CUTOFFS)
    return parser.parse_args()


def parse_documents(context: str) -> list[dict[str, str]]:
    parts = DOCUMENT_SPLIT.split(context)
    return [
        {"id": f"document-{parts[index]}", "text": parts[index + 1].strip()}
        for index in range(1, len(parts), 2)
        if parts[index + 1].strip()
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_rows = pq.read_table(args.parquet).to_pylist()
    cutoffs = sorted(set(args.cutoffs))
    max_cutoff = max(cutoffs)
    rows = []
    source_summaries = []

    for sample_index, sample in enumerate(dataset_rows):
        source = sample.get("metadata", {}).get("source", "")
        if not source.startswith("ruler_"):
            continue
        documents = parse_documents(sample["context"])
        index = Bm25(documents)
        source_rows = []
        for question_index, (question, answers) in enumerate(
            zip(sample["questions"], sample["answers"], strict=True)
        ):
            started = time.perf_counter()
            ranked = index.search(question, max_cutoff)
            latency_ms = (time.perf_counter() - started) * 1000
            rank = first_answer_rank(ranked, answers)
            row = {
                "source": source,
                "question_index": question_index,
                "question": question,
                "answers": answers,
                "answer_present_in_full_context": contains_answer(
                    sample["context"],
                    answers,
                ),
                "first_answer_rank": rank,
                "latency_ms": latency_ms,
            }
            rows.append(row)
            source_rows.append(row)

        source_summaries.append(
            {
                "source": source,
                "sample_index": sample_index,
                "context_characters": len(sample["context"]),
                "documents": len(documents),
                "questions": len(source_rows),
                "answer_present_in_full_context": statistics.fmean(
                    float(row["answer_present_in_full_context"])
                    for row in source_rows
                ),
                "hit_at_k": {
                    str(cutoff): statistics.fmean(
                        float(
                            row["first_answer_rank"] is not None
                            and int(row["first_answer_rank"]) <= cutoff
                        )
                        for row in source_rows
                    )
                    for cutoff in cutoffs
                },
                "mean_reciprocal_rank": statistics.fmean(
                    1.0 / int(row["first_answer_rank"])
                    if row["first_answer_rank"] is not None
                    else 0.0
                    for row in source_rows
                ),
                "mean_latency_ms": statistics.fmean(
                    row["latency_ms"] for row in source_rows
                ),
            }
        )

    return {
        "manifest": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": "ai-hyz/MemoryAgentBench",
            "split": "Accurate_Retrieval",
            "subset": "RULER QA",
            "source_file": str(args.parquet),
            "source_file_sha256": hashlib.sha256(args.parquet.read_bytes()).hexdigest(),
            "cutoffs": cutoffs,
            "sources": len(source_summaries),
            "questions": len(rows),
            "metric_scope": (
                "Deterministic answer-string document evidence coverage before "
                "generation; not the benchmark's official end-to-end score."
            ),
        },
        "sources": source_summaries,
        "rows": rows,
    }


if __name__ == "__main__":
    arguments = parse_args()
    write_result(arguments.output, run(arguments))
    print(arguments.output)
