#!/usr/bin/env python3
"""Compare DetectiveQA query construction before lexical multiple-choice reading."""

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
from memoryagentbench_eventqa_query_ablation import chunk_context, predict_option


TARGET_PATTERN = re.compile(
    r"Now Answer the Question:\s*(.*?)\s*Output:",
    re.DOTALL,
)
OPTION_PATTERN = re.compile(r"(?m)^([A-D])\.\s*(.+?)\s*$")
STRATEGIES = (
    "full_question_top5",
    "target_with_options_top5",
    "target_stem_top5",
    "target_stem_top20",
    "oracle_answer_top5",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-words", type=int, default=120)
    parser.add_argument("--stride-words", type=int, default=60)
    return parser.parse_args()


def parse_target(question: str) -> tuple[str, str, list[str]]:
    match = TARGET_PATTERN.search(question)
    if not match:
        raise ValueError("DetectiveQA target question was not found")
    target = match.group(1).strip()
    option_matches = OPTION_PATTERN.findall(target)
    if len(option_matches) != 4:
        raise ValueError("DetectiveQA expected exactly four options")
    options = [f"{label}. {text.strip()}" for label, text in option_matches]
    stem = OPTION_PATTERN.split(target, maxsplit=1)[0].strip()
    return target, stem, options


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_rows = pq.read_table(args.parquet).to_pylist()
    rows = []
    source_summaries = []

    for sample_index, sample in enumerate(dataset_rows):
        source = sample.get("metadata", {}).get("source", "")
        if source != "detective_qa":
            continue
        chunks = chunk_context(
            sample["context"],
            window_words=args.window_words,
            stride_words=args.stride_words,
        )
        index = Bm25(chunks)
        source_rows = []

        for question_index, (question, answers) in enumerate(
            zip(sample["questions"], sample["answers"], strict=True)
        ):
            target, stem, options = parse_target(question)
            gold = answers[0]
            started = time.perf_counter()
            retrieved = {
                "full_question_top5": index.search(question, 5),
                "target_with_options_top5": index.search(target, 5),
                "target_stem_top5": index.search(stem, 5),
                "target_stem_top20": index.search(stem, 20),
                "oracle_answer_top5": index.search(gold, 5),
            }
            latency_ms = (time.perf_counter() - started) * 1000
            predictions = {
                strategy: predict_option(
                    options,
                    " ".join(item["text"] for item in evidence),
                )
                for strategy, evidence in retrieved.items()
            }
            row = {
                "sample_index": sample_index,
                "question_index": question_index,
                "gold": gold,
                "predictions": predictions,
                "correct": {
                    strategy: prediction == gold
                    for strategy, prediction in predictions.items()
                },
                "retrieval_latency_ms": latency_ms,
            }
            rows.append(row)
            source_rows.append(row)

        source_summaries.append(
            {
                "sample_index": sample_index,
                "context_characters": len(sample["context"]),
                "chunks": len(chunks),
                "questions": len(source_rows),
                "accuracy": {
                    strategy: statistics.fmean(
                        float(row["correct"][strategy]) for row in source_rows
                    )
                    for strategy in STRATEGIES
                },
                "mean_retrieval_latency_ms_all_strategies": statistics.fmean(
                    row["retrieval_latency_ms"] for row in source_rows
                ),
            }
        )

    return {
        "manifest": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": "ai-hyz/MemoryAgentBench",
            "split": "Long_Range_Understanding",
            "subset": "detective_qa",
            "source_file": str(args.parquet),
            "source_file_sha256": hashlib.sha256(args.parquet.read_bytes()).hexdigest(),
            "window_words": args.window_words,
            "stride_words": args.stride_words,
            "sources": len(source_summaries),
            "questions": len(rows),
            "strategies": list(STRATEGIES),
            "metric_scope": (
                "Deterministic lexical multiple-choice ablation over retrieved "
                "windows. Oracle-answer retrieval is an upper bound, not an "
                "official DetectiveQA score."
            ),
        },
        "summary": {
            "accuracy": {
                strategy: statistics.fmean(
                    float(row["correct"][strategy]) for row in rows
                )
                for strategy in STRATEGIES
            },
            "mean_retrieval_latency_ms_all_strategies": statistics.fmean(
                row["retrieval_latency_ms"] for row in rows
            ),
        },
        "sources": source_summaries,
        "rows": rows,
    }


if __name__ == "__main__":
    arguments = parse_args()
    write_result(arguments.output, run(arguments))
    print(arguments.output)
