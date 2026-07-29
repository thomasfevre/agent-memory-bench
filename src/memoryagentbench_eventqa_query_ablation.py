#!/usr/bin/env python3
"""Compare EventQA query construction and sequential-neighborhood retrieval."""

from __future__ import annotations

import argparse
import ast
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


OPTIONS_PATTERN = re.compile(
    r"Below is a list of possible subsequent events:\s*\n"
    r"(\[.*?\])\s*\n\s*Your task",
    re.DOTALL,
)
STRATEGIES = (
    "full_question_top5",
    "previous_event_top5",
    "anchor5_next3",
    "oracle_answer_top5",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-words", type=int, default=120)
    parser.add_argument("--stride-words", type=int, default=60)
    return parser.parse_args()


def parse_options(question: str) -> list[str]:
    match = OPTIONS_PATTERN.search(question)
    if not match:
        raise ValueError("EventQA options were not found in question")
    options = ast.literal_eval(match.group(1))
    if not isinstance(options, list) or not all(
        isinstance(option, str) for option in options
    ):
        raise ValueError("EventQA options must be a list of strings")
    return options


def chunk_context(
    context: str,
    *,
    window_words: int,
    stride_words: int,
) -> list[dict[str, Any]]:
    words = context.split()
    return [
        {
            "id": f"chunk-{position}",
            "text": " ".join(words[position : position + window_words]),
            "position": position,
        }
        for position in range(0, len(words), stride_words)
        if words[position : position + window_words]
    ]


def predict_option(options: list[str], evidence: str) -> str | None:
    option_index = Bm25(
        [
            {"id": f"option-{index}", "text": option}
            for index, option in enumerate(options)
        ]
    )
    ranked = option_index.search(evidence, len(options))
    if not ranked:
        return None
    option_index_value = int(ranked[0]["id"].removeprefix("option-"))
    return options[option_index_value]


def anchor_neighborhood(
    index: Bm25,
    chunks: list[dict[str, Any]],
    query: str,
    *,
    anchors: int,
    following_chunks: int,
    stride_words: int,
) -> list[dict[str, Any]]:
    selected = []
    seen = set()
    for anchor in index.search(query, anchors):
        start = int(anchor["position"]) // stride_words
        for chunk in chunks[start : start + following_chunks]:
            if chunk["id"] in seen:
                continue
            selected.append(chunk)
            seen.add(chunk["id"])
    return selected


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_rows = pq.read_table(args.parquet).to_pylist()
    rows = []
    source_summaries = []

    for sample_index, sample in enumerate(dataset_rows):
        source = sample.get("metadata", {}).get("source", "")
        previous_events = sample.get("metadata", {}).get("previous_events")
        if not source.startswith("eventqa_") or not previous_events:
            continue
        chunks = chunk_context(
            sample["context"],
            window_words=args.window_words,
            stride_words=args.stride_words,
        )
        index = Bm25(chunks)
        source_rows = []

        for question_index, (question, answers, previous_event) in enumerate(
            zip(
                sample["questions"],
                sample["answers"],
                previous_events,
                strict=True,
            )
        ):
            options = parse_options(question)
            gold = answers[0]
            started = time.perf_counter()
            retrieved = {
                "full_question_top5": index.search(question, 5),
                "previous_event_top5": index.search(previous_event, 5),
                "anchor5_next3": anchor_neighborhood(
                    index,
                    chunks,
                    previous_event,
                    anchors=5,
                    following_chunks=3,
                    stride_words=args.stride_words,
                ),
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
                "source": source,
                "sample_index": sample_index,
                "question_index": question_index,
                "gold": gold,
                "predictions": predictions,
                "correct": {
                    strategy: prediction == gold
                    for strategy, prediction in predictions.items()
                },
                "retrieved_chunks": {
                    strategy: [item["id"] for item in evidence]
                    for strategy, evidence in retrieved.items()
                },
                "retrieval_latency_ms": latency_ms,
            }
            rows.append(row)
            source_rows.append(row)

        source_summaries.append(
            {
                "source": source,
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
            "split": "Accurate_Retrieval",
            "subset": "EventQA",
            "source_file": str(args.parquet),
            "source_file_sha256": hashlib.sha256(args.parquet.read_bytes()).hexdigest(),
            "window_words": args.window_words,
            "stride_words": args.stride_words,
            "sources": len(source_summaries),
            "questions": len(rows),
            "strategies": list(STRATEGIES),
            "metric_scope": (
                "Deterministic lexical multiple-choice ablation over retrieved "
                "windows. Oracle-answer retrieval is an upper bound, not a valid "
                "production strategy or official benchmark score."
            ),
        },
        "sources": source_summaries,
        "rows": rows,
    }


if __name__ == "__main__":
    arguments = parse_args()
    write_result(arguments.output, run(arguments))
    print(arguments.output)
