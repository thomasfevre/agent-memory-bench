#!/usr/bin/env python3
"""Measure ground-truth movie evidence retrieval on MemoryAgentBench ReDial."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from benchmark import Bm25
from graph_benchmark_common import write_result
from memoryagentbench_slice import normalize_answer


DIALOGUE_SPLIT = re.compile(r"(?m)^Dialogue\s+(\d+):\s*$")
DEFAULT_CUTOFFS = (1, 5, 10, 20, 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--entity-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=DEFAULT_CUTOFFS)
    return parser.parse_args()


def parse_dialogues(context: str) -> list[dict[str, str]]:
    parts = DIALOGUE_SPLIT.split(context)
    return [
        {"id": f"dialogue-{parts[index]}", "text": parts[index + 1].strip()}
        for index in range(1, len(parts), 2)
        if parts[index + 1].strip()
    ]


def extract_movie_name(entity: str) -> str:
    filename = entity.split("/")[-1]
    cleaned = filename.replace("_", " ").replace("-", " ").replace(">", " ")
    cleaned = re.sub(r"\([^()]*\)", "", cleaned)
    return " ".join(cleaned.split())


def movie_is_mentioned(movie: str, text: str) -> bool:
    normalized_movie = normalize_answer(movie)
    normalized_text = normalize_answer(text)
    return bool(normalized_movie and normalized_movie in normalized_text)


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_rows = pq.read_table(args.parquet).to_pylist()
    sample = next(
        row
        for row in dataset_rows
        if row.get("metadata", {}).get("source") == "recsys_redial_full"
    )
    name_to_id = json.loads(args.entity_map.read_text())
    id_to_name = {
        int(entity_id): extract_movie_name(entity)
        for entity, entity_id in name_to_id.items()
    }
    dialogues = parse_dialogues(sample["context"])
    index = Bm25(dialogues)
    cutoffs = sorted(set(args.cutoffs))
    max_cutoff = max(cutoffs)
    rows = []

    for question_index, (question, answers) in enumerate(
        zip(sample["questions"], sample["answers"], strict=True)
    ):
        gold_ids = [int(answer) for answer in answers]
        gold_movies = [id_to_name[movie_id] for movie_id in gold_ids]
        started = time.perf_counter()
        ranked = index.search(question, max_cutoff)
        latency_ms = (time.perf_counter() - started) * 1000
        recalls = {}
        first_ranks = {}
        for movie in gold_movies:
            first_ranks[movie] = next(
                (
                    rank
                    for rank, dialogue in enumerate(ranked, start=1)
                    if movie_is_mentioned(movie, dialogue["text"])
                ),
                None,
            )
        for cutoff in cutoffs:
            recalls[str(cutoff)] = statistics.fmean(
                float(
                    first_ranks[movie] is not None
                    and int(first_ranks[movie]) <= cutoff
                )
                for movie in gold_movies
            )
        rows.append(
            {
                "question_index": question_index,
                "question": question,
                "gold_ids": gold_ids,
                "gold_movies": gold_movies,
                "gold_movie_present_in_full_context": {
                    movie: movie_is_mentioned(movie, sample["context"])
                    for movie in gold_movies
                },
                "first_dialogue_rank": first_ranks,
                "recall_at_k": recalls,
                "latency_ms": latency_ms,
            }
        )

    return {
        "manifest": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": "ai-hyz/MemoryAgentBench",
            "split": "Test_Time_Learning",
            "subset": "recsys_redial_full",
            "source_file": str(args.parquet),
            "source_file_sha256": hashlib.sha256(args.parquet.read_bytes()).hexdigest(),
            "entity_map": str(args.entity_map),
            "entity_map_sha256": hashlib.sha256(
                args.entity_map.read_bytes()
            ).hexdigest(),
            "entities": len(name_to_id),
            "dialogues": len(dialogues),
            "questions": len(rows),
            "cutoffs": cutoffs,
            "metric_scope": (
                "Ground-truth movie-title evidence recall in BM25-retrieved "
                "dialogues before recommendation generation; not official "
                "recommendation recall."
            ),
        },
        "summary": {
            "gold_movie_present_in_full_context": statistics.fmean(
                float(present)
                for row in rows
                for present in row["gold_movie_present_in_full_context"].values()
            ),
            "recall_at_k": {
                str(cutoff): statistics.fmean(
                    row["recall_at_k"][str(cutoff)] for row in rows
                )
                for cutoff in cutoffs
            },
            "mean_latency_ms": statistics.fmean(row["latency_ms"] for row in rows),
        },
        "rows": rows,
    }


if __name__ == "__main__":
    arguments = parse_args()
    write_result(arguments.output, run(arguments))
    print(arguments.output)
