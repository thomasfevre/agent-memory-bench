#!/usr/bin/env python3
"""Provider-free retrieval evaluation on the public MemGym-DR corpus.

This runner deliberately stops before answer generation. It executes MemGym's
official IR memory managers and official memory-required-fact recall proxy on
the same stratified sample. BM25 can additionally be evaluated on the complete
1,194-row release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


STRATA = {
    "3hop": "3hop_verified.jsonl",
    "4hop": "4hop_paper_run.jsonl",
    "56hop": "56hop_clean.jsonl",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def deterministic_sample(
    rows: list[dict[str, Any]], sample_size: int, seed: int
) -> list[dict[str, Any]]:
    if sample_size <= 0 or sample_size >= len(rows):
        return list(rows)
    indices = sorted(random.Random(seed).sample(range(len(rows)), sample_size))
    return [rows[index] for index in indices]


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        return {
            "n": 0,
            "mean_fact_recall": 0.0,
            "mean_context_words": 0.0,
            "mean_ingestion_seconds": 0.0,
            "mean_query_ms": 0.0,
        }
    return {
        "n": len(rows),
        "mean_fact_recall": statistics.fmean(row["fact_recall"] for row in rows),
        "mean_context_words": statistics.fmean(
            row["context_words"] for row in rows
        ),
        "mean_ingestion_seconds": statistics.fmean(
            row["ingestion_seconds"] for row in rows
        ),
        "mean_query_ms": statistics.fmean(row["query_ms"] for row in rows),
    }


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_strategy(
    raw_rows: list[dict[str, Any]],
    strategy: str,
    top_ks: list[int],
) -> list[dict[str, Any]]:
    from memgym.memory.base import get_memory_model
    from memgym.pipelines.memgym_ir.eval.agent import (
        IRMemoryAdapter,
        _build_visible_docs,
    )
    from memgym.pipelines.memgym_ir.eval.memory import _compute_note_fact_recall
    from memgym.pipelines.memgym_ir.types.schemas import MemGymIRInstance

    output: list[dict[str, Any]] = []
    max_k = max(top_ks)
    manager = None

    for raw in raw_rows:
        instance = MemGymIRInstance(
            **{
                key: value
                for key, value in raw.items()
                if key != "_benchmark_stratum"
            }
        )
        if manager is None:
            manager = get_memory_model(
                strategy,
                question=instance.question,
                top_k=max_k,
            )
        else:
            manager.reset()
            if hasattr(manager, "_question"):
                manager._question = instance.question

        adapter = IRMemoryAdapter(manager, question=instance.question)
        ingestion_started = time.perf_counter()
        for turn_index, turn in enumerate(instance.turns):
            visible_docs, _ = _build_visible_docs(
                instance,
                turn_index,
                instance.eviction_policy,
            )
            adapter.process_turn(turn, visible_docs)
        ingestion_seconds = time.perf_counter() - ingestion_started

        for top_k in top_ks:
            query_started = time.perf_counter()
            context = manager.retrieve_for_question(instance.question, top_k=top_k)
            query_ms = (time.perf_counter() - query_started) * 1000
            output.append(
                {
                    "instance_id": instance.instance_id,
                    "stratum": raw["_benchmark_stratum"],
                    "num_hops": instance.num_hops,
                    "strategy": strategy,
                    "top_k": top_k,
                    "fact_recall": _compute_note_fact_recall(context, instance),
                    "context_words": len(context.split()),
                    "ingestion_seconds": ingestion_seconds,
                    "query_ms": query_ms,
                    "num_memory_required_facts": len(
                        instance.memory_required_facts
                    ),
                }
            )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--memgym-repo", type=Path, required=True)
    parser.add_argument("--sample-per-stratum", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 2, 5, 10])
    parser.add_argument("--bm25-full", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    top_ks = sorted(set(args.top_k))
    if not top_ks or min(top_ks) < 1:
        parser.error("--top-k must contain positive integers")

    per_stratum: dict[str, list[dict[str, Any]]] = {}
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    sample_ids: dict[str, list[str]] = {}
    for offset, (stratum, filename) in enumerate(STRATA.items()):
        rows = load_jsonl(args.dataset_dir / filename)
        counts[stratum] = len(rows)
        sample = deterministic_sample(
            rows, args.sample_per_stratum, args.seed + offset
        )
        tagged_rows = [
            {**row, "_benchmark_stratum": stratum}
            for row in rows
        ]
        tagged_sample = [
            {**row, "_benchmark_stratum": stratum}
            for row in sample
        ]
        per_stratum[stratum] = tagged_rows
        selected.extend(tagged_sample)
        sample_ids[stratum] = [row["instance_id"] for row in sample]

    sample_results: list[dict[str, Any]] = []
    for strategy in ("ir_bm25", "ir_naive_rag"):
        sample_results.extend(run_strategy(selected, strategy, top_ks))

    bm25_full_results: list[dict[str, Any]] = []
    if args.bm25_full:
        all_rows = [
            row
            for stratum in STRATA
            for row in per_stratum[stratum]
        ]
        bm25_full_results = run_strategy(all_rows, "ir_bm25", top_ks)

    def grouped_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        keys = sorted({(row["strategy"], row["top_k"]) for row in rows})
        for strategy, top_k in keys:
            subset = [
                row
                for row in rows
                if row["strategy"] == strategy and row["top_k"] == top_k
            ]
            result[f"{strategy}@{top_k}"] = {
                "overall": summarize_rows(subset),
                "by_stratum": {
                    stratum: summarize_rows(
                        [
                            row
                            for row in subset
                            if row["stratum"] == stratum
                        ]
                    )
                    for stratum in STRATA
                },
            }
        return result

    script_path = Path(__file__).resolve()
    payload = {
        "protocol": "memgym-dr-provider-free-retrieval-v1",
        "scope": (
            "Official MemGym IR managers and official lexical memory-required-"
            "fact recall proxy; no answer generation and no LLM judge"
        ),
        "provenance": {
            "memgym_commit": git_commit(args.memgym_repo),
            "dataset_snapshot": args.dataset_dir.name,
            "dataset_counts": counts,
            "runner_sha256": sha256(script_path),
            "python": sys.version,
            "platform": platform.platform(),
        },
        "configuration": {
            "seed": args.seed,
            "sample_per_stratum": args.sample_per_stratum,
            "sample_ids": sample_ids,
            "top_k": top_ks,
            "chunk_size": 512,
            "dense_model": "all-MiniLM-L6-v2",
            "bm25_full": args.bm25_full,
        },
        "sample_summary": grouped_summary(sample_results),
        "full_bm25_summary": grouped_summary(bm25_full_results),
        "sample_results": sample_results,
        "full_bm25_results": bm25_full_results,
        "limitations": [
            "The official fact-recall proxy uses substring matches for terms longer than four characters and a 40 percent threshold.",
            "The same observation chunking is used, but this runner stops before answer generation.",
            "The dense comparison is a seeded stratified sample, not all 1,194 rows.",
            "Top-k changes context volume; it is not a fixed-word-budget comparison.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({
        "output": str(args.output),
        "sample_summary": payload["sample_summary"],
        "full_bm25_summary": payload["full_bm25_summary"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
