from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("MEM0_TELEMETRY", "false")

from graph_benchmark_common import (
    load_jsonl,
    score_retrieval,
    source_ids_from_text,
    write_result,
)
from execution_order import interleaved_product


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen3:1.7b")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--execution-seed", type=int, default=20260729)
    parser.add_argument("--work-root", type=Path, default=Path("/tmp/mem0-common-benchmark"))
    return parser.parse_args()


def normalize_results(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("results", "memories"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def source_ids_from_result(result: dict[str, Any]) -> list[str]:
    source_ids: list[str] = []
    metadata = result.get("metadata") or {}
    candidates = []
    if isinstance(metadata, dict):
        if metadata.get("source_id"):
            candidates.append(metadata["source_id"])
        if isinstance(metadata.get("source_ids"), list):
            candidates.extend(metadata["source_ids"])
    candidates.extend(source_ids_from_text(result.get("memory", "")))
    for candidate in candidates:
        normalized = str(candidate).lower()
        if normalized.startswith("d") and normalized[1:].isdigit() and normalized not in source_ids:
            source_ids.append(normalized)
    return source_ids


def get_all_memories(memory: Any, *, user_id: str, expected_documents: int) -> list[dict[str, Any]]:
    top_k = max(expected_documents * 4, 100)
    return normalize_results(
        memory.get_all(filters={"user_id": user_id}, top_k=top_k)
    )


def build_config(root: Path, model: str, collection: str) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    return {
        "history_db_path": str(root / "history.db"),
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "path": str(root / "qdrant"),
                "collection_name": collection,
                "embedding_model_dims": 768,
            },
        },
        "llm": {
            "provider": "ollama",
            "config": {
                "model": model,
                "temperature": 0,
                "max_tokens": 2000,
                "ollama_base_url": "http://127.0.0.1:11434",
            },
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": "nomic-embed-text:latest",
                "embedding_dims": 768,
                "ollama_base_url": "http://127.0.0.1:11434",
            },
        },
    }


def run_mode(
    *,
    corpus: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    root: Path,
    model: str,
    top_k: int,
    infer: bool,
    user_id: str,
) -> dict[str, Any]:
    from mem0 import Memory

    memory = Memory.from_config(
        build_config(root, model, f"common_{'infer' if infer else 'raw'}")
    )
    ingestion_rows = []
    ingestion_started = time.perf_counter()
    for document in corpus:
        started = time.perf_counter()
        content = f"[SOURCE {document['id']}] {document['text']}"
        response = memory.add(
            [{"role": "user", "content": content}],
            user_id=user_id,
            metadata={
                "source_id": document["id"],
                "source_ids": [document["id"]],
                "timestamp": document["timestamp"],
                "kind": document["kind"],
                "confidence": document["confidence"],
            },
            infer=infer,
        )
        ingestion_rows.append(
            {
                "source_id": document["id"],
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "response": response,
            }
        )
    ingestion_seconds = time.perf_counter() - ingestion_started

    stored = get_all_memories(
        memory,
        user_id=user_id,
        expected_documents=len(corpus),
    )
    retained_sources = []
    for item in stored:
        for source_id in source_ids_from_result(item):
            if source_id not in retained_sources:
                retained_sources.append(source_id)

    rows = []
    for question in questions:
        started = time.perf_counter()
        response = memory.search(
            question["query"],
            top_k=top_k,
            filters={"user_id": user_id},
            threshold=0.1,
            rerank=False,
        )
        results = normalize_results(response)
        retrieved_sources = []
        for result in results:
            for source_id in source_ids_from_result(result):
                if source_id not in retrieved_sources:
                    retrieved_sources.append(source_id)
        rows.append(
            {
                "question_id": question["id"],
                "query": question["query"],
                "retrieved_source_ids": retrieved_sources[:top_k],
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "results": results,
            }
        )

    payload = {
        "infer": infer,
        "model": model,
        "user_id": user_id,
        "root": str(root),
        "stored_memories": len(stored),
        "retained_source_ids": retained_sources,
        "retained_source_count": len(retained_sources),
        "ingestion_seconds": round(ingestion_seconds, 3),
        "mean_ingestion_latency_ms": statistics.fmean(
            row["latency_ms"] for row in ingestion_rows
        ),
        "metrics": score_retrieval(questions, rows),
        "ingestion_rows": ingestion_rows,
        "stored": stored,
        "rows": rows,
    }
    del memory
    gc.collect()
    return payload


def aggregate(repeats: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for mode in ("raw", "infer"):
        runs = [repeat[mode] for repeat in repeats]
        metric_names = sorted(runs[0]["metrics"])
        summary[mode] = {
            "stored_memories": [run["stored_memories"] for run in runs],
            "retained_source_count": [run["retained_source_count"] for run in runs],
            "ingestion_seconds": [run["ingestion_seconds"] for run in runs],
            "metrics": {},
        }
        for name in metric_names:
            values = [float(run["metrics"][name]) for run in runs]
            summary[mode]["metrics"][name] = {
                "mean": statistics.fmean(values),
                "sample_stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
                "values": values,
            }
    return summary


def run(args: argparse.Namespace) -> None:
    corpus = load_jsonl(args.corpus)
    questions = load_jsonl(args.questions)
    args.work_root.mkdir(parents=True, exist_ok=True)
    repeat_roots = {
        repeat_index: Path(
            tempfile.mkdtemp(prefix=f"repeat-{repeat_index}-", dir=args.work_root)
        )
        for repeat_index in range(1, args.repetitions + 1)
    }
    repeats_by_index: dict[int, dict[str, Any]] = {
        repeat_index: {
            "repeat": repeat_index,
            "run_root": str(repeat_root),
        }
        for repeat_index, repeat_root in repeat_roots.items()
    }
    schedule = interleaved_product(
        range(1, args.repetitions + 1),
        ["raw", "infer"],
        seed=args.execution_seed,
    )
    for completed_runs, (repeat_index, mode) in enumerate(schedule, start=1):
        repeat_root = repeat_roots[repeat_index]
        result = run_mode(
            corpus=corpus,
            questions=questions,
            root=repeat_root / mode,
            model=args.model,
            top_k=args.top_k,
            infer=mode == "infer",
            user_id=f"common-{mode}-{repeat_index}",
        )
        repeats_by_index[repeat_index][mode] = result
        write_result(
            args.output.with_suffix(args.output.suffix + ".checkpoint.json"),
            {
                "status": "running",
                "completed_runs": completed_runs,
                "expected_runs": args.repetitions * 2,
                "execution_seed": args.execution_seed,
                "repeats": [
                    repeats_by_index[index]
                    for index in sorted(repeats_by_index)
                ],
            },
        )

    repeats = [
        repeats_by_index[index]
        for index in sorted(repeats_by_index)
    ]
    write_result(
        args.output,
        {
            "system": "Mem0 OSS",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "protocol": {
                "documents": len(corpus),
                "questions": len(questions),
                "top_k": args.top_k,
                "repetitions": args.repetitions,
                "execution_seed": args.execution_seed,
                "execution_order": "interleaved raw/infer modes and repetitions",
                "model": args.model,
                "embedder": "nomic-embed-text:latest",
                "vector_store": "embedded Qdrant",
                "modes": {
                    "raw": "Memory.add infer=False",
                    "infer": "Memory.add infer=True with local LLM extraction and updates",
                },
            },
            "repeats": repeats,
            "summary": aggregate(repeats),
        },
    )


if __name__ == "__main__":
    run(parse_args())
