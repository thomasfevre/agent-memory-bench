#!/usr/bin/env python3
"""Run local retrieval baselines on official LoCoMo and LongMemEval data."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark import Bm25, DenseIndex, MiniLm, MODEL_DIR, reciprocal_rank_fusion


def metrics(gold: set[str], candidates: list[dict[str, Any]]) -> dict[str, float]:
    ids = [candidate["id"] for candidate in candidates]
    found = gold & set(ids)
    recall = len(found) / len(gold) if gold else 0.0
    first = next((rank for rank, item_id in enumerate(ids, 1) if item_id in gold), None)
    return {"recall_at_5": recall, "mrr": 1.0 / first if first else 0.0}


def summarize(rows: list[dict[str, Any]], grouping: str) -> list[dict[str, Any]]:
    groups: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["strategy"], row[grouping])].append(row)
    result = []
    for (strategy, group), selected in sorted(groups.items()):
        result.append(
            {
                "strategy": strategy,
                grouping: group,
                "recall_at_5": statistics.fmean(
                    row["recall_at_5"] for row in selected
                ),
                "mrr": statistics.fmean(row["mrr"] for row in selected),
                "latency_ms": statistics.fmean(
                    row["latency_ms"] for row in selected
                ),
                "questions": len(selected),
            }
        )
    return result


def rankings(
    query: str, bm25: Bm25, dense: DenseIndex, limit: int = 5
) -> dict[str, tuple[list[dict[str, Any]], float]]:
    started = time.perf_counter()
    bm25_rows = bm25.search(query, limit * 2)
    bm25_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    dense_rows = dense.search(query, limit * 2)
    dense_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    hybrid_rows = reciprocal_rank_fusion(
        [bm25_rows, dense_rows], limit, "hybrid"
    )
    fusion_ms = (time.perf_counter() - started) * 1000
    return {
        "bm25": (bm25_rows[:limit], bm25_ms),
        "dense": (dense_rows[:limit], dense_ms),
        "hybrid": (hybrid_rows, bm25_ms + dense_ms + fusion_ms),
    }


def run_locomo(path: Path, encoder: MiniLm) -> dict[str, Any]:
    data = json.loads(path.read_text())
    rows = []
    conversation_count = 0
    for sample in data:
        candidates = []
        for key, value in sample["conversation"].items():
            if key.startswith("session_") and not key.endswith("_date_time"):
                for turn in value:
                    candidates.append(
                        {
                            "id": turn["dia_id"],
                            "text": f"{turn['speaker']}: {turn['text']}",
                        }
                    )
        if not candidates:
            continue
        conversation_count += 1
        bm25 = Bm25(candidates)
        dense = DenseIndex(candidates, encoder)
        for index, question in enumerate(sample["qa"]):
            gold = set(question.get("evidence", []))
            if not gold:
                continue
            result = rankings(question["question"], bm25, dense)
            for strategy, (candidates_found, latency_ms) in result.items():
                rows.append(
                    {
                        "dataset": "LoCoMo",
                        "sample_id": sample["sample_id"],
                        "question_id": f"{sample['sample_id']}:{index}",
                        "question_type": str(question["category"]),
                        "strategy": strategy,
                        "latency_ms": latency_ms,
                        **metrics(gold, candidates_found),
                    }
                )
    return {
        "dataset": "LoCoMo",
        "source": str(path),
        "conversations": conversation_count,
        "questions": len({row["question_id"] for row in rows}),
        "category_counts": dict(
            Counter(row["question_type"] for row in rows if row["strategy"] == "bm25")
        ),
        "rows": rows,
        "summaries": summarize(rows, "question_type"),
    }


def session_text(session: list[dict[str, str]]) -> str:
    return "\n".join(
        f"{message.get('role', '')}: {message.get('content', '')}"
        for message in session
    )


def run_longmemeval(path: Path, encoder: MiniLm) -> dict[str, Any]:
    data = json.loads(path.read_text())
    rows = []
    skipped_abstention = 0
    for item in data:
        if "_abs" in item["question_id"]:
            skipped_abstention += 1
            continue
        candidates = [
            {"id": session_id, "text": session_text(session)}
            for session_id, session in zip(
                item["haystack_session_ids"], item["haystack_sessions"], strict=True
            )
        ]
        gold = set(item.get("answer_session_ids", []))
        if not candidates or not gold:
            continue
        bm25 = Bm25(candidates)
        dense = DenseIndex(candidates, encoder)
        result = rankings(item["question"], bm25, dense)
        for strategy, (candidates_found, latency_ms) in result.items():
            rows.append(
                {
                    "dataset": "LongMemEval",
                    "question_id": item["question_id"],
                    "question_type": item["question_type"],
                    "strategy": strategy,
                    "latency_ms": latency_ms,
                    **metrics(gold, candidates_found),
                }
            )
    return {
        "dataset": "LongMemEval",
        "source": str(path),
        "questions": len({row["question_id"] for row in rows}),
        "skipped_abstention_questions": skipped_abstention,
        "category_counts": dict(
            Counter(row["question_type"] for row in rows if row["strategy"] == "bm25")
        ),
        "rows": rows,
        "summaries": summarize(rows, "question_type"),
    }


def overall(result: dict[str, Any]) -> list[dict[str, Any]]:
    summaries = []
    for strategy in ["bm25", "dense", "hybrid"]:
        selected = [row for row in result["rows"] if row["strategy"] == strategy]
        summaries.append(
            {
                "strategy": strategy,
                "recall_at_5": statistics.fmean(
                    row["recall_at_5"] for row in selected
                ),
                "mrr": statistics.fmean(row["mrr"] for row in selected),
                "latency_ms": statistics.fmean(
                    row["latency_ms"] for row in selected
                ),
                "questions": len(selected),
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--locomo", type=Path)
    parser.add_argument("--longmemeval", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.locomo and not args.longmemeval:
        parser.error("provide --locomo and/or --longmemeval")

    encoder = MiniLm(MODEL_DIR)
    payload: dict[str, Any] = {
        "manifest": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "embedding_model": "all-MiniLM-L6-v2 local ONNX",
            "top_k": 5,
        }
    }
    if args.locomo:
        payload["locomo"] = run_locomo(args.locomo, encoder)
        payload["locomo"]["overall"] = overall(payload["locomo"])
    if args.longmemeval:
        payload["longmemeval"] = run_longmemeval(args.longmemeval, encoder)
        payload["longmemeval"]["overall"] = overall(payload["longmemeval"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    for name in ("locomo", "longmemeval"):
        if name not in payload:
            continue
        print(f"\n{name.upper()} ({payload[name]['questions']} questions)")
        for row in payload[name]["overall"]:
            print(
                f"{row['strategy']:<8} recall@5={row['recall_at_5']:.3f} "
                f"mrr={row['mrr']:.3f} latency={row['latency_ms']:.1f}ms"
            )
    print(f"\n{args.output}")


if __name__ == "__main__":
    main()
