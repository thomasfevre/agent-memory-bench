#!/usr/bin/env python3
"""Build a deterministic semantic-audit sample from a retrieval result."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from graph_benchmark_common import write_result
from graphrag_bench_chunk_graph import (
    best_compatible_window_recall,
    chunk_text,
    parse_evidence_units,
    sentence_windows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["hybrid", "hybrid_graph"],
    )
    parser.add_argument("--per-stratum", type=int, default=5)
    return parser.parse_args()


def stable_key(row: dict[str, Any]) -> str:
    payload = "|".join(
        [
            row["source"],
            row["question_id"],
            row["question_type"],
            row["strategy"],
            str(row["evidence_index"]),
        ]
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def main() -> None:
    args = parse_args()
    result = json.loads(args.result.read_text())
    manifest = result["manifest"]
    threshold = manifest["evidence_token_recall_threshold"]
    corpora = {
        row["corpus_name"]: row["context"]
        for row in json.loads(args.corpus.read_text())
    }
    questions = {
        (row["source"], row["id"]): row
        for row in json.loads(args.questions.read_text())
    }
    chunks_by_source = {}
    for source, text in corpora.items():
        chunks = chunk_text(
            text,
            chunk_words=manifest["chunk_words"],
            overlap_words=manifest["overlap_words"],
        )
        chunks_by_source[source] = {row["id"]: row for row in chunks}
    candidates: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in result["rows"]:
        if row["strategy"] not in args.strategies:
            continue
        question = questions[(row["source"], row["question_id"])]
        evidence_units = parse_evidence_units(question["evidence"])
        chunks = chunks_by_source[row["source"]]
        for evidence_index, evidence_tokens in enumerate(evidence_units):
            matches = []
            for context_id in row["context_ids"]:
                chunk = chunks[context_id]
                recall = best_compatible_window_recall(
                    sentence_windows(chunk["text"]),
                    evidence_tokens,
                )
                if recall >= threshold:
                    matches.append((recall, context_id, chunk["text"]))
            if not matches:
                continue
            recall, context_id, text = max(matches)
            candidate = {
                "source": row["source"],
                "question_id": row["question_id"],
                "question_type": row["question_type"],
                "strategy": row["strategy"],
                "question": question["question"],
                "evidence_index": evidence_index,
                "official_evidence": question["evidence"].split(";")[
                    evidence_index
                ].strip(),
                "matched_chunk_id": context_id,
                "lexical_token_recall": recall,
                "matched_chunk": text,
                "semantic_audit_label": None,
                "semantic_audit_note": None,
            }
            candidates[(row["question_type"], row["strategy"])].append(
                candidate
            )
    sample = []
    for stratum, rows in sorted(candidates.items()):
        selected = sorted(rows, key=stable_key)[: args.per_stratum]
        sample.extend(selected)
    write_result(
        args.output,
        {
            "manifest": {
                "source_result": str(args.result.resolve()),
                "source_result_sha256": hashlib.sha256(
                    args.result.read_bytes()
                ).hexdigest(),
                "sampling": (
                    "SHA-256 deterministic ordering within question_type "
                    "and strategy strata"
                ),
                "strategies": args.strategies,
                "per_stratum": args.per_stratum,
                "evidence_token_recall_threshold": threshold,
                "limitations": [
                    "Labels are intentionally empty until an independent semantic review.",
                    "The sample contains lexical positives only and cannot estimate false-negative rates.",
                ],
            },
            "sample": sample,
        },
    )
    print(args.output)


if __name__ == "__main__":
    main()
