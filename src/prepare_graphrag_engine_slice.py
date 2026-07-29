#!/usr/bin/env python3
"""Build a bounded, reproducible GraphRAG-Bench slice for real graph engines."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmark import Bm25
from graph_benchmark_common import write_result
from graphrag_bench_chunk_graph import (
    chunk_text,
    parse_evidence_units,
    sentence_windows,
    unit_is_covered,
)


PROTOCOL_VERSION = "graphrag-real-engines-slice-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output-corpus", type=Path, required=True)
    parser.add_argument("--output-questions", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--source", default="Novel-30752")
    parser.add_argument("--question-count", type=int, default=10)
    parser.add_argument("--complex-count", type=int, default=5)
    parser.add_argument("--document-count", type=int, default=20)
    parser.add_argument("--chunk-words", type=int, default=160)
    parser.add_argument("--overlap-words", type=int, default=40)
    parser.add_argument("--evidence-threshold", type=float, default=0.85)
    return parser.parse_args()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    )


def representable_questions(
    questions: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    windows = {chunk["id"]: sentence_windows(chunk["text"]) for chunk in chunks}
    candidates = []
    for question in questions:
        units = parse_evidence_units(question.get("evidence", ""))
        if not units:
            continue
        gold_chunk_ids: list[str] = []
        unit_candidates: list[list[str]] = []
        for unit in units:
            matching = [
                chunk["id"]
                for chunk in chunks
                if unit_is_covered(windows[chunk["id"]], unit, threshold=threshold)
            ]
            if not matching:
                break
            unit_candidates.append(matching)
            gold_chunk_ids.append(matching[0])
        else:
            candidates.append(
                {
                    "question": question,
                    "gold_chunk_ids": list(dict.fromkeys(gold_chunk_ids)),
                    "unit_candidates": unit_candidates,
                }
            )
    return candidates


def select_questions(
    candidates: list[dict[str, Any]],
    *,
    question_count: int,
    complex_count: int,
) -> list[dict[str, Any]]:
    if not 0 <= complex_count <= question_count:
        raise ValueError("complex_count must fit within question_count")
    selected: list[dict[str, Any]] = []
    selected_gold: set[str] = set()

    def take(question_type: str, count: int) -> None:
        remaining = [
            candidate
            for candidate in candidates
            if candidate["question"]["question_type"] == question_type
            and candidate not in selected
        ]
        for _ in range(count):
            if not remaining:
                break
            chosen = min(
                remaining,
                key=lambda candidate: (
                    len(set(candidate["gold_chunk_ids"]) - selected_gold),
                    len(candidate["gold_chunk_ids"]),
                    candidate["question"]["id"],
                ),
            )
            selected.append(chosen)
            selected_gold.update(chosen["gold_chunk_ids"])
            remaining.remove(chosen)

    take("Complex Reasoning", complex_count)
    take("Fact Retrieval", question_count - len(selected))
    if len(selected) < question_count:
        remaining = [candidate for candidate in candidates if candidate not in selected]
        while remaining and len(selected) < question_count:
            chosen = min(
                remaining,
                key=lambda candidate: (
                    len(set(candidate["gold_chunk_ids"]) - selected_gold),
                    len(candidate["gold_chunk_ids"]),
                    candidate["question"]["id"],
                ),
            )
            selected.append(chosen)
            selected_gold.update(chosen["gold_chunk_ids"])
            remaining.remove(chosen)
    if len(selected) != question_count:
        raise RuntimeError(
            f"only {len(selected)} representable questions available, need {question_count}"
        )
    return selected


def select_distractors(
    chunks: list[dict[str, Any]],
    selected_questions: list[dict[str, Any]],
    *,
    count: int,
) -> list[str]:
    if count < 0:
        raise ValueError("distractor count cannot be negative")
    gold = {
        chunk_id
        for candidate in selected_questions
        for chunk_id in candidate["gold_chunk_ids"]
    }
    rankings = Bm25(chunks)
    scores: defaultdict[str, float] = defaultdict(float)
    maximum_rank = min(len(chunks), 80)
    for candidate in selected_questions:
        query = candidate["question"]["question"]
        for rank, row in enumerate(rankings.search(query, maximum_rank), start=1):
            if row["id"] not in gold:
                scores[row["id"]] += 1.0 / (60 + rank)
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    if len(ordered) < count:
        ordered.extend(
            chunk["id"]
            for chunk in chunks
            if chunk["id"] not in gold and chunk["id"] not in ordered
        )
    return ordered[:count]


def build_slice(args: argparse.Namespace) -> dict[str, Any]:
    corpora = {
        row["corpus_name"]: row["context"]
        for row in json.loads(args.corpus.read_text())
    }
    if args.source not in corpora:
        raise KeyError(f"source not found: {args.source}")
    source_questions = [
        row
        for row in json.loads(args.questions.read_text())
        if row["source"] == args.source
    ]
    chunks = chunk_text(
        corpora[args.source],
        chunk_words=args.chunk_words,
        overlap_words=args.overlap_words,
    )
    candidates = representable_questions(
        source_questions,
        chunks,
        threshold=args.evidence_threshold,
    )
    selected_questions = select_questions(
        candidates,
        question_count=args.question_count,
        complex_count=args.complex_count,
    )
    gold_chunk_ids = {
        chunk_id
        for candidate in selected_questions
        for chunk_id in candidate["gold_chunk_ids"]
    }
    if len(gold_chunk_ids) > args.document_count:
        raise RuntimeError(
            f"{len(gold_chunk_ids)} gold chunks exceed {args.document_count} documents"
        )
    distractors = select_distractors(
        chunks,
        selected_questions,
        count=args.document_count - len(gold_chunk_ids),
    )
    selected_chunk_ids = gold_chunk_ids | set(distractors)
    selected_chunks = [
        chunk for chunk in chunks if chunk["id"] in selected_chunk_ids
    ]
    if len(selected_chunks) != args.document_count:
        raise RuntimeError(
            f"selected {len(selected_chunks)} documents, need {args.document_count}"
        )

    source_id_by_chunk = {
        chunk["id"]: f"d{index:02d}"
        for index, chunk in enumerate(selected_chunks, start=1)
    }
    output_corpus = [
        {
            "id": source_id_by_chunk[chunk["id"]],
            "kind": "graphrag_bench_novel_chunk",
            "timestamp": f"2026-01-01T00:{index:02d}:00",
            "text": chunk["text"],
            "original_source": args.source,
            "original_chunk_id": chunk["id"],
            "word_start": chunk["word_start"],
            "word_end": chunk["word_end"],
        }
        for index, chunk in enumerate(selected_chunks)
    ]
    output_questions = [
        {
            "id": candidate["question"]["id"],
            "query": candidate["question"]["question"],
            "query_date": "2026-01-02T00:00:00",
            "category": candidate["question"]["question_type"].lower().replace(" ", "_"),
            "gold_source_ids": [
                source_id_by_chunk[chunk_id]
                for chunk_id in candidate["gold_chunk_ids"]
            ],
            "answer": candidate["question"]["answer"],
            "should_abstain": False,
            "original_source": args.source,
            "original_evidence": candidate["question"]["evidence"],
            "original_evidence_triple": candidate["question"].get("evidence_triple"),
        }
        for candidate in selected_questions
    ]
    write_jsonl(args.output_corpus, output_corpus)
    write_jsonl(args.output_questions, output_questions)
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "source": args.source,
        "source_files": {
            "corpus": {
                "path": str(args.corpus),
                "sha256": sha256_path(args.corpus),
            },
            "questions": {
                "path": str(args.questions),
                "sha256": sha256_path(args.questions),
            },
        },
        "parameters": {
            "question_count": args.question_count,
            "complex_count": args.complex_count,
            "document_count": args.document_count,
            "chunk_words": args.chunk_words,
            "overlap_words": args.overlap_words,
            "evidence_threshold": args.evidence_threshold,
            "distractor_strategy": "cross-question BM25 reciprocal-rank",
        },
        "selection": {
            "representable_questions": len(candidates),
            "question_ids": [row["id"] for row in output_questions],
            "question_types": [
                candidate["question"]["question_type"]
                for candidate in selected_questions
            ],
            "gold_original_chunk_ids": sorted(gold_chunk_ids),
            "distractor_original_chunk_ids": sorted(distractors),
            "selected_original_chunk_ids": [
                chunk["id"] for chunk in selected_chunks
            ],
        },
        "outputs": {
            "corpus": {
                "path": str(args.output_corpus),
                "sha256": sha256_path(args.output_corpus),
                "documents": len(output_corpus),
            },
            "questions": {
                "path": str(args.output_questions),
                "sha256": sha256_path(args.output_questions),
                "questions": len(output_questions),
            },
        },
    }
    write_result(args.output_manifest, manifest)
    return manifest


if __name__ == "__main__":
    build_slice(parse_args())
