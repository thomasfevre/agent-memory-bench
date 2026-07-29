#!/usr/bin/env python3
"""Run a bounded local slice of MemoryAgentBench Conflict Resolution."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import string
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from benchmark import Bm25
from execution_order import interleaved_product


def normalize_answer(text: str) -> str:
    lowered = text.lower()
    without_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def exact_match(prediction: str, answers: list[str]) -> bool:
    normalized = normalize_answer(prediction)
    return any(normalized == normalize_answer(answer) for answer in answers)


def substring_match(prediction: str, answers: list[str]) -> bool:
    normalized = normalize_answer(prediction)
    return any(normalize_answer(answer) in normalized for answer in answers)


def token_f1(prediction: str, answers: list[str]) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    if not prediction_tokens:
        return 0.0
    scores = []
    for answer in answers:
        answer_tokens = normalize_answer(answer).split()
        common = set(prediction_tokens) & set(answer_tokens)
        if not common:
            scores.append(0.0)
            continue
        precision = len(common) / len(prediction_tokens)
        recall = len(common) / len(answer_tokens)
        scores.append(2 * precision * recall / (precision + recall))
    return max(scores, default=0.0)


def split_facts(context: str) -> list[dict[str, str]]:
    facts = []
    for line_number, line in enumerate(context.splitlines()):
        match = re.match(r"\s*(\d+)\.\s+(.*)", line)
        if match:
            facts.append(
                {
                    "id": f"fact-{match.group(1)}",
                    "text": match.group(2).strip(),
                }
            )
        elif line.strip():
            facts.append({"id": f"line-{line_number}", "text": line.strip()})
    return facts


def ollama_answer(
    model: str,
    question: str,
    evidence: str,
    seed: int,
    num_ctx: int,
) -> dict[str, Any]:
    prompt = f"""Answer the question using only the facts below.
Return only the shortest answer, with no explanation and no label.

Facts:
{evidence}

Question: {question}
Answer:"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
            "seed": seed,
            "num_ctx": num_ctx,
            "num_predict": 64,
        },
    }
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=360) as response:
        raw = json.load(response)
    return {
        "answer": raw.get("message", {}).get("content", "").strip(),
        "latency_ms": (time.perf_counter() - started) * 1000,
        "prompt_tokens": raw.get("prompt_eval_count", 0),
        "output_tokens": raw.get("eval_count", 0),
    }


def find_sample(path: Path, source: str) -> dict[str, Any]:
    rows = pq.read_table(path).to_pylist()
    for row in rows:
        if row.get("metadata", {}).get("source") == source:
            return row
    available = sorted(
        row.get("metadata", {}).get("source", "<missing>") for row in rows
    )
    raise ValueError(f"source {source!r} not found; available={available}")


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for strategy in sorted({row["strategy"] for row in rows}):
        selected = [row for row in rows if row["strategy"] == strategy]
        output.append(
            {
                "strategy": strategy,
                "questions": len(selected),
                "exact_match": statistics.fmean(
                    float(row["exact_match"]) for row in selected
                ),
                "substring_exact_match": statistics.fmean(
                    float(row["substring_exact_match"]) for row in selected
                ),
                "token_f1": statistics.fmean(row["token_f1"] for row in selected),
                "latency_ms": statistics.fmean(
                    row["latency_ms"] for row in selected
                ),
                "prompt_tokens": sum(row["prompt_tokens"] for row in selected),
                "output_tokens": sum(row["output_tokens"] for row in selected),
            }
        )
    return output


def build_payload(
    *,
    args: argparse.Namespace,
    facts: list[dict[str, str]],
    selected_questions: list[tuple[str, list[str]]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_rows = len(selected_questions) * len(args.strategies)
    return {
        "manifest": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": "ai-hyz/MemoryAgentBench",
            "split": "Conflict_Resolution",
            "source": args.source,
            "source_file": str(args.parquet),
            "model": args.model,
            "questions": len(selected_questions),
            "question_offset": args.offset,
            "facts": len(facts),
            "top_k": args.top_k,
            "seed": args.seed,
            "execution_seed": getattr(args, "execution_seed", 20260729),
            "strategies": args.strategies,
            "official_primary_metric": "substring_exact_match",
            "completed_rows": len(rows),
            "expected_rows": expected_rows,
            "complete": len(rows) == expected_rows,
            "note": "Bounded local slice, not a reproduction of the paper table.",
        },
        "rows": rows,
        "summaries": summarize(rows),
    }


def validate_resume_manifest(
    manifest: dict[str, Any],
    args: argparse.Namespace,
    selected_questions: list[tuple[str, list[str]]],
) -> None:
    expected = {
        "source": args.source,
        "model": args.model,
        "questions": len(selected_questions),
        "question_offset": args.offset,
        "top_k": args.top_k,
        "seed": args.seed,
        "execution_seed": getattr(args, "execution_seed", 20260729),
        "strategies": args.strategies,
    }
    mismatches = {
        key: {"existing": manifest.get(key), "requested": value}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"resume manifest mismatch: {mismatches}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--model", default="qwen2.5:14b")
    parser.add_argument("--source", default="factconsolidation_mh_6k")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--questions", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--execution-seed", type=int, default=20260729)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=("long_context", "bm25"),
        default=["long_context", "bm25"],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    sample = find_sample(args.parquet, args.source)
    facts = split_facts(sample["context"])
    bm25 = Bm25(facts)
    all_questions = list(zip(sample["questions"], sample["answers"], strict=True))
    selected_questions = all_questions[args.offset : args.offset + args.questions]
    rows: list[dict[str, Any]] = []
    if args.resume and args.output.exists():
        existing = json.loads(args.output.read_text())
        validate_resume_manifest(existing["manifest"], args, selected_questions)
        rows = existing["rows"]
    completed = {
        (int(row["question_index"]), row["strategy"])
        for row in rows
    }

    indexed_questions = list(
        enumerate(selected_questions, start=args.offset)
    )
    schedule = interleaved_product(
        indexed_questions,
        args.strategies,
        seed=args.execution_seed,
    )
    for question_item, strategy in schedule:
        index, (question, answers) = question_item
        evidence_by_strategy = {
            "long_context": sample["context"],
            "bm25": "\n".join(
                f"{item['id']}: {item['text']}"
                for item in bm25.search(question, args.top_k)
            ),
        }
        evidence = evidence_by_strategy[strategy]
        if (index, strategy) in completed:
            print(f"{index:02d} {strategy:<12} resumed", flush=True)
            continue
        result = ollama_answer(
            args.model,
            question,
            evidence,
            args.seed,
            num_ctx=32768 if strategy == "long_context" else 8192,
        )
        answer = result["answer"]
        row = {
            "question_index": index,
            "question": question,
            "gold_answers": answers,
            "strategy": strategy,
            "exact_match": exact_match(answer, answers),
            "substring_exact_match": substring_match(answer, answers),
            "token_f1": token_f1(answer, answers),
            **result,
        }
        rows.append(row)
        completed.add((index, strategy))
        payload = build_payload(
            args=args,
            facts=facts,
            selected_questions=selected_questions,
            rows=rows,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        print(
            f"{index:02d} {strategy:<12} "
            f"substring={int(row['substring_exact_match'])} "
            f"answer={answer[:80]!r}",
            flush=True,
        )

    payload = build_payload(
        args=args,
        facts=facts,
        selected_questions=selected_questions,
        rows=rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["summaries"], indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
