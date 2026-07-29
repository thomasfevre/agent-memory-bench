#!/usr/bin/env python3
"""PROTOTYPE: compare bounded orchestration topologies with one local reader."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark import (
    DATA,
    REPEATS,
    RESULTS,
    DenseIndex,
    answer_matches,
    average,
    build_indexes,
    load_jsonl,
    ollama_answer,
    parse_json_object,
    retrieve,
    source_ids,
)
from execution_order import interleaved_product


QUESTION_IDS = {"q01", "q03", "q05", "q06", "q09", "q10", "q13", "q18"}


def pipeline_retrieve(
    question: dict[str, Any], indexes: Any, limit: int = 5
) -> list[dict[str, Any]]:
    shortlist = indexes.document_bm25.search(question["query"], 12)
    if not shortlist:
        return []
    return DenseIndex(shortlist, indexes.document_dense.encoder).search(
        question["query"], limit
    )


def conditional_retrieve(
    question: dict[str, Any], indexes: Any, limit: int = 5
) -> list[dict[str, Any]]:
    query = question["query"].lower()
    if (
        " on 2026-" in query
        or "currently" in query
        or query.startswith("who was ")
        or query.startswith("who is ")
    ):
        return retrieve("facts", question, indexes, limit)
    if "shared" in query or "approved shared rule" in query:
        return retrieve("context_shards", question, indexes, limit)
    if "who" in query and " and " in query:
        return retrieve("graph", question, indexes, limit)
    return retrieve("hybrid", question, indexes, limit)


def model_call(model: str, prompt: str, seed: int) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.1, "seed": seed, "num_ctx": 8192},
    }
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as response:
        raw = json.load(response)
    return {
        "text": raw.get("message", {}).get("content", ""),
        "latency_ms": (time.perf_counter() - started) * 1000,
        "prompt_tokens": raw.get("prompt_eval_count", 0),
        "output_tokens": raw.get("eval_count", 0),
    }


def evidence_text(candidates: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"[{item['id']}] {item['text']} Sources={sorted(source_ids(item))}"
        for item in candidates
    )


def verify_answer(
    model: str,
    question: dict[str, Any],
    candidates: list[dict[str, Any]],
    answer: dict[str, Any],
    seed: int,
    adversarial: bool,
) -> dict[str, Any]:
    stance = (
        "Try hard to refute the proposed answer. Accept it only if direct evidence survives."
        if adversarial
        else "Check whether direct evidence fully supports the proposed answer."
    )
    prompt = f"""{stance}
Return exactly one JSON object: {{"accept": boolean, "reason": string}}.

Question: {question['query']}
Proposed answer: {json.dumps(answer.get('parsed'))}
Evidence:
{evidence_text(candidates)}
"""
    result = model_call(model, prompt, seed)
    parsed = parse_json_object(result["text"])
    return {
        **result,
        "parsed": parsed,
        "accept": bool(parsed and parsed.get("accept")),
    }


def score_final(question: dict[str, Any], parsed: dict[str, Any] | None) -> bool:
    if parsed is None:
        return False
    abstained = bool(parsed.get("abstain")) or parsed.get("answer") is None
    if question["should_abstain"]:
        return abstained
    if abstained:
        return False
    return answer_matches(question["answer"], parsed.get("answer"))


def aggregate_calls(calls: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "latency_ms": sum(call["latency_ms"] for call in calls),
        "prompt_tokens": sum(call["prompt_tokens"] for call in calls),
        "output_tokens": sum(call["output_tokens"] for call in calls),
        "calls": len(calls),
    }


def run_topology(
    topology: str,
    model: str,
    question: dict[str, Any],
    indexes: Any,
    seed: int,
) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []

    if topology == "single_hybrid":
        candidates = retrieve("hybrid", question, indexes)
        answer = ollama_answer(model, question, candidates, seed)
        calls.append(answer)
        final = answer["parsed"]

    elif topology == "pipeline":
        candidates = pipeline_retrieve(question, indexes)
        answer = ollama_answer(model, question, candidates, seed)
        calls.append(answer)
        final = answer["parsed"]

    elif topology == "conditional_route":
        candidates = conditional_retrieve(question, indexes)
        answer = ollama_answer(model, question, candidates, seed)
        calls.append(answer)
        final = answer["parsed"]

    elif topology == "adversarial_gate":
        candidates = retrieve("hybrid", question, indexes)
        answer = ollama_answer(model, question, candidates, seed)
        calls.append(answer)
        checks = [
            verify_answer(model, question, candidates, answer, seed + 101, False),
            verify_answer(model, question, candidates, answer, seed + 211, True),
        ]
        calls.extend(checks)
        final = (
            answer["parsed"]
            if all(check["accept"] for check in checks)
            else {"answer": None, "abstain": True, "evidence_ids": []}
        )

    elif topology == "verify_retry_loop":
        candidates = retrieve("bm25", question, indexes)
        first = ollama_answer(model, question, candidates, seed)
        calls.append(first)
        check = verify_answer(model, question, candidates, first, seed + 307, True)
        calls.append(check)
        final = first["parsed"]
        if not check["accept"]:
            expanded = retrieve("parallel_merge", question, indexes, 8)
            retry = ollama_answer(model, question, expanded, seed + 401)
            calls.append(retry)
            final = retry["parsed"]

    elif topology == "supervisor_swarm":
        branches = ["bm25", "dense", "facts"]

        def run_branch(strategy: str) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
            candidates = retrieve(strategy, question, indexes)
            return strategy, candidates, ollama_answer(
                model, question, candidates, seed + len(strategy)
            )

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=3) as executor:
            branch_results = list(executor.map(run_branch, branches))
        parallel_elapsed = (time.perf_counter() - started) * 1000
        for _, _, answer in branch_results:
            calls.append(answer)
        candidates_by_id: dict[str, dict[str, Any]] = {}
        proposals = []
        for strategy, candidates, answer in branch_results:
            proposals.append({"strategy": strategy, "answer": answer["parsed"]})
            for candidate in candidates:
                candidates_by_id[candidate["id"]] = candidate
        candidates = list(candidates_by_id.values())
        prompt = f"""Act as a supervisor. Select or repair the best supported answer.
Return exactly one JSON object: {{"answer": string|null, "abstain": boolean, "evidence_ids": [string]}}.

Question: {question['query']}
Worker proposals: {json.dumps(proposals)}
Evidence:
{evidence_text(candidates)}
"""
        synthesis = model_call(model, prompt, seed + 503)
        synthesis["parsed"] = parse_json_object(synthesis["text"])
        calls.append(synthesis)
        final = synthesis["parsed"]
        totals = aggregate_calls(calls)
        totals["latency_ms"] = parallel_elapsed + synthesis["latency_ms"]
        return {
            **totals,
            "correct": score_final(question, final),
            "abstained": bool(
                final
                and (final.get("abstain") or final.get("answer") is None)
            ),
            "final": final,
        }
    else:
        raise ValueError(topology)

    totals = aggregate_calls(calls)
    return {
        **totals,
        "correct": score_final(question, final),
        "abstained": bool(
            final and (final.get("abstain") or final.get("answer") is None)
        ),
        "final": final,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3:1.7b")
    parser.add_argument("--execution-seed", type=int, default=20260729)
    args = parser.parse_args()

    indexes = build_indexes()
    questions = [
        question
        for question in load_jsonl(DATA / "questions.jsonl")
        if question["id"] in QUESTION_IDS
    ]
    topologies = [
        "single_hybrid",
        "pipeline",
        "conditional_route",
        "adversarial_gate",
        "verify_retry_loop",
        "supervisor_swarm",
    ]
    rows = []
    schedule = interleaved_product(
        list(enumerate(REPEATS, start=1)),
        topologies,
        questions,
        seed=args.execution_seed,
    )
    for repeat_seed, topology, question in schedule:
        repeat, seed = repeat_seed
        result = run_topology(topology, args.model, question, indexes, seed)
        row = {
            "model": args.model,
            "repeat": repeat,
            "seed": seed,
            "topology": topology,
            "question_id": question["id"],
            "category": question["category"],
            **result,
        }
        rows.append(row)
        print(
            f"{topology:<20} {question['id']} repeat={repeat} "
            f"correct={result['correct']} calls={result['calls']}",
            flush=True,
        )
    summaries = []
    for topology in topologies:
        selected = [row for row in rows if row["topology"] == topology]
        summaries.append(
            {
                "model": args.model,
                "topology": topology,
                "accuracy": average(float(row["correct"]) for row in selected),
                "abstention_rate": average(
                    float(row["abstained"]) for row in selected
                ),
                "latency_ms": average(row["latency_ms"] for row in selected),
                "prompt_tokens": sum(row["prompt_tokens"] for row in selected),
                "output_tokens": sum(row["output_tokens"] for row in selected),
                "calls": sum(row["calls"] for row in selected),
                "runs": len(selected),
            }
        )
    payload = {
        "manifest": {
            "prototype": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "questions": len(questions),
            "repetitions": len(REPEATS),
            "execution_seed": args.execution_seed,
            "topologies": topologies,
        },
        "rows": rows,
        "summaries": summaries,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RESULTS / f"TOPOLOGY-{timestamp}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (RESULTS / "TOPOLOGY-latest.json").write_text(path.read_text())
    print("\nTOPOLOGY SUMMARY")
    for summary in summaries:
        print(
            f"{summary['topology']:<20} accuracy={summary['accuracy']:.3f} "
            f"latency={summary['latency_ms']:.0f}ms calls={summary['calls']} "
            f"tokens={summary['prompt_tokens'] + summary['output_tokens']}"
        )
    print(path)


if __name__ == "__main__":
    main()
