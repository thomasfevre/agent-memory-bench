#!/usr/bin/env python3
"""PROTOTYPE: compare local-model extraction and repeated-pattern candidates."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark import DATA, REPEATS, RESULTS, load_jsonl
from topology_benchmark import model_call


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "only",
    "the",
    "to",
    "with",
}


def tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9_]+", text.lower())
        if token not in STOPWORDS
    }


def fact_f1(expected: str, actual: str) -> float:
    gold = tokens(expected)
    predicted = tokens(actual)
    if not gold or not predicted:
        return 0.0
    overlap = len(gold & predicted)
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def parse_array(text: str, key: str) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    value = parsed.get(key, []) if isinstance(parsed, dict) else []
    return [item for item in value if isinstance(item, dict)]


def extraction_prompt(document: dict[str, Any]) -> str:
    return f"""Extract only explicit durable operational facts from this source.
Preserve dates, validity windows, exceptions, and supersession. Do not infer.
Return exactly one JSON object:
{{"facts":[{{"text":string,"valid_from":string|null,"valid_to":string|null}}]}}

Source id: {document['id']}
Source timestamp: {document['timestamp']}
Source text: {document['text']}
"""


def shard_prompt(documents: list[dict[str, Any]]) -> str:
    rendered = "\n".join(
        f"[{document['id']}] {document['timestamp']}: {document['text']}"
        for document in documents
    )
    return f"""Find behavioral or team-context patterns repeated by at least two independent observations.
Repeated mention is not approval. Return candidates for later human review.
Return exactly one JSON object:
{{"candidates":[{{"text":string,"source_ids":[string],"occurrences":integer}}]}}

Observations:
{rendered}
"""


def run_model(
    model: str,
    documents: list[dict[str, Any]],
    facts: list[dict[str, Any]],
) -> dict[str, Any]:
    facts_by_source: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for fact in facts:
        for source_id in fact["source_ids"]:
            facts_by_source[source_id].append(fact)
    document_map = {document["id"]: document for document in documents}
    selected_ids = sorted(facts_by_source)
    rows = []
    for repeat, seed in enumerate(REPEATS, 1):
        for source_id in selected_ids:
            document = document_map[source_id]
            call = model_call(model, extraction_prompt(document), seed)
            extracted = parse_array(call["text"], "facts")
            extracted_text = " ".join(
                str(item.get("text", "")) for item in extracted
            )
            gold_facts = facts_by_source[source_id]
            scores = [
                max(
                    (
                        fact_f1(gold["text"], str(item.get("text", "")))
                        for item in extracted
                    ),
                    default=0.0,
                )
                for gold in gold_facts
            ]
            temporal = []
            for gold in gold_facts:
                best = max(
                    extracted,
                    key=lambda item: fact_f1(
                        gold["text"], str(item.get("text", ""))
                    ),
                    default={},
                )
                temporal.append(
                    best.get("valid_from") == gold.get("valid_from")
                    and best.get("valid_to") == gold.get("valid_to")
                )
            rows.append(
                {
                    "model": model,
                    "repeat": repeat,
                    "seed": seed,
                    "source_id": source_id,
                    "gold_fact_ids": [fact["id"] for fact in gold_facts],
                    "extracted": extracted,
                    "fact_f1": sum(scores) / len(scores),
                    "temporal_exact": sum(temporal) / len(temporal),
                    "empty": not extracted_text,
                    "latency_ms": call["latency_ms"],
                    "prompt_tokens": call["prompt_tokens"],
                    "output_tokens": call["output_tokens"],
                }
            )

    shard_documents = [
        document for document in documents if document["id"] in {"d09", "d10", "d11", "d12"}
    ]
    shard_rows = []
    for repeat, seed in enumerate(REPEATS, 1):
        call = model_call(model, shard_prompt(shard_documents), seed)
        candidates = parse_array(call["text"], "candidates")
        combined = " ".join(str(item.get("text", "")) for item in candidates)
        shard_rows.append(
            {
                "model": model,
                "repeat": repeat,
                "seed": seed,
                "candidates": candidates,
                "utc_detected": "utc" in combined.lower(),
                "dark_mode_detected": "dark" in combined.lower(),
                "review_status_invented": any(
                    any(
                        word in str(item.get("text", "")).lower()
                        for word in ("approved", "rejected")
                    )
                    for item in candidates
                ),
                "latency_ms": call["latency_ms"],
                "prompt_tokens": call["prompt_tokens"],
                "output_tokens": call["output_tokens"],
            }
        )
    return {"rows": rows, "shard_rows": shard_rows}


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["qwen3:1.7b"])
    args = parser.parse_args()
    documents = load_jsonl(DATA / "corpus.jsonl")
    facts = load_jsonl(DATA / "facts.jsonl")
    payload: dict[str, Any] = {
        "manifest": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "models": args.models,
            "repetitions": len(REPEATS),
            "source_documents": len(documents),
            "gold_facts": len(facts),
        },
        "models": {},
    }
    for model in args.models:
        result = run_model(model, documents, facts)
        result["summary"] = {
            "fact_f1": mean(result["rows"], "fact_f1"),
            "temporal_exact": mean(result["rows"], "temporal_exact"),
            "empty_rate": mean(result["rows"], "empty"),
            "latency_ms": mean(result["rows"], "latency_ms"),
            "prompt_tokens": sum(row["prompt_tokens"] for row in result["rows"]),
            "output_tokens": sum(row["output_tokens"] for row in result["rows"]),
            "utc_candidate_rate": mean(result["shard_rows"], "utc_detected"),
            "dark_mode_candidate_rate": mean(
                result["shard_rows"], "dark_mode_detected"
            ),
            "invented_review_rate": mean(
                result["shard_rows"], "review_status_invented"
            ),
        }
        payload["models"][model] = result

    RESULTS.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RESULTS / f"INGESTION-{timestamp}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (RESULTS / "INGESTION-latest.json").write_text(path.read_text())
    for model, result in payload["models"].items():
        summary = result["summary"]
        print(
            f"{model}: fact_f1={summary['fact_f1']:.3f} "
            f"temporal={summary['temporal_exact']:.3f} "
            f"utc_candidate={summary['utc_candidate_rate']:.3f} "
            f"dark_candidate={summary['dark_mode_candidate_rate']:.3f}"
        )
    print(path)


if __name__ == "__main__":
    main()
