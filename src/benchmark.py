#!/usr/bin/env python3
"""PROTOTYPE: compare memory views over one immutable synthetic corpus."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import time
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from execution_order import interleaved_product


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MODEL_DIR = Path(
    os.environ.get(
        "AMB_MINILM_DIR",
        ROOT / ".cache" / "models" / "all-MiniLM-L6-v2",
    )
)
DEFAULT_K = 5
REPEATS = (11, 23, 37)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def source_ids(candidate: dict[str, Any]) -> set[str]:
    if "source_ids" in candidate:
        return set(candidate["source_ids"])
    return {candidate["id"]}


def is_valid_on(candidate: dict[str, Any], query_date: str) -> bool:
    valid_from = candidate.get("valid_from")
    valid_to = candidate.get("valid_to")
    if valid_from and query_date < valid_from:
        return False
    if valid_to and query_date > valid_to:
        return False
    return candidate.get("confidence", 1.0) >= 0.8


class Bm25:
    def __init__(self, candidates: list[dict[str, Any]]) -> None:
        self.candidates = candidates
        self.tokens = [tokenize(item["text"]) for item in candidates]
        self.counts = [Counter(tokens) for tokens in self.tokens]
        self.df: Counter[str] = Counter()
        for tokens in self.tokens:
            self.df.update(set(tokens))
        self.avgdl = statistics.fmean(len(tokens) for tokens in self.tokens) or 1.0

    def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        query_terms = set(tokenize(query))
        n = len(self.candidates)
        scored: list[tuple[float, dict[str, Any]]] = []
        for item, tokens, counts in zip(
            self.candidates,
            self.tokens,
            self.counts,
            strict=True,
        ):
            score = 0.0
            for term in query_terms:
                frequency = counts[term]
                if not frequency:
                    continue
                document_frequency = self.df[term]
                inverse_frequency = math.log(
                    1.0 + (n - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                denominator = frequency + 1.2 * (
                    1.0 - 0.75 + 0.75 * len(tokens) / self.avgdl
                )
                score += inverse_frequency * frequency * 2.2 / denominator
            scored.append((score, item))
        scored.sort(key=lambda row: (-row[0], row[1]["id"]))
        return [
            {**item, "_score": float(score), "_retriever": "bm25"}
            for score, item in scored[:limit]
            if score > 0
        ]


class MiniLm:
    def __init__(self, model_dir: Path) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        if not (model_dir / "model.onnx").exists():
            raise RuntimeError(f"MiniLM model missing at {model_dir}")
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding()
        self.session = ort.InferenceSession(
            str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        self.input_names = {entry.name for entry in self.session.get_inputs()}

    def encode(self, texts: list[str]) -> np.ndarray:
        encoded = self.tokenizer.encode_batch(texts)
        input_ids = np.asarray([item.ids for item in encoded], dtype=np.int64)
        attention = np.asarray([item.attention_mask for item in encoded], dtype=np.int64)
        inputs: dict[str, np.ndarray] = {
            "input_ids": input_ids,
            "attention_mask": attention,
        }
        if "token_type_ids" in self.input_names:
            inputs["token_type_ids"] = np.zeros_like(input_ids)
        hidden = self.session.run(None, inputs)[0]
        mask = attention[..., None].astype(np.float32)
        pooled = (hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1e-9)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return pooled / np.maximum(norms, 1e-9)


class DenseIndex:
    def __init__(self, candidates: list[dict[str, Any]], encoder: MiniLm) -> None:
        self.candidates = candidates
        self.encoder = encoder
        self.embeddings = encoder.encode([item["text"] for item in candidates])

    def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        query_embedding = self.encoder.encode([query])[0]
        scores = self.embeddings @ query_embedding
        order = np.argsort(-scores, kind="stable")[:limit]
        return [
            {
                **self.candidates[int(index)],
                "_score": float(scores[int(index)]),
                "_retriever": "dense",
            }
            for index in order
        ]


def reciprocal_rank_fusion(
    rankings: Iterable[list[dict[str, Any]]], limit: int, label: str
) -> list[dict[str, Any]]:
    scores: defaultdict[str, float] = defaultdict(float)
    records: dict[str, dict[str, Any]] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item["id"]] += 1.0 / (60 + rank)
            records[item["id"]] = item
    ordered = sorted(scores, key=lambda item_id: (-scores[item_id], item_id))[:limit]
    return [
        {**records[item_id], "_score": scores[item_id], "_retriever": label}
        for item_id in ordered
    ]


@dataclass
class IndexSet:
    documents: list[dict[str, Any]]
    facts: list[dict[str, Any]]
    shards: list[dict[str, Any]]
    document_bm25: Bm25
    fact_bm25: Bm25
    shard_bm25: Bm25
    document_dense: DenseIndex
    fact_dense: DenseIndex
    shard_dense: DenseIndex
    temporal_fact_indexes: dict[str, tuple[Bm25, DenseIndex, list[dict[str, Any]]]] = field(
        default_factory=dict
    )
    reviewed_shard_indexes: tuple[Bm25, DenseIndex, list[dict[str, Any]]] | None = None


def build_indexes(model_dir: Path = MODEL_DIR) -> IndexSet:
    documents = load_jsonl(DATA / "corpus.jsonl")
    facts = load_jsonl(DATA / "facts.jsonl")
    shards = load_jsonl(DATA / "shards.jsonl")
    shard_records = [
        {
            **item,
            "text": f"Review status {item['review']}: {item['text']}",
        }
        for item in shards
    ]
    encoder = MiniLm(model_dir)
    return IndexSet(
        documents=documents,
        facts=facts,
        shards=shard_records,
        document_bm25=Bm25(documents),
        fact_bm25=Bm25(facts),
        shard_bm25=Bm25(shard_records),
        document_dense=DenseIndex(documents, encoder),
        fact_dense=DenseIndex(facts, encoder),
        shard_dense=DenseIndex(shard_records, encoder),
    )


def retrieve(
    strategy: str, question: dict[str, Any], indexes: IndexSet, limit: int = DEFAULT_K
) -> list[dict[str, Any]]:
    query = question["query"]
    query_date = question["query_date"]
    if strategy == "long_context":
        return [
            {**item, "_score": 1.0, "_retriever": "long_context"}
            for item in indexes.documents
        ]
    if strategy == "bm25":
        return indexes.document_bm25.search(query, limit)
    if strategy == "dense":
        return indexes.document_dense.search(query, limit)
    if strategy == "hybrid":
        return reciprocal_rank_fusion(
            [
                indexes.document_bm25.search(query, limit * 2),
                indexes.document_dense.search(query, limit * 2),
            ],
            limit,
            "hybrid",
        )
    if strategy in {"facts", "graph"}:
        if query_date not in indexes.temporal_fact_indexes:
            valid_facts = [
                item for item in indexes.facts if is_valid_on(item, query_date)
            ]
            indexes.temporal_fact_indexes[query_date] = (
                Bm25(valid_facts),
                DenseIndex(valid_facts, indexes.fact_dense.encoder),
                valid_facts,
            )
        fact_bm25, fact_dense, valid_facts = indexes.temporal_fact_indexes[query_date]
        initial = reciprocal_rank_fusion(
            [fact_bm25.search(query, limit * 2), fact_dense.search(query, limit * 2)],
            limit if strategy == "facts" else 3,
            "facts",
        )
        if strategy == "facts":
            return initial
        by_id = {item["id"]: item for item in valid_facts}
        expanded = list(initial)
        seen = {item["id"] for item in expanded}
        for item in list(initial):
            for linked_id in item.get("links", []):
                if linked_id in by_id and linked_id not in seen:
                    expanded.append(
                        {
                            **by_id[linked_id],
                            "_score": item["_score"] * 0.8,
                            "_retriever": "graph_expand",
                        }
                    )
                    seen.add(linked_id)
        return expanded[:limit]
    if strategy == "context_shards":
        if indexes.reviewed_shard_indexes is None:
            candidates = [
                item
                for item in indexes.shards
                if item["occurrences"] >= 2
                and item["review"] in {"approved", "rejected"}
            ]
            indexes.reviewed_shard_indexes = (
                Bm25(candidates),
                DenseIndex(candidates, indexes.shard_dense.encoder),
                candidates,
            )
        shard_bm25, shard_dense, candidates = indexes.reviewed_shard_indexes
        return reciprocal_rank_fusion(
            [shard_bm25.search(query, limit), shard_dense.search(query, limit)],
            limit,
            "context_shards",
        )
    if strategy == "routed":
        category = question["category"]
        if category in {"temporal", "multi_hop_temporal", "rare_exception", "conflict"}:
            return retrieve("facts", question, indexes, limit)
        if category == "multi_hop":
            return retrieve("graph", question, indexes, limit)
        if category in {"context_shard", "rejected_shard"}:
            return retrieve("context_shards", question, indexes, limit)
        if category == "semantic":
            return retrieve("dense", question, indexes, limit)
        return retrieve("bm25", question, indexes, limit)
    if strategy == "parallel_merge":
        return reciprocal_rank_fusion(
            [
                retrieve("bm25", question, indexes, limit * 2),
                retrieve("dense", question, indexes, limit * 2),
                retrieve("facts", question, indexes, limit * 2),
                retrieve("graph", question, indexes, limit * 2),
                retrieve("context_shards", question, indexes, limit * 2),
            ],
            limit,
            "parallel_merge",
        )
    raise ValueError(f"unknown strategy: {strategy}")


TEMPORAL_CONFLICTS = {
    "q01": {"d01", "d20"},
    "q02": {"d02", "d20"},
    "q03": {"d05"},
    "q07": {"d05"},
    "q08": {"d06"},
}


def retrieval_metrics(
    question: dict[str, Any], candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    gold = set(question["gold_source_ids"])
    retrieved_sets = [source_ids(item) for item in candidates]
    retrieved_union = set().union(*retrieved_sets) if retrieved_sets else set()
    if gold:
        recall = len(gold & retrieved_union) / len(gold)
        relevant_flags = [bool(gold & item_sources) for item_sources in retrieved_sets]
        precision = sum(relevant_flags) / len(relevant_flags) if relevant_flags else 0.0
        first_rank = next(
            (rank for rank, relevant in enumerate(relevant_flags, start=1) if relevant), None
        )
        reciprocal_rank = 1.0 / first_rank if first_rank else 0.0
    else:
        recall = None
        precision = 1.0 if not candidates else 0.0
        reciprocal_rank = None
    temporal_conflicts = TEMPORAL_CONFLICTS.get(question["id"], set())
    temporal_exact = None
    if question["id"] in TEMPORAL_CONFLICTS:
        temporal_exact = gold <= retrieved_union and not bool(
            temporal_conflicts & retrieved_union
        )
    return {
        "recall": recall,
        "context_precision": precision,
        "mrr": reciprocal_rank,
        "temporal_exact": temporal_exact,
        "retrieved_source_ids": sorted(retrieved_union),
        "candidate_ids": [item["id"] for item in candidates],
    }


def normalized_answer(value: Any) -> str:
    if value is None:
        return ""
    text = re.sub(r"[^a-z0-9_]+", " ", str(value).lower()).strip()
    number_words = {
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    return " ".join(number_words.get(token, token) for token in text.split())


def answer_matches(expected_value: Any, actual_value: Any) -> bool:
    expected_tokens = normalized_answer(expected_value).split()
    actual_tokens = set(normalized_answer(actual_value).split())
    return bool(expected_tokens) and all(token in actual_tokens for token in expected_tokens)


def parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def ollama_answer(
    model: str,
    question: dict[str, Any],
    candidates: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    evidence = "\n".join(
        f"[{item['id']}] {item['text']} Sources={sorted(source_ids(item))}"
        for item in candidates
    )
    prompt = f"""Answer only from the evidence below.
If the evidence does not support an answer, abstain.
Return exactly one JSON object: {{"answer": string|null, "abstain": boolean, "evidence_ids": [string]}}.

Question date: {question['query_date']}
Question: {question['query']}

Evidence:
{evidence}
"""
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
    elapsed_ms = (time.perf_counter() - started) * 1000
    text = raw.get("message", {}).get("content", "")
    parsed = parse_json_object(text)
    expected = normalized_answer(question["answer"])
    if parsed is None:
        correct = False
        abstained = False
    else:
        abstained = bool(parsed.get("abstain")) or parsed.get("answer") is None
        if question["should_abstain"]:
            correct = abstained
        elif abstained:
            correct = False
        else:
            correct = answer_matches(expected, parsed.get("answer"))
    return {
        "parsed": parsed,
        "raw": text,
        "correct": correct,
        "abstained": abstained,
        "latency_ms": elapsed_ms,
        "prompt_tokens": raw.get("prompt_eval_count", 0),
        "output_tokens": raw.get("eval_count", 0),
    }


def average(values: Iterable[float | int | None]) -> float | None:
    concrete = [float(value) for value in values if value is not None]
    return statistics.fmean(concrete) if concrete else None


def run_retrieval(indexes: IndexSet, questions: list[dict[str, Any]]) -> dict[str, Any]:
    strategies = [
        "long_context",
        "bm25",
        "dense",
        "hybrid",
        "facts",
        "graph",
        "context_shards",
        "routed",
        "parallel_merge",
    ]
    rows: list[dict[str, Any]] = []
    for repeat, seed in enumerate(REPEATS, start=1):
        for strategy in strategies:
            for question in questions:
                started = time.perf_counter()
                candidates = retrieve(strategy, question, indexes)
                latency_ms = (time.perf_counter() - started) * 1000
                rows.append(
                    {
                        "repeat": repeat,
                        "seed": seed,
                        "strategy": strategy,
                        "question_id": question["id"],
                        "category": question["category"],
                        "latency_ms": latency_ms,
                        "context_words": sum(
                            len(tokenize(candidate["text"])) for candidate in candidates
                        ),
                        **retrieval_metrics(question, candidates),
                    }
                )
    summaries = []
    for strategy in strategies:
        selected = [row for row in rows if row["strategy"] == strategy]
        summaries.append(
            {
                "strategy": strategy,
                "recall": average(row["recall"] for row in selected),
                "context_precision": average(
                    row["context_precision"] for row in selected
                ),
                "mrr": average(row["mrr"] for row in selected),
                "temporal_exact": average(
                    float(row["temporal_exact"])
                    if row["temporal_exact"] is not None
                    else None
                    for row in selected
                ),
                "latency_ms": average(row["latency_ms"] for row in selected),
                "runs": len(selected),
            }
        )
    return {"rows": rows, "summaries": summaries}


def run_ollama(
    indexes: IndexSet,
    questions: list[dict[str, Any]],
    models: list[str],
    execution_seed: int,
) -> dict[str, Any]:
    strategies = ["long_context", "hybrid", "routed", "parallel_merge"]
    selected_ids = {
        "q01",
        "q02",
        "q03",
        "q05",
        "q06",
        "q09",
        "q10",
        "q13",
        "q17",
        "q18",
    }
    selected_questions = [
        question for question in questions if question["id"] in selected_ids
    ]
    rows: list[dict[str, Any]] = []
    model_order = [
        item[0] for item in interleaved_product(models, seed=execution_seed)
    ]
    for model_index, model in enumerate(model_order):
        schedule = interleaved_product(
            list(enumerate(REPEATS, start=1)),
            strategies,
            selected_questions,
            seed=execution_seed + model_index + 1,
        )
        for repeat_seed, strategy, question in schedule:
            repeat, seed = repeat_seed
            candidates = retrieve(strategy, question, indexes)
            answer = ollama_answer(model, question, candidates, seed)
            rows.append(
                {
                    "model": model,
                    "repeat": repeat,
                    "seed": seed,
                    "strategy": strategy,
                    "question_id": question["id"],
                    "category": question["category"],
                    **answer,
                }
            )
    summaries = []
    for model in models:
        for strategy in strategies:
            selected = [
                row
                for row in rows
                if row["model"] == model and row["strategy"] == strategy
            ]
            summaries.append(
                {
                    "model": model,
                    "strategy": strategy,
                    "accuracy": average(float(row["correct"]) for row in selected),
                    "abstention_rate": average(
                        float(row["abstained"]) for row in selected
                    ),
                    "latency_ms": average(row["latency_ms"] for row in selected),
                    "prompt_tokens": sum(row["prompt_tokens"] for row in selected),
                    "output_tokens": sum(row["output_tokens"] for row in selected),
                    "runs": len(selected),
                }
            )
    return {"rows": rows, "summaries": summaries}


def write_results(payload: dict[str, Any]) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RESULTS / f"PROTOTYPE-{timestamp}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    latest = RESULTS / "PROTOTYPE-latest.json"
    latest.write_text(path.read_text())
    return path


def print_summary(payload: dict[str, Any], path: Path) -> None:
    print("\nPROTOTYPE STATE")
    print(f"  corpus documents: {payload['manifest']['documents']}")
    print(f"  atomic facts:     {payload['manifest']['facts']}")
    print(f"  questions:        {payload['manifest']['questions']}")
    print(f"  repetitions:      {payload['manifest']['repetitions']}")
    print("\nRETRIEVAL")
    for row in payload["retrieval"]["summaries"]:
        print(
            f"  {row['strategy']:<16} "
            f"recall={row['recall']:.3f} "
            f"precision={row['context_precision']:.3f} "
            f"temporal={row['temporal_exact']:.3f} "
            f"latency={row['latency_ms']:.2f}ms"
        )
    if payload.get("ollama"):
        print("\nLOCAL READERS")
        for row in payload["ollama"]["summaries"]:
            print(
                f"  {row['model']:<12} {row['strategy']:<16} "
                f"accuracy={row['accuracy']:.3f} "
                f"latency={row['latency_ms']:.0f}ms "
                f"tokens={row['prompt_tokens'] + row['output_tokens']}"
            )
    print(f"\nRESULT FILE\n  {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-ollama", nargs="+", default=[])
    parser.add_argument("--minilm-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--execution-seed", type=int, default=20260729)
    args = parser.parse_args()

    questions = load_jsonl(DATA / "questions.jsonl")
    indexes = build_indexes(args.minilm_dir)
    payload: dict[str, Any] = {
        "manifest": {
            "prototype": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "documents": len(indexes.documents),
            "facts": len(indexes.facts),
            "shards": len(indexes.shards),
            "questions": len(questions),
            "repetitions": len(REPEATS),
            "embedding_model": "all-MiniLM-L6-v2 local ONNX",
            "reader_models": args.with_ollama,
            "execution_seed": args.execution_seed,
            "model_execution": "grouped to avoid local model reload confounding",
            "raw_data_immutable": True,
        },
        "retrieval": run_retrieval(indexes, questions),
    }
    if args.with_ollama:
        payload["ollama"] = run_ollama(
            indexes,
            questions,
            args.with_ollama,
            args.execution_seed,
        )
    path = write_results(payload)
    print_summary(payload, path)


if __name__ == "__main__":
    main()
