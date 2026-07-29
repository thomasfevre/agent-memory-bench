#!/usr/bin/env python3
"""Compare lexical, dense, hybrid, and chunk-graph retrieval on GraphRAG-Bench."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import re
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

from benchmark import Bm25, DenseIndex, MiniLm, MODEL_DIR, tokenize
from graph_benchmark_common import write_result
from locomo_hybrid_fusion import weighted_rrf


PROTOCOL_VERSION = "graphrag-bench-chunk-neighbor-v4"
STRATEGIES = ("bm25", "dense", "hybrid", "bm25_graph", "hybrid_graph")
POLARITY_TOKENS = {"no", "not", "never", "neither", "nor", "without"}
STOPWORDS = {
    "about",
    "after",
    "again",
    "against",
    "also",
    "among",
    "because",
    "before",
    "being",
    "between",
    "could",
    "first",
    "from",
    "have",
    "into",
    "other",
    "over",
    "said",
    "than",
    "that",
    "their",
    "there",
    "these",
    "they",
    "this",
    "through",
    "under",
    "upon",
    "were",
    "which",
    "while",
    "with",
    "would",
}


def normalize_phrase(text: str) -> str:
    return " ".join(tokenize(text))


def chunk_text(
    text: str,
    *,
    chunk_words: int,
    overlap_words: int,
) -> list[dict[str, Any]]:
    if chunk_words <= 0 or not 0 <= overlap_words < chunk_words:
        raise ValueError("chunk and overlap sizes are inconsistent")
    words = text.split()
    step = chunk_words - overlap_words
    chunks = []
    for start in range(0, len(words), step):
        selected = words[start : start + chunk_words]
        if not selected:
            continue
        word_end = start + len(selected)
        if chunks and word_end <= chunks[-1]["word_end"]:
            break
        chunks.append(
            {
                "id": f"c{len(chunks):05d}",
                "text": " ".join(selected),
                "word_start": start,
                "word_end": word_end,
            }
        )
    return chunks


def parse_evidence_units(evidence: str) -> list[tuple[str, ...]]:
    return [
        tuple(tokenize(statement))
        for statement in re.split(r"\s*;\s*", evidence or "")
        if tokenize(statement)
    ]


def evidence_token_recall(
    chunk_tokens: Counter[str],
    evidence_tokens: tuple[str, ...],
) -> float:
    evidence_counts = Counter(evidence_tokens)
    overlap = sum(
        min(count, chunk_tokens.get(token, 0))
        for token, count in evidence_counts.items()
    )
    return overlap / len(evidence_tokens) if evidence_tokens else 0.0


def sentence_windows(text: str) -> list[tuple[str, ...]]:
    sentences = [
        tuple(tokenize(sentence))
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if tokenize(sentence)
    ]
    return sentences + [
        left + right for left, right in zip(sentences, sentences[1:])
    ]


def best_compatible_window_recall(
    windows: list[tuple[str, ...]],
    unit: tuple[str, ...],
) -> float:
    evidence_polarity = set(unit) & POLARITY_TOKENS
    return max(
        (
            evidence_token_recall(Counter(window), unit)
            for window in windows
            if (set(window) & POLARITY_TOKENS) == evidence_polarity
        ),
        default=0.0,
    )


def unit_is_covered(
    windows: list[tuple[str, ...]],
    unit: tuple[str, ...],
    *,
    threshold: float,
) -> bool:
    return best_compatible_window_recall(windows, unit) >= threshold


def score_context(
    selected: list[dict[str, Any]],
    all_chunks: list[dict[str, Any]],
    evidence_units: list[tuple[str, ...]],
    *,
    evidence_token_recall_threshold: float,
    chunk_sentence_windows: dict[str, list[tuple[str, ...]]] | None = None,
    resolvable_units: list[tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    if chunk_sentence_windows is None:
        chunk_sentence_windows = {
            chunk["id"]: sentence_windows(chunk["text"]) for chunk in all_chunks
        }
    resolvable = (
        resolvable_units
        if resolvable_units is not None
        else [
            unit
            for unit in evidence_units
            if any(
                unit_is_covered(
                    counts,
                    unit,
                    threshold=evidence_token_recall_threshold,
                )
                for counts in chunk_sentence_windows.values()
            )
        ]
    )
    covered = [
        unit
        for unit in resolvable
        if any(
            unit_is_covered(
                chunk_sentence_windows[chunk["id"]],
                unit,
                threshold=evidence_token_recall_threshold,
            )
            for chunk in selected
        )
    ]
    useful_chunks = sum(
        any(
            unit_is_covered(
                chunk_sentence_windows[chunk["id"]],
                unit,
                threshold=evidence_token_recall_threshold,
            )
            for unit in resolvable
        )
        for chunk in selected
    )
    fully_representable = (
        bool(evidence_units) and len(resolvable) == len(evidence_units)
    )
    return {
        "official_evidence_units": len(evidence_units),
        "resolvable_evidence_units": len(resolvable),
        "covered_evidence_units": len(covered),
        "all_official_evidence_recall": (
            len(covered) / len(evidence_units) if evidence_units else 0.0
        ),
        "conditional_resolvable_evidence_recall": (
            len(covered) / len(resolvable) if resolvable else None
        ),
        "fully_representable": fully_representable,
        "full_official_evidence_coverage": (
            bool(evidence_units) and len(covered) == len(evidence_units)
        ),
        "context_precision": useful_chunks / len(selected) if selected else 0.0,
    }


def build_chunk_graph(
    chunks: list[dict[str, Any]],
    *,
    minimum_document_frequency: int,
    maximum_document_frequency: int,
    maximum_neighbors: int,
) -> dict[str, dict[str, float]]:
    if maximum_neighbors < 0:
        raise ValueError("maximum_neighbors cannot be negative")
    required_sequence_neighbors = min(2, max(0, len(chunks) - 1))
    if maximum_neighbors < required_sequence_neighbors:
        raise ValueError("maximum_neighbors must preserve both sequence neighbors")
    postings: defaultdict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        terms = {
            term
            for term in tokenize(chunk["text"])
            if len(term) >= 5 and term not in STOPWORDS
        }
        for term in terms:
            postings[term].append(chunk["id"])
    pair_scores: Counter[tuple[str, str]] = Counter()
    for chunk_ids in postings.values():
        if not minimum_document_frequency <= len(chunk_ids) <= maximum_document_frequency:
            continue
        for left, right in combinations(sorted(chunk_ids), 2):
            pair_scores[(left, right)] += 1
    graph: dict[str, dict[str, float]] = {chunk["id"]: {} for chunk in chunks}
    for left, right in zip(chunks, chunks[1:]):
        graph[left["id"]][right["id"]] = 1.0
        graph[right["id"]][left["id"]] = 1.0
    for (left, right), score in sorted(
        pair_scores.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        if (
            len(graph[left]) >= maximum_neighbors
            or len(graph[right]) >= maximum_neighbors
        ):
            continue
        graph[left][right] = float(score)
        graph[right][left] = float(score)
    return graph


def graph_expand(
    ranking: list[dict[str, Any]],
    graph: dict[str, dict[str, float]],
    chunks_by_id: dict[str, dict[str, Any]],
    *,
    seed_count: int,
    limit: int,
    graph_slots: int = 2,
) -> list[dict[str, Any]]:
    if not ranking:
        return []
    if limit <= 0:
        raise ValueError("limit must be positive")
    if not 0 <= seed_count <= limit:
        raise ValueError("seed_count must fit within limit")
    if not 0 <= graph_slots <= limit:
        raise ValueError("graph_slots must fit within limit")
    retained_count = max(seed_count, limit - graph_slots)
    retained = ranking[:retained_count]
    retained_ids = {row["id"] for row in retained}
    scores: defaultdict[str, float] = defaultdict(float)
    for rank, seed in enumerate(ranking[:seed_count], start=1):
        neighbors = graph.get(seed["id"], {})
        maximum = max(neighbors.values(), default=1.0)
        for neighbor, weight in neighbors.items():
            if neighbor in retained_ids:
                continue
            scores[neighbor] += 0.8 * (weight / maximum) / rank
    graph_candidates = sorted(
        scores,
        key=lambda item: (-scores[item], item),
    )[: max(0, limit - len(retained))]
    selected = [
        {
            **row,
            "_retriever": "chunk_graph:retained",
        }
        for row in retained
    ] + [
        {
            **chunks_by_id[item_id],
            "_score": scores[item_id],
            "_retriever": "chunk_graph",
        }
        for item_id in graph_candidates
    ]
    selected_ids = {row["id"] for row in selected}
    for row in ranking:
        if len(selected) >= limit:
            break
        if row["id"] not in selected_ids:
            selected.append(row)
            selected_ids.add(row["id"])
    return selected


def graph_statistics(graph: dict[str, dict[str, float]]) -> dict[str, float]:
    edges = {
        tuple(sorted((left, right)))
        for left, neighbors in graph.items()
        for right in neighbors
    }
    degrees = [len(neighbors) for neighbors in graph.values()]
    return {
        "nodes": len(graph),
        "undirected_edges": len(edges),
        "mean_degree": statistics.fmean(degrees) if degrees else 0.0,
        "isolated_node_rate": (
            statistics.fmean(float(degree == 0) for degree in degrees)
            if degrees
            else 0.0
        ),
    }


def exact_mcnemar_p(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(left_only, right_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2 * tail)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["strategy"]].append(row)
    output = []
    for strategy, selected in sorted(grouped.items()):
        resolvable = [
            row
            for row in selected
            if row["conditional_resolvable_evidence_recall"] is not None
        ]
        fully_representable = [
            row for row in selected if row["fully_representable"]
        ]
        official_units = sum(row["official_evidence_units"] for row in selected)
        resolvable_units = sum(
            row["resolvable_evidence_units"] for row in selected
        )
        covered_units = sum(row["covered_evidence_units"] for row in selected)
        output.append(
            {
                "strategy": strategy,
                "questions": len(selected),
                "official_evidence_units": official_units,
                "resolvable_evidence_units": resolvable_units,
                "evidence_unit_representability_rate": (
                    resolvable_units / official_units if official_units else 0.0
                ),
                "questions_with_resolvable_evidence": len(resolvable),
                "resolvable_question_rate": len(resolvable) / len(selected),
                "fully_representable_questions": len(fully_representable),
                "fully_representable_question_rate": (
                    len(fully_representable) / len(selected)
                ),
                "pooled_all_official_evidence_recall": (
                    covered_units / official_units if official_units else 0.0
                ),
                "mean_per_question_all_official_evidence_recall": statistics.fmean(
                    row["all_official_evidence_recall"] for row in selected
                ),
                "mean_conditional_resolvable_evidence_recall": statistics.fmean(
                    row["conditional_resolvable_evidence_recall"]
                    for row in resolvable
                )
                if resolvable
                else 0.0,
                "full_official_evidence_coverage_rate_all_questions": (
                    statistics.fmean(
                        float(row["full_official_evidence_coverage"])
                        for row in selected
                    )
                ),
                "full_official_evidence_coverage_rate_fully_representable": (
                    statistics.fmean(
                        float(row["full_official_evidence_coverage"])
                        for row in fully_representable
                    )
                    if fully_representable
                    else 0.0
                ),
                "mean_context_precision": statistics.fmean(
                    row["context_precision"] for row in selected
                )
                if selected
                else 0.0,
                "mean_context_words": statistics.fmean(
                    row["context_words"] for row in selected
                ),
                "mean_query_latency_ms": statistics.fmean(
                    row["query_latency_ms"] for row in selected
                ),
            }
        )
    return output


def summarize_dimension(
    rows: list[dict[str, Any]],
    dimension: str,
) -> list[dict[str, Any]]:
    grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row[dimension], row["strategy"])].append(row)
    output = []
    for (value, strategy), selected in sorted(grouped.items()):
        summary = summarize(selected)[0]
        output.append({dimension: value, **summary})
    return output


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile needs at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def cluster_bootstrap_deltas(
    values: list[dict[str, Any]],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    if repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    by_source: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in values:
        by_source[row["source"]].append(row)
    sources = sorted(by_source)
    randomizer = random.Random(seed)
    recall_deltas = []
    coverage_deltas = []
    for _ in range(repetitions):
        sampled = [randomizer.choice(sources) for _ in sources]
        replicate = [
            row
            for source in sampled
            for row in by_source[source]
        ]
        recall_deltas.append(
            statistics.fmean(
                row["left_recall"] - row["right_recall"] for row in replicate
            )
        )
        coverage_replicate = [
            row for row in replicate if row["fully_representable"]
        ]
        if coverage_replicate:
            coverage_deltas.append(
                statistics.fmean(
                    float(row["left_full"]) - float(row["right_full"])
                    for row in coverage_replicate
                )
            )
    return {
        "cluster_unit": "source_book",
        "estimand": "question_weighted_macro_difference",
        "clusters": len(sources),
        "repetitions": repetitions,
        "seed": seed,
        "recall_delta_95_percentile_interval": [
            percentile(recall_deltas, 0.025),
            percentile(recall_deltas, 0.975),
        ],
        "full_coverage_delta_95_percentile_interval": [
            percentile(coverage_deltas, 0.025),
            percentile(coverage_deltas, 0.975),
        ]
        if coverage_deltas
        else None,
    }


def paired_comparisons(
    rows: list[dict[str, Any]],
    *,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    by_key = {}
    for row in rows:
        key = (row["source"], row["question_id"], row["strategy"])
        if key in by_key:
            raise ValueError(f"duplicate result key: {key}")
        by_key[key] = row
    pairs = (("bm25_graph", "bm25"), ("hybrid_graph", "hybrid"))
    output = []
    for left, right in pairs:
        values = [
            {
                "source": source,
                "fully_representable": row["fully_representable"],
                "left_full": bool(row["full_official_evidence_coverage"]),
                "right_full": bool(
                    by_key[(source, question_id, right)][
                        "full_official_evidence_coverage"
                    ]
                ),
                "left_recall": row["all_official_evidence_recall"],
                "right_recall": by_key[(source, question_id, right)][
                    "all_official_evidence_recall"
                ],
            }
            for (source, question_id, strategy), row in by_key.items()
            if strategy == left
            and (source, question_id, right) in by_key
        ]
        if not values:
            continue
        coverage_values = [
            row for row in values if row["fully_representable"]
        ]
        left_only = sum(
            row["left_full"] and not row["right_full"] for row in values
            if row["fully_representable"]
        )
        right_only = sum(
            row["right_full"] and not row["left_full"] for row in values
            if row["fully_representable"]
        )
        output.append(
            {
                "left": left,
                "right": right,
                "paired_questions": len(values),
                "paired_fully_representable_questions": len(coverage_values),
                "mean_per_question_all_official_evidence_recall_delta": statistics.fmean(
                    row["left_recall"] - row["right_recall"] for row in values
                ),
                "full_official_evidence_coverage_rate_delta_fully_representable": statistics.fmean(
                    float(row["left_full"]) - float(row["right_full"])
                    for row in coverage_values
                ),
                "left_only_full_coverage": left_only,
                "right_only_full_coverage": right_only,
                "mcnemar_unit": "fully_representable_question",
                "mcnemar_exact_two_sided_p": exact_mcnemar_p(
                    left_only,
                    right_only,
                ),
                "source_cluster_bootstrap": cluster_bootstrap_deltas(
                    values,
                    repetitions=bootstrap_repetitions,
                    seed=bootstrap_seed,
                ),
            }
        )
    return output


def validate_inputs(
    corpora: list[dict[str, Any]],
    questions: list[dict[str, Any]],
) -> None:
    corpus_names = [row["corpus_name"] for row in corpora]
    if len(corpus_names) != len(set(corpus_names)):
        raise ValueError("corpus_name values must be unique")
    question_keys = [(row["source"], row["id"]) for row in questions]
    if len(question_keys) != len(set(question_keys)):
        raise ValueError("(source, question id) values must be unique")
    missing_sources = sorted(
        {row["source"] for row in questions} - set(corpus_names)
    )
    if missing_sources:
        raise ValueError(f"questions reference missing corpora: {missing_sources}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    corpora = json.loads(args.corpus.read_text())
    questions = json.loads(args.questions.read_text())
    if args.sources:
        source_set = set(args.sources)
        available_sources = {row["corpus_name"] for row in corpora}
        unknown_sources = sorted(source_set - available_sources)
        if unknown_sources:
            raise ValueError(f"unknown requested sources: {unknown_sources}")
        corpora = [row for row in corpora if row["corpus_name"] in source_set]
        questions = [row for row in questions if row["source"] in source_set]
    if args.limit_questions is not None:
        if args.limit_questions <= 0:
            raise ValueError("limit_questions must be positive")
        questions = questions[: args.limit_questions]
    if not corpora or not questions:
        raise ValueError("benchmark workload cannot be empty")
    validate_inputs(corpora, questions)
    if not 0 < args.evidence_token_recall_threshold <= 1:
        raise ValueError("evidence token recall threshold must be in (0, 1]")
    if args.top_k > args.ranking_depth:
        raise ValueError("top_k cannot exceed ranking_depth")
    if not 0 <= args.graph_seeds <= args.top_k:
        raise ValueError("graph_seeds must fit within top_k")
    if not 0 <= args.graph_slots <= args.top_k:
        raise ValueError("graph_slots must fit within top_k")
    questions_by_source: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for question in questions:
        questions_by_source[question["source"]].append(question)
    encoder_started = time.perf_counter()
    encoder = MiniLm(args.model_dir)
    encoder_load_seconds = time.perf_counter() - encoder_started
    rows = []
    corpus_stats = []
    for corpus in corpora:
        source = corpus["corpus_name"]
        selected_questions = questions_by_source.get(source, [])
        if not selected_questions:
            continue
        indexing_started = time.perf_counter()
        component_started = time.perf_counter()
        chunks = chunk_text(
            corpus["context"],
            chunk_words=args.chunk_words,
            overlap_words=args.overlap_words,
        )
        chunking_seconds = time.perf_counter() - component_started
        chunks_by_id = {chunk["id"]: chunk for chunk in chunks}
        chunk_sentence_windows = {
            chunk["id"]: sentence_windows(chunk["text"]) for chunk in chunks
        }
        component_started = time.perf_counter()
        bm25 = Bm25(chunks)
        bm25_index_seconds = time.perf_counter() - component_started
        component_started = time.perf_counter()
        dense = DenseIndex(chunks, encoder)
        dense_index_seconds = time.perf_counter() - component_started
        component_started = time.perf_counter()
        graph = build_chunk_graph(
            chunks,
            minimum_document_frequency=args.graph_min_df,
            maximum_document_frequency=args.graph_max_df,
            maximum_neighbors=args.graph_max_neighbors,
        )
        graph_index_seconds = time.perf_counter() - component_started
        corpus_stats.append(
            {
                "source": source,
                "words": len(corpus["context"].split()),
                "chunks": len(chunks),
                "indexing_seconds": time.perf_counter() - indexing_started,
                "chunking_seconds": chunking_seconds,
                "bm25_index_seconds": bm25_index_seconds,
                "dense_index_seconds": dense_index_seconds,
                "graph_index_seconds": graph_index_seconds,
                **graph_statistics(graph),
            }
        )
        for question in selected_questions:
            started = time.perf_counter()
            bm25_ranking = bm25.search(question["question"], args.ranking_depth)
            bm25_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            dense_ranking = dense.search(question["question"], args.ranking_depth)
            dense_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            hybrid_ranking = weighted_rrf(
                bm25_ranking,
                dense_ranking,
                alpha=args.alpha_bm25,
                limit=args.ranking_depth,
            )
            fusion_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            bm25_graph = graph_expand(
                bm25_ranking,
                graph,
                chunks_by_id,
                seed_count=args.graph_seeds,
                limit=args.top_k,
                graph_slots=args.graph_slots,
            )
            bm25_graph_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            hybrid_graph = graph_expand(
                hybrid_ranking,
                graph,
                chunks_by_id,
                seed_count=args.graph_seeds,
                limit=args.top_k,
                graph_slots=args.graph_slots,
            )
            hybrid_graph_ms = (time.perf_counter() - started) * 1000
            rankings = {
                "bm25": bm25_ranking[: args.top_k],
                "dense": dense_ranking[: args.top_k],
                "hybrid": hybrid_ranking[: args.top_k],
                "bm25_graph": bm25_graph,
                "hybrid_graph": hybrid_graph,
            }
            latencies = {
                "bm25": bm25_ms,
                "dense": dense_ms,
                "hybrid": bm25_ms + dense_ms + fusion_ms,
                "bm25_graph": bm25_ms + bm25_graph_ms,
                "hybrid_graph": (
                    bm25_ms + dense_ms + fusion_ms + hybrid_graph_ms
                ),
            }
            units = parse_evidence_units(question.get("evidence", ""))
            resolvable_units = [
                unit
                for unit in units
                if any(
                    unit_is_covered(
                        counts,
                        unit,
                        threshold=args.evidence_token_recall_threshold,
                    )
                    for counts in chunk_sentence_windows.values()
                )
            ]
            for strategy, context in rankings.items():
                score = score_context(
                    context,
                    chunks,
                    units,
                    evidence_token_recall_threshold=(
                        args.evidence_token_recall_threshold
                    ),
                    chunk_sentence_windows=chunk_sentence_windows,
                    resolvable_units=resolvable_units,
                )
                rows.append(
                    {
                        "question_id": question["id"],
                        "source": source,
                        "question_type": question["question_type"],
                        "strategy": strategy,
                        "context_ids": [row["id"] for row in context],
                        "context_words": sum(
                            len(row["text"].split()) for row in context
                        ),
                        "query_latency_ms": latencies[strategy],
                        **score,
                    }
                )
    return {
        "rows": rows,
        "summaries": summarize(rows),
        "question_type_summaries": summarize_dimension(
            rows,
            "question_type",
        ),
        "source_summaries": summarize_dimension(rows, "source"),
        "paired_comparisons": paired_comparisons(
            rows,
            bootstrap_repetitions=args.bootstrap_repetitions,
            bootstrap_seed=args.bootstrap_seed,
        ),
        "encoder_load_seconds": encoder_load_seconds,
        "corpora": corpus_stats,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--sources", nargs="+")
    parser.add_argument("--limit-questions", type=int)
    parser.add_argument("--chunk-words", type=int, default=160)
    parser.add_argument("--overlap-words", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--ranking-depth", type=int, default=20)
    parser.add_argument("--alpha-bm25", type=float, default=0.5)
    parser.add_argument("--graph-seeds", type=int, default=2)
    parser.add_argument("--graph-slots", type=int, default=2)
    parser.add_argument("--graph-min-df", type=int, default=2)
    parser.add_argument("--graph-max-df", type=int, default=8)
    parser.add_argument("--graph-max-neighbors", type=int, default=8)
    parser.add_argument(
        "--evidence-token-recall-threshold",
        type=float,
        default=0.85,
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260729)
    parser.add_argument(
        "--dataset-repository-url",
        default="https://github.com/GraphRAG-Bench/GraphRAG-Benchmark",
    )
    parser.add_argument(
        "--dataset-commit",
        default="fdbab5959b18c96532580877ffe27d112bccc0ec",
    )
    return parser.parse_args()


def installed_versions(names: list[str]) -> dict[str, str]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def detected_git_commit(path: Path) -> str:
    repository = path.resolve().parents[2]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return completed.stdout.strip()


def main() -> None:
    args = parse_args()
    actual_dataset_commit = detected_git_commit(args.corpus)
    if actual_dataset_commit != args.dataset_commit:
        raise ValueError(
            "dataset commit mismatch: "
            f"expected {args.dataset_commit}, found {actual_dataset_commit}"
        )
    payload = run(args)
    payload["manifest"] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_version": PROTOCOL_VERSION,
        "scope": (
            "Retrieval-only GraphRAG-Bench Novel comparison using corpus-only "
            "lexical chunk-neighbor expansion and official evidence statements"
        ),
        "dataset_repository_url": args.dataset_repository_url,
        "dataset_commit": actual_dataset_commit,
        "corpus": str(args.corpus.resolve()),
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "questions": str(args.questions.resolve()),
        "questions_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest(),
        "sources": args.sources,
        "limit_questions": args.limit_questions,
        "strategies": list(STRATEGIES),
        "chunk_words": args.chunk_words,
        "overlap_words": args.overlap_words,
        "top_k": args.top_k,
        "ranking_depth": args.ranking_depth,
        "alpha_bm25": args.alpha_bm25,
        "graph_seeds": args.graph_seeds,
        "graph_slots": args.graph_slots,
        "graph_min_df": args.graph_min_df,
        "graph_max_df": args.graph_max_df,
        "graph_max_neighbors": args.graph_max_neighbors,
        "evidence_token_recall_threshold": (
            args.evidence_token_recall_threshold
        ),
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_seed": args.bootstrap_seed,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "local_module_sha256": {
            name: hashlib.sha256(
                (Path(__file__).parent / name).read_bytes()
            ).hexdigest()
            for name in [
                "benchmark.py",
                "graph_benchmark_common.py",
                "locomo_hybrid_fusion.py",
            ]
        },
        "python_version": sys.version,
        "platform": platform.platform(),
        "dependency_versions": installed_versions(
            ["numpy", "onnxruntime", "tokenizers"]
        ),
        "embedding_model_dir": str(args.model_dir.resolve()),
        "embedding_model_sha256": hashlib.sha256(
            (args.model_dir / "model.onnx").read_bytes()
        ).hexdigest(),
        "embedding_tokenizer_sha256": hashlib.sha256(
            (args.model_dir / "tokenizer.json").read_bytes()
        ).hexdigest(),
        "limitations": [
            "This is a deterministic lexical chunk-neighbor operator, not LightRAG, HippoRAG, FastGraphRAG, or a knowledge graph extracted by an LLM.",
            "Official evidence statements define retrieval units; final answer generation and official LLM judging are not run.",
            "A statement is lexically representable when one sentence or adjacent-sentence window inside a fixed chunk contains at least the configured fraction of its tokens and has the same explicit polarity markers; this is not semantic entailment.",
            "All-official-evidence recall counts unrepresentable statements as missed; conditional metrics and fully representable-question metrics are reported separately.",
            "Operator parameters match the recorded pilot, but the evidence metric was corrected after independent review and the run is not preregistered.",
            "The same MiniLM encoder is used for dense and hybrid retrieval.",
            "Question-level McNemar results are exploratory; source-book cluster bootstrap intervals are the primary uncertainty estimate.",
            "Latency is component-accounted from one fixed-order execution per question, not repeated end-to-end wall-clock latency.",
        ],
    }
    write_result(args.output, payload)
    print(args.output)


if __name__ == "__main__":
    main()
