from __future__ import annotations

import argparse
import asyncio
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
os.environ.setdefault("EMBEDDING_DIM", "768")
os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")

from redislite.async_falkordb_client import AsyncFalkorDB

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client import LLMConfig, OpenAIClient
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF
from graphiti_core.search.search_filters import (
    ComparisonOperator,
    DateFilter,
    SearchFilters,
)
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

from graph_benchmark_common import canonical_record, load_jsonl, score_retrieval, write_result


class IdentityCrossEncoder(CrossEncoderClient):
    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        return [(passage, 1.0 - index / max(len(passages), 1)) for index, passage in enumerate(passages)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen2.5:14b")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit-docs", type=int, default=0)
    parser.add_argument("--limit-questions", type=int, default=0)
    parser.add_argument("--ingestion-timeout", type=float, default=600.0)
    parser.add_argument("--document-timeout", type=float, default=180.0)
    parser.add_argument("--query-timeout", type=float, default=60.0)
    return parser.parse_args()


def as_utc(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def temporal_filters(query_date: datetime) -> SearchFilters:
    return SearchFilters(
        valid_at=[
            [
                DateFilter(
                    date=query_date,
                    comparison_operator=ComparisonOperator.less_than_equal,
                )
            ],
            [DateFilter(comparison_operator=ComparisonOperator.is_null)],
        ],
        invalid_at=[
            [
                DateFilter(
                    date=query_date,
                    comparison_operator=ComparisonOperator.greater_than,
                )
            ],
            [DateFilter(comparison_operator=ComparisonOperator.is_null)],
        ],
    )


async def run(args: argparse.Namespace) -> None:
    corpus = load_jsonl(args.corpus)
    questions = load_jsonl(args.questions)
    if args.limit_docs:
        corpus = corpus[: args.limit_docs]
    allowed_sources = {document["id"] for document in corpus}
    questions = [
        question
        for question in questions
        if set(question["gold_source_ids"]).issubset(allowed_sources)
    ]
    if args.limit_questions:
        questions = questions[: args.limit_questions]

    db_path = str(Path(tempfile.mkdtemp(prefix="graphiti_common_")) / "graphiti.db")
    falkor_db = AsyncFalkorDB(dbfilename=db_path)
    driver = FalkorDriver(falkor_db=falkor_db)
    llm_config = LLMConfig(
        api_key="ollama",
        model=args.model,
        small_model=args.model,
        base_url="http://127.0.0.1:11434/v1",
        temperature=0,
        max_tokens=4096,
    )
    llm_client = OpenAIClient(config=llm_config)
    embedder = OpenAIEmbedder(
        OpenAIEmbedderConfig(
            api_key="ollama",
            base_url="http://127.0.0.1:11434/v1",
            embedding_model="nomic-embed-text",
            embedding_dim=768,
        )
    )
    graphiti = Graphiti(
        graph_driver=driver,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=IdentityCrossEncoder(),
        max_coroutines=1,
    )
    group_id = f"common-{uuid4().hex}"
    episode_to_source: dict[str, str] = {}

    started = time.perf_counter()
    try:
        await clear_data(graphiti.driver)
        await graphiti.build_indices_and_constraints()
        previous_episode_uuids: list[str] = []
        ingestion_errors: list[dict[str, str]] = []
        for document_index, document in enumerate(corpus):
            remaining_budget = args.ingestion_timeout - (time.perf_counter() - started)
            if remaining_budget <= 0:
                ingestion_errors.extend(
                    {
                        "source_id": pending["id"],
                        "error": (
                            "not attempted: total ingestion budget "
                            f"{args.ingestion_timeout:.1f}s exhausted"
                        ),
                    }
                    for pending in corpus[document_index:]
                )
                break
            try:
                result = await asyncio.wait_for(
                    graphiti.add_episode(
                        name=f"SOURCE {document['id']}",
                        episode_body=canonical_record(document),
                        source=EpisodeType.json,
                        source_description=f"{document['kind']} source {document['id']}",
                        reference_time=as_utc(document["timestamp"]),
                        group_id=group_id,
                        previous_episode_uuids=previous_episode_uuids[-3:],
                    ),
                    timeout=min(args.document_timeout, remaining_budget),
                )
                episode_to_source[result.episode.uuid] = document["id"]
                previous_episode_uuids.append(result.episode.uuid)
            except TimeoutError:
                ingestion_errors.append(
                    {
                        "source_id": document["id"],
                        "error": f"timeout after {args.document_timeout:.1f}s",
                    }
                )
            except Exception as error:
                ingestion_errors.append({"source_id": document["id"], "error": str(error)})
            write_result(
                args.output.with_suffix(args.output.suffix + ".checkpoint.json"),
                {
                    "system": "Graphiti",
                    "model": args.model,
                    "documents_requested": len(corpus),
                    "documents_completed": len(episode_to_source),
                    "documents_attempted": len(episode_to_source) + len(ingestion_errors),
                    "ingestion_errors": ingestion_errors,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                },
            )
        ingestion_seconds = time.perf_counter() - started

        rows = []
        search_config = COMBINED_HYBRID_SEARCH_RRF.model_copy(deep=True)
        search_config.limit = args.top_k
        for question in questions:
            query_started = time.perf_counter()
            query_date = as_utc(question["query_date"])
            try:
                search_results = await asyncio.wait_for(
                    graphiti.search_(
                        query=question["query"],
                        config=search_config,
                        group_ids=[group_id],
                        search_filter=temporal_filters(query_date),
                    ),
                    timeout=args.query_timeout,
                )
            except TimeoutError:
                rows.append(
                    {
                        "question_id": question["id"],
                        "query": question["query"],
                        "retrieved_source_ids": [],
                        "latency_ms": round((time.perf_counter() - query_started) * 1000, 3),
                        "error": f"timeout after {args.query_timeout:.1f}s",
                        "results": [],
                    }
                )
                continue
            valid_edges = search_results.edges
            source_ids: list[str] = []
            if "temporal" not in question["category"]:
                for episode in search_results.episodes:
                    source_id = episode_to_source.get(episode.uuid)
                    if source_id and source_id not in source_ids:
                        source_ids.append(source_id)
            for edge in valid_edges:
                for episode_uuid in edge.episodes:
                    source_id = episode_to_source.get(episode_uuid)
                    if source_id and source_id not in source_ids:
                        source_ids.append(source_id)
            rows.append(
                {
                    "question_id": question["id"],
                    "query": question["query"],
                    "retrieved_source_ids": source_ids,
                    "latency_ms": round((time.perf_counter() - query_started) * 1000, 3),
                    "results": [
                        {
                            "fact": edge.fact,
                            "valid_at": edge.valid_at.isoformat() if edge.valid_at else None,
                            "invalid_at": edge.invalid_at.isoformat() if edge.invalid_at else None,
                            "source_ids": [
                                episode_to_source[episode_uuid]
                                for episode_uuid in edge.episodes
                                if episode_uuid in episode_to_source
                            ],
                        }
                        for edge in valid_edges
                    ],
                    "episodes": [
                        {
                            "source_id": episode_to_source.get(episode.uuid),
                            "valid_at": episode.valid_at.isoformat(),
                        }
                        for episode in search_results.episodes
                    ],
                }
            )

        total_usage = graphiti.token_tracker.get_total_usage()
        usage_by_prompt = graphiti.token_tracker.get_usage()
        payload = {
            "system": "Graphiti",
            "model": args.model,
            "embedding_model": "nomic-embed-text",
            "backend": "FalkorDB Lite",
            "top_k": args.top_k,
            "documents": len(corpus),
            "budget": {
                "ingestion_timeout_seconds": args.ingestion_timeout,
                "document_timeout_seconds": args.document_timeout,
                "query_timeout_seconds": args.query_timeout,
            },
            "ingestion_seconds": round(ingestion_seconds, 3),
            "ingestion_errors": ingestion_errors,
            "llm_tokens": {
                "input": total_usage.input_tokens,
                "output": total_usage.output_tokens,
                "total": total_usage.total_tokens,
                "by_prompt": {
                    name: {
                        "calls": usage.call_count,
                        "input": usage.total_input_tokens,
                        "output": usage.total_output_tokens,
                        "total": usage.total_tokens,
                    }
                    for name, usage in usage_by_prompt.items()
                },
            },
            "metrics": score_retrieval(questions, rows),
            "rows": rows,
        }
        write_result(args.output, payload)
    finally:
        await graphiti.close()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
