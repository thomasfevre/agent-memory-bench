from __future__ import annotations

import asyncio
import os
import tempfile
import time
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

_DATA_DIR = Path(tempfile.mkdtemp(prefix="cognee_common_"))
os.environ["DATA_ROOT_DIRECTORY"] = str(_DATA_DIR / "data")
os.environ["SYSTEM_ROOT_DIRECTORY"] = str(_DATA_DIR / "system")
os.environ["CACHE_ROOT_DIRECTORY"] = str(_DATA_DIR / "cache")
os.environ["COGNEE_LOGS_DIR"] = str(_DATA_DIR / "logs")
os.environ["COGNEE_LOG_FILE"] = "false"
os.environ["COGNEE_CLI_MODE"] = "true"
os.environ["COGNEE_TRACING_ENABLED"] = "false"
os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"
os.environ["CACHING"] = "false"
os.environ["TELEMETRY_DISABLED"] = "1"
os.environ["LLM_PROVIDER"] = "ollama"
os.environ.setdefault("LLM_MODEL", "qwen2.5:14b")
os.environ["LLM_ENDPOINT"] = "http://127.0.0.1:11434/v1"
os.environ["LLM_API_KEY"] = "ollama"
os.environ["LLM_TEMPERATURE"] = "0.0"
os.environ["EMBEDDING_PROVIDER"] = "ollama"
os.environ["EMBEDDING_MODEL"] = "nomic-embed-text"
os.environ["EMBEDDING_ENDPOINT"] = "http://127.0.0.1:11434/api/embed"
os.environ["EMBEDDING_DIMENSIONS"] = "768"
os.environ["HUGGINGFACE_TOKENIZER"] = "nomic-ai/nomic-embed-text-v1.5"

import cognee
from cognee import SearchType
from cognee.tasks.ingestion.data_item import DataItem

from graph_benchmark_common import (
    canonical_record,
    flatten_text,
    load_jsonl,
    score_retrieval,
    source_ids_from_text,
    write_result,
)
from graph_engine_cli import parse_cognee_args

cognee.config.set_graph_database_provider("kuzu")
cognee.config.set_vector_db_provider("lancedb")
cognee.config.data_root_directory(str(_DATA_DIR / "data"))
cognee.config.system_root_directory(str(_DATA_DIR / "system"))


async def run(args) -> None:
    os.environ["LLM_MODEL"] = args.model
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

    dataset = "common_graph_benchmark"
    items = [
        DataItem(
            data=canonical_record(document),
            label=document["id"],
            data_id=uuid5(NAMESPACE_URL, f"memory-benchmark:{document['id']}"),
            external_metadata={
                "source_id": document["id"],
                "timestamp": document["timestamp"],
            },
        )
        for document in corpus
    ]
    started = time.perf_counter()
    ingestion_error = None

    async def ingest() -> None:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
        await cognee.add(
            items,
            dataset_name=dataset,
            run_in_background=False,
            data_cache=False,
        )
        await cognee.cognify(
            datasets=[dataset],
            temporal_cognify=True,
            run_in_background=False,
            data_cache=False,
            chunks_per_batch=args.chunks_per_batch,
            data_per_batch=args.data_per_batch,
        )

    try:
        await asyncio.wait_for(
            ingest(),
            timeout=args.ingestion_timeout,
        )
    except TimeoutError:
        ingestion_error = f"timeout after {args.ingestion_timeout:.1f}s"
    except Exception as error:
        ingestion_error = str(error)
    ingestion_seconds = time.perf_counter() - started

    rows = []
    if ingestion_error is None:
        for question in questions:
            query_started = time.perf_counter()
            query_type = (
                SearchType.TEMPORAL
                if "temporal" in question["category"]
                else SearchType.GRAPH_COMPLETION
            )
            try:
                results = await asyncio.wait_for(
                    cognee.search(
                        query_type=query_type,
                        query_text=question["query"],
                        datasets=[dataset],
                        top_k=args.top_k,
                        only_context=True,
                        include_references=True,
                        verbose=True,
                    ),
                    timeout=args.query_timeout,
                )
            except TimeoutError:
                rows.append(
                    {
                        "question_id": question["id"],
                        "query": question["query"],
                        "query_type": query_type.value,
                        "retrieved_source_ids": [],
                        "latency_ms": round((time.perf_counter() - query_started) * 1000, 3),
                        "error": f"timeout after {args.query_timeout:.1f}s",
                        "results_text": "",
                    }
                )
                continue
            result_text = flatten_text(results)
            rows.append(
                {
                    "question_id": question["id"],
                    "query": question["query"],
                    "query_type": query_type.value,
                    "retrieved_source_ids": source_ids_from_text(result_text),
                    "latency_ms": round((time.perf_counter() - query_started) * 1000, 3),
                    "results_text": result_text,
                }
            )

    payload = {
        "system": "Cognee",
        "model": args.model,
        "embedding_model": "nomic-embed-text",
        "backend": "Kuzu and LanceDB embedded",
        "top_k": args.top_k,
        "documents": len(corpus),
        "budget": {
            "ingestion_timeout_seconds": args.ingestion_timeout,
            "query_timeout_seconds": args.query_timeout,
        },
        "engine_configuration": {
            "chunks_per_batch": args.chunks_per_batch,
            "data_per_batch": args.data_per_batch,
        },
        "ingestion_seconds": round(ingestion_seconds, 3),
        "ingestion_error": ingestion_error,
        "metrics": score_retrieval(questions, rows) if rows else None,
        "rows": rows,
    }
    write_result(args.output, payload)


if __name__ == "__main__":
    asyncio.run(run(parse_cognee_args()))
