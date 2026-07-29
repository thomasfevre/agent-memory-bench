from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graph_benchmark_common import load_jsonl, score_retrieval, write_result


MEMORY_ID_PATTERN = re.compile(r"(?m)^\s*id:\s*(d\d{2})\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jcode", type=Path, required=True)
    parser.add_argument("--native-bench", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, default=Path("/tmp/jcode-common-benchmark"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--command-timeout", type=float, default=120.0)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def trust_for_confidence(confidence: float) -> str:
    if confidence >= 0.85:
        return "high"
    if confidence <= 0.3:
        return "low"
    return "medium"


def build_memory_entries(corpus: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries = []
    for document in corpus:
        timestamp = f"{document['timestamp']}T00:00:00Z"
        entries.append(
            {
                "id": document["id"],
                "category": "fact",
                "content": document["text"],
                "tags": [document["kind"], f"source:{document['id']}"],
                "search_text": "",
                "created_at": timestamp,
                "updated_at": timestamp,
                "access_count": 0,
                "source": f"benchmark:{document['id']}",
                "trust": trust_for_confidence(float(document["confidence"])),
                "strength": 1,
                "active": True,
                "superseded_by": None,
                "reinforcements": [],
                "embedding": None,
                "embedding_model": None,
                "confidence": float(document["confidence"]),
            }
        )
    return entries


def parse_search_output(output: str, top_k: int) -> list[str]:
    result = []
    for source_id in MEMORY_ID_PATTERN.findall(output):
        normalized = source_id.lower()
        if normalized not in result:
            result.append(normalized)
        if len(result) >= top_k:
            break
    return result


def run_command(
    command: list[str],
    *,
    env: dict[str, str],
    timeout: float,
) -> tuple[subprocess.CompletedProcess[str], float]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed, elapsed_ms


def prepare_home(home: Path, model_dir: Path) -> None:
    home.mkdir(parents=True, exist_ok=False)
    models_parent = home / "models"
    models_parent.mkdir(parents=True)
    (models_parent / "all-MiniLM-L6-v2").symlink_to(model_dir, target_is_directory=True)


def load_graph_summary(home: Path) -> dict[str, Any]:
    graph_path = home / "memory" / "global.json"
    graph = json.loads(graph_path.read_text())
    memories = graph.get("memories", {})
    return {
        "path": str(graph_path),
        "sha256": sha256_file(graph_path),
        "memory_count": len(memories),
        "memory_ids": sorted(memories),
        "reinforcement_count": sum(
            len(memory.get("reinforcements", [])) for memory in memories.values()
        ),
    }


def search_cli(
    *,
    jcode: Path,
    home: Path,
    questions: list[dict[str, Any]],
    top_k: int,
    semantic: bool,
    timeout: float,
) -> list[dict[str, Any]]:
    env = {
        **os.environ,
        "JCODE_HOME": str(home),
        "JCODE_NO_TELEMETRY": "1",
    }
    rows = []
    for question in questions:
        command = [
            str(jcode),
            "--no-update",
            "--quiet",
            "memory",
            "search",
            question["query"],
        ]
        if semantic:
            command.append("--semantic")
        completed, latency_ms = run_command(command, env=env, timeout=timeout)
        rows.append(
            {
                "question_id": question["id"],
                "query": question["query"],
                "retrieved_source_ids": parse_search_output(
                    completed.stdout,
                    top_k=top_k,
                ),
                "latency_ms": round(latency_ms, 3),
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
    return rows


def search_native_hybrid(
    *,
    native_bench: Path,
    home: Path,
    questions_path: Path,
    top_k: int,
    timeout: float,
    import_path: Path | None = None,
) -> tuple[list[dict[str, Any]], float]:
    env = {
        **os.environ,
        "JCODE_HOME": str(home),
        "JCODE_NO_TELEMETRY": "1",
    }
    command = [
        str(native_bench),
        "--questions",
        str(questions_path),
        "--top-k",
        str(top_k),
    ]
    if import_path is not None:
        command.extend(["--import", str(import_path)])
    completed, wall_ms = run_command(
        command,
        env=env,
        timeout=timeout,
    )
    native_rows = json.loads(completed.stdout)
    rows = [
        {
            "question_id": row["question_id"],
            "query": row["query"],
            "retrieved_source_ids": [item["id"] for item in row["retrieved"]],
            "latency_ms": round(float(row["latency_ms"]), 3),
            "results": row["retrieved"],
        }
        for row in native_rows
    ]
    return rows, wall_ms


def numeric_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and key != "questions"
    }


def aggregate_repetitions(repeats: list[dict[str, Any]]) -> dict[str, Any]:
    strategies = sorted(
        {
            strategy
            for repeat in repeats
            for strategy in repeat.get("strategies", {})
        }
    )
    summary: dict[str, Any] = {}
    for strategy in strategies:
        metric_rows = [
            numeric_metrics(repeat["strategies"][strategy]["metrics"])
            for repeat in repeats
            if strategy in repeat.get("strategies", {})
        ]
        metric_names = sorted({key for row in metric_rows for key in row})
        summary[strategy] = {}
        for metric_name in metric_names:
            values = [row[metric_name] for row in metric_rows if metric_name in row]
            summary[strategy][metric_name] = {
                "mean": statistics.fmean(values),
                "sample_stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
                "values": values,
            }
    return summary


def run(args: argparse.Namespace) -> None:
    corpus = load_jsonl(args.corpus)
    questions = load_jsonl(args.questions)
    entries = build_memory_entries(corpus)
    args.work_root.mkdir(parents=True, exist_ok=True)
    repeats = []

    for repeat_index in range(1, args.repetitions + 1):
        run_dir = Path(
            tempfile.mkdtemp(
                prefix=f"repeat-{repeat_index}-",
                dir=args.work_root,
            )
        )
        home = run_dir / "home"
        prepare_home(home, args.model_dir)
        preserved_home = run_dir / "preserved-home"
        prepare_home(preserved_home, args.model_dir)
        import_path = run_dir / "memories.json"
        import_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n")
        env = {
            **os.environ,
            "JCODE_HOME": str(home),
            "JCODE_NO_TELEMETRY": "1",
        }
        imported, import_latency_ms = run_command(
            [
                str(args.jcode),
                "--no-update",
                "--quiet",
                "memory",
                "import",
                str(import_path),
                "--scope",
                "global",
            ],
            env=env,
            timeout=args.command_timeout,
        )
        graph_summary = load_graph_summary(home)

        strategy_rows: dict[str, list[dict[str, Any]]] = {}
        strategy_rows["cli_keyword"] = search_cli(
            jcode=args.jcode,
            home=home,
            questions=questions,
            top_k=args.top_k,
            semantic=False,
            timeout=args.command_timeout,
        )
        strategy_rows["cli_semantic"] = search_cli(
            jcode=args.jcode,
            home=home,
            questions=questions,
            top_k=args.top_k,
            semantic=True,
            timeout=args.command_timeout,
        )
        default_native_rows, default_native_wall_ms = search_native_hybrid(
            native_bench=args.native_bench,
            home=home,
            questions_path=args.questions,
            top_k=args.top_k,
            timeout=args.command_timeout,
        )
        strategy_rows["prod_hybrid_default_import"] = default_native_rows
        preserved_native_rows, preserved_native_wall_ms = search_native_hybrid(
            native_bench=args.native_bench,
            home=preserved_home,
            questions_path=args.questions,
            top_k=args.top_k,
            timeout=args.command_timeout,
            import_path=import_path,
        )
        strategy_rows["prod_hybrid_preserved_upsert"] = preserved_native_rows
        preserved_graph_summary = load_graph_summary(preserved_home)

        strategies = {}
        for strategy, rows in strategy_rows.items():
            metrics = score_retrieval(questions, rows)
            warm_latencies = [float(row["latency_ms"]) for row in rows[1:]]
            strategies[strategy] = {
                "metrics": metrics,
                "warm_mean_latency_ms_excluding_first_query": (
                    statistics.fmean(warm_latencies) if warm_latencies else 0.0
                ),
                "rows": rows,
            }
        strategies["prod_hybrid_default_import"]["process_wall_ms"] = round(
            default_native_wall_ms,
            3,
        )
        strategies["prod_hybrid_preserved_upsert"]["process_wall_ms"] = round(
            preserved_native_wall_ms,
            3,
        )

        repeat = {
            "repeat": repeat_index,
            "run_dir": str(run_dir),
            "import": {
                "mode": "official CLI import with semantic deduplication",
                "stdout": imported.stdout,
                "stderr": imported.stderr,
                "latency_ms": round(import_latency_ms, 3),
                **graph_summary,
            },
            "preserved_import": {
                "mode": "shipped upsert_global_memory API, preserving stable IDs",
                **preserved_graph_summary,
            },
            "strategies": strategies,
        }
        repeats.append(repeat)
        checkpoint = {
            "status": "running",
            "completed_repetitions": repeat_index,
            "expected_repetitions": args.repetitions,
            "repeats": repeats,
        }
        write_result(args.output.with_suffix(args.output.suffix + ".checkpoint.json"), checkpoint)

    payload = {
        "system": "jcode",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "documents": len(corpus),
            "questions": len(questions),
            "top_k": args.top_k,
            "repetitions": args.repetitions,
            "scope": "isolated global memory per repetition",
            "embedding_model": "all-MiniLM-L6-v2 local ONNX",
            "strategies": {
                "cli_keyword": "official CLI substring search, not BM25",
                "cli_semantic": "official CLI dense search with cosine floor 0.3",
                "prod_hybrid_default_import": (
                    "real MemoryManager.find_similar_hybrid after official CLI import "
                    "and its storage deduplication"
                ),
                "prod_hybrid_preserved_upsert": (
                    "same production hybrid retriever after the shipped upsert API "
                    "preserves all stable source IDs"
                ),
            },
            "latency_note": (
                "CLI latency includes a fresh process and model load per query; "
                "prod_hybrid runs all queries in one persistent process."
            ),
        },
        "artifacts": {
            "jcode_binary": str(args.jcode),
            "jcode_binary_sha256": sha256_file(args.jcode),
            "native_benchmark_binary": str(args.native_bench),
            "native_benchmark_binary_sha256": sha256_file(args.native_bench),
            "corpus": str(args.corpus),
            "corpus_sha256": sha256_file(args.corpus),
            "questions": str(args.questions),
            "questions_sha256": sha256_file(args.questions),
        },
        "repeats": repeats,
        "summary": aggregate_repetitions(repeats),
    }
    write_result(args.output, payload)


if __name__ == "__main__":
    run(parse_args())
