from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from graph_benchmark_common import load_jsonl, score_retrieval, write_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--prototype", type=Path, required=True)
    parser.add_argument("--jcode", type=Path, required=True)
    parser.add_argument("--mem0", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = sorted({name for run in runs for name in run if name != "questions"})
    summary = {}
    for name in metric_names:
        values = [float(run[name]) for run in runs]
        summary[name] = {
            "mean": statistics.fmean(values),
            "sample_stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }
    return summary


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    questions = load_jsonl(args.questions)
    prototype = json.loads(args.prototype.read_text())
    jcode = json.loads(args.jcode.read_text())
    mem0 = json.loads(args.mem0.read_text())

    prototype_rows = prototype["retrieval"]["rows"]
    strategies = sorted({row["strategy"] for row in prototype_rows})
    repetitions = sorted({int(row["repeat"]) for row in prototype_rows})
    comparison = {}
    for strategy in strategies:
        runs = []
        for repetition in repetitions:
            rows = [
                row
                for row in prototype_rows
                if row["strategy"] == strategy and int(row["repeat"]) == repetition
            ]
            runs.append(score_retrieval(questions, rows))
        comparison[strategy] = {
            "system": "prototype",
            "summary": summarize_runs(runs),
        }

    for strategy, summary in jcode["summary"].items():
        comparison[f"jcode_{strategy}"] = {
            "system": "jcode",
            "summary": summary,
        }

    for strategy, summary in mem0["summary"].items():
        comparison[f"mem0_{strategy}"] = {
            "system": "Mem0 OSS",
            "summary": summary["metrics"],
        }

    first_repeat = jcode["repeats"][0]
    default_ids = set(first_repeat["import"]["memory_ids"])
    preserved_ids = set(first_repeat["preserved_import"]["memory_ids"])
    first_mem0_repeat = mem0["repeats"][0]
    mem0_raw_stored = int(first_mem0_repeat["raw"]["stored_memories"])
    mem0_infer_stored = int(first_mem0_repeat["infer"]["stored_memories"])
    return {
        "protocol": {
            "questions": len(questions),
            "top_k": jcode["protocol"]["top_k"],
            "repetitions": len(repetitions),
            "metric_contract": "graph_benchmark_common.score_retrieval",
            "latency_warning": (
                "Implementations and process boundaries differ; compare quality directly, "
                "but treat latency as descriptive rather than a fair leaderboard."
            ),
        },
        "ingestion": {
            "jcode_default_import_memory_count": len(default_ids),
            "jcode_preserved_upsert_memory_count": len(preserved_ids),
            "jcode_default_retention": len(default_ids) / len(preserved_ids),
            "lost_stable_ids": sorted(preserved_ids - default_ids),
            "reinforcement_count": first_repeat["import"]["reinforcement_count"],
            "mem0_raw_memory_count": mem0_raw_stored,
            "mem0_infer_memory_count": mem0_infer_stored,
            "mem0_infer_extra_records_over_sources": (
                mem0_infer_stored - mem0["protocol"]["documents"]
            ),
            "mem0_raw_retained_sources": first_mem0_repeat["raw"][
                "retained_source_count"
            ],
            "mem0_infer_retained_sources": first_mem0_repeat["infer"][
                "retained_source_count"
            ],
        },
        "comparison": comparison,
    }


if __name__ == "__main__":
    arguments = parse_args()
    write_result(arguments.output, analyze(arguments))
