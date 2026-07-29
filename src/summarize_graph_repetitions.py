from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from graph_benchmark_common import write_result


METRIC_FIELDS = (
    "mean_recall",
    "mean_context_precision",
    "temporal_correctness",
    "temporal_context_precision",
    "temporal_exact_source_set",
    "mean_latency_ms",
)


def summarize_values(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "population_stddev": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def completed_documents(payload: dict[str, Any]) -> int:
    documents = int(payload["documents"])
    if "ingestion_errors" in payload:
        return documents - len(payload["ingestion_errors"])
    return documents if not payload.get("ingestion_error") else 0


def summarize_system(paths: list[Path]) -> dict[str, Any]:
    payloads = [json.loads(path.read_text()) for path in paths]
    run_rows = []
    for path, payload in zip(paths, payloads, strict=True):
        metrics = payload["metrics"]
        run_rows.append(
            {
                "file": path.name,
                "documents_completed": completed_documents(payload),
                "documents_requested": int(payload["documents"]),
                "ingestion_seconds": float(payload["ingestion_seconds"]),
                **{field: float(metrics[field]) for field in METRIC_FIELDS},
            }
        )

    numeric_fields = (
        "documents_completed",
        "ingestion_seconds",
        *METRIC_FIELDS,
    )
    return {
        "runs": run_rows,
        "summary": {
            field: summarize_values([float(row[field]) for row in run_rows])
            for field in numeric_fields
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cognee", nargs="+", type=Path, required=True)
    parser.add_argument("--graphiti", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    write_result(
        args.output,
        {
            "protocol": "common-temporal-graph",
            "repetitions_per_system": {
                "cognee": len(args.cognee),
                "graphiti": len(args.graphiti),
            },
            "systems": {
                "cognee": summarize_system(args.cognee),
                "graphiti": summarize_system(args.graphiti),
            },
        },
    )


if __name__ == "__main__":
    main()
