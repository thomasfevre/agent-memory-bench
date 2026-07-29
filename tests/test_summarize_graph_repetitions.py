from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from summarize_graph_repetitions import completed_documents, summarize_system


def result_payload(recall: float, errors: list[dict[str, str]] | None = None) -> dict:
    payload = {
        "documents": 8,
        "ingestion_seconds": 100,
        "metrics": {
            "mean_recall": recall,
            "mean_context_precision": 0.5,
            "temporal_correctness": 0.6,
            "temporal_context_precision": 0.4,
            "temporal_exact_source_set": 0.2,
            "mean_latency_ms": 25,
        },
    }
    if errors is not None:
        payload["ingestion_errors"] = errors
    else:
        payload["ingestion_error"] = None
    return payload


class SummarizeGraphRepetitionsTests(unittest.TestCase):
    def test_completed_documents_handles_partial_graphiti_run(self) -> None:
        payload = result_payload(0.5, [{"source_id": "d07", "error": "timeout"}])
        self.assertEqual(completed_documents(payload), 7)

    def test_summary_reports_variance_and_range(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, recall in enumerate((0.9, 0.7, 0.5), start=1):
                path = Path(directory) / f"run-{index}.json"
                path.write_text(json.dumps(result_payload(recall)))
                paths.append(path)

            summary = summarize_system(paths)["summary"]["mean_recall"]

        self.assertAlmostEqual(summary["mean"], 0.7)
        self.assertAlmostEqual(summary["population_stddev"], 0.1632993162)
        self.assertEqual(summary["min"], 0.5)
        self.assertEqual(summary["max"], 0.9)


if __name__ == "__main__":
    unittest.main()
