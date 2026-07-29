from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_tool(name: str):
    path = ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registry_has_unique_runs_and_declared_evidence() -> None:
    registry = json.loads(
        (ROOT / "results" / "published" / "registry.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (ROOT / "results" / "published" / "raw-evidence-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    ids = [run["id"] for run in registry["runs"]]
    assert len(ids) == len(set(ids))
    evidence = {item["path"] for item in manifest["artifacts"]}
    assert evidence
    for run in registry["runs"]:
        assert run["metrics"]
        assert set(run["evidence_files"]) <= evidence
        assert run["budget"]

    for artifact in manifest["artifacts"]:
        if artifact["published"]:
            published_path = ROOT / artifact["published_path"]
            assert published_path.is_file()
            assert tool_sha256(published_path) == artifact["sha256"]


def tool_sha256(path: Path) -> str:
    tool = load_tool("build_evidence_manifest")
    return tool.sha256(path)


def test_dashboard_registry_matches_canonical_registry() -> None:
    tool = load_tool("build_dashboard_data")
    assert tool.canonical_bytes(tool.SOURCE) == tool.DESTINATION.read_bytes()
    assert (
        tool.canonical_bytes(tool.MANIFEST_SOURCE)
        == tool.MANIFEST_DESTINATION.read_bytes()
    )


def test_public_registry_validator_passes() -> None:
    tool = load_tool("validate_public_registry")
    assert tool.main() == 0


def test_interleaved_product_is_complete_and_reproducible() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from execution_order import interleaved_product

    first = interleaved_product(["a", "b"], [1, 2], ["x", "y"], seed=17)
    second = interleaved_product(["a", "b"], [1, 2], ["x", "y"], seed=17)
    assert first == second
    assert len(first) == 8
    assert len(set(first)) == 8
    assert first != sorted(first)


def test_graph_engine_record_preserves_ingestion_timeout_without_fake_quality_score() -> None:
    tool = load_tool("update_p2_registry")
    record = tool.graph_engine_record(
        {
            "system": "Cognee",
            "model": "qwen2.5:14b",
            "embedding_model": "nomic-embed-text",
            "backend": "Kuzu and LanceDB embedded",
            "top_k": 5,
            "documents": 20,
            "budget": {
                "ingestion_timeout_seconds": 1800.0,
                "query_timeout_seconds": 60.0,
            },
            "ingestion_seconds": 1800.015,
            "ingestion_error": "timeout after 1800.0s",
            "metrics": None,
            "rows": [],
        },
        run_id="cognee-primary",
        evidence_file="results/cognee.json",
        variant="default concurrency",
    )

    assert record["metrics"]["operational_status"] == "ingestion_timeout"
    assert record["metrics"]["retrieval"] is None
    assert record["metrics"]["queries_completed"] == 0


def test_graph_engine_record_marks_partial_index_metrics() -> None:
    tool = load_tool("update_p2_registry")
    record = tool.graph_engine_record(
        {
            "system": "Graphiti",
            "model": "qwen2.5:14b",
            "embedding_model": "nomic-embed-text",
            "backend": "FalkorDB Lite",
            "top_k": 5,
            "documents": 20,
            "budget": {
                "ingestion_timeout_seconds": 1800.0,
                "document_timeout_seconds": 180.0,
                "query_timeout_seconds": 60.0,
            },
            "ingestion_seconds": 1800.007,
            "ingestion_errors": [
                {"source_id": f"d{index}", "error": "timeout"}
                for index in range(7)
            ],
            "metrics": {
                "questions": 10,
                "mean_recall": 0.0833333,
                "mean_context_precision": 0.0333333,
                "temporal_correctness": 0.0,
            },
            "llm_tokens": {"total": 124735},
            "rows": [{"question_id": f"q{index}"} for index in range(10)],
        },
        run_id="graphiti-primary",
        evidence_file="results/graphiti.json",
        variant="primary",
    )

    assert record["metrics"]["operational_status"] == "partial_index"
    assert record["metrics"]["documents_completed"] == 13
    assert record["metrics"]["retrieval"]["mean_recall"] == 0.083
    assert record["metrics"]["llm_tokens"] == 124735


def test_memgym_record_keeps_semantic_judge_provisional() -> None:
    tool = load_tool("update_p2_registry")
    record = tool.memgym_record(
        {
            "manifest": {
                "created_at": "2026-07-29T20:00:00+00:00",
                "per_stratum": 10,
                "repetitions": 1,
                "expected_reader_calls": 120,
                "successful_reader_calls": 120,
                "expected_judge_calls": 120,
                "successful_judge_calls": 120,
            },
            "rows": [
                {
                    "instance_id": "i1",
                    "stratum": "3hop",
                    "architecture": "visible_only",
                    "repetition": 0,
                    "reader_model": "reader",
                    "judges": [{"ok": True, "response": {"score": 0.5}}],
                },
                {
                    "instance_id": "i1",
                    "stratum": "3hop",
                    "architecture": "bm25_k5",
                    "repetition": 0,
                    "reader_model": "reader",
                    "judges": [{"ok": True, "response": {"score": 0.7}}],
                },
            ],
            "summaries": [
                {
                    "stratum": "3hop",
                    "architecture": "visible_only",
                    "attempted_calls": 10,
                    "successful_reader_calls": 10,
                    "mean_token_f1": 0.2,
                    "mean_judge_score": 0.5,
                    "judged_calls": 10,
                    "mean_context_words": 18000,
                    "mean_reader_latency_seconds": 12,
                    "mean_reader_tokens": 35000,
                },
                {
                    "stratum": "3hop",
                    "architecture": "bm25_k5",
                    "attempted_calls": 10,
                    "successful_reader_calls": 10,
                    "mean_token_f1": 0.3,
                    "mean_judge_score": 0.7,
                    "judged_calls": 10,
                    "mean_context_words": 20000,
                    "mean_reader_latency_seconds": 15,
                    "mean_reader_tokens": 39000,
                },
            ],
        }
    )

    assert record["metrics"]["architecture_macro_judge"]["bm25_k5"] == 0.7
    assert record["metrics"]["architecture_macro_judge"]["visible_only"] == 0.5
    assert record["metrics"]["paired_vs_visible"]["bm25_k5"]["pairs"] == 1
    assert (
        record["metrics"]["paired_vs_visible"]["bm25_k5"][
            "mean_judge_difference"
        ]
        == 0.2
    )
    assert "not human-calibrated" in record["limitation"]
