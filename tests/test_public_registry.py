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


def test_dashboard_exposes_harness_compatibility_summary() -> None:
    html = (ROOT / "site" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "site" / "assets" / "app.js").read_text(encoding="utf-8")

    assert 'id="harness-comparison"' in html
    assert "priority3-coding-harness-qwen25-14b-20260730" in script
    assert "not a quality ranking" in script


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


def test_temporal_record_exposes_stale_state_hidden_by_selected_answer() -> None:
    tool = load_tool("update_p3_registry")
    record = tool.temporal_record(
        {
            "extraction_metrics": {
                "attempts": 180,
                "successful_attempts": 180,
                "event_type_accuracy": 1.0,
                "field_accuracy": 0.935185,
                "provenance_exact": 1.0,
                "temporal_window_exact": 0.933333,
                "text_observable_field_accuracy": 0.961759,
                "text_observable_score_is_posthoc": True,
                "stable_event_fraction": 1.0,
            },
            "schedule_metrics": {
                "query_count_per_repetition_schedule": 152,
                "schedule_stability_by_repetition": {
                    "1": True,
                    "2": True,
                    "3": True,
                },
                "rows": [
                    {
                        "schedule": "chronological",
                        "repetition": 1,
                        "selected_final_value_exact": 1.0,
                        "final_state_exact": 0.85,
                        "historical_active_state_accuracy": 0.960526,
                        "stale_record_leakage_rate": 0.038961,
                        "abstention_after_invalidation": 1.0,
                        "duplicate_amplification": 0.0,
                    }
                ],
            },
            "source_extraction_sha256": "abc123",
        }
    )

    metrics = record["metrics"]
    assert metrics["extraction"]["strict_field_accuracy"] == 0.935
    assert metrics["extraction"]["posthoc_observable_field_accuracy"] == 0.962
    assert metrics["extraction"]["posthoc_metric"] is True
    assert metrics["schedule"]["selected_final_value_exact"] == 1.0
    assert metrics["schedule"]["complete_active_state_exact"] == 0.85
    assert metrics["schedule"]["stale_record_leakage_rate"] == 0.039
    assert "hidden" in record["conclusion"]


def test_crash_record_requires_every_declared_boundary_to_recover() -> None:
    tool = load_tool("update_p3_registry")
    record = tool.crash_record(
        {
            "protocol": "priority3-derived-recovery-v1",
            "all_scenarios_pass": True,
            "wall_time_seconds": 0.353454,
            "scenarios": [
                {
                    "boundary": f"boundary-{index}",
                    "crash_observed": True,
                    "recovery_action": "rebuilt",
                    "semantic_signature_matches_uninterrupted": True,
                    "source_ids_match_uninterrupted": True,
                    "orphan_generations": [],
                }
                for index in range(8)
            ],
        }
    )

    assert record["metrics"]["passed_boundaries"] == 8
    assert record["metrics"]["declared_boundaries"] == 8
    assert record["metrics"]["orphan_generations"] == 0
    assert record["metrics"]["all_scenarios_pass"] is True


def test_deletion_record_keeps_backup_and_ssd_limits_visible() -> None:
    tool = load_tool("update_p3_registry")
    record = tool.deletion_record(
        {
            "protocol": "priority3-deletion-compaction-v1",
            "all_active_surfaces_clean": True,
            "audit_events_use_hashed_targets": True,
            "immutable_audit_events_retained": 4,
            "old_backup_contains_deleted_payloads": True,
            "secure_flash_erasure_claimed": False,
            "signed_generation_valid": True,
            "verification": {
                "active_export_clean": True,
                "dense_neighbors_clean": True,
                "direct_current_state_clean": True,
                "full_text_bm25_clean": True,
                "graph_traversal_clean": True,
                "new_backup_clean": True,
                "vacuumed_database_pages_clean": True,
            },
            "wall_time_seconds": 0.010918,
        }
    )

    assert record["metrics"]["clean_surfaces"] == 7
    assert record["metrics"]["old_backup_contains_deleted_payloads"] is True
    assert record["metrics"]["secure_flash_erasure_claimed"] is False
    assert "backup" in record["limitation"].lower()
    assert "SSD" in record["limitation"]


def test_harness_record_reports_compatibility_failure_not_quality_ranking() -> None:
    tool = load_tool("update_p3_registry")
    record = tool.harness_record(
        {
            "harnesses": [
                {
                    "harness": "jcode",
                    "classification": "tool_protocol_incompatible",
                    "attempts": 2,
                    "distinct_tasks": 2,
                    "tasks_completed": 0,
                    "hidden_tests_passed": 1,
                    "hidden_tests_total": 5,
                    "production_files_changed": 0,
                    "known_total_tokens": 63969,
                    "known_wall_time_seconds": 340.415,
                }
            ],
            "fixed_budget": {
                "rows": [],
                "time_rows": [],
                "tool_call_ceilings": [20, 40, 80],
                "pareto_efficient_configurations": [],
            },
        }
    )

    assert record["metrics"]["harnesses"][0]["tasks_completed"] == 0
    assert (
        record["metrics"]["harnesses"][0]["classification"]
        == "tool_protocol_incompatible"
    )
    assert "not a quality ranking" in record["limitation"]
