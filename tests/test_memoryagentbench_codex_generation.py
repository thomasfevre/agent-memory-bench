from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memoryagentbench_codex_generation import (
    architecture_comparisons,
    load_seed_rows,
    inherited_legacy_execution_sources,
    rank_facts,
    score_response,
    stability_summary,
    summarize,
    validate_compatible_manifest,
    validate_resume_manifest,
)


class MemoryAgentBenchCodexGenerationTests(unittest.TestCase):
    def test_long_context_preserves_all_facts(self):
        facts = [
            {"id": "fact-1", "text": "Alpha is linked to Beta."},
            {"id": "fact-2", "text": "Gamma is linked to Delta."},
        ]
        self.assertEqual(
            facts,
            rank_facts(
                facts,
                "Alpha",
                "long_context",
                encoder=None,
                top_k=1,
                ranking_depth=2,
                alpha_bm25=0.5,
            ),
        )

    def test_bm25_selects_bounded_evidence(self):
        facts = [
            {"id": "fact-1", "text": "Alpha is linked to Beta."},
            {"id": "fact-2", "text": "Gamma is linked to Delta."},
        ]
        ranked = rank_facts(
            facts,
            "Alpha Beta",
            "bm25",
            encoder=None,
            top_k=1,
            ranking_depth=2,
            alpha_bm25=0.5,
        )
        self.assertEqual(["fact-1"], [row["id"] for row in ranked])

    def test_scoring_keeps_official_metric_separate_from_citations(self):
        context = [{"id": "fact-1", "text": "The answer is Belgium."}]
        valid = score_response(
            {
                "answer": "Belgium",
                "abstain": False,
                "evidence_ids": ["fact-1"],
                "confidence": 0.9,
            },
            ["Belgium"],
            context,
        )
        invalid_citation = score_response(
            {
                "answer": "Belgium",
                "abstain": False,
                "evidence_ids": ["missing"],
                "confidence": 0.9,
            },
            ["Belgium"],
            context,
        )
        self.assertTrue(valid["substring_exact_match"])
        self.assertTrue(valid["citation_ids_valid"])
        self.assertTrue(invalid_citation["substring_exact_match"])
        self.assertFalse(invalid_citation["citation_ids_valid"])
        self.assertFalse(invalid_citation["formally_cited_correct"])

    def test_summary_reports_official_and_grounded_scores(self):
        rows = [
            {
                "ok": True,
                "source": "s",
                "model": "m",
                "architecture": "a",
                "substring_exact_match": True,
                "exact_match": True,
                "token_f1": 1.0,
                "citation_ids_valid": True,
                "formally_cited_correct": True,
                "abstained": False,
                "context_facts": 10,
                "context_words": 100,
                "latency_seconds": 2.0,
                "tokens_used": 200,
            },
            {
                "ok": True,
                "source": "s",
                "model": "m",
                "architecture": "a",
                "substring_exact_match": False,
                "exact_match": False,
                "token_f1": 0.0,
                "citation_ids_valid": True,
                "formally_cited_correct": False,
                "abstained": True,
                "context_facts": 10,
                "context_words": 100,
                "latency_seconds": 4.0,
                "tokens_used": 400,
            },
        ]
        result = summarize(rows)[0]
        self.assertEqual(2, result["attempted_calls"])
        self.assertEqual(1.0, result["provider_success_rate"])
        self.assertEqual(0.5, result["substring_exact_match"])
        self.assertEqual(0.5, result["formally_cited_accuracy"])
        self.assertEqual(0.5, result["abstention_rate"])
        self.assertEqual(300, result["mean_tokens_used"])

    def test_stability_summary_groups_repetitions_by_question(self):
        rows = []
        for question_index, values in {
            1: [True, True, True],
            2: [True, False, True],
        }.items():
            for repetition, correct in enumerate(values):
                rows.append(
                    {
                        "ok": True,
                        "source": "s",
                        "model": "m",
                        "architecture": "a",
                        "question_index": question_index,
                        "repetition": repetition,
                        "substring_exact_match": correct,
                        "abstained": not correct,
                        "response": {"answer": "yes" if correct else "no"},
                    }
                )
        result = stability_summary(rows)[0]
        self.assertEqual(2, result["questions"])
        self.assertEqual(3, result["repetitions_per_question"])
        self.assertEqual(0.5, result["correctness_unanimity_rate"])
        self.assertEqual(0.5, result["abstention_unanimity_rate"])

    def test_architecture_comparison_is_question_paired(self):
        rows = []
        for architecture, values in {
            "bm25": [True, False, True],
            "long_context": [False, False, True],
        }.items():
            for index, correct in enumerate(values):
                rows.append(
                    {
                        "ok": True,
                        "source": "s",
                        "model": "m",
                        "architecture": architecture,
                        "question_index": index,
                        "repetition": 0,
                        "substring_exact_match": correct,
                    }
                )
        result = architecture_comparisons(
            rows,
            metric="substring_exact_match",
            resamples=100,
            seed=1,
        )[0]
        self.assertEqual(3, result["paired_question_repetitions"])
        self.assertEqual(1, result["left_only_correct"])
        self.assertEqual(0, result["right_only_correct"])
        self.assertEqual("question", result["mcnemar_unit"])

    def test_architecture_comparison_uses_question_majority_for_repetitions(self):
        rows = []
        for architecture, values in {
            "bm25": {
                1: [True, True, False],
                2: [False, False, True],
            },
            "long_context": {
                1: [False, False, True],
                2: [False, True, True],
            },
        }.items():
            for question_index, repetitions in values.items():
                for repetition, correct in enumerate(repetitions):
                    rows.append(
                        {
                            "ok": True,
                            "source": "s",
                            "model": "m",
                            "architecture": architecture,
                            "question_index": question_index,
                            "repetition": repetition,
                            "substring_exact_match": correct,
                        }
                    )
        result = architecture_comparisons(
            rows,
            metric="substring_exact_match",
            resamples=100,
            seed=1,
        )[0]
        self.assertEqual("question_majority", result["mcnemar_unit"])
        self.assertEqual(1, result["question_majority_left_only_correct"])
        self.assertEqual(1, result["question_majority_right_only_correct"])

    def test_resume_manifest_rejects_protocol_drift(self):
        existing = {
            "dataset_sha256": "sha",
            "sources": ["s"],
            "models": ["m"],
            "architectures": ["bm25"],
            "question_indices": [0],
            "repetitions": 1,
            "top_k": 20,
            "ranking_depth": 100,
            "alpha_bm25": 0.5,
            "reasoning_effort": "low",
        }
        requested = dict(existing)
        requested["top_k"] = 10
        with self.assertRaisesRegex(ValueError, "resume manifest mismatch"):
            validate_resume_manifest(existing, requested)

    def test_seed_rows_are_filtered_to_allowed_run_keys(self):
        rows = load_seed_rows(
            [
                {
                    "rows": [
                        {"run_key": "allowed", "ok": True},
                        {"run_key": "outside", "ok": True},
                    ]
                }
            ],
            {"allowed"},
        )
        self.assertEqual(["allowed"], sorted(rows))

    def test_compatible_manifest_rejects_runtime_drift(self):
        expected = {
            "dataset_sha256": "dataset",
            "top_k": 20,
            "ranking_depth": 100,
            "alpha_bm25": 0.5,
            "reasoning_effort": "low",
            "dry_run": False,
            "schema_sha256": "schema",
            "reader_prompt_version": "v1",
            "embedding_model_sha256": "embedding",
            "codex_version": "codex 1",
            "execution_seed": 17,
        }
        existing = dict(expected)
        existing["embedding_model_sha256"] = "other"
        with self.assertRaisesRegex(ValueError, "incompatible result manifest"):
            validate_compatible_manifest(existing, expected)

    def test_compatible_manifest_requires_runtime_fields(self):
        with self.assertRaisesRegex(ValueError, "missing compatibility fields"):
            validate_compatible_manifest(
                {"dataset_sha256": "dataset"},
                {
                    "dataset_sha256": "dataset",
                    "reader_prompt_version": "v1",
                },
            )

    def test_mixed_legacy_provenance_survives_a_second_resume(self):
        payload = {
            "manifest": {
                "execution_seed": 17,
                "execution_order": "mixed-legacy-and-interleaved",
                "legacy_execution_order_sources": ["original.json"],
            }
        }
        self.assertEqual(
            ["original.json"],
            inherited_legacy_execution_sources(
                Path("migrated.json"),
                payload,
            ),
        )


if __name__ == "__main__":
    unittest.main()
