from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmark import MiniLm
from longmemeval_codex_generation import (
    ChunkRetriever,
    answer_matches_spec,
    context_cache_fingerprint,
    deterministic_answer_match,
    exact_mcnemar_p,
    load_existing_rows,
    model_comparisons,
    pair_consistency,
    parse_tokens_used,
    rank_chunks,
    score_response,
    select_context,
    token_f1,
    upsert_run_row,
)


class FakeEncoder:
    def encode(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vector = np.asarray(
                [
                    lowered.count("train"),
                    lowered.count("cake"),
                    lowered.count("dollar"),
                ],
                dtype=np.float32,
            )
            norm = np.linalg.norm(vector)
            vectors.append(vector / norm if norm else vector)
        return np.asarray(vectors, dtype=np.float32)


class LongMemEvalCodexGenerationTests(unittest.TestCase):
    def test_minilm_encode_batches_without_changing_order(self):
        encoder = MiniLm.__new__(MiniLm)
        calls = []

        def fake_batch(texts):
            calls.append(list(texts))
            return np.asarray(
                [[float(int(text.removeprefix("t")))] for text in texts],
                dtype=np.float32,
            )

        encoder._encode_batch = fake_batch
        result = encoder.encode(["t0", "t1", "t2", "t3", "t4"], batch_size=2)
        self.assertEqual([["t0", "t1"], ["t2", "t3"], ["t4"]], calls)
        self.assertEqual([0.0, 1.0, 2.0, 3.0, 4.0], result[:, 0].tolist())

    def test_chunk_retriever_matches_one_shot_ranking(self):
        chunks = [
            {"id": "u1", "text": "train dollar", "role": "user"},
            {"id": "a1", "text": "cake", "role": "assistant"},
            {"id": "u2", "text": "train cake", "role": "user"},
        ]
        encoder = FakeEncoder()
        retriever = ChunkRetriever(chunks, encoder)
        for architecture in ("hybrid_chunks", "hybrid_user_chunks"):
            expected = rank_chunks(
                chunks,
                "train dollar",
                architecture,
                encoder,
                3,
                0.5,
            )
            actual = retriever.rank(
                "train dollar",
                architecture,
                3,
                0.5,
            )
            self.assertEqual(
                [row["id"] for row in expected],
                [row["id"] for row in actual],
            )

    def test_select_context_never_exceeds_budget(self):
        ranking = [
            {
                "id": "a",
                "text": "one two three",
                "session_id": "s1",
                "date": "d1",
                "role": "user",
            },
            {
                "id": "b",
                "text": "four five six",
                "session_id": "s2",
                "date": "d2",
                "role": "user",
            },
            {
                "id": "c",
                "text": "seven eight",
                "session_id": "s3",
                "date": "d3",
                "role": "user",
            },
        ]
        selected = select_context(ranking, 11)
        self.assertEqual(["a", "b"], [row["id"] for row in selected])
        self.assertLessEqual(
            sum(
                3 + len(row["text"].split())
                for row in selected
            ),
            11,
        )

    def test_user_only_bm25_excludes_assistant_chunks(self):
        chunks = [
            {
                "id": "u",
                "text": "train costs ten dollars",
                "role": "user",
            },
            {
                "id": "a",
                "text": "train costs ten dollars",
                "role": "assistant",
            },
        ]
        ranking = rank_chunks(
            chunks,
            "train dollars",
            "bm25_user_chunks",
            None,
            10,
            0.5,
        )
        self.assertEqual(["u"], [row["id"] for row in ranking])

    def test_answer_matching_normalizes_articles_and_punctuation(self):
        self.assertTrue(
            deterministic_answer_match("It was the lemon blueberry cake.", "a lemon blueberry cake")
        )
        self.assertEqual(1.0, token_f1("The Luna", "Luna"))

    def test_answer_spec_can_require_two_temporal_states(self):
        spec = {
            "required_all": [
                ["every week", "weekly"],
                ["every other week", "biweekly"],
            ]
        }
        self.assertTrue(
            answer_matches_spec(
                "Previously weekly, and now every other week.",
                "unused",
                spec,
            )
        )
        self.assertFalse(
            answer_matches_spec("Previously weekly.", "unused", spec)
        )

    def test_abstention_requires_empty_evidence(self):
        context = [{"id": "s1:m0:c0", "text": "Luna", "session_id": "s1"}]
        valid = score_response(
            {
                "answer": "INSUFFICIENT_EVIDENCE",
                "abstain": True,
                "evidence_ids": [],
                "confidence": 0.9,
            },
            "unused",
            True,
            context,
        )
        invalid = score_response(
            {
                "answer": "INSUFFICIENT_EVIDENCE",
                "abstain": True,
                "evidence_ids": ["missing"],
                "confidence": 0.9,
            },
            "unused",
            True,
            context,
        )
        self.assertTrue(valid["correct"])
        self.assertFalse(invalid["correct"])

    def test_abstention_requires_exact_sentinel(self):
        result = score_response(
            {
                "answer": "I do not know",
                "abstain": True,
                "evidence_ids": [],
                "confidence": 0.9,
            },
            "unused",
            True,
            [],
        )
        self.assertFalse(result["correct"])
        self.assertFalse(result["sentinel_valid"])

    def test_parse_tokens_used_handles_narrow_spaces(self):
        self.assertEqual(16163, parse_tokens_used("tokens used\n16\u202f163"))

    def test_exact_mcnemar_is_one_when_no_disagreement(self):
        self.assertEqual(1.0, exact_mcnemar_p(0, 0))

    def test_context_cache_fingerprint_changes_with_budget(self):
        first = context_cache_fingerprint(
            "sha",
            ["a"],
            ["bm25_chunks"],
            4000,
            224,
            100,
            0.5,
        )
        second = context_cache_fingerprint(
            "sha",
            ["a"],
            ["bm25_chunks"],
            3999,
            224,
            100,
            0.5,
        )
        self.assertNotEqual(first, second)

    def test_pair_consistency_requires_both_members(self):
        rows = [
            {
                "ok": True,
                "model": "m",
                "architecture": "a",
                "pair_id": "p",
                "repetition": 0,
                "correct": True,
            },
            {
                "ok": True,
                "model": "m",
                "architecture": "a",
                "pair_id": "p",
                "repetition": 0,
                "correct": False,
            },
        ]
        result = pair_consistency(rows)[0]
        self.assertEqual(1, result["complete_pair_repetitions"])
        self.assertEqual(0.0, result["both_correct_rate"])

    def test_model_comparison_keeps_pair_repetitions_together(self):
        rows = []
        for model, values in {
            "light": [False, True],
            "strong": [True, True],
        }.items():
            for question_id, correct in zip(
                ["p", "p_abs"],
                values,
                strict=True,
            ):
                rows.append(
                    {
                        "ok": True,
                        "architecture": "bm25",
                        "model": model,
                        "question_id": question_id,
                        "repetition": 0,
                        "correct": correct,
                    }
                )
        result = model_comparisons(rows, resamples=100, seed=1)[0]
        self.assertEqual(1, result["pair_repetitions"])
        self.assertEqual(-0.5, result["observed_accuracy_difference_left_minus_right"])

    def test_existing_rows_survive_partial_resume_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seed.json"
            path.write_text(
                json.dumps(
                    {
                        "manifest": {"execution_seed": 17},
                        "rows": [
                            {"run_key": "first", "ok": True},
                            {"run_key": "later", "ok": True},
                        ],
                    }
                )
            )
            rows, legacy, unverified = load_existing_rows(
                [path],
                {"first", "later"},
                {"execution_seed": 17},
            )
        self.assertEqual({"first", "later"}, set(rows))
        self.assertEqual([], legacy)
        self.assertEqual([], unverified)

    def test_existing_rows_reject_execution_seed_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seed.json"
            path.write_text(
                json.dumps(
                    {
                        "manifest": {"execution_seed": 11},
                        "rows": [{"run_key": "first", "ok": True}],
                    }
                )
            )
            with self.assertRaisesRegex(
                ValueError,
                "protocol mismatch for execution_seed",
            ):
                load_existing_rows(
                    [path],
                    {"first"},
                    {"execution_seed": 17},
                )

    def test_legacy_existing_rows_are_explicitly_labeled(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seed.json"
            path.write_text(
                json.dumps(
                    {
                        "manifest": {},
                        "rows": [{"run_key": "first", "ok": True}],
                    }
                )
            )
            rows, legacy, unverified = load_existing_rows(
                [path],
                {"first"},
                {"execution_seed": 17},
                allow_legacy=True,
            )
        self.assertEqual({"first"}, set(rows))
        self.assertEqual([str(path)], legacy)
        self.assertEqual([str(path)], unverified)

    def test_retry_replaces_existing_row_instead_of_duplicating_it(self):
        existing = {"same": {"run_key": "same", "ok": False}}
        rows = upsert_run_row(
            existing,
            {"run_key": "same", "ok": True},
        )
        self.assertEqual(1, len(rows))
        self.assertTrue(rows[0]["ok"])

    def test_mixed_legacy_provenance_survives_a_second_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "migrated.json"
            path.write_text(
                json.dumps(
                    {
                        "manifest": {
                            "execution_seed": 17,
                            "execution_order": "mixed-legacy-and-interleaved",
                            "legacy_execution_order_sources": ["original.json"],
                        },
                        "rows": [{"run_key": "first", "ok": True}],
                    }
                )
            )
            _, legacy, unverified = load_existing_rows(
                [path],
                {"first"},
                {"execution_seed": 17},
            )
        self.assertEqual(["original.json"], legacy)
        self.assertEqual([], unverified)

    def test_legacy_missing_protocol_is_rejected_without_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            path.write_text(
                json.dumps(
                    {
                        "manifest": {},
                        "rows": [{"run_key": "first", "ok": True}],
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "missing protocol field"):
                load_existing_rows(
                    [path],
                    {"first"},
                    {"reader_prompt_version": "v1"},
                )

    def test_context_cache_fingerprint_changes_with_embedding(self):
        first = context_cache_fingerprint(
            "dataset",
            ["pair"],
            ["hybrid"],
            4000,
            224,
            100,
            0.5,
            "embedding-a",
            "tokenizer-a",
        )
        second = context_cache_fingerprint(
            "dataset",
            ["pair"],
            ["hybrid"],
            4000,
            224,
            100,
            0.5,
            "embedding-b",
            "tokenizer-a",
        )
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
