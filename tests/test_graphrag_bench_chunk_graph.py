from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graphrag_bench_chunk_graph import (
    build_chunk_graph,
    chunk_text,
    paired_comparisons,
    graph_expand,
    parse_evidence_units,
    score_context,
    validate_inputs,
)


class GraphRagBenchChunkGraphTests(unittest.TestCase):
    def test_chunk_text_preserves_overlap_and_order(self):
        text = " ".join(f"w{index}" for index in range(12))
        chunks = chunk_text(text, chunk_words=5, overlap_words=2)
        self.assertEqual(
            [
                "w0 w1 w2 w3 w4",
                "w3 w4 w5 w6 w7",
                "w6 w7 w8 w9 w10",
                "w9 w10 w11",
            ],
            [chunk["text"] for chunk in chunks],
        )
        self.assertEqual(12, chunks[-1]["word_end"])

    def test_evidence_units_preserve_relation_and_commas(self):
        units = parse_evidence_units(
            "Dr. Inglis graduated M.B., C.M. in 1899 from Edinburgh; "
            "Launcelot was one of the Knights of the Round Table."
        )
        self.assertEqual(
            [
                (
                    "dr",
                    "inglis",
                    "graduated",
                    "m",
                    "b",
                    "c",
                    "m",
                    "in",
                    "1899",
                    "from",
                    "edinburgh",
                ),
                (
                    "launcelot",
                    "was",
                    "one",
                    "of",
                    "the",
                    "knights",
                    "of",
                    "the",
                    "round",
                    "table",
                ),
            ],
            units,
        )

    def test_graph_contains_sequence_and_rare_term_edges(self):
        chunks = [
            {"id": "c0", "text": "Alice meets the quartz falcon."},
            {"id": "c1", "text": "The quartz falcon guides Bob."},
            {"id": "c2", "text": "Carol watches the sea."},
        ]
        graph = build_chunk_graph(
            chunks,
            minimum_document_frequency=2,
            maximum_document_frequency=2,
            maximum_neighbors=4,
        )
        self.assertIn("c1", graph["c0"])
        self.assertIn("c0", graph["c1"])
        self.assertIn("c2", graph["c1"])

    def test_graph_adds_nonadjacent_rare_term_edge(self):
        chunks = [
            {"id": "c0", "text": "Alice meets the quartz falcon."},
            {"id": "c1", "text": "Bob watches the sea."},
            {"id": "c2", "text": "The quartz falcon guides Carol."},
        ]
        graph = build_chunk_graph(
            chunks,
            minimum_document_frequency=2,
            maximum_document_frequency=2,
            maximum_neighbors=4,
        )
        self.assertIn("c2", graph["c0"])

    def test_graph_expansion_can_add_neighbor_outside_top_k(self):
        chunks = [
            {"id": "c0", "text": "seed"},
            {"id": "c1", "text": "neighbor"},
            {"id": "c2", "text": "other"},
        ]
        ranking = [
            {**chunks[0], "_score": 4.0},
            {**chunks[2], "_score": 3.0},
        ]
        selected = graph_expand(
            ranking,
            {"c0": {"c1": 3.0}, "c1": {"c0": 3.0}, "c2": {}},
            {chunk["id"]: chunk for chunk in chunks},
            seed_count=1,
            limit=2,
        )
        self.assertEqual(["c0", "c1"], [row["id"] for row in selected])

    def test_score_context_uses_official_evidence_statement(self):
        all_chunks = [
            {"id": "c0", "text": "Sir Launcelot served King Arthur."},
            {"id": "c1", "text": "The sea was calm."},
        ]
        units = [
            ("sir", "launcelot", "served", "king", "arthur"),
        ]
        score = score_context(
            [all_chunks[0]],
            all_chunks,
            units,
            evidence_token_recall_threshold=0.8,
        )
        self.assertEqual(1, score["resolvable_evidence_units"])
        self.assertEqual(1.0, score["all_official_evidence_recall"])
        self.assertTrue(score["full_official_evidence_coverage"])

    def test_score_context_does_not_ignore_relation_tokens(self):
        all_chunks = [
            {"id": "c0", "text": "Alice rejected the proposal from Bob."},
        ]
        score = score_context(
            all_chunks,
            all_chunks,
            [("alice", "approved", "the", "proposal", "from", "bob")],
            evidence_token_recall_threshold=0.85,
        )
        self.assertEqual(0, score["resolvable_evidence_units"])
        self.assertEqual(0.0, score["all_official_evidence_recall"])

    def test_score_context_requires_matching_explicit_polarity(self):
        all_chunks = [
            {"id": "c0", "text": "Alice did not approve the proposal."},
        ]
        score = score_context(
            all_chunks,
            all_chunks,
            [("alice", "did", "approve", "the", "proposal")],
            evidence_token_recall_threshold=0.8,
        )
        self.assertEqual(0, score["resolvable_evidence_units"])

    def test_paired_comparison_keeps_duplicate_question_ids_across_sources(self):
        rows = []
        for source, left_full, right_full in (
            ("book-a", True, False),
            ("book-b", False, True),
        ):
            for strategy, full in (
                ("bm25_graph", left_full),
                ("bm25", right_full),
            ):
                rows.append(
                    {
                        "source": source,
                        "question_id": "duplicate-id",
                        "strategy": strategy,
                        "fully_representable": True,
                        "full_official_evidence_coverage": full,
                        "all_official_evidence_recall": float(full),
                    }
                )
        comparison = paired_comparisons(
            rows,
            bootstrap_repetitions=100,
            bootstrap_seed=7,
        )[0]
        self.assertEqual(2, comparison["paired_questions"])
        self.assertEqual(1, comparison["left_only_full_coverage"])
        self.assertEqual(1, comparison["right_only_full_coverage"])
        self.assertEqual(
            2,
            comparison["source_cluster_bootstrap"]["clusters"],
        )

    def test_graph_expansion_rejects_budget_violation(self):
        with self.assertRaises(ValueError):
            graph_expand(
                [{"id": "c0", "text": "seed"}],
                {"c0": {}},
                {"c0": {"id": "c0", "text": "seed"}},
                seed_count=2,
                limit=1,
            )

    def test_validate_inputs_rejects_duplicate_question_key(self):
        corpora = [{"corpus_name": "book", "context": "text"}]
        question = {"source": "book", "id": "q"}
        with self.assertRaises(ValueError):
            validate_inputs(corpora, [question, question])


if __name__ == "__main__":
    unittest.main()
