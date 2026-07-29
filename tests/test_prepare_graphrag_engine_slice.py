from __future__ import annotations

from prepare_graphrag_engine_slice import select_distractors, select_questions


def candidate(
    question_id: str,
    question_type: str,
    gold_chunk_ids: list[str],
) -> dict:
    return {
        "question": {
            "id": question_id,
            "question": f"question {question_id} rare{question_id}",
            "question_type": question_type,
        },
        "gold_chunk_ids": gold_chunk_ids,
    }


def test_select_questions_balances_types_and_reuses_gold_chunks() -> None:
    candidates = [
        candidate("c2", "Complex Reasoning", ["x2"]),
        candidate("c1", "Complex Reasoning", ["x1"]),
        candidate("c3", "Complex Reasoning", ["x1"]),
        candidate("f2", "Fact Retrieval", ["x3"]),
        candidate("f1", "Fact Retrieval", ["x1"]),
    ]

    selected = select_questions(candidates, question_count=4, complex_count=2)

    assert [row["question"]["id"] for row in selected] == ["c1", "c3", "f1", "f2"]
    assert sum(
        row["question"]["question_type"] == "Complex Reasoning"
        for row in selected
    ) == 2


def test_select_distractors_excludes_gold_and_is_deterministic() -> None:
    chunks = [
        {"id": "x1", "text": "alpha gold"},
        {"id": "x2", "text": "alpha rarec1"},
        {"id": "x3", "text": "question rarec1 distractor"},
        {"id": "x4", "text": "unrelated fallback"},
    ]
    selected = [candidate("c1", "Complex Reasoning", ["x1"])]

    first = select_distractors(chunks, selected, count=2)
    second = select_distractors(chunks, selected, count=2)

    assert first == second
    assert "x1" not in first
    assert first[0] in {"x2", "x3"}
