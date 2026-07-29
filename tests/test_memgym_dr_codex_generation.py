from __future__ import annotations

from memgym_dr_codex_generation import (
    build_observations,
    render_reader_prompt,
    retrieve_notes,
    selected_ids,
    visible_documents,
)


def fixture_instance():
    return {
        "question": "Which city is the capital of France?",
        "answer": "Paris",
        "eviction_policy": {
            "mode": "full_eviction",
            "window_size": 1,
        },
        "turns": [
            {
                "sub_query": "France capital",
                "documents": [
                    {
                        "title": "France",
                        "text": "Paris is the capital city of France.",
                    }
                ],
            },
            {
                "sub_query": "Unrelated",
                "documents": [
                    {
                        "title": "Noise",
                        "text": "Berlin is a city in Germany.",
                    }
                ],
            },
            {
                "sub_query": "More unrelated",
                "documents": [
                    {
                        "title": "More noise",
                        "text": "Madrid is a city in Spain.",
                    }
                ],
            },
        ],
    }


def test_selected_ids_is_deterministic_and_stratified():
    manifest = {
        "configuration": {
            "sample_ids": {
                "3hop": ["a", "b"],
                "4hop": ["c", "d"],
                "56hop": ["e", "f"],
            }
        }
    }
    first = selected_ids(manifest, 1)
    second = selected_ids(manifest, 1)
    assert first == second
    assert set(first) == {"3hop", "4hop", "56hop"}
    assert all(len(ids) == 1 for ids in first.values())
    assert first["3hop"][0] in {"a", "b"}
    assert first["4hop"][0] in {"c", "d"}
    assert first["56hop"][0] in {"e", "f"}


def test_full_eviction_keeps_only_current_turn_visible():
    instance = fixture_instance()
    assert [row["title"] for row in visible_documents(instance, 2)] == [
        "More noise"
    ]
    observations = build_observations(instance)
    assert "Paris is the capital" in observations[0]
    assert "Paris is the capital" not in observations[1]


def test_bm25_notes_recover_evicted_answer():
    instance = fixture_instance()
    notes = retrieve_notes(instance, "bm25_k1")
    assert "Paris is the capital city of France" in notes
    assert retrieve_notes(instance, "visible_only") == ""


def test_reader_prompt_keeps_notes_and_last_visible_documents_separate():
    prompt = render_reader_prompt(fixture_instance(), "Paris is the capital.")
    assert "Your accumulated notes" in prompt
    assert "Paris is the capital." in prompt
    assert "Documents still visible" in prompt
    assert "Madrid is a city in Spain." in prompt
