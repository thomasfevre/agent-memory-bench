from __future__ import annotations

from unittest.mock import patch

from ingestion_benchmark import run_campaign


def test_run_campaign_groups_local_models_but_interleaves_tasks() -> None:
    calls: list[tuple[str, str, int]] = []

    def fake_model_call(model: str, prompt: str, seed: int) -> dict[str, object]:
        task = "shards" if prompt.startswith("Find behavioral") else "extract"
        calls.append((model, task, seed))
        payload = (
            '{"candidates":[]}'
            if task == "shards"
            else '{"facts":[{"text":"Alpha","valid_from":"2026-01-01","valid_to":null}]}'
        )
        return {
            "text": payload,
            "latency_ms": 1.0,
            "prompt_tokens": 1,
            "output_tokens": 1,
        }

    documents = [
        {
            "id": "d1",
            "timestamp": "2026-01-01",
            "text": "Alpha",
        }
    ]
    facts = [
        {
            "id": "f1",
            "text": "Alpha",
            "source_ids": ["d1"],
            "valid_from": "2026-01-01",
            "valid_to": None,
        }
    ]
    with patch("ingestion_benchmark.model_call", side_effect=fake_model_call):
        result = run_campaign(["small", "large"], documents, facts, 17)

    assert set(result) == {"small", "large"}
    assert len(calls) == 12
    model_sequence = [model for model, _, _ in calls]
    assert model_sequence[:6] == [model_sequence[0]] * 6
    assert model_sequence[6:] == [model_sequence[6]] * 6
    assert model_sequence[0] != model_sequence[6]
    for model in ("small", "large"):
        tasks = [task for called_model, task, _ in calls if called_model == model]
        assert tasks.count("extract") == 3
        assert tasks.count("shards") == 3
        assert tasks != sorted(tasks)
