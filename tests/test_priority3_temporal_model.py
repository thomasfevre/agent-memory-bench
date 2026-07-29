import json
from pathlib import Path

import pytest

from priority3_temporal_model import (
    build_extraction_prompt,
    run_extraction_campaign,
    validate_extraction,
)


VALID_EXTRACTION = {
    "entity_key": "entity-01.operating_mode",
    "value": "mode-01-v1",
    "asserted_at": "2026-01-02T09:00:00Z",
    "effective_from": "2026-01-02T09:00:00Z",
    "effective_until": None,
    "event_type": "assertion",
    "target_event_id": None,
    "confidence": 0.96,
    "source_id": "p3-source-01-1",
}


def test_validate_extraction_requires_the_exact_contract():
    assert validate_extraction(VALID_EXTRACTION) == VALID_EXTRACTION

    missing = dict(VALID_EXTRACTION)
    missing.pop("source_id")
    with pytest.raises(ValueError, match="fields"):
        validate_extraction(missing)

    extra = dict(VALID_EXTRACTION, explanation="not allowed")
    with pytest.raises(ValueError, match="fields"):
        validate_extraction(extra)

    invalid_type = dict(VALID_EXTRACTION, confidence="high")
    with pytest.raises(ValueError, match="confidence"):
        validate_extraction(invalid_type)


def test_prompt_contains_only_the_observation_and_public_contract():
    prompt = build_extraction_prompt("Source alpha says entity-x became blue.")

    assert "Source alpha says entity-x became blue." in prompt
    assert "expected" not in prompt.lower()
    assert "Return exactly one JSON object" in prompt
    assert "target_event_id" in prompt


def test_campaign_extracts_each_observation_once_per_repetition_and_resumes(
    tmp_path: Path,
):
    observations_path = tmp_path / "observations.jsonl"
    output_path = tmp_path / "result.json"
    observations = [
        {"id": "event-1", "text": "first"},
        {"id": "event-2", "text": "second"},
    ]
    observations_path.write_text(
        "".join(json.dumps(row) + "\n" for row in observations),
        encoding="utf-8",
    )
    calls = []

    def fake_extractor(text, *, model, seed, endpoint, timeout_seconds):
        calls.append((text, model, seed, endpoint, timeout_seconds))
        extraction = dict(VALID_EXTRACTION)
        extraction["source_id"] = f"source-{text}"
        return {
            "extraction": extraction,
            "metrics": {
                "latency_ms": 12.0,
                "prompt_tokens": 10,
                "output_tokens": 5,
            },
        }

    first = run_extraction_campaign(
        observations_path=observations_path,
        output_path=output_path,
        model="test-model",
        repetitions=3,
        seed=100,
        extractor=fake_extractor,
    )
    second = run_extraction_campaign(
        observations_path=observations_path,
        output_path=output_path,
        model="test-model",
        repetitions=3,
        seed=100,
        extractor=fake_extractor,
    )

    assert first["status"] == "complete"
    assert len(first["rows"]) == 6
    assert len(calls) == 6
    assert second == first
    assert {(row["repetition"], row["event_id"]) for row in first["rows"]} == {
        (1, "event-1"),
        (1, "event-2"),
        (2, "event-1"),
        (2, "event-2"),
        (3, "event-1"),
        (3, "event-2"),
    }
    assert [call[2] for call in calls] == [100, 101, 100, 101, 100, 101]
