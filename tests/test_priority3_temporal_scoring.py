from priority3_temporal_scoring import (
    TemporalMemory,
    evaluate_schedule_matrix,
    score_extraction_rows,
)
from priority3_temporal_extraction import (
    build_arrival_schedules,
    generate_observations,
)


EXPECTED = {
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


def test_extraction_scores_measure_fields_temporality_provenance_and_stability():
    observations = [{"id": "event-1", "expected": EXPECTED}]
    wrong = dict(EXPECTED, event_type="late_arrival", source_id="wrong")
    rows = [
        {
            "event_id": "event-1",
            "repetition": 1,
            "status": "success",
            "extraction": EXPECTED,
        },
        {
            "event_id": "event-1",
            "repetition": 2,
            "status": "success",
            "extraction": EXPECTED,
        },
        {
            "event_id": "event-1",
            "repetition": 3,
            "status": "success",
            "extraction": wrong,
        },
    ]

    scores = score_extraction_rows(observations, rows)

    assert scores["attempts"] == 3
    assert scores["successful_attempts"] == 3
    assert scores["field_accuracy"] == 25 / 27
    assert scores["text_observable_field_accuracy"] == 25 / 27
    assert scores["temporal_window_exact"] == 1.0
    assert scores["event_type_accuracy"] == 2 / 3
    assert scores["provenance_exact"] == 2 / 3
    assert scores["stable_event_fraction"] == 0.0


def test_posthoc_observable_score_excludes_unstated_expiration_fields():
    expiration = {
        **EXPECTED,
        "event_type": "expiration",
        "target_event_id": "event-0",
        "effective_until": "2026-01-05T09:00:00Z",
    }
    extracted = {
        **expiration,
        "confidence": 1.0,
        "effective_from": "2026-01-05T09:00:00Z",
    }

    scores = score_extraction_rows(
        [{"id": "event-1", "expected": expiration}],
        [
            {
                "event_id": "event-1",
                "repetition": 1,
                "status": "success",
                "extraction": extracted,
            }
        ],
    )

    assert scores["field_accuracy"] == 7 / 9
    assert scores["text_observable_field_accuracy"] == 1.0
    assert scores["text_observable_score_is_posthoc"] is True


def test_temporal_memory_is_order_independent_and_deduplicates_retries():
    assertion = {
        "entity_key": "entity-x.mode",
        "value": "v1",
        "asserted_at": "2026-01-02T00:00:00Z",
        "effective_from": "2026-01-02T00:00:00Z",
        "effective_until": None,
        "event_type": "assertion",
        "target_event_id": None,
        "confidence": 0.96,
        "source_id": "source-1",
    }
    correction = {
        "entity_key": "entity-x.mode",
        "value": "v2",
        "asserted_at": "2026-02-01T00:00:00Z",
        "effective_from": "2026-01-15T00:00:00Z",
        "effective_until": None,
        "event_type": "correction",
        "target_event_id": "event-1",
        "confidence": 0.96,
        "source_id": "source-2",
    }

    first = TemporalMemory()
    first.ingest("event-1", assertion)
    first.ingest("event-2", correction)
    first.ingest("event-2", correction)

    second = TemporalMemory()
    second.ingest("event-2", correction)
    second.ingest("event-1", assertion)

    assert first.query("entity-x.mode", "2026-01-10T00:00:00Z") == "v1"
    assert first.query("entity-x.mode", "2026-01-20T00:00:00Z") == "v2"
    assert first.query("entity-x.mode", "2026-03-01T00:00:00Z") == "v2"
    assert first.signature() == second.signature()
    assert first.unique_event_count == 2
    assert first.duplicate_delivery_count == 1


def test_active_state_exposes_stale_record_hidden_by_selected_value():
    assertion = {
        **EXPECTED,
        "entity_key": "entity-x.mode",
        "value": "v1",
        "source_id": "source-1",
    }
    correct_correction = {
        **EXPECTED,
        "entity_key": "entity-x.mode",
        "value": "v2",
        "asserted_at": "2026-02-01T00:00:00Z",
        "effective_from": "2026-01-15T00:00:00Z",
        "event_type": "correction",
        "target_event_id": "event-1",
        "source_id": "source-2",
    }
    broken_correction = {**correct_correction, "target_event_id": None}
    expected_memory = TemporalMemory()
    expected_memory.ingest("event-1", assertion)
    expected_memory.ingest("event-2", correct_correction)
    broken_memory = TemporalMemory()
    broken_memory.ingest("event-1", assertion)
    broken_memory.ingest("event-2", broken_correction)

    query_at = "2026-03-01T00:00:00Z"
    assert expected_memory.query("entity-x.mode", query_at) == "v2"
    assert broken_memory.query("entity-x.mode", query_at) == "v2"
    assert expected_memory.active_state("entity-x.mode", query_at) != (
        broken_memory.active_state("entity-x.mode", query_at)
    )


def test_expiration_retraction_and_low_confidence_do_not_invent_active_state():
    memory = TemporalMemory()
    memory.ingest(
        "event-1",
        {
            **EXPECTED,
            "entity_key": "entity-x.mode",
            "value": "trusted",
            "effective_from": "2026-01-01T00:00:00Z",
        },
    )
    memory.ingest(
        "event-2",
        {
            **EXPECTED,
            "entity_key": "entity-x.mode",
            "value": "rumor",
            "effective_from": "2026-01-10T00:00:00Z",
            "event_type": "low_confidence_contradiction",
            "confidence": 0.55,
            "source_id": "source-rumor",
        },
    )
    memory.ingest(
        "event-3",
        {
            **EXPECTED,
            "entity_key": "entity-x.mode",
            "value": "trusted",
            "effective_from": "2026-02-01T00:00:00Z",
            "event_type": "retraction",
            "target_event_id": "event-1",
            "source_id": "source-retraction",
        },
    )

    assert memory.query("entity-x.mode", "2026-01-20T00:00:00Z") == "trusted"
    assert memory.query("entity-x.mode", "2026-02-02T00:00:00Z") is None


def test_perfect_extraction_is_invariant_across_all_five_schedules():
    observations = generate_observations()
    rows = [
        {
            "event_id": observation["id"],
            "repetition": repetition,
            "status": "success",
            "extraction": observation["expected"],
        }
        for repetition in (1, 2, 3)
        for observation in observations
    ]

    result = evaluate_schedule_matrix(
        observations,
        build_arrival_schedules(observations),
        rows,
    )

    assert len(result["rows"]) == 15
    assert all(row["final_state_exact"] == 1.0 for row in result["rows"])
    assert all(
        row["historical_query_accuracy"] == 1.0 for row in result["rows"]
    )
    assert all(
        row["duplicate_amplification"] == 0.0 for row in result["rows"]
    )
    assert all(result["schedule_stability_by_repetition"].values())
