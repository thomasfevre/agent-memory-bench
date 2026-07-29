from priority3_temporal_extraction import generate_observations


def test_generated_temporal_dataset_covers_adversarial_event_types():
    observations = generate_observations()

    assert len(observations) >= 60
    assert len({row["expected"]["entity_key"] for row in observations}) == 20
    assert {
        "assertion",
        "correction",
        "late_arrival",
        "duplicate",
        "expiration",
        "retraction",
        "low_confidence_contradiction",
        "shard_approval",
        "shard_rejection",
    } <= {row["expected"]["event_type"] for row in observations}
    for row in observations:
        assert row["id"]
        assert row["text"]
        assert row["expected"]["source_id"]
        assert row["expected"]["asserted_at"]
        assert row["expected"]["effective_from"]
