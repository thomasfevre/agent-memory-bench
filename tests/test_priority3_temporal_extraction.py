from priority3_temporal_extraction import (
    build_arrival_schedules,
    generate_observations,
)


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


def test_arrival_schedules_are_deterministic_and_cover_protocol():
    observations = generate_observations()

    first = build_arrival_schedules(observations)
    second = build_arrival_schedules(observations)

    assert first == second
    assert set(first) == {
        "chronological",
        "late_10pct",
        "late_25pct",
        "reverse_windows_5",
        "duplicate_retry",
    }

    expected_ids = {row["id"] for row in observations}
    for name in (
        "chronological",
        "late_10pct",
        "late_25pct",
        "reverse_windows_5",
    ):
        deliveries = first[name]["deliveries"]
        assert len(deliveries) == len(observations)
        assert {row["event_id"] for row in deliveries} == expected_ids
        assert all(row["attempt"] == 1 for row in deliveries)


def test_late_schedules_move_the_preregistered_fraction_later():
    schedules = build_arrival_schedules(generate_observations())
    chronological = [
        row["event_id"] for row in schedules["chronological"]["deliveries"]
    ]

    for name, expected_count in (("late_10pct", 6), ("late_25pct", 15)):
        schedule = schedules[name]
        moved_ids = schedule["late_event_ids"]
        order = [row["event_id"] for row in schedule["deliveries"]]

        assert len(moved_ids) == expected_count
        assert set(order) == set(chronological)
        assert all(order.index(event_id) > chronological.index(event_id) for event_id in moved_ids)


def test_reverse_windows_and_retry_schedule_have_expected_shape():
    schedules = build_arrival_schedules(generate_observations())
    chronological = schedules["chronological"]["deliveries"]
    reversed_deliveries = schedules["reverse_windows_5"]["deliveries"]
    retries = schedules["duplicate_retry"]["deliveries"]

    assert [row["event_id"] for row in reversed_deliveries[:5]] == [
        row["event_id"] for row in reversed(chronological[:5])
    ]
    assert len(retries) == 72
    assert sum(row["attempt"] == 2 for row in retries) == 12
    assert len({row["delivery_id"] for row in retries}) == len(retries)
