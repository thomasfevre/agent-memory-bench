from pathlib import Path

from incremental_memory_lifecycle import load_jsonl
from priority3_derived_recovery import (
    CRASH_BOUNDARIES,
    DerivedIndexStore,
    run_crash_matrix,
)


def fixture_events() -> list[dict]:
    root = Path(__file__).resolve().parents[1]
    return load_jsonl(root / "data" / "incremental-memory-events.jsonl")


def test_uninterrupted_generation_validates_all_four_views(tmp_path: Path):
    events = fixture_events()
    database = tmp_path / "derived.sqlite3"
    store = DerivedIndexStore(database)

    generation = store.rebuild(events)
    validation = store.validate_generation(generation, events)

    assert validation["valid"]
    assert validation["manifest_matches_journal"]
    assert validation["temporal_count"] > 0
    assert validation["full_text_count"] == validation["temporal_count"]
    assert validation["vector_count"] == validation["temporal_count"]
    assert validation["graph_count"] == validation["temporal_count"] * 2
    store.close()


def test_all_eight_crash_boundaries_recover_to_uninterrupted_signature():
    payload = run_crash_matrix(fixture_events())

    assert set(CRASH_BOUNDARIES) == {
        "before_transaction_begin",
        "after_temporal_update",
        "after_full_text_update",
        "after_vector_update",
        "after_graph_update",
        "before_manifest_commit",
        "immediately_after_commit",
        "during_full_rebuild",
    }
    assert len(payload["scenarios"]) == 8
    assert payload["all_scenarios_pass"]
    assert payload["wall_time_seconds"] > 0
    assert all(row["recovered_generation_valid"] for row in payload["scenarios"])
    assert all(
        row["semantic_signature_matches_uninterrupted"]
        for row in payload["scenarios"]
    )
    assert all(
        row["source_ids_match_uninterrupted"]
        for row in payload["scenarios"]
    )


def test_tampered_view_is_rejected_and_rebuilt(tmp_path: Path):
    events = fixture_events()
    store = DerivedIndexStore(tmp_path / "derived.sqlite3")
    generation = store.rebuild(events)
    store.connection.execute(
        """
        DELETE FROM dense_vectors
        WHERE rowid = (
            SELECT rowid FROM dense_vectors
            WHERE generation_id = ? LIMIT 1
        )
        """,
        (generation,),
    )
    store.connection.commit()

    assert not store.validate_generation(generation, events)["valid"]
    recovery = store.recover(events)

    assert recovery["action"] == "rebuilt"
    assert recovery["validation"]["valid"]
    store.close()
