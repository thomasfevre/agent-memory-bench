from pathlib import Path

from priority3_deletion_compaction import run_deletion_verification


def test_deletion_compaction_covers_every_preregistered_surface(
    tmp_path: Path,
):
    payload = run_deletion_verification(tmp_path)

    assert payload["all_active_surfaces_clean"]
    assert payload["signed_generation_valid"]
    assert payload["operations"] == {
        "source_tombstones": 1,
        "fact_retractions": 1,
        "shard_expirations": 1,
        "retention_deletions": 1,
        "retention_days": 30,
    }
    assert payload["verification"] == {
        "direct_current_state_clean": True,
        "full_text_bm25_clean": True,
        "dense_neighbors_clean": True,
        "graph_traversal_clean": True,
        "active_export_clean": True,
        "vacuumed_database_pages_clean": True,
        "new_backup_clean": True,
    }
    assert payload["old_backup_contains_deleted_payloads"]
    assert payload["immutable_audit_events_retained"] == 4
    assert payload["secure_flash_erasure_claimed"] is False
