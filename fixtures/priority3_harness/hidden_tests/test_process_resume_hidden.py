import json

import pytest

from memory_harness.process_resume.campaign import apply_campaign
from memory_harness.process_resume import journal as journal_module
from memory_harness.process_resume.journal import append_ack, load_acks


def test_duplicate_ack_is_idempotent(tmp_path):
    journal = tmp_path / "acks.jsonl"
    append_ack(journal, {"event_id": "e1"})
    append_ack(journal, {"event_id": "e1"})

    assert load_acks(journal) == [{"event_id": "e1"}]
    assert len(journal.read_text().splitlines()) == 1


def test_acknowledgement_is_fsynced_before_return(tmp_path, monkeypatch):
    journal = tmp_path / "acks.jsonl"
    calls = []
    monkeypatch.setattr(journal_module.os, "fsync", calls.append)

    append_ack(journal, {"event_id": "e1"})

    assert len(calls) == 1


def test_one_torn_final_line_is_recovered(tmp_path):
    journal = tmp_path / "acks.jsonl"
    journal.write_text('{"event_id":"e1"}\n{"event_id":', encoding="utf-8")

    assert load_acks(journal) == [{"event_id": "e1"}]


def test_middle_corruption_fails_closed(tmp_path):
    journal = tmp_path / "acks.jsonl"
    journal.write_text(
        '{"event_id":"e1"}\nnot-json\n{"event_id":"e2"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="journal"):
        load_acks(journal)


def test_resume_skips_durable_acknowledgements(tmp_path):
    journal = tmp_path / "acks.jsonl"
    append_ack(journal, {"event_id": "e1"})
    calls = []

    result = apply_campaign(["e1", "e2"], calls.append, journal)

    assert calls == ["e2"]
    assert result.applied == 1
    assert result.replayed == 1
