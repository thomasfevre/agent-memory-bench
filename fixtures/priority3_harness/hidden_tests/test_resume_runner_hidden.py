import json

import pytest

from memory_harness.resume_runner.checkpoint import load_completed
from memory_harness.resume_runner.runner import run_items


def test_progress_survives_processor_failure(tmp_path):
    checkpoint = tmp_path / "checkpoint.json"
    first_calls = []

    def fail_on_c(item_id):
        first_calls.append(item_id)
        if item_id == "c":
            raise RuntimeError("forced")

    with pytest.raises(RuntimeError, match="forced"):
        run_items(["a", "b", "c", "d"], fail_on_c, checkpoint)

    assert load_completed(checkpoint) == {"a", "b"}
    second_calls = []
    summary = run_items(
        ["a", "b", "c", "d"],
        second_calls.append,
        checkpoint,
    )
    assert second_calls == ["c", "d"]
    assert summary.completed == 2
    assert summary.skipped == 2


def test_malformed_checkpoint_fails_closed(tmp_path):
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(json.dumps({"completed": "not-a-list"}))

    with pytest.raises(ValueError, match="checkpoint"):
        load_completed(checkpoint)
