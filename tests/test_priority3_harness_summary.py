from priority3_harness_summary import summarize_attempts


def attempt(harness: str, task: str, tokens: int | None, seconds: float):
    return {
        "task_id": task,
        "attempt": {
            "harness": harness,
            "total_tokens": tokens,
            "wall_time_seconds": seconds,
        },
        "changed_files": [],
        "production_files_changed": [],
        "public": {"passed": 1, "tests": 1},
        "hidden": {"passed": 0, "tests": 2},
        "task_complete": False,
    }


def test_two_independent_no_change_failures_are_retained_as_incompatible():
    summary = summarize_attempts(
        [
            attempt("jcode", "resume_runner", 51000, 300),
            attempt("jcode", "provenance", 12000, 30),
        ]
    )

    harness = summary["harnesses"][0]
    assert harness["classification"] == "tool_protocol_incompatible"
    assert harness["distinct_tasks"] == 2
    assert harness["tasks_completed"] == 0
    assert harness["hidden_tests_passed"] == 0
    assert harness["production_files_changed"] == 0


def test_fixed_budget_table_never_turns_failure_into_success():
    summary = summarize_attempts(
        [
            attempt("jcode", "resume_runner", 51000, 300),
            attempt("jcode", "provenance", 12000, 30),
        ]
    )

    rows = summary["fixed_budget"]["rows"]
    assert {row["token_ceiling"] for row in rows} == {
        25000,
        50000,
        100000,
    }
    assert all(row["tasks_completed"] == 0 for row in rows)


def test_unknown_token_totals_remain_unknown():
    summary = summarize_attempts(
        [
            attempt("Codex CLI", "resume_runner", None, 300),
            attempt("Codex CLI", "provenance", None, 30),
        ]
    )

    assert summary["harnesses"][0]["known_total_tokens"] is None
