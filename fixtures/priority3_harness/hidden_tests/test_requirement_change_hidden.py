from memory_harness.requirement_change.policy import decide_access
from memory_harness.requirement_change.reporting import render_decision


def test_unknown_source_is_denied_after_requirement_change():
    decision = decide_access("unknown", {"trusted"})
    report = render_decision(decision)

    assert decision.allowed is False
    assert "deny" in decision.reason
    assert report["allowed"] is False
    assert report["policy_version"] == "default-deny-v2"


def test_trusted_source_remains_allowed():
    decision = decide_access("trusted", {"trusted"})

    assert decision.allowed is True
    assert decision.reason == "trusted_source"
