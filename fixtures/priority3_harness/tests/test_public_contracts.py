from memory_harness.process_resume.contracts import CampaignResult
from memory_harness.provenance.contracts import Evidence
from memory_harness.requirement_change.contracts import AccessDecision
from memory_harness.resume_runner.contracts import RunSummary


def test_public_contracts_remain_importable():
    assert RunSummary(completed=1, skipped=0, failed=0).completed == 1
    assert Evidence(text="fact", source_ids=("s1",)).source_ids == ("s1",)
    assert CampaignResult(applied=1, replayed=0).applied == 1
    assert AccessDecision(allowed=True, reason="known").allowed
