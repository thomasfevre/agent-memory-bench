from memory_harness.provenance.contracts import Evidence
from memory_harness.provenance.retrieval import merge_evidence
from memory_harness.provenance.serialization import from_payload, to_payload


def test_merge_deduplicates_sources_in_deterministic_order():
    merged = merge_evidence(
        [
            Evidence("first", ("s2", "s1")),
            Evidence("second", ("s1", "s3")),
        ]
    )

    assert merged.text == "first\nsecond"
    assert merged.source_ids == ("s1", "s2", "s3")


def test_legacy_payload_roundtrips_without_losing_sources():
    restored = from_payload(
        {"content": "legacy fact", "sources": ["old-2", "old-1", "old-1"]}
    )

    assert restored == Evidence(
        text="legacy fact",
        source_ids=("old-1", "old-2"),
    )
    assert to_payload(restored) == {
        "text": "legacy fact",
        "source_ids": ["old-1", "old-2"],
    }


def test_plain_string_does_not_invent_provenance():
    assert from_payload("legacy text") == Evidence("legacy text", ())
