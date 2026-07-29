from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_tool(name: str):
    path = ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registry_has_unique_runs_and_declared_evidence() -> None:
    registry = json.loads(
        (ROOT / "results" / "published" / "registry.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (ROOT / "results" / "published" / "raw-evidence-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    ids = [run["id"] for run in registry["runs"]]
    assert len(ids) == len(set(ids))
    evidence = {item["path"] for item in manifest["artifacts"]}
    assert evidence
    for run in registry["runs"]:
        assert run["metrics"]
        assert set(run["evidence_files"]) <= evidence


def test_dashboard_registry_matches_canonical_registry() -> None:
    tool = load_tool("build_dashboard_data")
    assert tool.canonical_bytes(tool.SOURCE) == tool.DESTINATION.read_bytes()
    assert (
        tool.canonical_bytes(tool.MANIFEST_SOURCE)
        == tool.MANIFEST_DESTINATION.read_bytes()
    )


def test_public_registry_validator_passes() -> None:
    tool = load_tool("validate_public_registry")
    assert tool.main() == 0
