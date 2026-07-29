import subprocess
import sys
from pathlib import Path


def test_priority3_harness_public_contracts_pass():
    root = Path(__file__).resolve().parents[1]
    fixture = root / "fixtures" / "priority3_harness"
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=fixture,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
