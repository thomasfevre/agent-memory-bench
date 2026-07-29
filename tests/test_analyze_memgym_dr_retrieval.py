import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyze_memgym_dr_retrieval import paired_bootstrap, percentile


def test_percentile_uses_sorted_values():
    values = [3.0, 1.0, 2.0]
    assert percentile(values, 0.0) == 1.0
    assert percentile(values, 1.0) == 3.0


def test_paired_bootstrap_constant_difference():
    result = paired_bootstrap([0.25, 0.25, 0.25], 100, 7)
    assert result == {
        "n": 3,
        "mean_difference": 0.25,
        "ci95_low": 0.25,
        "ci95_high": 0.25,
    }
