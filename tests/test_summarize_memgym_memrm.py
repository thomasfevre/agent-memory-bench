import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from summarize_memgym_memrm import compare_predictions, pearson


def test_pearson_identical_vectors():
    assert pearson([0.1, 0.2, 0.3], [0.1, 0.2, 0.3]) == pytest.approx(1.0)


def test_compare_predictions_checks_alignment_and_metrics():
    rerun = [
        {
            "instance_id": "a",
            "step": 1,
            "source": "s",
            "perturbation": "p",
            "prob_safe": 0.8,
            "pred_label": 1,
        },
        {
            "instance_id": "b",
            "step": 2,
            "source": "s",
            "perturbation": "p",
            "prob_safe": 0.2,
            "pred_label": 0,
        },
    ]
    official = [
        {**rerun[1], "prob_safe": 0.3},
        {**rerun[0], "prob_safe": 0.7},
    ]
    result = compare_predictions(rerun, official)
    assert result["n"] == 2
    assert result["prediction_agreement"] == 1.0
    assert abs(result["prob_safe_mae"] - 0.1) < 1e-12
    assert result["prob_safe_pearson"] == pytest.approx(1.0)


def test_compare_predictions_rejects_different_identity_sets():
    base = {
        "instance_id": "a",
        "step": 1,
        "source": "s",
        "perturbation": "p",
        "prob_safe": 0.8,
        "pred_label": 1,
    }
    with pytest.raises(ValueError, match="identity sets differ"):
        compare_predictions(
            [base],
            [{**base, "instance_id": "different"}],
        )
