import numpy as np
import pandas as pd
import pytest
from scipy import stats

from piu_severity.evaluation.inference import negligibility_test
from piu_severity.evaluation.selection import SELECTION_ALPHA
from piu_severity.evaluation.superiority import (
    SUPERIORITY_CONFIDENCE,
    superiority_conclusion,
    superiority_lower_bound,
)
from piu_severity.evaluation.validation import summarize_paired


def _differences(values: np.ndarray) -> pd.DataFrame:
    """Paired fold differences with the sizes of the 5 x 5 outer folds."""
    return pd.DataFrame(
        {"difference": values, "n_train": 2_189, "n_test": 547},
    )


def test_lower_bound_matches_the_corrected_formula():
    rng = np.random.default_rng(0)
    values = 0.07 + rng.normal(scale=0.02, size=25)
    result = superiority_lower_bound(_differences(values))
    se = np.sqrt((1 / 25 + 547 / 2_189) * values.std(ddof=1) ** 2)
    expected = values.mean() - stats.t.ppf(SUPERIORITY_CONFIDENCE, df=24) * se
    assert result["mean_difference"] == pytest.approx(values.mean())
    assert result["lower_bound"] == pytest.approx(expected)
    assert result["confidence"] == SUPERIORITY_CONFIDENCE


def test_lower_bound_mirrors_the_negligibility_upper_bound():
    rng = np.random.default_rng(1)
    differences = _differences(0.03 + rng.normal(scale=0.02, size=25))
    lower = superiority_lower_bound(differences)
    upper = negligibility_test(differences, margin=0.03)
    mean = lower["mean_difference"]
    assert mean - lower["lower_bound"] == pytest.approx(upper["upper_bound"] - mean)


@pytest.mark.parametrize("center", [0.002, 0.01, 0.08])
def test_lower_bound_above_zero_agrees_with_the_one_sided_test(center):
    rng = np.random.default_rng(2)
    differences = _differences(center + rng.normal(scale=0.02, size=25))
    above_zero = superiority_lower_bound(differences)["lower_bound"] > 0
    significant = (
        summarize_paired(differences)["p_value_one_sided"] < 1 - SUPERIORITY_CONFIDENCE
    )
    assert above_zero == significant


def test_lower_bound_needs_two_folds():
    with pytest.raises(ValueError, match="two folds"):
        superiority_lower_bound(_differences(np.array([0.1])))


def test_conclusion_comes_from_the_main_analysis_and_is_confirmed():
    assert superiority_conclusion(0.001, 0.002) == {
        "supported": True,
        "sensitivity_supported": True,
        "confirmed": True,
    }
    assert superiority_conclusion(0.001, 0.2) == {
        "supported": True,
        "sensitivity_supported": False,
        "confirmed": False,
    }
    assert superiority_conclusion(0.3, 0.4)["confirmed"]
    assert not superiority_conclusion(0.3, 0.4)["supported"]


def test_conclusion_uses_the_selection_alpha_and_checks_p_values():
    assert not superiority_conclusion(SELECTION_ALPHA, 0.0)["supported"]
    with pytest.raises(ValueError, match="p_main"):
        superiority_conclusion(1.5, 0.1)
