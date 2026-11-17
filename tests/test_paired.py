import numpy as np
import pandas as pd
import pytest

from piu_severity.evaluation.metrics import underestimation_rate
from piu_severity.evaluation.paired import (
    compare_participant_rates,
    participant_levels_referral_rates,
    participant_underestimation_rates,
)
from piu_severity.evaluation.screening import referral_metrics

Y_TRUE = [0, 1, 2, 2, 3, 3]


def _predictions(y_pred_by_repeat: list[list[int]]) -> pd.DataFrame:
    rows = [
        {"repeat": r, "fold": 0, "position": p, "y_true": t, "y_pred": y}
        for r, y_pred in enumerate(y_pred_by_repeat)
        for p, (t, y) in enumerate(zip(Y_TRUE, y_pred, strict=True))
    ]
    return pd.DataFrame(rows)


PREDICTIONS = _predictions([[0, 1, 1, 2, 2, 3], [1, 0, 2, 2, 1, 2]])


def test_underestimation_rates_average_to_the_rate_of_each_repeat():
    rates = participant_underestimation_rates(PREDICTIONS)
    assert rates.to_dict() == pytest.approx({2: 0.5, 3: 0.0, 4: 1.0, 5: 0.5})
    by_repeat = [
        underestimation_rate(g["y_true"], g["y_pred"])
        for _, g in PREDICTIONS.groupby("repeat")
    ]
    assert rates.mean() == pytest.approx(np.mean(by_repeat))
    assert participant_underestimation_rates(PREDICTIONS, (3,)).index.tolist() == [
        4,
        5,
    ]


def test_levels_referral_rates_give_the_screening_recall():
    rates = participant_levels_referral_rates(PREDICTIONS, (2, 3))
    assert rates.index.tolist() == [2, 3, 4, 5]
    recall = [
        referral_metrics(g["y_true"], g["y_pred"])["recall"]
        for _, g in PREDICTIONS.groupby("repeat")
    ]
    assert rates.mean() == pytest.approx(np.mean(recall))


def test_identical_rates_have_no_difference():
    rates = pd.Series([0.2, 0.6, 1.0], index=[0, 1, 2])
    result = compare_participant_rates(rates, rates)
    assert result["difference"] == 0
    assert result["interval_low"] == result["interval_high"] == 0
    assert result["p_value_two_sided"] == 1.0
    assert result["participants_differing"] == 0


def test_higher_rates_give_a_positive_difference():
    rng = np.random.default_rng(0)
    b = pd.Series(rng.uniform(0, 0.5, size=60))
    a = b + 0.4
    result = compare_participant_rates(a, b)
    assert result["mean_a"] - result["mean_b"] == pytest.approx(0.4)
    assert result["difference"] == pytest.approx(0.4)
    assert result["interval_low"] == pytest.approx(0.4)
    assert result["p_value_two_sided"] < 0.01
    assert result["participants"] == 60


def test_rates_of_other_participants_are_rejected():
    a = pd.Series([0.5, 0.5], index=[0, 1])
    b = pd.Series([0.5, 0.5], index=[0, 2])
    with pytest.raises(ValueError, match="same participants"):
        compare_participant_rates(a, b)
