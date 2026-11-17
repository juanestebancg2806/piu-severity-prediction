import numpy as np
import pandas as pd
import pytest

from piu_severity.evaluation.metrics import wilson_interval
from piu_severity.evaluation.screening import (
    SCREENING_METRICS,
    level_referral_by_repeat,
    metrics_from_counts,
    participant_bootstrap_screening,
    participant_referral_counts,
    participant_referral_rates,
    referral_consistency,
    referral_counts,
    referral_metrics,
    screening_by_repeat,
)

Y_TRUE = [0, 0, 1, 2, 3, 3]
Y_PRED = [0, 2, 2, 1, 2, 3]


def _predictions(y_pred_by_repeat: list[list[int]], y_true=Y_TRUE) -> pd.DataFrame:
    """Predictions of the same participants in several repeats."""
    rows = [
        {
            "repeat": repeat,
            "fold": 0,
            "position": position,
            "y_true": true,
            "y_pred": pred,
        }
        for repeat, y_pred in enumerate(y_pred_by_repeat)
        for position, (true, pred) in enumerate(zip(y_true, y_pred, strict=True))
    ]
    return pd.DataFrame(rows)


def test_referral_counts_and_metrics_of_a_known_example():
    assert referral_counts(Y_TRUE, Y_PRED) == {"tp": 2, "fn": 1, "fp": 2, "tn": 1}
    metrics = referral_metrics(Y_TRUE, Y_PRED)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["specificity"] == pytest.approx(1 / 3)
    assert metrics["ppv"] == pytest.approx(1 / 2)
    assert metrics["referred_share"] == pytest.approx(4 / 6)


def test_referral_level_sets_positives_and_referrals():
    assert referral_counts(Y_TRUE, Y_PRED, referral_level=3) == {
        "tp": 1,
        "fn": 1,
        "fp": 0,
        "tn": 4,
    }


def test_undefined_metrics_are_nan():
    nobody_referred = referral_metrics([0, 2, 3], [0, 1, 1])
    assert np.isnan(nobody_referred["ppv"])
    assert nobody_referred["referred_share"] == 0
    no_negatives = referral_metrics([2, 3], [2, 1])
    assert np.isnan(no_negatives["specificity"])
    arrays = metrics_from_counts([1, 0], [1, 0], [0, 0], [2, 0])
    assert arrays["recall"][0] == pytest.approx(0.5)
    assert np.isnan(arrays["recall"][1])


def test_screening_by_repeat_matches_each_repeat():
    second = [0, 0, 1, 2, 3, 2]
    table = screening_by_repeat(_predictions([Y_PRED, second]))
    assert list(table.columns) == list(SCREENING_METRICS)
    assert table.loc[0].to_dict() == pytest.approx(referral_metrics(Y_TRUE, Y_PRED))
    assert table.loc[1].to_dict() == pytest.approx(referral_metrics(Y_TRUE, second))


def test_level_referral_by_repeat_counts_with_wilson():
    table = level_referral_by_repeat(_predictions([Y_PRED, [0] * 6]), level=3)
    assert table.loc[0, "cases"] == 2
    assert table.loc[0, "referred"] == 2
    assert table.loc[1, "referred"] == 0
    low, high = wilson_interval(0.0, 2)
    assert table.loc[1, "wilson_low"] == pytest.approx(low)
    assert table.loc[1, "wilson_high"] == pytest.approx(high)
    with pytest.raises(ValueError, match="no participant of level"):
        level_referral_by_repeat(_predictions([Y_PRED], y_true=[0] * 6), level=3)


def test_participant_referral_rates_and_consistency():
    predictions = _predictions([Y_PRED, [0, 0, 1, 2, 1, 3], [0, 0, 1, 2, 1, 3]])
    rates = participant_referral_rates(predictions, level=3)
    assert rates.to_dict() == pytest.approx({4: 1 / 3, 5: 1.0})
    assert referral_consistency(rates) == {
        "always": 1,
        "never": 0,
        "sometimes": 1,
        "participants": 2,
    }


def test_participant_counts_require_every_repeat():
    predictions = _predictions([Y_PRED, Y_PRED])
    counts = participant_referral_counts(predictions)
    assert counts.sum().to_dict() == {"tp": 4, "fn": 2, "fp": 4, "tn": 2}
    with pytest.raises(ValueError, match="once in every repeat"):
        participant_referral_counts(predictions.iloc[1:])


def test_bootstrap_is_reproducible_and_contains_the_estimate():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 4, size=200)
    repeats = [np.clip(y_true + rng.integers(-1, 2, size=200), 0, 3) for _ in range(3)]
    predictions = _predictions([list(r) for r in repeats], y_true=list(y_true))
    first = participant_bootstrap_screening(
        predictions, n_bootstrap=400, chunk_size=150
    )
    second = participant_bootstrap_screening(predictions, n_bootstrap=400)
    pd.testing.assert_frame_equal(first, second)
    assert (first["interval_low"] <= first["estimate"]).all()
    assert (first["estimate"] <= first["interval_high"]).all()
    mean_recall = screening_by_repeat(predictions)["recall"].mean()
    assert first.loc["recall", "estimate"] == pytest.approx(mean_recall)


def test_bootstrap_of_perfect_predictions_is_a_point():
    predictions = _predictions([Y_TRUE, Y_TRUE])
    table = participant_bootstrap_screening(predictions, n_bootstrap=200)
    assert table.loc["recall", "interval_low"] == pytest.approx(1.0)
    assert table.loc["recall", "interval_high"] == pytest.approx(1.0)


def test_missing_columns_are_rejected():
    with pytest.raises(KeyError, match="lack the columns"):
        screening_by_repeat(pd.DataFrame({"repeat": [0], "y_true": [0]}))
