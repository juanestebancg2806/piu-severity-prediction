import numpy as np
import pandas as pd
import pytest
from scipy import stats

from piu_severity.evaluation.inference import (
    assigned_level_distribution,
    holm_adjust,
    level_hits_by_repeat,
    mcnemar_by_repeat,
    negligibility_test,
    paired_participant_test,
    participant_bootstrap_ci,
    participant_hit_rates,
)
from piu_severity.evaluation.metrics import wilson_interval
from piu_severity.evaluation.validation import outer_splits, paired_differences

SEVERE = 3


def _predictions(hits_per_repeat: list[list[int]], n_other: int = 6) -> pd.DataFrame:
    """Predictions of len(hits) severe participants and n_other of level 0.

    hits_per_repeat[r][i] is 1 when severe participant i is predicted severe in
    repeat r; otherwise it is predicted moderate. Level-0 participants are
    always predicted 0.
    """
    rows = []
    for repeat, hits in enumerate(hits_per_repeat):
        for position, hit in enumerate(hits):
            rows.append((repeat, position, SEVERE, SEVERE if hit else 2))
        for offset in range(n_other):
            rows.append((repeat, len(hits) + offset, 0, 0))
    return pd.DataFrame(rows, columns=["repeat", "position", "y_true", "y_pred"])


def test_level_hits_by_repeat_counts_hits_and_wilson_intervals():
    predictions = _predictions([[1, 1, 0, 0], [1, 0, 0, 0]])
    table = level_hits_by_repeat(predictions, SEVERE)
    assert list(table["hits"]) == [2, 1]
    assert list(table["cases"]) == [4, 4]
    assert list(table["recall"]) == [0.5, 0.25]
    assert (table.loc[0, "wilson_low"], table.loc[0, "wilson_high"]) == (
        pytest.approx(wilson_interval(0.5, 4))
    )


def test_level_hits_by_repeat_requires_cases_of_the_level():
    predictions = _predictions([[1, 0]]).query("y_true != 3")
    with pytest.raises(ValueError, match="no participant"):
        level_hits_by_repeat(predictions, SEVERE)


def test_zero_hits_give_a_wilson_interval_starting_at_zero():
    table = level_hits_by_repeat(_predictions([[0] * 34]), SEVERE)
    assert table.loc[0, "wilson_low"] == 0.0
    assert 0 < table.loc[0, "wilson_high"] < 0.15


def test_assigned_level_distribution_adds_up_to_one():
    predictions = _predictions([[1, 1, 0, 0], [1, 0, 0, 0]])
    shares = assigned_level_distribution(predictions, SEVERE)
    assert shares.sum() == pytest.approx(1.0)
    assert shares[SEVERE] == pytest.approx((0.5 + 0.25) / 2)
    assert shares[2] == pytest.approx((0.5 + 0.75) / 2)


def test_participant_hit_rates_average_over_repeats():
    rates = participant_hit_rates(_predictions([[1, 1, 0], [1, 0, 0]]), SEVERE)
    assert list(rates) == [1.0, 0.5, 0.0]


def test_bootstrap_interval_contains_the_recall_and_is_deterministic():
    rates = pd.Series(np.r_[np.ones(10), np.zeros(24)])
    low, high = participant_bootstrap_ci(rates)
    assert low < rates.mean() < high
    assert (low, high) == participant_bootstrap_ci(rates)
    assert participant_bootstrap_ci(pd.Series(np.zeros(34))) == (0.0, 0.0)


def test_paired_participant_test_detects_a_clear_difference():
    better = pd.Series(np.r_[np.ones(20), np.zeros(14)])
    worse = pd.Series(np.zeros(34))
    result = paired_participant_test(better, worse)
    assert result["mean_difference"] == pytest.approx(20 / 34)
    assert result["participants_differing"] == 20
    # Twenty differences of the same sign: the exact p-value is 2 / 2**20.
    assert result["p_value_two_sided"] < 0.001


def test_paired_participant_test_without_differences_is_not_significant():
    rates = pd.Series(np.r_[np.ones(5), np.zeros(29)])
    result = paired_participant_test(rates, rates.copy())
    assert result["mean_difference"] == 0
    assert result["p_value_two_sided"] == 1.0


def test_paired_participant_test_with_balanced_differences_is_not_significant():
    a = pd.Series(np.r_[np.ones(6), np.zeros(28)])
    b = pd.Series(np.r_[np.zeros(6), np.ones(6), np.zeros(22)])
    assert paired_participant_test(a, b)["p_value_two_sided"] > 0.5


def test_paired_participant_test_requires_the_same_participants():
    with pytest.raises(ValueError, match="same participants"):
        paired_participant_test(
            pd.Series([1.0, 0.0], index=[0, 1]), pd.Series([1.0], index=[0])
        )


def test_mcnemar_by_repeat_uses_only_discordant_participants():
    a = _predictions([[1, 1, 1, 0], [1, 1, 0, 0]])
    b = _predictions([[1, 0, 0, 0], [1, 1, 0, 0]])
    table = mcnemar_by_repeat(a, b, SEVERE)
    assert list(table["only_a"]) == [2, 0]
    assert list(table["only_b"]) == [0, 0]
    assert table.loc[0, "p_value_two_sided"] == pytest.approx(
        stats.binomtest(2, 2, 0.5).pvalue
    )
    assert table.loc[1, "p_value_two_sided"] == 1.0


def test_mcnemar_requires_the_same_predictions():
    a = _predictions([[1, 0]])
    with pytest.raises(ValueError):
        mcnemar_by_repeat(a, a.iloc[:-1], SEVERE)


def test_holm_adjustment():
    adjusted = holm_adjust({"H2": 0.01, "H3": 0.04})
    assert adjusted["H2"] == pytest.approx(0.02)
    assert adjusted["H3"] == pytest.approx(0.04)
    # Adjusted values never decrease along the ordered p-values and stay <= 1.
    adjusted = holm_adjust({"a": 0.04, "b": 0.03, "c": 0.5})
    assert adjusted["b"] == pytest.approx(0.09)
    assert adjusted["a"] == pytest.approx(0.09)
    assert adjusted["c"] == pytest.approx(0.5)
    assert holm_adjust({"a": 0.9, "b": 0.8})["b"] == 1.0


def _differences(values: np.ndarray) -> pd.DataFrame:
    y = np.repeat(np.arange(4), (60, 30, 20, 10))
    splits = outer_splits(y)
    base = pd.DataFrame(
        {
            "repeat": [s.repeat for s in splits],
            "fold": [s.fold for s in splits],
            "n_train": [len(s.train) for s in splits],
            "n_test": [len(s.test) for s in splits],
            "test_hash": [str(i) for i in range(len(splits))],
        }
    )
    return paired_differences(
        base.assign(qwk=0.3 + values), base.assign(qwk=np.full(len(splits), 0.3))
    )


def test_negligibility_test_accepts_small_and_rejects_large_differences():
    rng = np.random.default_rng(0)
    noise = rng.normal(scale=0.005, size=25)
    small = negligibility_test(_differences(0.002 + noise - noise.mean()), 0.03)
    assert small["upper_bound"] < 0.03
    assert small["p_value_one_sided"] < 0.05
    large = negligibility_test(_differences(0.05 + noise - noise.mean()), 0.03)
    assert large["upper_bound"] > 0.03
    assert large["p_value_one_sided"] > 0.5


def test_negligibility_test_with_constant_differences():
    assert (
        negligibility_test(_differences(np.full(25, 0.01)), 0.03)["p_value_one_sided"]
        == 0.0
    )
    assert (
        negligibility_test(_differences(np.full(25, 0.05)), 0.03)["p_value_one_sided"]
        == 1.0
    )
