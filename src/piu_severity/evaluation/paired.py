"""Paired comparison of two evaluations of the same participants.

Two configurations evaluated on the same outer splits predict every
participant once per repeat. Each participant gets a rate per configuration
(the share of repeats with a hit, an under-estimation or a referral), and the
configurations are compared through the per-participant differences of those
rates: a percentile bootstrap over participants gives the interval of the mean
difference, and a sign-flip test its two-sided p-value. These comparisons are
descriptive; they are not part of any family of hypothesis tests.
"""

from collections.abc import Sequence

import pandas as pd

from piu_severity.evaluation.inference import (
    paired_participant_test,
    participant_bootstrap_ci,
)
from piu_severity.evaluation.metrics import UNDERESTIMATION_LEVELS
from piu_severity.evaluation.screening import (
    REFERRAL_LEVEL,
    participant_referral_rates,
)


def participant_underestimation_rates(
    predictions: pd.DataFrame, levels: Sequence[int] = UNDERESTIMATION_LEVELS
) -> pd.Series:
    """Return, for each participant of the given levels, the share of repeats under-estimated.

    Their mean is the under-estimation rate of those levels averaged over the
    repeats, because every participant is predicted once per repeat.
    """
    cases = predictions.loc[predictions["y_true"].isin(list(levels))]
    lower = (cases["y_pred"] < cases["y_true"]).astype(float)
    return lower.groupby(cases["position"]).mean().rename("underestimation_rate")


def participant_levels_referral_rates(
    predictions: pd.DataFrame,
    levels: Sequence[int],
    referral_level: int = REFERRAL_LEVEL,
) -> pd.Series:
    """Return the referral rate of every participant whose true level is in levels."""
    rates = [
        participant_referral_rates(predictions, level, referral_level)
        for level in levels
    ]
    return pd.concat(rates).sort_index().rename("referral_rate")


def compare_participant_rates(
    rates_a: pd.Series, rates_b: pd.Series, confidence: float = 0.95
) -> dict[str, float]:
    """Compare two configurations through the per-participant difference a − b.

    Both series must cover the same participants. Returns the mean rate of
    each configuration, the mean difference with its bootstrap interval, and
    the sign-flip p-value with the number of participants whose rate differs.
    """
    test = paired_participant_test(rates_a, rates_b)
    a, b = rates_a.align(rates_b, join="outer")
    low, high = participant_bootstrap_ci(a - b, confidence=confidence)
    return {
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "difference": test["mean_difference"],
        "interval_low": low,
        "interval_high": high,
        "p_value_two_sided": test["p_value_two_sided"],
        "participants": test["participants"],
        "participants_differing": test["participants_differing"],
    }
