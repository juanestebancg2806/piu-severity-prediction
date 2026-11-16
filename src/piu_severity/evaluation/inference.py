"""Uncertainty of per-level results and paired tests between configurations.

The severe level has 34 participants, so its recall is reported per repeat with
a Wilson interval and, across repeats, with a bootstrap over participants.
Two configurations evaluated on the same splits are compared on those
participants with a paired test whose unit is the participant, because the
repeats reuse the same people. The QWK comparisons of the hypotheses use the
Nadeau–Bengio corrected t test, here also in its negligibility form, and the
p-values of each family of tests are adjusted with Holm's method.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy import stats

from piu_severity.config import RANDOM_SEED
from piu_severity.evaluation.metrics import wilson_interval

SEVERE_LEVEL = 3
# The severe recall is unstable when its standard deviation across repeats
# exceeds this value (about three of the 34 severe cases); the evaluation of
# the selected configuration is then extended to EXTENDED_REPEATS repeats.
SEVERE_RECALL_MAX_SD = 0.10
EXTENDED_REPEATS = 10
N_BOOTSTRAP = 10_000
N_PERMUTATIONS = 10_000


def level_hits_by_repeat(predictions: pd.DataFrame, level: int) -> pd.DataFrame:
    """Return the hits of one level in each repeat, with its recall and Wilson interval.

    In a repeat every participant is predicted once, so the hits are whole
    counts over the participants of that level.
    """
    rows = []
    for repeat, group in predictions.groupby("repeat"):
        cases = group["y_true"] == level
        n_cases = int(cases.sum())
        if n_cases == 0:
            raise ValueError(f"Repeat {repeat} has no participant of level {level}.")
        hits = int((group.loc[cases, "y_pred"] == level).sum())
        low, high = wilson_interval(hits / n_cases, n_cases)
        rows.append(
            {
                "repeat": repeat,
                "cases": n_cases,
                "hits": hits,
                "recall": hits / n_cases,
                "wilson_low": low,
                "wilson_high": high,
            }
        )
    return pd.DataFrame(rows).set_index("repeat")


def assigned_level_distribution(predictions: pd.DataFrame, level: int) -> pd.Series:
    """Return the share of the participants of one level assigned to each level.

    The shares are averaged across repeats, so they add up to one.
    """
    cases = predictions.loc[predictions["y_true"] == level]
    shares = (
        cases.groupby("repeat")["y_pred"]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
        .mean()
    )
    return shares.rename("share").rename_axis("assigned_level")


def participant_hit_rates(predictions: pd.DataFrame, level: int) -> pd.Series:
    """Return, for each participant of one level, the share of repeats with a hit."""
    cases = predictions.loc[predictions["y_true"] == level]
    hits = (cases["y_pred"] == level).astype(float)
    return hits.groupby(cases["position"]).mean().rename("hit_rate")


def participant_bootstrap_ci(
    hit_rates: pd.Series,
    confidence: float = 0.95,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    """Return a percentile bootstrap interval of the recall, resampling participants.

    The recall over all repeats is the mean of the participants' hit rates; the
    participants, not the repeats, are the independent units.
    """
    rates = hit_rates.to_numpy()
    rng = np.random.default_rng(seed)
    samples = rng.choice(rates, size=(n_bootstrap, len(rates)), replace=True)
    means = samples.mean(axis=1)
    tail = (1 - confidence) / 2
    return float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail))


def paired_participant_test(
    rates_a: pd.Series,
    rates_b: pd.Series,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = RANDOM_SEED,
) -> dict[str, float]:
    """Compare two configurations on the same participants with a sign-flip test.

    Each participant contributes the difference between its hit rates under a
    and b. Under the null hypothesis of no difference, the sign of each
    difference is exchangeable, so the two-sided p-value is the share of random
    sign flips whose mean difference is at least as large in absolute value as
    the observed one (with the observed arrangement counted once).
    """
    a, b = rates_a.align(rates_b, join="outer")
    if a.isna().any() or b.isna().any():
        raise ValueError("Both configurations must cover the same participants.")
    differences = (a - b).to_numpy()
    observed = float(differences.mean())
    nonzero = differences[differences != 0]
    if len(nonzero) == 0:
        p_value = 1.0
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1.0, 1.0], size=(n_permutations, len(nonzero)))
        null = np.abs((signs * nonzero).sum(axis=1) / len(differences))
        p_value = float(
            (1 + np.sum(null >= abs(observed) - 1e-12)) / (1 + n_permutations)
        )
    return {
        "mean_difference": observed,
        "participants": len(differences),
        "participants_differing": len(nonzero),
        "p_value_two_sided": p_value,
    }


def mcnemar_by_repeat(
    predictions_a: pd.DataFrame, predictions_b: pd.DataFrame, level: int
) -> pd.DataFrame:
    """Return the exact McNemar test of the hits of one level in each repeat.

    Only the participants that one configuration hits and the other misses
    count; their split is tested against 50/50 with a two-sided binomial test.
    """
    keys = ["repeat", "position"]
    merged = predictions_a[[*keys, "y_true", "y_pred"]].merge(
        predictions_b[[*keys, "y_true", "y_pred"]],
        on=keys,
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    if len(merged) != len(predictions_a) or len(merged) != len(predictions_b):
        raise ValueError("Both configurations must cover the same predictions.")
    if (merged["y_true_a"] != merged["y_true_b"]).any():
        raise ValueError("The true levels of the two configurations differ.")
    rows = []
    for repeat, group in merged.loc[merged["y_true_a"] == level].groupby("repeat"):
        hit_a = group["y_pred_a"] == level
        hit_b = group["y_pred_b"] == level
        only_a = int((hit_a & ~hit_b).sum())
        only_b = int((~hit_a & hit_b).sum())
        discordant = only_a + only_b
        p_value = (
            1.0
            if discordant == 0
            else float(stats.binomtest(only_a, discordant, 0.5).pvalue)
        )
        rows.append(
            {
                "repeat": repeat,
                "only_a": only_a,
                "only_b": only_b,
                "p_value_two_sided": p_value,
            }
        )
    return pd.DataFrame(rows).set_index("repeat")


def holm_adjust(p_values: Mapping[str, float]) -> pd.Series:
    """Return Holm's step-down adjusted p-values of one family of tests."""
    names = list(p_values)
    values = np.array([p_values[name] for name in names], dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return pd.Series(adjusted, index=names, name="p_value_holm")


def negligibility_test(
    differences: pd.DataFrame, margin: float, confidence: float = 0.95
) -> dict[str, float]:
    """Test whether a QWK difference is smaller than a relevant margin.

    Uses the Nadeau–Bengio corrected standard error, as summarize_paired does:
    the difference is negligible when the one-sided upper bound of the mean
    lies below margin, which is the corrected t test of mean < margin.
    """
    diff = differences["difference"].to_numpy()
    n_folds = len(diff)
    mean = float(diff.mean())
    std = float(diff.std(ddof=1))
    ratio = float((differences["n_test"] / differences["n_train"]).mean())
    corrected_se = float(np.sqrt((1 / n_folds + ratio) * std**2))
    critical = float(stats.t.ppf(confidence, df=n_folds - 1))
    upper = mean + critical * corrected_se
    if corrected_se > 0:
        t_statistic = (mean - margin) / corrected_se
        p_value = float(stats.t.cdf(t_statistic, df=n_folds - 1))
    else:
        t_statistic = float("nan")
        p_value = 0.0 if mean < margin else 1.0
    return {
        "mean_difference": mean,
        "upper_bound": upper,
        "margin": margin,
        "t_statistic": float(t_statistic),
        "p_value_one_sided": p_value,
    }
