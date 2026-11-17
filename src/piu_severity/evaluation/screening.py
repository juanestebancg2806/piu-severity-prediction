"""Screening scenario: referral of every participant estimated moderate or severe.

The predictions of the selected configuration are translated into a referral
decision. A participant is positive when the true SII reaches the referral
level and is referred when the estimated level reaches it. The metrics are
computed on the predictions of each repeat, where every participant is
predicted once; their stability is the deviation across repeats, and their
uncertainty a percentile bootstrap over participants, which are the
independent units because the repeats reuse the same people.
"""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from piu_severity.config import RANDOM_SEED
from piu_severity.evaluation.inference import N_BOOTSTRAP
from piu_severity.evaluation.metrics import as_levels, wilson_interval

# Fixed before the results are seen: a participant is referred when the
# estimated level is moderate (2) or severe (3), as the formulation states.
REFERRAL_LEVEL = 2
SCREENING_CONFIDENCE = 0.95
# Bootstrap samples drawn at once; bounds the memory of the resampling.
BOOTSTRAP_CHUNK_SIZE = 500
SCREENING_METRICS: tuple[str, ...] = (
    "recall",
    "specificity",
    "ppv",
    "referred_share",
)
COUNT_COLUMNS: tuple[str, ...] = ("tp", "fn", "fp", "tn")
PREDICTION_COLUMNS: tuple[str, ...] = ("repeat", "position", "y_true", "y_pred")


def _check_predictions(predictions: pd.DataFrame) -> None:
    """Raise if the predictions lack the columns the screening needs."""
    missing = [c for c in PREDICTION_COLUMNS if c not in predictions.columns]
    if missing:
        raise KeyError(f"Predictions lack the columns {missing}.")


def referral_counts(
    y_true: ArrayLike, y_pred: ArrayLike, referral_level: int = REFERRAL_LEVEL
) -> dict[str, int]:
    """Return the true and false positives and negatives of the referral."""
    positive = as_levels(y_true) >= referral_level
    referred = as_levels(y_pred) >= referral_level
    return {
        "tp": int(np.sum(positive & referred)),
        "fn": int(np.sum(positive & ~referred)),
        "fp": int(np.sum(~positive & referred)),
        "tn": int(np.sum(~positive & ~referred)),
    }


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Return numerator / denominator, NaN where the denominator is zero."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    result = np.full(np.broadcast(numerator, denominator).shape, np.nan)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def metrics_from_counts(
    tp: ArrayLike, fn: ArrayLike, fp: ArrayLike, tn: ArrayLike
) -> dict[str, np.ndarray]:
    """Return recall, specificity, PPV and referred share from referral counts.

    Works element-wise on arrays, so it also serves the bootstrap. A metric
    whose denominator is zero is NaN.
    """
    tp, fn, fp, tn = (np.asarray(v, dtype=float) for v in (tp, fn, fp, tn))
    return {
        "recall": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "ppv": _ratio(tp, tp + fp),
        "referred_share": _ratio(tp + fp, tp + fn + fp + tn),
    }


def referral_metrics(
    y_true: ArrayLike, y_pred: ArrayLike, referral_level: int = REFERRAL_LEVEL
) -> dict[str, float]:
    """Return recall, specificity, PPV and referred share of the referral."""
    counts = referral_counts(y_true, y_pred, referral_level)
    return {name: float(value) for name, value in metrics_from_counts(**counts).items()}


def screening_by_repeat(
    predictions: pd.DataFrame, referral_level: int = REFERRAL_LEVEL
) -> pd.DataFrame:
    """Return the screening metrics computed on the predictions of each repeat."""
    _check_predictions(predictions)
    rows = {
        repeat: referral_metrics(group["y_true"], group["y_pred"], referral_level)
        for repeat, group in predictions.groupby("repeat")
    }
    table = pd.DataFrame.from_dict(rows, orient="index")
    return table[list(SCREENING_METRICS)].rename_axis("repeat")


def level_referral_by_repeat(
    predictions: pd.DataFrame,
    level: int,
    referral_level: int = REFERRAL_LEVEL,
    confidence: float = SCREENING_CONFIDENCE,
) -> pd.DataFrame:
    """Return, per repeat, how many participants of one true level are referred.

    The share comes with its Wilson interval over the participants of that
    level, as the severe recall does.
    """
    _check_predictions(predictions)
    rows = []
    for repeat, group in predictions.groupby("repeat"):
        cases = group["y_true"] == level
        n_cases = int(cases.sum())
        if n_cases == 0:
            raise ValueError(f"Repeat {repeat} has no participant of level {level}.")
        referred = int((group.loc[cases, "y_pred"] >= referral_level).sum())
        low, high = wilson_interval(referred / n_cases, n_cases, confidence)
        rows.append(
            {
                "repeat": repeat,
                "cases": n_cases,
                "referred": referred,
                "share": referred / n_cases,
                "wilson_low": low,
                "wilson_high": high,
            }
        )
    return pd.DataFrame(rows).set_index("repeat")


def participant_referral_rates(
    predictions: pd.DataFrame, level: int, referral_level: int = REFERRAL_LEVEL
) -> pd.Series:
    """Return, for each participant of one true level, the share of repeats referred."""
    _check_predictions(predictions)
    cases = predictions.loc[predictions["y_true"] == level]
    referred = (cases["y_pred"] >= referral_level).astype(float)
    return referred.groupby(cases["position"]).mean().rename("referral_rate")


def referral_consistency(rates: pd.Series) -> dict[str, int]:
    """Count the participants referred in every repeat, in none and in some."""
    always = int((rates == 1).sum())
    never = int((rates == 0).sum())
    return {
        "always": always,
        "never": never,
        "sometimes": len(rates) - always - never,
        "participants": len(rates),
    }


def participant_referral_counts(
    predictions: pd.DataFrame, referral_level: int = REFERRAL_LEVEL
) -> pd.DataFrame:
    """Return each participant's referral counts summed over the repeats.

    Every participant must be predicted once in every repeat, so that the
    pooled counts weigh the repeats equally.
    """
    _check_predictions(predictions)
    per_participant = predictions.groupby("position")["repeat"].agg(
        ["count", "nunique"]
    )
    n_repeats = predictions["repeat"].nunique()
    if (per_participant["count"] != n_repeats).any() or (
        per_participant["nunique"] != n_repeats
    ).any():
        raise ValueError("Every participant must be predicted once in every repeat.")
    positive = predictions["y_true"] >= referral_level
    referred = predictions["y_pred"] >= referral_level
    indicators = pd.DataFrame(
        {
            "tp": positive & referred,
            "fn": positive & ~referred,
            "fp": ~positive & referred,
            "tn": ~positive & ~referred,
        }
    ).astype(int)
    return indicators.groupby(predictions["position"]).sum()[list(COUNT_COLUMNS)]


def participant_bootstrap_screening(
    predictions: pd.DataFrame,
    referral_level: int = REFERRAL_LEVEL,
    confidence: float = SCREENING_CONFIDENCE,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
    chunk_size: int = BOOTSTRAP_CHUNK_SIZE,
) -> pd.DataFrame:
    """Return a percentile bootstrap interval of each screening metric.

    Participants are resampled with replacement, each with its counts over all
    repeats, and every metric is computed on the pooled counts of the sample.
    The estimate is the metric on the pooled counts of all participants, which
    equals the mean across repeats for every metric but the PPV. Samples in
    which a metric is undefined (no positive, negative or referred
    participant) are left out of its interval.
    """
    counts = participant_referral_counts(predictions, referral_level).to_numpy()
    rng = np.random.default_rng(seed)
    n_participants = len(counts)
    samples = {name: [] for name in SCREENING_METRICS}
    for start in range(0, n_bootstrap, chunk_size):
        size = min(chunk_size, n_bootstrap - start)
        index = rng.integers(0, n_participants, size=(size, n_participants))
        totals = counts[index].sum(axis=1)
        metrics = metrics_from_counts(*totals.T)
        for name in SCREENING_METRICS:
            samples[name].append(metrics[name])
    estimate = metrics_from_counts(*counts.sum(axis=0))
    tail = (1 - confidence) / 2
    rows = {}
    for name in SCREENING_METRICS:
        values = np.concatenate(samples[name])
        values = values[~np.isnan(values)]
        low, high = (
            (np.quantile(values, tail), np.quantile(values, 1 - tail))
            if len(values)
            else (np.nan, np.nan)
        )
        rows[name] = {
            "estimate": float(estimate[name]),
            "interval_low": float(low),
            "interval_high": float(high),
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("metric")
