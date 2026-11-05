"""Metrics for the ordinal SII target: agreement, per-level recall and under-estimation."""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import norm
from sklearn.metrics import cohen_kappa_score, confusion_matrix, make_scorer

SII_LEVELS: tuple[int, ...] = (0, 1, 2, 3)
UNDERESTIMATION_LEVELS: tuple[int, ...] = (2, 3)


def as_levels(values: ArrayLike) -> np.ndarray:
    """Return SII values as an integer array; reject missing or non-integer values."""
    array = np.asarray(values, dtype=float)
    if np.isnan(array).any() or not np.array_equal(array, np.round(array)):
        raise ValueError("SII levels must be integers without missing values.")
    return array.astype(int)


def quadratic_weighted_kappa(
    y_true: ArrayLike, y_pred: ArrayLike, levels: tuple[int, ...] = SII_LEVELS
) -> float:
    """Return Cohen's kappa with quadratic weights over a fixed set of levels."""
    return float(
        cohen_kappa_score(
            as_levels(y_true),
            as_levels(y_pred),
            labels=list(levels),
            weights="quadratic",
        )
    )


def level_confusion_matrix(
    y_true: ArrayLike, y_pred: ArrayLike, levels: tuple[int, ...] = SII_LEVELS
) -> pd.DataFrame:
    """Return the confusion matrix with every level as a row and a column."""
    matrix = confusion_matrix(as_levels(y_true), as_levels(y_pred), labels=list(levels))
    return pd.DataFrame(
        matrix,
        index=pd.Index(levels, name="true"),
        columns=pd.Index(levels, name="predicted"),
    )


def recall_by_level(
    y_true: ArrayLike, y_pred: ArrayLike, levels: tuple[int, ...] = SII_LEVELS
) -> pd.Series:
    """Return the recall of each level; NaN for levels absent from y_true."""
    matrix = level_confusion_matrix(y_true, y_pred, levels).to_numpy()
    support = matrix.sum(axis=1)
    hits = np.diag(matrix)
    recall = np.full(len(levels), np.nan)
    np.divide(hits, support, out=recall, where=support > 0)
    return pd.Series(recall, index=pd.Index(levels, name="level"), name="recall")


def underestimation_rate(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    levels: tuple[int, ...] = UNDERESTIMATION_LEVELS,
) -> float:
    """Return the share of participants in the given levels assigned a lower level.

    Returns NaN when no participant belongs to the given levels.
    """
    true = as_levels(y_true)
    pred = as_levels(y_pred)
    selected = np.isin(true, levels)
    if not selected.any():
        return float("nan")
    return float(np.mean(pred[selected] < true[selected]))


def wilson_interval(
    proportion: float, n: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Return the Wilson score interval of a proportion estimated from n cases.

    Unlike the normal (Wald) interval, its limits stay within [0, 1] and its
    coverage holds with small samples and proportions close to 0 or 1.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if not 0 <= proportion <= 1:
        raise ValueError("proportion must lie in [0, 1].")
    z = norm.ppf(1 - (1 - confidence) / 2)
    denominator = 1 + z**2 / n
    center = (proportion + z**2 / (2 * n)) / denominator
    margin = (z / denominator) * np.sqrt(
        proportion * (1 - proportion) / n + z**2 / (4 * n**2)
    )
    # Clip rounding error so that the limits never leave [0, 1].
    return float(max(center - margin, 0.0)), float(min(center + margin, 1.0))


qwk_scorer = make_scorer(quadratic_weighted_kappa)
