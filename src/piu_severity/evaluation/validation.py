"""Repeated nested cross-validation, per-repeat summaries and paired comparisons."""

import hashlib
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold

from piu_severity.config import RANDOM_SEED
from piu_severity.evaluation.metrics import (
    SEVERE_LEVELS,
    SII_LEVELS,
    as_levels,
    level_confusion_matrix,
    quadratic_weighted_kappa,
    recall_by_level,
    underestimation_rate,
)

N_OUTER_SPLITS = 5
N_OUTER_REPEATS = 5
N_INNER_SPLITS = 3


@dataclass(frozen=True)
class Split:
    """One outer fold: positional indices of its training and evaluation rows."""

    repeat: int
    fold: int
    train: np.ndarray
    test: np.ndarray


def outer_splits(
    y: ArrayLike,
    n_splits: int = N_OUTER_SPLITS,
    n_repeats: int = N_OUTER_REPEATS,
    seed: int = RANDOM_SEED,
) -> list[Split]:
    """Return stratified outer folds; repeat r uses seed + r, so repeats are reusable."""
    levels = as_levels(y)
    placeholder = np.zeros(len(levels))
    splits = []
    for repeat in range(n_repeats):
        cv = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=seed + repeat
        )
        for fold, (train, test) in enumerate(cv.split(placeholder, levels)):
            splits.append(Split(repeat, fold, train, test))
    return splits


def inner_cv(
    n_splits: int = N_INNER_SPLITS, seed: int = RANDOM_SEED
) -> StratifiedKFold:
    """Return the stratified splitter used inside each outer training fold."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def split_level_counts(
    splits: list[Split], y: ArrayLike, levels: tuple[int, ...] = SII_LEVELS
) -> pd.DataFrame:
    """Return the number of evaluation cases of each level in every outer fold."""
    values = as_levels(y)
    rows = [
        {
            "repeat": split.repeat,
            "fold": split.fold,
            **{level: int(np.sum(values[split.test] == level)) for level in levels},
        }
        for split in splits
    ]
    return pd.DataFrame(rows).set_index(["repeat", "fold"])


def _hash_indices(indices: np.ndarray) -> str:
    """Return a fingerprint of a set of row positions, independent of their order."""
    ordered = np.sort(np.asarray(indices, dtype=np.int64))
    return hashlib.sha1(ordered.tobytes()).hexdigest()


def run_nested_cv(
    estimator: BaseEstimator, X: pd.DataFrame, y: ArrayLike, splits: list[Split]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit a fresh clone of estimator on every outer fold and evaluate it.

    The estimator carries its own inner search (for example, a GridSearchCV), so
    every tuning decision uses only the outer training rows.
    Returns one row of results per fold and the out-of-fold predictions. The
    predictions are row-level: keep them out of version control.
    """
    levels = as_levels(y)
    fold_rows = []
    prediction_frames = []
    for split in splits:
        model = clone(estimator)
        start = time.perf_counter()
        model.fit(X.iloc[split.train], levels[split.train])
        fit_seconds = time.perf_counter() - start
        y_true = levels[split.test]
        y_pred = as_levels(model.predict(X.iloc[split.test]))
        fold_rows.append(
            {
                "repeat": split.repeat,
                "fold": split.fold,
                "n_train": len(split.train),
                "n_test": len(split.test),
                "test_hash": _hash_indices(split.test),
                "qwk": quadratic_weighted_kappa(y_true, y_pred),
                "fit_seconds": fit_seconds,
                "best_params": getattr(model, "best_params_", None),
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "repeat": split.repeat,
                    "fold": split.fold,
                    "position": split.test,
                    "y_true": y_true,
                    "y_pred": y_pred,
                }
            )
        )
    return pd.DataFrame(fold_rows), pd.concat(prediction_frames, ignore_index=True)


def summarize_by_repeat(folds: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Return one row of metrics per repeat.

    The QWK is the mean of the fold values of the repeat. Per-level sensitivity and
    under-estimation rates are computed on the pooled out-of-fold predictions of
    the repeat, where every participant is predicted exactly once.
    """
    rows = []
    for repeat, group in predictions.groupby("repeat"):
        recall = recall_by_level(group["y_true"], group["y_pred"])
        rows.append(
            {
                "repeat": repeat,
                "qwk": folds.loc[folds["repeat"] == repeat, "qwk"].mean(),
                **{f"recall_{level}": float(value) for level, value in recall.items()},
                "underestimation": underestimation_rate(
                    group["y_true"], group["y_pred"]
                ),
                "underestimation_severe": underestimation_rate(
                    group["y_true"], group["y_pred"], SEVERE_LEVELS
                ),
            }
        )
    return pd.DataFrame(rows).set_index("repeat")


def aggregate_repeats(summary: pd.DataFrame) -> pd.DataFrame:
    """Return the mean and standard deviation across repeats of each metric."""
    return summary.agg(["mean", "std"]).T


def mean_confusion_matrix(
    predictions: pd.DataFrame, normalize: bool = True
) -> pd.DataFrame:
    """Return the confusion matrix averaged across repeats.

    With normalize, each row is divided by its total, so the diagonal holds the
    per-level sensitivity.
    """
    matrices = [
        level_confusion_matrix(group["y_true"], group["y_pred"])
        for _, group in predictions.groupby("repeat")
    ]
    mean = sum(matrices) / len(matrices)
    if normalize:
        mean = mean.div(mean.sum(axis="columns"), axis="index")
    return mean


def paired_differences(folds_a: pd.DataFrame, folds_b: pd.DataFrame) -> pd.DataFrame:
    """Return the per-fold QWK difference a − b; both must use the same splits."""
    keys = ["repeat", "fold"]
    merged = folds_a[[*keys, "n_train", "n_test", "test_hash", "qwk"]].merge(
        folds_b[[*keys, "test_hash", "qwk"]],
        on=keys,
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    same_folds = len(merged) == len(folds_a) == len(folds_b)
    if not same_folds or (merged["test_hash_a"] != merged["test_hash_b"]).any():
        raise ValueError("Both configurations must be evaluated on the same splits.")
    merged["difference"] = merged["qwk_a"] - merged["qwk_b"]
    return merged[[*keys, "n_train", "n_test", "qwk_a", "qwk_b", "difference"]]


def summarize_paired(differences: pd.DataFrame) -> dict[str, float]:
    """Summarize paired fold differences and test whether their mean exceeds zero.

    The t statistic uses the Nadeau–Bengio variance correction for overlapping
    training sets, (1/J + n_test/n_train) times the variance of the J differences;
    the p-value is one-sided.
    """
    diff = differences["difference"].to_numpy()
    n_folds = len(diff)
    mean = float(diff.mean())
    std = float(diff.std(ddof=1))
    test_train_ratio = float((differences["n_test"] / differences["n_train"]).mean())
    corrected_se = np.sqrt((1 / n_folds + test_train_ratio) * std**2)
    t_statistic = mean / corrected_se if corrected_se > 0 else float("nan")
    return {
        "mean_difference": mean,
        "std_difference": std,
        "folds_improved": int(np.sum(diff > 0)),
        "n_folds": n_folds,
        "corrected_t": float(t_statistic),
        "p_value_one_sided": float(stats.t.sf(t_statistic, df=n_folds - 1)),
    }


def permutation_null(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: ArrayLike,
    n_permutations: int,
    n_splits: int = N_OUTER_SPLITS,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Return the mean fold QWK of the estimator for labels permuted at random.

    Each permutation is evaluated with the same procedure as the observed score:
    one repeat of stratified outer folds, built with the seed of repeat 0.
    """
    rng = np.random.default_rng(seed)
    levels = as_levels(y)
    scores = np.empty(n_permutations)
    for index in range(n_permutations):
        permuted = rng.permutation(levels)
        splits = outer_splits(permuted, n_splits=n_splits, n_repeats=1, seed=seed)
        folds, _ = run_nested_cv(estimator, X, permuted, splits)
        scores[index] = folds["qwk"].mean()
    return scores


def permutation_p_value(observed: float, null_scores: np.ndarray) -> float:
    """Return the permutation p-value of an observed score, (1 + #null ≥ obs)/(1 + B)."""
    return float((1 + np.sum(null_scores >= observed)) / (1 + len(null_scores)))
