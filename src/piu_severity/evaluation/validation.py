"""Repeated nested cross-validation, per-repeat summaries and paired comparisons."""

import hashlib
import resource
import sys
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import ArrayLike
from scipy import stats
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold

from piu_severity.config import RANDOM_SEED
from piu_severity.evaluation.metrics import (
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
FOLD_COLUMNS: tuple[str, ...] = (
    "repeat",
    "fold",
    "n_train",
    "n_test",
    "test_hash",
    "qwk",
    "fit_seconds",
    "best_params",
)
IMPUTER_WARNING_MARKER = "IterativeImputer"


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


def hash_indices(indices: np.ndarray) -> str:
    """Return a fingerprint of a set of row positions, independent of their order."""
    ordered = np.sort(np.asarray(indices, dtype=np.int64))
    return hashlib.sha1(ordered.tobytes()).hexdigest()


def peak_memory_mb() -> float:
    """Return the peak resident memory of the current process, in megabytes."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports kilobytes.
    return peak / 1024**2 if sys.platform == "darwin" else peak / 1024


def _count_pipeline_fits(model: BaseEstimator) -> int:
    """Return the pipeline fits of an inner search, refit included; 1 otherwise."""
    if not hasattr(model, "cv_results_"):
        return 1
    refit = 1 if getattr(model, "refit", False) else 0
    return len(model.cv_results_["params"]) * model.n_splits_ + refit


def _fit_recording_convergence(
    model: BaseEstimator, X: pd.DataFrame, y: np.ndarray, show: bool = False
) -> tuple[int, int]:
    """Fit the model and count its convergence warnings by source.

    Warnings raised in parallel workers never reach the parent process, so they
    are counted where the fit happens. Other warnings are emitted again, and so
    are convergence warnings when show is True.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X, y)
    model_warnings = imputer_warnings = 0
    for warning in caught:
        is_convergence = issubclass(warning.category, ConvergenceWarning)
        if show or not is_convergence:
            warnings.warn_explicit(
                warning.message, warning.category, warning.filename, warning.lineno
            )
        if not is_convergence:
            continue
        if IMPUTER_WARNING_MARKER in str(warning.message):
            imputer_warnings += 1
        else:
            model_warnings += 1
    return model_warnings, imputer_warnings


def _cv_results_table(model: BaseEstimator) -> pd.DataFrame | None:
    """Return the inner-search results without the column of parameter objects."""
    if not hasattr(model, "cv_results_"):
        return None
    table = pd.DataFrame(model.cv_results_)
    return table.drop(columns="params")


def evaluate_fold(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    levels: np.ndarray,
    split: Split,
    keep_estimator: bool = False,
    show_convergence_warnings: bool = False,
) -> dict:
    """Fit a fresh clone of estimator on one outer fold and evaluate it.

    Returns the fold row (with timing, convergence warnings and peak memory), the
    out-of-fold predictions, the inner-search results and, when requested, the
    fitted best estimator.
    """
    model = clone(estimator)
    start = time.perf_counter()
    model_warnings, imputer_warnings = _fit_recording_convergence(
        model, X.iloc[split.train], levels[split.train], show_convergence_warnings
    )
    fit_seconds = time.perf_counter() - start
    y_true = levels[split.test]
    y_pred = as_levels(model.predict(X.iloc[split.test]))
    row = {
        "repeat": split.repeat,
        "fold": split.fold,
        "n_train": len(split.train),
        "n_test": len(split.test),
        "test_hash": hash_indices(split.test),
        "qwk": quadratic_weighted_kappa(y_true, y_pred),
        "fit_seconds": fit_seconds,
        "best_params": getattr(model, "best_params_", None),
        "n_pipeline_fits": _count_pipeline_fits(model),
        "convergence_warnings_model": model_warnings,
        "convergence_warnings_imputer": imputer_warnings,
        "peak_memory_mb": peak_memory_mb(),
    }
    predictions = pd.DataFrame(
        {
            "repeat": split.repeat,
            "fold": split.fold,
            "position": split.test,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    best = getattr(model, "best_estimator_", model) if keep_estimator else None
    if best is not None and hasattr(best, "memory"):
        # The cache folder of the fitting machine is not needed to predict.
        best.set_params(memory=None)
    return {
        "row": row,
        "predictions": predictions,
        "cv_results": _cv_results_table(model),
        "estimator": best,
    }


def run_nested_cv(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: ArrayLike,
    splits: list[Split],
    n_jobs: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit a fresh clone of estimator on every outer fold and evaluate it.

    The estimator carries its own inner search (for example, a GridSearchCV), so
    every tuning decision uses only the outer training rows. With n_jobs, the
    outer folds run in parallel; keep the inner search sequential then.
    Convergence warnings are shown as they occur, as before this option existed.
    Returns one row of results per fold and the out-of-fold predictions. The
    predictions are row-level: keep them out of version control.
    """
    levels = as_levels(y)
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_fold)(
            estimator, X, levels, split, show_convergence_warnings=True
        )
        for split in splits
    )
    folds = pd.DataFrame([result["row"] for result in results])[list(FOLD_COLUMNS)]
    predictions = pd.concat(
        [result["predictions"] for result in results], ignore_index=True
    )
    return folds, predictions


def summarize_by_repeat(folds: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Return one row of metrics per repeat.

    The QWK is the mean of the fold values of the repeat. Per-level recall and
    under-estimation rate are computed on the pooled out-of-fold predictions of
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
    per-level recall.
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
