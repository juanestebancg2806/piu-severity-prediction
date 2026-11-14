"""Formulations of the ordinal SII as scikit-learn estimators.

Each estimator weights observations inversely to the frequency of their level in
the data it is fitted on, raised to weight_power (1 gives the usual balanced
weights, n / (K n_k)). Preprocessing stays outside these estimators, in the
pipeline that contains them.
"""

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

from piu_severity.config import RANDOM_SEED
from piu_severity.evaluation.metrics import SII_LEVELS, as_levels

N_THRESHOLD_CANDIDATES = 200
INITIAL_THRESHOLDS: tuple[float, ...] = (0.5, 1.5, 2.5)
MAX_COORDINATE_SWEEPS = 50
N_THRESHOLD_SPLITS = 3


def level_weights(y: ArrayLike, power: float = 1.0) -> np.ndarray:
    """Return one weight per observation, (n / (K n_k)) ** power for its level k.

    K is the number of levels present in y, so power = 1 matches scikit-learn's
    balanced weights and power = 0 gives equal weights.
    """
    levels = as_levels(y)
    present, inverse, counts = np.unique(
        levels, return_inverse=True, return_counts=True
    )
    per_level = (len(levels) / (len(present) * counts)) ** power
    return per_level[inverse]


def _rows(X: ArrayLike, index: np.ndarray) -> ArrayLike:
    """Return the rows of X at the given positions, for DataFrames and arrays."""
    return X.iloc[index] if hasattr(X, "iloc") else np.asarray(X)[index]


def _check_sii_levels(y: ArrayLike) -> np.ndarray:
    """Return y as integer levels, rejecting values outside the SII scale."""
    levels = as_levels(y)
    if not np.isin(levels, SII_LEVELS).all():
        raise ValueError(f"SII levels must be in {SII_LEVELS}.")
    return levels


def quadratic_weighted_kappa_fast(
    y_true: np.ndarray, y_pred: np.ndarray, n_levels: int = len(SII_LEVELS)
) -> float:
    """Return the quadratic weighted kappa of integer levels 0..n_levels-1.

    Equivalent to metrics.quadratic_weighted_kappa without input validation, so
    the threshold search can evaluate it thousands of times. Returns 0 when the
    agreement expected by chance leaves no room for improvement.
    """
    observed = np.bincount(
        y_true * n_levels + y_pred, minlength=n_levels * n_levels
    ).reshape(n_levels, n_levels)
    index = np.arange(n_levels)
    weights = (index[:, None] - index[None, :]) ** 2 / (n_levels - 1) ** 2
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / len(y_true)
    denominator = float((weights * expected).sum())
    if denominator == 0:
        return 0.0
    return 1.0 - float((weights * observed).sum()) / denominator


def assign_levels(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Return the level of each score: the number of thresholds at or below it."""
    return np.searchsorted(thresholds, scores, side="right")


def _coordinate_search(
    scores: np.ndarray,
    levels: np.ndarray,
    start: np.ndarray,
    candidates: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Improve increasing thresholds one at a time over a set of candidate values.

    Each threshold may move to any candidate strictly between its neighbours, so
    the order is preserved. Sweeps stop when no threshold improves the QWK.
    """
    thresholds = np.array(start, dtype=float)
    best = quadratic_weighted_kappa_fast(levels, assign_levels(scores, thresholds))
    for _ in range(MAX_COORDINATE_SWEEPS):
        improved = False
        for j in range(len(thresholds)):
            low = thresholds[j - 1] if j > 0 else -np.inf
            high = thresholds[j + 1] if j < len(thresholds) - 1 else np.inf
            options = candidates[(candidates > low) & (candidates < high)]
            for value in options:
                trial = thresholds.copy()
                trial[j] = value
                score = quadratic_weighted_kappa_fast(
                    levels, assign_levels(scores, trial)
                )
                if score > best + 1e-12:
                    best, thresholds, improved = score, trial, True
        if not improved:
            break
    return thresholds, best


def _starting_thresholds(scores: np.ndarray, levels: np.ndarray) -> list[np.ndarray]:
    """Return the fixed start and the start that reproduces the level proportions."""
    starts = [np.array(INITIAL_THRESHOLDS, dtype=float)]
    proportions = np.bincount(levels, minlength=len(SII_LEVELS)) / len(levels)
    quantiles = np.quantile(scores, np.cumsum(proportions)[:-1])
    if np.all(np.diff(quantiles) > 0):
        starts.append(quantiles)
    return starts


def optimize_thresholds(scores: ArrayLike, levels: ArrayLike) -> np.ndarray:
    """Return three increasing thresholds that maximize the QWK of the scores.

    The QWK is a step function of the thresholds, so a local optimizer started at
    one point can stay there. The search runs from several starts over candidate
    values at the quantiles of the scores and keeps the best result; ties go to
    the first start.
    """
    scores = np.asarray(scores, dtype=float)
    levels = as_levels(levels)
    candidates = np.unique(
        np.quantile(scores, np.linspace(0, 1, N_THRESHOLD_CANDIDATES))
    )
    best_thresholds, best_score = None, -np.inf
    for start in _starting_thresholds(scores, levels):
        thresholds, score = _coordinate_search(scores, levels, start, candidates)
        if score > best_score:
            best_thresholds, best_score = thresholds, score
    return best_thresholds


class MulticlassClassifier(ClassifierMixin, BaseEstimator):
    """Nominal multiclass formulation with level weights computed in each fit."""

    def __init__(self, estimator: BaseEstimator = None, weight_power: float = 1.0):
        self.estimator = estimator
        self.weight_power = weight_power

    def fit(self, X: ArrayLike, y: ArrayLike) -> "MulticlassClassifier":
        """Fit the estimator with weights from the level frequencies of y."""
        levels = _check_sii_levels(y)
        self.classes_ = np.unique(levels)
        self.estimator_ = clone(self.estimator).fit(
            X, levels, sample_weight=level_weights(levels, self.weight_power)
        )
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Return the predicted level."""
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)

    def _estimator_has_proba(self) -> bool:
        return hasattr(self.estimator, "predict_proba")

    @available_if(_estimator_has_proba)
    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Return the probability of each level, when the estimator provides it."""
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X)


class FrankHallClassifier(ClassifierMixin, BaseEstimator):
    """Frank and Hall ordinal decomposition into binary classifiers of P(SII > k).

    Each binary classifier weights its two classes inversely to their frequency.
    The cumulative probabilities are made non-increasing in k with a cumulative
    minimum before the level probabilities are obtained by differences.
    """

    def __init__(self, estimator: BaseEstimator = None, weight_power: float = 1.0):
        self.estimator = estimator
        self.weight_power = weight_power

    def fit(self, X: ArrayLike, y: ArrayLike) -> "FrankHallClassifier":
        """Fit one binary classifier per cut point k = 0, ..., K - 2."""
        levels = _check_sii_levels(y)
        self.classes_ = np.asarray(SII_LEVELS)
        self.estimators_ = []
        for cut in self.classes_[:-1]:
            above = (levels > cut).astype(int)
            if len(np.unique(above)) < 2:
                raise ValueError(f"Both sides of the cut SII > {cut} are needed.")
            weights = level_weights(above, self.weight_power)
            self.estimators_.append(
                clone(self.estimator).fit(X, above, sample_weight=weights)
            )
        return self

    def binary_proba(self, X: ArrayLike, k: int) -> np.ndarray:
        """Return P(SII > k) from the binary classifier k, without any correction.

        Unlike the corrected probabilities, this output is a single model, so it
        can be explained exactly (for example, with SHAP).
        """
        check_is_fitted(self, "estimators_")
        binary = self.estimators_[k]
        positive = list(binary.classes_).index(1)
        return binary.predict_proba(X)[:, positive]

    def cumulative_proba(self, X: ArrayLike) -> np.ndarray:
        """Return P(SII > k) for every k, made non-increasing in k."""
        raw = np.column_stack(
            [self.binary_proba(X, k) for k in range(len(self.estimators_))]
        )
        return np.minimum.accumulate(raw, axis=1)

    def proba_above(self, X: ArrayLike, k: int = 1) -> np.ndarray:
        """Return the corrected P(SII > k); k = 1 is a moderate or severe level."""
        return self.cumulative_proba(X)[:, k]

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Return the probability of each level by differences of P(SII > k)."""
        cumulative = self.cumulative_proba(X)
        n = cumulative.shape[0]
        upper = np.column_stack([np.ones(n), cumulative])
        lower = np.column_stack([cumulative, np.zeros(n)])
        return upper - lower

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Return the level with the highest probability."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class ThresholdRegressor(ClassifierMixin, BaseEstimator):
    """Regression of the SII followed by three thresholds that maximize the QWK.

    The thresholds are optimized on out-of-fold predictions of the data passed to
    fit, with folds stratified by level, so no evaluation data is involved. The
    regressor is then refitted on all the data passed to fit.
    """

    def __init__(
        self,
        estimator: BaseEstimator = None,
        weight_power: float = 1.0,
        n_splits: int = N_THRESHOLD_SPLITS,
        random_state: int = RANDOM_SEED,
    ):
        self.estimator = estimator
        self.weight_power = weight_power
        self.n_splits = n_splits
        self.random_state = random_state

    def _fit_regressor(self, X: ArrayLike, levels: np.ndarray) -> BaseEstimator:
        """Fit a fresh regressor with the level weights of the given rows."""
        weights = level_weights(levels, self.weight_power)
        return clone(self.estimator).fit(X, levels.astype(float), sample_weight=weights)

    def _out_of_fold_scores(self, X: ArrayLike, levels: np.ndarray) -> np.ndarray:
        """Return continuous predictions of each row by a regressor that did not see it."""
        scores = np.empty(len(levels))
        folds = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
        for train, validation in folds.split(np.zeros(len(levels)), levels):
            model = self._fit_regressor(_rows(X, train), levels[train])
            scores[validation] = model.predict(_rows(X, validation))
        return scores

    def fit(self, X: ArrayLike, y: ArrayLike) -> "ThresholdRegressor":
        """Optimize the thresholds on out-of-fold predictions, then refit on all rows."""
        levels = _check_sii_levels(y)
        self.classes_ = np.asarray(SII_LEVELS)
        self.thresholds_ = optimize_thresholds(
            self._out_of_fold_scores(X, levels), levels
        )
        self.estimator_ = self._fit_regressor(X, levels)
        return self

    def predict_continuous(self, X: ArrayLike) -> np.ndarray:
        """Return the continuous estimate of the SII before thresholding."""
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Return the level given by the thresholds."""
        return self.classes_[
            assign_levels(self.predict_continuous(X), self.thresholds_)
        ]