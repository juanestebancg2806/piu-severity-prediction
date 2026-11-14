from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from piu_severity.evaluation.metrics import quadratic_weighted_kappa, qwk_scorer
from piu_severity.models.ordinal import (
    FrankHallClassifier,
    MulticlassClassifier,
    ThresholdRegressor,
    _coordinate_search,
    assign_levels,
    level_weights,
    optimize_thresholds,
    quadratic_weighted_kappa_fast,
)

CLASS_SIZES = (120, 60, 30, 9)


def _data(seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.permutation(np.repeat(np.arange(len(CLASS_SIZES)), CLASS_SIZES))
    X = pd.DataFrame(
        {
            "signal": y + rng.normal(scale=0.6, size=len(y)),
            "noise": rng.normal(size=len(y)),
        }
    )
    return X, y


class WeightSpyClassifier(ClassifierMixin, BaseEstimator):
    """Record the labels and weights of each fit; predict the majority class."""

    fits: ClassVar[list] = []

    def fit(self, X, y, sample_weight=None):
        WeightSpyClassifier.fits.append((np.asarray(y), np.asarray(sample_weight)))
        self.classes_ = np.unique(y)
        self.majority_ = self.classes_[np.argmax(np.bincount(y))]
        return self

    def predict(self, X):
        return np.full(len(X), self.majority_)

    def predict_proba(self, X):
        proba = np.zeros((len(X), len(self.classes_)))
        proba[:, list(self.classes_).index(self.majority_)] = 1.0
        return proba


class WeightSpyRegressor(RegressorMixin, BaseEstimator):
    """Record the targets and weights of each fit; predict the first feature."""

    fits: ClassVar[list] = []

    def fit(self, X, y, sample_weight=None):
        WeightSpyRegressor.fits.append((np.asarray(y), np.asarray(sample_weight)))
        return self

    def predict(self, X):
        return np.asarray(X)[:, 0]


class FixedProbability(ClassifierMixin, BaseEstimator):
    """Return the same probability of the positive class for every row."""

    def __init__(self, probability: float = 0.5):
        self.probability = probability
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        return np.column_stack(
            [np.full(len(X), 1 - self.probability), np.full(len(X), self.probability)]
        )


@pytest.fixture(autouse=True)
def _reset_spies():
    WeightSpyClassifier.fits = []
    WeightSpyRegressor.fits = []


def test_level_weights_match_balanced_weights_and_power():
    _, y = _data()
    assert np.allclose(level_weights(y), compute_sample_weight("balanced", y))
    assert np.allclose(level_weights(y, 0.5), np.sqrt(level_weights(y)))
    assert np.allclose(level_weights(y, 0.0), 1.0)


def test_fast_qwk_matches_the_project_metric():
    rng = np.random.default_rng(1)
    for _ in range(20):
        y_true = rng.integers(0, 4, size=200)
        y_pred = rng.integers(0, 4, size=200)
        assert quadratic_weighted_kappa_fast(y_true, y_pred) == pytest.approx(
            quadratic_weighted_kappa(y_true, y_pred)
        )


def test_multiclass_passes_level_weights():
    X, y = _data()
    MulticlassClassifier(WeightSpyClassifier(), weight_power=0.5).fit(X, y)
    ((labels, weights),) = WeightSpyClassifier.fits
    assert np.array_equal(labels, y)
    assert np.allclose(weights, level_weights(y, 0.5))


def test_multiclass_exposes_probabilities_only_when_available():
    X, y = _data()
    model = MulticlassClassifier(LogisticRegression(max_iter=1_000)).fit(X, y)
    assert model.predict_proba(X).shape == (len(y), 4)
    assert not hasattr(MulticlassClassifier(LinearRegression()), "predict_proba")


def test_frank_hall_weights_each_binary_on_its_own_two_classes():
    X, y = _data()
    FrankHallClassifier(WeightSpyClassifier(), weight_power=1.0).fit(X, y)
    assert len(WeightSpyClassifier.fits) == 3
    for cut, (labels, weights) in enumerate(WeightSpyClassifier.fits):
        above = (y > cut).astype(int)
        assert np.array_equal(labels, above)
        assert np.allclose(weights, compute_sample_weight("balanced", above))


def test_frank_hall_corrects_non_monotone_cumulative_probabilities():
    X, y = _data()
    model = FrankHallClassifier(WeightSpyClassifier()).fit(X, y)
    model.estimators_ = [FixedProbability(p) for p in (0.6, 0.7, 0.2)]
    assert np.allclose(model.binary_proba(X, 1), 0.7)
    assert np.allclose(model.cumulative_proba(X)[0], [0.6, 0.6, 0.2])
    assert np.allclose(model.proba_above(X, k=1), 0.6)
    proba = model.predict_proba(X)
    assert np.allclose(proba[0], [0.4, 0.0, 0.4, 0.2])
    assert (proba >= 0).all()
    assert np.allclose(proba.sum(axis=1), 1.0)
    # Levels 0 and 2 tie; argmax keeps the first.
    assert (model.predict(X) == 0).all()


def test_frank_hall_learns_an_ordinal_signal():
    X, y = _data()
    model = FrankHallClassifier(LogisticRegression(max_iter=1_000)).fit(X, y)
    assert quadratic_weighted_kappa(y, model.predict(X)) > 0.6
    cumulative = model.cumulative_proba(X)
    assert (np.diff(cumulative, axis=1) <= 0).all()


def test_frank_hall_rejects_a_cut_without_both_classes():
    X, y = _data()
    with pytest.raises(ValueError):
        FrankHallClassifier(LogisticRegression()).fit(X, np.minimum(y, 2))


def test_thresholds_fall_in_the_gaps_between_separated_levels():
    levels = np.repeat(np.arange(4), [50, 30, 20, 5])
    rng = np.random.default_rng(2)
    scores = 20 * levels + rng.uniform(0, 1, size=len(levels))
    thresholds = optimize_thresholds(scores, levels)
    for k in range(3):
        assert (
            scores[levels == k].max() < thresholds[k] <= scores[levels == k + 1].min()
        )
    assert np.array_equal(assign_levels(scores, thresholds), levels)


def test_coordinate_search_escapes_a_distant_start():
    levels = np.repeat(np.arange(4), [50, 30, 20, 5])
    rng = np.random.default_rng(3)
    scores = 20 * levels + rng.uniform(0, 1, size=len(levels))
    candidates = np.unique(np.quantile(scores, np.linspace(0, 1, 200)))
    # All scores of levels 1 to 3 lie above the start, so every one of them is
    # assigned level 3 and the QWK is flat near the start.
    thresholds, score = _coordinate_search(
        scores, levels, np.array([0.5, 1.5, 2.5]), candidates
    )
    assert score == pytest.approx(1.0)
    assert np.all(np.diff(thresholds) > 0)


def test_threshold_regressor_uses_level_weights_and_stratified_folds():
    X, y = _data()
    model = ThresholdRegressor(WeightSpyRegressor(), weight_power=0.5, n_splits=3)
    model.fit(X, y)
    *internal, final = WeightSpyRegressor.fits
    assert len(internal) == 3
    for targets, weights in internal:
        assert set(np.unique(targets)) == {0, 1, 2, 3}
        assert np.allclose(weights, level_weights(targets.astype(int), 0.5))
    targets, weights = final
    assert np.array_equal(targets, y.astype(float))
    assert np.allclose(weights, level_weights(y, 0.5))


def test_threshold_regressor_internal_folds_are_stratified_by_level():
    X, y = _data()
    ThresholdRegressor(WeightSpyRegressor(), n_splits=3).fit(X, y)
    expected = StratifiedKFold(3, shuffle=True, random_state=42).split(X, y)
    for (targets, _), (train, _) in zip(
        WeightSpyRegressor.fits, expected, strict=False
    ):
        assert np.array_equal(targets, y[train].astype(float))


def test_threshold_regressor_predicts_ordered_levels():
    X, y = _data()
    model = ThresholdRegressor(LinearRegression()).fit(X, y)
    assert np.all(np.diff(model.thresholds_) > 0)
    assert set(np.unique(model.predict(X))) <= {0, 1, 2, 3}
    assert quadratic_weighted_kappa(y, model.predict(X)) > 0.6
    assert model.predict_continuous(X).dtype == float


@pytest.mark.parametrize(
    "factory",
    [
        lambda: MulticlassClassifier(LogisticRegression(max_iter=1_000)),
        lambda: FrankHallClassifier(LogisticRegression(max_iter=1_000)),
        lambda: ThresholdRegressor(LinearRegression()),
    ],
)
def test_formulations_work_inside_a_grid_search(factory):
    X, y = _data()
    grid = {"weight_power": [0.5, 1.0]}
    search = GridSearchCV(factory(), grid, scoring=qwk_scorer, cv=3).fit(X, y)
    assert search.best_params_["weight_power"] in (0.5, 1.0)
    assert clone(search.best_estimator_).get_params()["weight_power"] in (0.5, 1.0)


def test_nested_estimator_parameters_are_tunable():
    model = FrankHallClassifier(LogisticRegression())
    model.set_params(estimator__C=0.1)
    assert model.get_params()["estimator__C"] == 0.1


@pytest.mark.parametrize(
    "factory",
    [
        lambda: MulticlassClassifier(RandomForestClassifier(20, random_state=0)),
        lambda: FrankHallClassifier(RandomForestClassifier(20, random_state=0)),
        lambda: ThresholdRegressor(RandomForestRegressor(20, random_state=0)),
    ],
)
def test_formulations_are_deterministic(factory):
    X, y = _data()
    first = factory().fit(X, y).predict(X)
    second = factory().fit(X, y).predict(X)
    assert np.array_equal(first, second)


def test_levels_outside_the_sii_scale_are_rejected():
    X, y = _data()
    with pytest.raises(ValueError):
        MulticlassClassifier(LogisticRegression()).fit(X, y + 1)