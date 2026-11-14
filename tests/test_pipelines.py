import json

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV

from piu_severity.evaluation.metrics import qwk_scorer
from piu_severity.evaluation.validation import outer_splits, run_nested_cv
from piu_severity.features.preprocessing import MissingDataPreprocessor
from piu_severity.models.pipelines import (
    ALGORITHMS,
    BOOSTING_GRID,
    CLASSIFIER_GRIDS,
    FORMULATIONS,
    LOGISTIC_GRID,
    MAX_ITER_INCREASE,
    MEMORY_BUDGET_GB,
    MEMORY_SAFETY_FACTOR,
    available_workers,
    choose_cut_level,
    complexity_key,
    make_boosting_search,
    make_logistic_search,
    make_search,
    max_iter_factor_for,
    model_grid,
    n_candidates,
    n_forest_trees,
    project_search_hours,
    search_grid,
    uses_random_search,
    weight_powers,
    workers_within_memory,
)

CLASS_SIZES = (100, 50, 30, 20)


def _data(seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.permutation(np.repeat(np.arange(len(CLASS_SIZES)), CLASS_SIZES))
    n = len(y)
    X = pd.DataFrame(
        {
            "Basic_Demos-Enroll_Season": rng.choice(["Fall", "Spring"], size=n),
            "Basic_Demos-Age": 8 + 3 * y + rng.normal(scale=1.0, size=n),
            "Basic_Demos-Sex": rng.integers(0, 2, size=n).astype(float),
            "Physical-Height": rng.normal(55, 8, size=n),
            "FGC-FGC_CU_Zone": rng.integers(0, 2, size=n).astype(float),
            "BIA-BIA_Frame_num": rng.integers(1, 4, size=n).astype(float),
        }
    )
    X.loc[:29, "Physical-Height"] = np.nan
    X.loc[30:59, ["FGC-FGC_CU_Zone", "BIA-BIA_Frame_num"]] = np.nan
    return X, y


@pytest.mark.parametrize(
    ("factory", "grid"),
    [(make_logistic_search, LOGISTIC_GRID), (make_boosting_search, BOOSTING_GRID)],
)
def test_search_fits_and_predicts_levels(factory, grid):
    X, y = _data()
    search = factory(X.columns).fit(X, y)
    assert set(search.best_params_) == set(grid)
    assert set(np.unique(search.predict(X))) <= {0, 1, 2, 3}


def test_logistic_search_learns_an_informative_feature():
    X, y = _data()
    splits = outer_splits(y, n_repeats=1)
    folds, _ = run_nested_cv(make_logistic_search(X.columns), X, y, splits)
    assert folds["qwk"].mean() > 0.5
    assert folds["best_params"].notna().all()


def _full_search_data(seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    X, y = _data(seed)
    X["BIA-BIA_Fat"] = np.random.default_rng(seed).gamma(3.0, 5.0, size=len(y))
    X.loc[60:89, "BIA-BIA_Fat"] = np.nan
    return X, y


def _single_candidate(search, algorithm, formulation):
    """Keep the first value of every grid entry, so a smoke fit is fast."""
    grid = search_grid(algorithm, formulation)
    search.set_params(param_grid={name: values[:1] for name, values in grid.items()})
    if algorithm == "random_forest":
        search.estimator.set_params(model__estimator__n_estimators=20)
    return search


CONFIGURATIONS = [(a, f) for a in ALGORITHMS for f in FORMULATIONS]


def test_there_are_twelve_configurations():
    assert len(CONFIGURATIONS) == 12


@pytest.mark.parametrize(("algorithm", "formulation"), CONFIGURATIONS)
def test_every_grid_candidate_is_a_valid_parameter_setting(algorithm, formulation):
    X, _ = _full_search_data()
    pipeline = make_search(algorithm, formulation, X.columns).estimator
    for candidate in ParameterGrid(search_grid(algorithm, formulation)):
        clone(pipeline).set_params(**candidate)


@pytest.mark.parametrize(("algorithm", "formulation"), CONFIGURATIONS)
def test_grid_values_are_plain_json(algorithm, formulation):
    grid = search_grid(algorithm, formulation)
    assert json.loads(json.dumps(grid)) == grid
    for values in grid.values():
        assert all(isinstance(v, (int, float, str)) for v in values)


def test_native_missing_values_only_in_gradient_boosting():
    for algorithm, formulation in CONFIGURATIONS:
        strategies = search_grid(algorithm, formulation)["preprocess__strategy"]
        assert ("native" in strategies) == (algorithm == "gradient_boosting")


def test_weight_powers_are_tuned_in_every_configuration():
    for algorithm, formulation in CONFIGURATIONS:
        assert search_grid(algorithm, formulation)["model__weight_power"] == [0.5, 1.0]


@pytest.mark.parametrize(("algorithm", "formulation"), CONFIGURATIONS)
def test_inner_search_runs_sequentially_with_qwk(algorithm, formulation):
    search = make_search(algorithm, formulation, ["Basic_Demos-Age"])
    assert search.n_jobs == 1
    assert search.scoring is qwk_scorer
    assert search.cv.get_n_splits() == 3


@pytest.mark.parametrize(("algorithm", "formulation"), CONFIGURATIONS)
def test_every_configuration_fits_and_predicts_levels(algorithm, formulation):
    X, y = _full_search_data()
    search = _single_candidate(
        make_search(algorithm, formulation, X.columns), algorithm, formulation
    )
    predictions = search.fit(X, y).predict(X)
    assert set(np.unique(predictions)) <= {0, 1, 2, 3}


@pytest.mark.parametrize(("algorithm", "formulation"), CONFIGURATIONS)
def test_every_configuration_is_deterministic(algorithm, formulation):
    X, y = _full_search_data()
    first, second = (
        _single_candidate(
            make_search(algorithm, formulation, X.columns), algorithm, formulation
        )
        .fit(X, y)
        .predict(X)
        for _ in range(2)
    )
    assert np.array_equal(first, second)


def test_svm_frank_hall_uses_seeded_sigmoid_calibration():
    model = make_search("svm", "frank_hall", ["Basic_Demos-Age"]).estimator["model"]
    calibration = model.estimator
    assert calibration.method == "sigmoid"
    assert calibration.ensemble is False
    assert calibration.cv.random_state == 42


def test_seeds_and_limits_of_the_base_estimators():
    columns = ["Basic_Demos-Age"]
    logistic = make_search("linear", "multiclass", columns, max_iter_factor=5)
    assert logistic.estimator["model"].estimator.max_iter == 50_000
    assert logistic.estimator["model"].estimator.random_state == 42
    elastic = make_search("linear", "threshold_regression", columns)
    assert elastic.estimator["model"].estimator.max_iter == 10_000
    forest = make_search("random_forest", "frank_hall", columns).estimator["model"]
    assert forest.estimator.random_state == 42
    assert forest.estimator.n_estimators == 500
    boosting = make_search("gradient_boosting", "multiclass", columns)
    base = boosting.estimator["model"].estimator
    assert base.early_stopping is True
    assert base.random_state == 42
    assert boosting.estimator["preprocess"].seed == 42


def test_cut_levels_apply_the_documented_reductions():
    assert weight_powers("svm", 0) == [0.5, 1.0]
    assert weight_powers("svm", 1) == [1.0]
    assert weight_powers("linear", 5) == [0.5, 1.0]
    assert model_grid("svm", "multiclass", 1)["C"] == CLASSIFIER_GRIDS["svm"]["C"]
    reduced = model_grid("svm", "threshold_regression", 2)
    assert reduced["C"] == [0.1, 1.0, 10.0, 100.0]
    assert reduced["gamma"] == [1e-3, 1e-2, 1e-1, 1.0]
    assert reduced["epsilon"] == [0.1, 0.5]
    assert n_forest_trees(2) == 500
    assert n_forest_trees(3) == 300
    assert "l2_regularization" in model_grid("gradient_boosting", "multiclass", 3)
    assert "l2_regularization" not in model_grid("gradient_boosting", "multiclass", 4)
    assert not uses_random_search("gradient_boosting", 4)
    assert uses_random_search("gradient_boosting", 5)
    random_search = make_search(
        "gradient_boosting", "multiclass", ["Basic_Demos-Age"], cut_level=5
    )
    assert isinstance(random_search, RandomizedSearchCV)
    assert random_search.n_iter == 40
    assert random_search.random_state == 42


def test_candidate_counts_follow_the_grids():
    assert n_candidates("linear", "multiclass") == 2 * 2 * 7 * 3
    assert n_candidates("svm", "threshold_regression") == 2 * 2 * 6 * 5 * 2
    assert n_candidates("svm", "frank_hall", 2) == 2 * 1 * 4 * 4
    assert n_candidates("gradient_boosting", "frank_hall") == 3 * 2 * 2 * 3 * 3 * 3
    assert n_candidates("gradient_boosting", "frank_hall", 5) == 40


def test_unknown_configuration_or_cut_level_raises():
    with pytest.raises(ValueError):
        make_search("knn", "multiclass", ["Basic_Demos-Age"])
    with pytest.raises(ValueError):
        make_search("linear", "ordinal", ["Basic_Demos-Age"])
    with pytest.raises(ValueError):
        make_search("linear", "multiclass", ["Basic_Demos-Age"], cut_level=6)


def test_complexity_order_is_total_over_configurations():
    keys = [complexity_key(a, f, "simple") for a, f in CONFIGURATIONS]
    assert len(set(keys)) == len(keys)
    ordered = sorted(CONFIGURATIONS, key=lambda c: complexity_key(*c, "simple"))
    assert ordered[0] == ("linear", "threshold_regression")
    assert ordered[-1] == ("gradient_boosting", "frank_hall")
    assert complexity_key("svm", "multiclass", "native") == complexity_key(
        "svm", "multiclass", "simple"
    )
    assert complexity_key("svm", "multiclass", "iterative") > complexity_key(
        "svm", "multiclass", "simple"
    )


def test_pipeline_memory_reuses_the_preprocessing_across_candidates(
    tmp_path, monkeypatch
):
    X, y = _full_search_data()
    calls = []
    original_fit = MissingDataPreprocessor.fit

    def counting_fit(self, X, y=None):
        calls.append(self.strategy)
        return original_fit(self, X, y)

    monkeypatch.setattr(MissingDataPreprocessor, "fit", counting_fit)
    grid = {
        "preprocess__strategy": ["simple", "iterative"],
        "model__weight_power": [0.5, 1.0],
        "model__estimator__C": [0.1, 1.0],
    }
    predictions = {}
    for memory in (None, str(tmp_path / "pipeline_memory")):
        calls.clear()
        search = make_search("linear", "multiclass", X.columns, memory=memory)
        search.set_params(param_grid=grid).fit(X, y)
        predictions[memory is not None] = search.predict(X)
        if memory is None:
            assert len(calls) == 8 * 3 + 1
        else:
            # One fit per inner fold and strategy, plus the final refit.
            assert len(calls) == 2 * 3 + 1
    assert np.array_equal(predictions[True], predictions[False])


def test_available_workers_leave_cores_free():
    assert available_workers("linear", 15) == 12
    assert available_workers("linear", 8) == 6
    assert available_workers("random_forest", 15) == 4
    assert available_workers("svm", 2) == 1


def test_memory_rule_lowers_workers_only_when_needed():
    budget_mb = MEMORY_BUDGET_GB * 1024
    assert workers_within_memory(12, peak_mb_per_worker=500) == 12
    peak = budget_mb / (4 * MEMORY_SAFETY_FACTOR)
    assert workers_within_memory(12, peak_mb_per_worker=peak) == 4
    assert workers_within_memory(12, peak_mb_per_worker=10 * budget_mb) == 1


def test_convergence_rule():
    assert max_iter_factor_for(0.0) == 1
    assert max_iter_factor_for(0.05) == 1
    assert max_iter_factor_for(0.06) == MAX_ITER_INCREASE


def _fold_seconds(seconds: float) -> dict[tuple[str, str], float]:
    return dict.fromkeys(CONFIGURATIONS, seconds)


def test_projection_counts_waves_of_parallel_folds():
    workers = dict.fromkeys(ALGORITHMS, 25)
    assert project_search_hours(_fold_seconds(300.0), workers, 0) == pytest.approx(1.0)
    workers = dict.fromkeys(ALGORITHMS, 12)
    assert project_search_hours(_fold_seconds(300.0), workers, 0) == pytest.approx(3.0)


def test_projection_scales_with_the_cuts():
    workers = dict.fromkeys(ALGORITHMS, 25)
    seconds = {("svm", "multiclass"): 3_600.0}
    assert project_search_hours(seconds, workers, 1) == pytest.approx(0.5)
    seconds = {("random_forest", "multiclass"): 3_600.0}
    assert project_search_hours(seconds, workers, 3) == pytest.approx(0.6)


def test_choose_cut_level_returns_the_lowest_level_within_budget():
    workers = dict.fromkeys(ALGORITHMS, 25)
    assert choose_cut_level(_fold_seconds(60.0), workers, budget_hours=10) == 0
    level = choose_cut_level(_fold_seconds(3_000.0), workers, budget_hours=8)
    assert project_search_hours(_fold_seconds(3_000.0), workers, level) <= 8
    assert project_search_hours(_fold_seconds(3_000.0), workers, level - 1) > 8
    assert choose_cut_level(_fold_seconds(1e6), workers, budget_hours=1) == 5