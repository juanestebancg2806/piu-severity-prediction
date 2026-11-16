import numpy as np
import pandas as pd
import pytest

from piu_severity.evaluation.persistence import (
    EvaluationTask,
    evaluation_fields,
    run_configurations_cached,
)
from piu_severity.evaluation.validation import (
    candidate_key,
    evaluate_fold,
    evaluate_fold_extension,
    outer_splits,
)
from piu_severity.models.pipelines import make_search

CLASS_SIZES = (100, 50, 30, 20)
C_NAME = "model__estimator__C"
BASE_GRID = {
    "preprocess__strategy": ["simple"],
    "model__weight_power": [1.0],
    C_NAME: [1e-3, 1e-2],
    "model__estimator__l1_ratio": [0.0],
}


def _data(seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.permutation(np.repeat(np.arange(len(CLASS_SIZES)), CLASS_SIZES))
    n = len(y)
    X = pd.DataFrame(
        {
            "Basic_Demos-Age": 8 + 3 * y + rng.normal(scale=2.0, size=n),
            "Basic_Demos-Sex": rng.integers(0, 2, size=n).astype(float),
            "Physical-Height": rng.normal(55, 8, size=n),
        }
    )
    X.loc[:29, "Physical-Height"] = np.nan
    return X, y


def _search(extra_c: list[float] | None = None):
    grid = dict(BASE_GRID)
    if extra_c:
        grid[C_NAME] = sorted([*grid[C_NAME], *extra_c])
    X, _ = _data()
    search = make_search("linear", "multiclass", X.columns)
    return search.set_params(param_grid=grid)


def _scores(result: dict) -> list[float]:
    return list(result["cv_results"]["mean_test_score"])


@pytest.mark.parametrize(
    "extra_c",
    [
        pytest.param([10.0], id="new candidate wins"),
        pytest.param([1e-4], id="previous candidate kept"),
        pytest.param([1e-4, 10.0], id="both edges"),
    ],
)
def test_extension_matches_the_full_extended_search(extra_c):
    X, y = _data()
    splits = outer_splits(y, n_splits=5, n_repeats=1)
    base, extended = _search(), _search(extra_c)
    for split in splits:
        previous = evaluate_fold(base, X, y, split)
        incremental = evaluate_fold_extension(extended, previous, X, y, split)
        full = evaluate_fold(extended, X, y, split)
        assert candidate_key(incremental["row"]["best_params"]) == candidate_key(
            full["row"]["best_params"]
        )
        assert incremental["row"]["qwk"] == full["row"]["qwk"]
        pd.testing.assert_frame_equal(incremental["predictions"], full["predictions"])
        assert _scores(incremental) == _scores(full)
        assert list(incremental["cv_results"]["rank_test_score"]) == list(
            full["cv_results"]["rank_test_score"]
        )


def test_extension_counts_only_the_new_candidates():
    X, y = _data()
    split = outer_splits(y, n_splits=5, n_repeats=1)[0]
    previous = evaluate_fold(_search(), X, y, split)
    kept = evaluate_fold_extension(_search([1e-4]), previous, X, y, split)
    assert kept["row"]["new_candidates"] == 1
    assert not kept["row"]["best_changed"]
    # One new candidate in three inner folds, and no refit.
    assert kept["row"]["n_pipeline_fits"] == 3
    won = evaluate_fold_extension(_search([10.0]), previous, X, y, split)
    assert won["row"]["best_changed"]
    assert won["row"]["n_pipeline_fits"] == 3 + 1


def test_extension_keeps_the_estimator_on_request():
    X, y = _data()
    split = outer_splits(y, n_splits=5, n_repeats=1)[0]
    previous = evaluate_fold(_search(), X, y, split)
    result = evaluate_fold_extension(
        _search([1e-4]), previous, X, y, split, keep_estimator=True
    )
    assert result["estimator"] is not None
    assert result["estimator"].memory is None
    np.testing.assert_array_equal(
        result["estimator"].predict(X.iloc[split.test]),
        result["predictions"]["y_pred"],
    )
    pd.testing.assert_frame_equal(result["predictions"], previous["predictions"])


def test_extension_rejects_another_split_or_an_unrelated_grid():
    X, y = _data()
    first, second = outer_splits(y, n_splits=5, n_repeats=1)[:2]
    previous = evaluate_fold(_search(), X, y, first)
    with pytest.raises(ValueError, match="another split"):
        evaluate_fold_extension(_search([10.0]), previous, X, y, second)
    other = _search().set_params(param_grid={**BASE_GRID, C_NAME: [0.5, 10.0]})
    with pytest.raises(ValueError, match="not contained"):
        evaluate_fold_extension(other, previous, X, y, first)


def _task(name, search, X, y, splits, previous=None):
    fields = evaluation_fields(
        search, X, y, config={"name": "extension"}, evaluation="search", n_repeats=1
    )
    return EvaluationTask(name, search, X, y, splits, fields, previous=previous)


def test_cached_tasks_with_previous_folds_match_the_full_search(tmp_path):
    X, y = _data()
    splits = outer_splits(y, n_splits=5, n_repeats=1)
    base = run_configurations_cached(
        [_task("base", _search(), X, y, splits)], cache_dir=tmp_path / "base"
    )["base"]
    extended = _search([1e-4, 10.0])
    incremental = run_configurations_cached(
        [_task("extended", extended, X, y, splits, base.fold_results())],
        cache_dir=tmp_path / "incremental",
        n_jobs=2,
    )["extended"]
    full = run_configurations_cached(
        [_task("extended", extended, X, y, splits)], cache_dir=tmp_path / "full"
    )["extended"]
    pd.testing.assert_frame_equal(incremental.predictions, full.predictions)
    pd.testing.assert_series_equal(incremental.folds["qwk"], full.folds["qwk"])
    # Same signature: a later call without the previous folds loads them.
    reloaded = run_configurations_cached(
        [_task("extended", extended, X, y, splits)], cache_dir=tmp_path / "incremental"
    )["extended"]
    assert reloaded.counts() == {"loaded": 5, "computed": 0}


def test_fold_results_round_trip(tmp_path):
    X, y = _data()
    splits = outer_splits(y, n_splits=5, n_repeats=1)
    cached = run_configurations_cached(
        [_task("base", _search(), X, y, splits)], cache_dir=tmp_path
    )["base"]
    folds = cached.fold_results()
    assert set(folds) == {(0, fold) for fold in range(5)}
    for split in splits:
        result = folds[(split.repeat, split.fold)]
        np.testing.assert_array_equal(result["predictions"]["position"], split.test)
        assert result["cv_results"] is not None


def test_task_without_previous_folds_for_a_split_raises():
    X, y = _data()
    splits = outer_splits(y, n_splits=5, n_repeats=1)
    task = _task("extended", _search([10.0]), X, y, splits, previous={})
    with pytest.raises(KeyError, match="no previous result"):
        task.previous_for(splits[0])
