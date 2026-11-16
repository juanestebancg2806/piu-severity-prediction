import os
import sys
import warnings
from pathlib import Path
from typing import ClassVar

import cloudpickle
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from piu_severity.evaluation import persistence
from piu_severity.evaluation.persistence import (
    CacheWarning,
    EvaluationTask,
    cached_size,
    clear_pipeline_memory,
    directory_size,
    evaluation_fields,
    fingerprint_frame,
    load_or_compute,
    run_configurations_cached,
    run_nested_cv_cached,
    signature_hash,
)
from piu_severity.evaluation.validation import outer_splits, run_nested_cv
from piu_severity.features.preprocessing import RoundedIterativeImputer
from piu_severity.models.pipelines import make_search

CLASS_SIZES = (40, 20, 12, 8)
CONFIG = {"algorithm": "spy", "formulation": "multiclass", "cut_level": 0}


def _data(seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.permutation(np.repeat(np.arange(len(CLASS_SIZES)), CLASS_SIZES))
    X = pd.DataFrame({"x": y + rng.normal(scale=0.5, size=len(y))})
    return X, y


class CountingClassifier(ClassifierMixin, BaseEstimator):
    """Round x + offset to the nearest level and count the fits."""

    fits: ClassVar[list] = []

    def __init__(self, offset: float = 0.0):
        self.offset = offset

    def fit(self, X, y):
        CountingClassifier.fits.append(len(y))
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.clip(np.round(X["x"] + self.offset), 0, 3).astype(int)


@pytest.fixture(autouse=True)
def _reset_counter():
    CountingClassifier.fits = []


def _run(cache_dir, estimator=None, X=None, y=None, splits=None, **options):
    """Run a cached evaluation with defaults that each test can override."""
    default_X, default_y = _data()
    X = default_X if X is None else X
    y = default_y if y is None else y
    estimator = CountingClassifier() if estimator is None else estimator
    splits = outer_splits(y, n_splits=5, n_repeats=1) if splits is None else splits
    field_options = {
        key: options.pop(key)
        for key in ("config", "evaluation", "n_repeats", "outer_seed", "extra")
        if key in options
    }
    fields = evaluation_fields(
        estimator,
        X,
        y,
        config=field_options.get("config", CONFIG),
        evaluation=field_options.get("evaluation", "search"),
        n_repeats=field_options.get("n_repeats", 1),
        outer_seed=field_options.get("outer_seed", 42),
        extra=field_options.get("extra"),
    )
    return run_nested_cv_cached(
        estimator,
        X,
        y,
        splits,
        name="spy",
        cache_dir=cache_dir,
        fields=fields,
        **options,
    )


def _comparable(folds: pd.DataFrame) -> pd.DataFrame:
    """Drop the columns that legitimately differ between runs."""
    return folds.drop(columns=["fit_seconds", "peak_memory_mb", "status"])


def test_second_call_loads_without_fitting(tmp_path):
    first = _run(tmp_path)
    assert len(CountingClassifier.fits) == 5
    assert first.counts() == {"loaded": 0, "computed": 5}
    second = _run(tmp_path)
    assert len(CountingClassifier.fits) == 5
    assert second.counts() == {"loaded": 5, "computed": 0}


def test_loaded_results_equal_computed_results(tmp_path):
    first = _run(tmp_path)
    second = _run(tmp_path)
    pd.testing.assert_frame_equal(_comparable(first.folds), _comparable(second.folds))
    pd.testing.assert_frame_equal(first.predictions, second.predictions)
    assert np.allclose(first.folds["fit_seconds"], second.folds["fit_seconds"])


def test_cached_results_match_run_nested_cv(tmp_path):
    X, y = _data()
    splits = outer_splits(y, n_splits=5, n_repeats=1)
    folds, predictions = run_nested_cv(CountingClassifier(), X, y, splits)
    cached = _run(tmp_path)
    pd.testing.assert_frame_equal(
        cached.folds[folds.columns].drop(columns="fit_seconds"),
        folds.drop(columns="fit_seconds"),
    )
    pd.testing.assert_frame_equal(cached.predictions, predictions)


def _change_x_value(X):
    changed = X.copy()
    changed.iloc[0, 0] += 1.0
    return changed


SIGNATURE_CHANGES = {
    "estimator": {"estimator": CountingClassifier(offset=0.1)},
    "columns": {"X": lambda X: X.assign(extra=0.0)},
    "X_hash": {"X": _change_x_value},
    "y_hash": {"y": lambda y: np.where(np.arange(len(y)) == 0, (y + 1) % 4, y)},
    "test_hash": {"splits": lambda y: outer_splits(y, n_splits=5, n_repeats=1, seed=7)},
    "config": {"config": {**CONFIG, "cut_level": 1}},
    "evaluation": {"evaluation": "retrain"},
    "n_repeats": {"n_repeats": 5},
    "outer_seed": {"outer_seed": 7},
    "extra": {"extra": {"subset": "actigraphy"}},
}


@pytest.mark.parametrize("field", sorted(SIGNATURE_CHANGES))
def test_any_change_in_the_signature_refits_and_names_the_field(tmp_path, field):
    X, y = _data()
    _run(tmp_path)
    options = dict(SIGNATURE_CHANGES[field])
    if "X" in options:
        options["X"] = options["X"](X)
    if "y" in options:
        options["y"] = options["y"](y)
    if "splits" in options:
        options["splits"] = options["splits"](y)
    CountingClassifier.fits = []
    with pytest.warns(CacheWarning, match=field):
        result = _run(tmp_path, **options)
    assert len(CountingClassifier.fits) == 5
    assert result.counts()["computed"] == 5


@pytest.mark.parametrize(
    ("attribute", "value", "field"),
    [
        ("RANDOM_SEED", 7, "random_seed"),
        ("SKLEARN_VERSION", "0.0.0", "sklearn_version"),
        ("CACHE_VERSION", 999, "cache_version"),
    ],
)
def test_seed_and_versions_are_part_of_the_signature(
    tmp_path, monkeypatch, attribute, value, field
):
    _run(tmp_path)
    monkeypatch.setattr(persistence, attribute, value)
    CountingClassifier.fits = []
    with pytest.warns(CacheWarning, match=field):
        _run(tmp_path)
    assert len(CountingClassifier.fits) == 5


def test_force_refits_every_fold(tmp_path):
    _run(tmp_path)
    CountingClassifier.fits = []
    result = _run(tmp_path, force=True)
    assert len(CountingClassifier.fits) == 5
    assert result.counts()["computed"] == 5


def test_only_missing_folds_are_fitted(tmp_path):
    _, y = _data()
    splits = outer_splits(y, n_splits=5, n_repeats=1)
    _run(tmp_path, splits=splits[:3])
    CountingClassifier.fits = []
    result = _run(tmp_path, splits=splits)
    assert len(CountingClassifier.fits) == 2
    assert result.counts() == {"loaded": 3, "computed": 2}


def test_orphan_temporary_file_is_ignored(tmp_path):
    (tmp_path / "spy_r0_f0.joblib.tmp").write_bytes(b"interrupted dump")
    _run(tmp_path)
    CountingClassifier.fits = []
    with warnings.catch_warnings():
        warnings.simplefilter("error", CacheWarning)
        result = _run(tmp_path)
    assert result.counts()["loaded"] == 5


def test_corrupt_file_is_refitted_without_raising(tmp_path):
    _run(tmp_path)
    (tmp_path / "spy_r0_f2.joblib").write_bytes(b"not a joblib file")
    CountingClassifier.fits = []
    with pytest.warns(CacheWarning, match="unreadable"):
        result = _run(tmp_path)
    assert len(CountingClassifier.fits) == 1
    assert result.counts() == {"loaded": 4, "computed": 1}


def test_failed_save_keeps_previous_file_and_returns_results(tmp_path, monkeypatch):
    _run(tmp_path)
    path = tmp_path / "spy_r0_f0.joblib"
    previous = path.read_bytes()

    def failing_dump(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(persistence.joblib, "dump", failing_dump)
    with pytest.warns(CacheWarning, match="could not save"):
        result = _run(tmp_path, force=True)
    assert path.read_bytes() == previous
    assert len(result.folds) == 5


def test_estimators_are_kept_only_on_request(tmp_path):
    without = _run(tmp_path)
    assert all(model is None for model in without.estimators.values())
    CountingClassifier.fits = []
    with pytest.warns(CacheWarning, match="lacks"):
        kept = _run(tmp_path, keep_estimators=True)
    assert len(CountingClassifier.fits) == 5
    assert all(isinstance(m, CountingClassifier) for m in kept.estimators.values())
    CountingClassifier.fits = []
    reloaded = _run(tmp_path, keep_estimators=True)
    assert len(CountingClassifier.fits) == 0
    assert all(m is not None for m in reloaded.estimators.values())


def test_parallel_and_sequential_runs_agree(tmp_path):
    estimator = LogisticRegression(max_iter=1_000)
    sequential = _run(tmp_path / "sequential", estimator=estimator, n_jobs=1)
    parallel = _run(tmp_path / "parallel", estimator=estimator, n_jobs=2)
    pd.testing.assert_frame_equal(
        _comparable(sequential.folds), _comparable(parallel.folds)
    )
    pd.testing.assert_frame_equal(sequential.predictions, parallel.predictions)


def _convergence_pipeline() -> Pipeline:
    """Return a pipeline whose imputer and model stop before converging."""
    return Pipeline(
        [
            ("impute", RoundedIterativeImputer(max_iter=1)),
            ("model", LogisticRegression(solver="saga", max_iter=1)),
        ]
    )


def test_convergence_warnings_are_counted_in_parallel_workers(tmp_path):
    X, y = _data()
    rng = np.random.default_rng(1)
    X = X.assign(z=X["x"] + rng.normal(size=len(y)), w=rng.normal(size=len(y)))
    X.loc[::4, "z"] = np.nan
    counts = {}
    for n_jobs in (1, 2):
        result = _run(
            tmp_path / str(n_jobs),
            estimator=_convergence_pipeline(),
            X=X,
            n_jobs=n_jobs,
        )
        counts[n_jobs] = result.folds[
            ["convergence_warnings_model", "convergence_warnings_imputer"]
        ]
    assert (counts[1]["convergence_warnings_model"] > 0).all()
    assert (counts[1]["convergence_warnings_imputer"] > 0).all()
    pd.testing.assert_frame_equal(counts[1], counts[2])


def test_peak_memory_and_fit_counts_are_recorded(tmp_path):
    result = _run(tmp_path)
    assert (result.folds["peak_memory_mb"] > 0).all()
    assert (result.folds["n_pipeline_fits"] == 1).all()


def _memory_entries(folder: Path) -> int:
    """Return the number of preprocessing fits stored by Pipeline(memory)."""
    return len(list(folder.rglob("output.pkl")))


def test_pipeline_memory_is_shared_safely_by_parallel_workers(tmp_path):
    rng = np.random.default_rng(0)
    y = rng.permutation(np.repeat(np.arange(4), (60, 30, 20, 10)))
    X = pd.DataFrame(
        {
            "Basic_Demos-Age": y + rng.normal(scale=0.8, size=len(y)),
            "BIA-BIA_Fat": rng.normal(size=len(y)),
        }
    )
    X.loc[:40, "BIA-BIA_Fat"] = np.nan
    splits = outer_splits(y, n_splits=5, n_repeats=1)[:2]
    grid = {
        "preprocess__strategy": ["simple", "iterative"],
        "model__weight_power": [0.5, 1.0],
        "model__estimator__C": [0.1, 1.0],
    }
    memory = tmp_path / "pipeline_memory"

    def search(memory_folder):
        return make_search(
            "linear", "multiclass", X.columns, memory=memory_folder
        ).set_params(param_grid=grid)

    plain = _run(tmp_path / "plain", estimator=search(None), X=X, y=y, splits=splits)
    shared = _run(
        tmp_path / "shared",
        estimator=search(str(memory)),
        X=X,
        y=y,
        splits=splits,
        n_jobs=2,
    )
    pd.testing.assert_frame_equal(plain.predictions, shared.predictions)
    # Per outer fold: one entry per inner fold and strategy, plus the refit.
    assert _memory_entries(memory) == 2 * (3 * 2 + 1)
    again = _run(
        tmp_path / "shared",
        estimator=search(str(memory)),
        X=X,
        y=y,
        splits=splits,
        n_jobs=2,
        force=True,
    )
    assert _memory_entries(memory) == 2 * (3 * 2 + 1)
    pd.testing.assert_frame_equal(plain.predictions, again.predictions)


def test_load_or_compute_reuses_matching_results(tmp_path):
    calls = []

    def compute():
        calls.append(1)
        return {"value": len(calls)}

    path = tmp_path / "evaluation.joblib"
    fields = {"evaluation": "permutation", "n": 10}
    assert load_or_compute(path, fields, compute) == ({"value": 1}, "computed")
    assert load_or_compute(path, fields, compute) == ({"value": 1}, "loaded")
    with pytest.warns(CacheWarning, match="n"):
        result, status = load_or_compute(path, {**fields, "n": 20}, compute)
    assert (result, status) == ({"value": 2}, "computed")
    assert load_or_compute(path, {**fields, "n": 20}, compute, force=True)[0] == {
        "value": 3
    }


def test_signature_hash_ignores_key_order_and_run_only_parameters():
    assert signature_hash({"a": 1, "b": [1, 2]}) == signature_hash(
        {"b": [1, 2], "a": 1}
    )
    first = evaluation_fields(
        make_search("linear", "multiclass", ["x"], memory="a", n_jobs=1),
        *_data(),
        config=CONFIG,
        evaluation="search",
        n_repeats=1,
    )
    second = evaluation_fields(
        make_search("linear", "multiclass", ["x"], memory="b", n_jobs=4),
        *_data(),
        config=CONFIG,
        evaluation="search",
        n_repeats=1,
    )
    assert signature_hash(first) == signature_hash(second)


def test_frame_fingerprint_detects_values_names_and_types():
    X, _ = _data()
    assert fingerprint_frame(X) == fingerprint_frame(X.copy())
    assert fingerprint_frame(X) != fingerprint_frame(_change_x_value(X))
    assert fingerprint_frame(X) != fingerprint_frame(X.rename(columns={"x": "y"}))
    assert fingerprint_frame(X) != fingerprint_frame(X.astype("float32"))


def test_directory_size_counts_nested_files(tmp_path):
    assert directory_size(tmp_path / "missing") == 0
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "file.bin").write_bytes(b"12345")
    (tmp_path / "other.bin").write_bytes(b"123")
    assert directory_size(tmp_path) == 8


def test_clear_pipeline_memory_never_touches_the_results_cache(tmp_path):
    results = tmp_path / "03_busqueda_modelos" / "cache"
    memory = tmp_path / "03_busqueda_modelos" / "pipeline_memory" / "v1"
    for folder in (results, memory):
        folder.mkdir(parents=True)
        (folder / "file.bin").write_bytes(b"data")
    clear_pipeline_memory(memory)
    assert not memory.exists()
    assert (results / "file.bin").exists()
    with pytest.raises(ValueError):
        clear_pipeline_memory(results)
    clear_pipeline_memory(memory)


# Classes defined in this module are sent to the workers by value, so the
# parallel tests can use them although the tests folder is not a package.
cloudpickle.register_pickle_by_value(sys.modules[__name__])


class FailOnFoldClassifier(CountingClassifier):
    """Raise when row 0 is not in the training data: exactly one fold per repeat."""

    attempts: ClassVar[list] = []

    def fit(self, X, y):
        FailOnFoldClassifier.attempts.append(len(y))
        if 0 not in X.index:
            raise ValueError("deliberate failure")
        return super().fit(X, y)


class CrashOnceClassifier(CountingClassifier):
    """Kill its worker the first time it fits, then behave normally."""

    def __init__(self, offset: float = 0.0, marker: str = ""):
        super().__init__(offset)
        self.marker = marker

    def fit(self, X, y):
        marker = Path(self.marker)
        if not marker.exists():
            marker.write_text("crashed")
            os._exit(1)
        return super().fit(X, y)


class AlwaysCrashClassifier(CountingClassifier):
    """Kill its worker on every fit."""

    def fit(self, X, y):
        os._exit(1)


def _task(name, estimator, X=None, y=None, splits=None, keep_estimators=False):
    """Return an evaluation task with the default data and five splits."""
    default_X, default_y = _data()
    X = default_X if X is None else X
    y = default_y if y is None else y
    splits = outer_splits(y, n_splits=5, n_repeats=1) if splits is None else splits
    fields = evaluation_fields(
        estimator, X, y, config={"name": name}, evaluation="search", n_repeats=1
    )
    return EvaluationTask(name, estimator, X, y, splits, fields, keep_estimators)


def test_pooled_tasks_match_separate_runs(tmp_path):
    tasks = [
        _task("logistic", LogisticRegression(max_iter=1_000)),
        _task("shifted", CountingClassifier(offset=0.2)),
    ]
    pooled = run_configurations_cached(tasks, cache_dir=tmp_path / "pooled", n_jobs=2)
    assert set(pooled) == {"logistic", "shifted"}
    for task in tasks:
        separate = run_nested_cv_cached(
            task.estimator,
            task.X,
            task.y,
            task.splits,
            name=task.name,
            cache_dir=tmp_path / "separate",
            fields=task.fields,
        )
        pd.testing.assert_frame_equal(
            _comparable(pooled[task.name].folds), _comparable(separate.folds)
        )
        pd.testing.assert_frame_equal(
            pooled[task.name].predictions, separate.predictions
        )


def test_pooled_tasks_load_their_folds_on_a_second_call(tmp_path):
    tasks = [
        _task("first", CountingClassifier()),
        _task("second", CountingClassifier(offset=0.1)),
    ]
    run_configurations_cached(tasks, cache_dir=tmp_path)
    CountingClassifier.fits = []
    again = run_configurations_cached(tasks, cache_dir=tmp_path)
    assert len(CountingClassifier.fits) == 0
    assert all(result.counts()["loaded"] == 5 for result in again.values())


def test_task_names_must_be_unique(tmp_path):
    tasks = [_task("same", CountingClassifier()), _task("same", CountingClassifier())]
    with pytest.raises(ValueError, match="unique"):
        run_configurations_cached(tasks, cache_dir=tmp_path)


def test_a_failing_fold_does_not_stop_the_others(tmp_path):
    task = _task("failing", FailOnFoldClassifier())
    with pytest.raises(RuntimeError, match="1 folds failed"):
        run_configurations_cached([task], cache_dir=tmp_path)
    assert len(list(tmp_path.glob("failing_*.joblib"))) == 4
    FailOnFoldClassifier.attempts = []
    with pytest.raises(RuntimeError, match="deliberate failure"):
        run_configurations_cached([task], cache_dir=tmp_path)
    # Only the failing fold is attempted again; the other four are loaded.
    assert len(FailOnFoldClassifier.attempts) == 1


def test_a_dead_worker_is_recovered_with_fewer_workers(tmp_path):
    marker = tmp_path / "crashed.txt"
    task = _task("crash", CrashOnceClassifier(marker=str(marker)))
    with pytest.warns(CacheWarning, match="worker died"):
        results = run_configurations_cached(
            [task], cache_dir=tmp_path / "cache", n_jobs=2
        )
    assert marker.exists()
    assert len(results["crash"].folds) == 5
    assert len(list((tmp_path / "cache").glob("crash_*.joblib"))) == 5


def test_workers_that_keep_dying_raise_and_keep_saved_folds(tmp_path, monkeypatch):
    monkeypatch.setattr(persistence, "MAX_WORKER_RETRIES", 1)
    task = _task("doomed", AlwaysCrashClassifier())
    with (
        pytest.warns(CacheWarning, match="worker died"),
        pytest.raises(RuntimeError, match="kept dying"),
    ):
        run_configurations_cached([task], cache_dir=tmp_path, n_jobs=4)


def test_cached_size_counts_only_the_files_of_one_task(tmp_path):
    first, second = (
        _task("first", CountingClassifier()),
        _task("second", CountingClassifier()),
    )
    assert cached_size(first, tmp_path) == 0
    run_configurations_cached([first], cache_dir=tmp_path)
    size = cached_size(first, tmp_path)
    assert size == directory_size(tmp_path) > 0
    run_configurations_cached([second], cache_dir=tmp_path)
    assert cached_size(first, tmp_path) == size
    assert directory_size(tmp_path) > size
