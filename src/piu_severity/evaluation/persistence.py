"""Disk cache of expensive evaluations, keyed by a signature of everything they depend on.

Each outer fold of a nested cross-validation is stored in its own file, so an
interrupted run loses only the folds in progress. A file is loaded only when its
signature matches the current one: the configuration, the estimator and its
grid, the predictors and their values, the target, the split, the seeds and the
versions of scikit-learn and of this cache. Files hold row-level predictions, so
they live in an ignored folder under data/interim.
"""

import hashlib
import json
import shutil
import subprocess
import time
import traceback
import warnings
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator

from piu_severity.config import PROJECT_ROOT, RANDOM_SEED
from piu_severity.evaluation.metrics import as_levels
from piu_severity.evaluation.validation import (
    FOLD_COLUMNS,
    Split,
    evaluate_fold,
    evaluate_fold_extension,
    hash_indices,
)

# Increase by hand whenever the code of the estimators or the preprocessing
# changes in a way that can change results; every cached file is then refitted.
CACHE_VERSION = 1
SKLEARN_VERSION = sklearn.__version__
PIPELINE_MEMORY_FOLDER = "pipeline_memory"
IGNORED_PARAMETERS = frozenset({"memory", "n_jobs", "verbose", "pre_dispatch"})
STORED_KEYS = frozenset({"fields", "fields_hash", "result", "metadata"})
# After a worker dies (for example, killed for lack of memory), the folds that
# were not saved are retried with half the workers, at most this many times.
MAX_WORKER_RETRIES = 2


class CacheWarning(UserWarning):
    """A cached file was not used, or a result could not be saved."""


def fingerprint_frame(X: pd.DataFrame) -> str:
    """Return a hash of the values, index, column names and types of a DataFrame."""
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(X, index=True).to_numpy().tobytes())
    digest.update(json.dumps([str(c) for c in X.columns]).encode())
    digest.update(json.dumps([str(t) for t in X.dtypes]).encode())
    return digest.hexdigest()


def fingerprint_levels(y: ArrayLike) -> str:
    """Return a hash of the target levels."""
    return hashlib.sha256(as_levels(y).astype(np.int64).tobytes()).hexdigest()


def canonical(value: object) -> object:
    """Return a JSON-compatible version of value with a stable representation.

    Estimators become their class and parameters (without those that do not
    change the result, such as memory or n_jobs), recursively.
    """
    if isinstance(value, BaseEstimator):
        parameters = {
            name: parameter
            for name, parameter in value.get_params(deep=False).items()
            if name not in IGNORED_PARAMETERS
        }
        return {"class": type(value).__name__, "params": canonical(parameters)}
    if isinstance(value, Mapping):
        return {str(key): canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]
    if isinstance(value, np.ndarray):
        return canonical(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def signature_hash(fields: Mapping) -> str:
    """Return the SHA-256 hash of the canonical JSON of the signature fields."""
    text = json.dumps(canonical(fields), sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def evaluation_fields(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: ArrayLike,
    *,
    config: Mapping,
    evaluation: str,
    n_repeats: int,
    outer_seed: int = RANDOM_SEED,
    extra: Mapping | None = None,
) -> dict:
    """Return the signature fields shared by every fold of one evaluation.

    config names the configuration (for example, algorithm, formulation, cut
    level and feature set); the estimator fingerprint adds its full grid, seeds
    and limits.
    """
    cv = getattr(estimator, "cv", None)
    return {
        "evaluation": evaluation,
        "config": dict(config),
        "columns": [str(c) for c in X.columns],
        "X_hash": fingerprint_frame(X),
        "y_hash": fingerprint_levels(y),
        "estimator": canonical(estimator),
        "n_repeats": n_repeats,
        "n_inner_splits": cv.get_n_splits() if hasattr(cv, "get_n_splits") else None,
        "outer_seed": outer_seed,
        "random_seed": RANDOM_SEED,
        "sklearn_version": SKLEARN_VERSION,
        "cache_version": CACHE_VERSION,
        "extra": dict(extra or {}),
    }


def fold_fields(fields: Mapping, split: Split) -> dict:
    """Return the signature fields of one outer fold."""
    return {
        **fields,
        "repeat": split.repeat,
        "fold": split.fold,
        "test_hash": hash_indices(split.test),
        "n_train": len(split.train),
        "n_test": len(split.test),
    }


def _git_commit() -> str | None:
    """Return the current commit of the repository, if it can be read."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _changed_fields(stored: Mapping, current: Mapping) -> list[str]:
    """Return the names of the signature fields whose values differ."""
    names = sorted(set(stored) | set(current))
    return [name for name in names if stored.get(name) != current.get(name)]


def read_cached(
    path: Path,
    fields: Mapping,
    require: Callable[[object], bool] | None = None,
) -> tuple[object | None, str]:
    """Return the cached result and its status: loaded, missing, corrupt or invalid.

    A file whose signature differs, or whose result fails require, is never
    used; a warning names the fields that changed.
    """
    if not path.is_file():
        return None, "missing"
    try:
        stored = joblib.load(path)
        if not isinstance(stored, dict) or not STORED_KEYS <= set(stored):
            raise ValueError("unexpected content")
    except Exception as error:  # noqa: BLE001 - any unreadable file is recomputed
        warnings.warn(
            f"{path.name}: unreadable cache file ({error}); recomputing.",
            CacheWarning,
            stacklevel=2,
        )
        return None, "corrupt"
    current = canonical(fields)
    if stored["fields_hash"] != signature_hash(fields):
        changed = _changed_fields(stored["fields"], current)
        warnings.warn(
            f"{path.name}: signature changed in {', '.join(changed)}; recomputing.",
            CacheWarning,
            stacklevel=2,
        )
        return None, "invalid"
    if require is not None and not require(stored["result"]):
        warnings.warn(
            f"{path.name}: the cached result lacks what was requested; recomputing.",
            CacheWarning,
            stacklevel=2,
        )
        return None, "invalid"
    return stored["result"], "loaded"


def write_cached(
    path: Path, fields: Mapping, result: object, minutes: float
) -> str | None:
    """Save a result atomically; return an error message instead of raising.

    The file is written to a temporary name and renamed only when complete, so
    an interrupted or failed write leaves any previous file intact.
    """
    payload = {
        "fields": canonical(fields),
        "fields_hash": signature_hash(fields),
        "result": result,
        "metadata": {
            "saved_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "minutes": minutes,
            "commit": _git_commit(),
            "sklearn_version": SKLEARN_VERSION,
        },
    }
    temporary = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, temporary, compress=3)
        temporary.replace(path)
    except Exception as error:  # noqa: BLE001 - keep the result in memory instead
        return f"{path.name}: could not save the result ({error})."
    return None


def load_or_compute(
    path: Path,
    fields: Mapping,
    compute: Callable[[], object],
    force: bool = False,
) -> tuple[object, str]:
    """Return a cached result when its signature matches, or compute and save it.

    For expensive evaluations that are not nested cross-validations. The status
    is loaded, or computed after the file was missing, corrupt or invalid.
    """
    if not force:
        result, status = read_cached(path, fields)
        if status == "loaded":
            return result, status
    start = time.perf_counter()
    result = compute()
    error = write_cached(path, fields, result, (time.perf_counter() - start) / 60)
    if error:
        warnings.warn(error, CacheWarning, stacklevel=2)
    return result, "computed"


@dataclass(frozen=True)
class CachedEvaluation:
    """Results of a cached nested cross-validation.

    folds and predictions match the output of run_nested_cv, with extra fold
    columns: status, pipeline fits, convergence warnings and peak memory.
    cv_results and estimators are keyed by (repeat, fold).
    """

    folds: pd.DataFrame
    predictions: pd.DataFrame
    cv_results: dict[tuple[int, int], pd.DataFrame | None]
    estimators: dict[tuple[int, int], BaseEstimator | None]

    def counts(self) -> dict[str, int]:
        """Return how many folds were loaded and how many were computed."""
        status = self.folds["status"]
        return {
            "loaded": int((status == "loaded").sum()),
            "computed": int((status == "computed").sum()),
        }

    def fold_results(self) -> dict[tuple[int, int], dict]:
        """Return each fold as evaluate_fold returns it, keyed by (repeat, fold)."""
        results = {}
        for row in self.folds.to_dict("records"):
            key = (row["repeat"], row["fold"])
            selected = (self.predictions["repeat"] == key[0]) & (
                self.predictions["fold"] == key[1]
            )
            results[key] = {
                "row": row,
                "predictions": self.predictions.loc[selected].reset_index(drop=True),
                "cv_results": self.cv_results[key],
                "estimator": self.estimators[key],
            }
        return results


def _fold_path(cache_dir: Path, name: str, split: Split) -> Path:
    """Return the cache file of one outer fold of an evaluation."""
    return cache_dir / f"{name}_r{split.repeat}_f{split.fold}.joblib"


@dataclass(frozen=True)
class EvaluationTask:
    """One cached nested cross-validation: an estimator, its data and its splits.

    name identifies the evaluation in the cache (for example, algorithm,
    formulation and feature set); fields comes from evaluation_fields. When
    previous holds the folds of the same evaluation with a smaller grid, keyed
    by (repeat, fold), only the new candidates are evaluated, with the result of
    the full search (evaluate_fold_extension).
    """

    name: str
    estimator: BaseEstimator
    X: pd.DataFrame
    y: ArrayLike
    splits: list[Split]
    fields: Mapping
    keep_estimators: bool = False
    previous: Mapping[tuple[int, int], Mapping] | None = field(default=None, repr=False)
    levels: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "levels", as_levels(self.y))

    def previous_for(self, split: Split) -> Mapping | None:
        """Return the previous result of a split, or None without a previous grid."""
        if self.previous is None:
            return None
        key = (split.repeat, split.fold)
        if key not in self.previous:
            raise KeyError(f"{self.name}: no previous result for fold {key}.")
        return self.previous[key]

    def require(self) -> Callable[[object], bool] | None:
        """Return the check a cached fold must pass, or None."""
        if not self.keep_estimators:
            return None
        return lambda result: result["estimator"] is not None


@dataclass(frozen=True)
class _PendingFold:
    """A fold that has to be fitted: its task, split, file and signature."""

    task: int
    split: Split
    path: Path
    fields: Mapping


def _compute_fold(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    levels: np.ndarray,
    split: Split,
    path: Path,
    fields: Mapping,
    keep_estimator: bool,
    previous: Mapping | None = None,
) -> tuple[dict | None, str | None, str | None]:
    """Evaluate and save one fold in a worker.

    With previous, only the candidates missing from it are evaluated. Returns
    the result, a save error and a fit error. A fit error is returned instead of
    raised, so one failing fold does not stop the others.
    """
    start = time.perf_counter()
    try:
        if previous is None:
            result = evaluate_fold(estimator, X, levels, split, keep_estimator)
        else:
            result = evaluate_fold_extension(
                estimator, previous, X, levels, split, keep_estimator
            )
    except Exception:  # noqa: BLE001 - reported for every failing fold at the end
        return None, None, traceback.format_exc()
    error = write_cached(path, fields, result, (time.perf_counter() - start) / 60)
    return result, error, None


def _plan_folds(
    tasks: Sequence[EvaluationTask], cache_dir: Path, force: bool
) -> tuple[dict[tuple[str, int, int], dict], list[_PendingFold]]:
    """Load the folds already cached and list those that have to be fitted."""
    loaded: dict[tuple[str, int, int], dict] = {}
    pending: list[_PendingFold] = []
    for index, task in enumerate(tasks):
        for split in task.splits:
            path = _fold_path(cache_dir, task.name, split)
            fields = fold_fields(task.fields, split)
            if not force:
                result, state = read_cached(path, fields, task.require())
                if state == "loaded":
                    loaded[(task.name, split.repeat, split.fold)] = result
                    continue
            pending.append(_PendingFold(index, split, path, fields))
    return loaded, pending


def _fit_pending(
    tasks: Sequence[EvaluationTask],
    pending: list[_PendingFold],
    n_jobs: int | None,
    verbose: int,
) -> tuple[dict[tuple[str, int, int], dict], dict[tuple[str, int, int], str]]:
    """Fit the pending folds in one pool of workers, recovering from dead workers.

    Each worker saves its fold as soon as it finishes. If a worker dies, the
    folds saved before the failure are read back and the rest are retried with
    half the workers, up to MAX_WORKER_RETRIES times.
    """
    computed: dict[tuple[str, int, int], dict] = {}
    failures: dict[tuple[str, int, int], str] = {}
    workers = n_jobs
    for attempt in range(MAX_WORKER_RETRIES + 1):
        try:
            outputs = Parallel(n_jobs=workers, verbose=verbose)(
                delayed(_compute_fold)(
                    tasks[fold.task].estimator,
                    tasks[fold.task].X,
                    tasks[fold.task].levels,
                    fold.split,
                    fold.path,
                    fold.fields,
                    tasks[fold.task].keep_estimators,
                    tasks[fold.task].previous_for(fold.split),
                )
                for fold in pending
            )
        except BrokenProcessPool as error:
            if attempt == MAX_WORKER_RETRIES:
                raise RuntimeError(
                    f"Workers kept dying after {MAX_WORKER_RETRIES} retries; the "
                    "folds already saved are kept. Lower the number of workers."
                ) from error
            pending = _recover_saved(tasks, pending, computed)
            workers = max(1, (workers or 1) // 2)
            warnings.warn(
                f"A worker died ({error}); retrying {len(pending)} folds with "
                f"{workers} workers.",
                CacheWarning,
                stacklevel=2,
            )
            continue
        for fold, (result, save_error, fit_error) in zip(pending, outputs, strict=True):
            key = (tasks[fold.task].name, fold.split.repeat, fold.split.fold)
            if fit_error:
                failures[key] = fit_error
                continue
            if save_error:
                warnings.warn(save_error, CacheWarning, stacklevel=2)
            computed[key] = result
        break
    return computed, failures


def _recover_saved(
    tasks: Sequence[EvaluationTask],
    pending: list[_PendingFold],
    computed: dict[tuple[str, int, int], dict],
) -> list[_PendingFold]:
    """Read back the folds saved before a worker died; return those still missing."""
    remaining = []
    for fold in pending:
        task = tasks[fold.task]
        result, state = read_cached(fold.path, fold.fields, task.require())
        if state == "loaded":
            computed[(task.name, fold.split.repeat, fold.split.fold)] = result
        else:
            remaining.append(fold)
    return remaining


def _assemble(
    task: EvaluationTask, results: Mapping, status: Mapping
) -> CachedEvaluation:
    """Collect the folds of one task into the tables of run_nested_cv."""
    keys = [(task.name, split.repeat, split.fold) for split in task.splits]
    folds = pd.DataFrame(
        [{**results[key]["row"], "status": status[key]} for key in keys]
    )
    extra_columns = [c for c in folds.columns if c not in FOLD_COLUMNS]
    return CachedEvaluation(
        folds=folds[[*FOLD_COLUMNS, *extra_columns]],
        predictions=pd.concat(
            [results[key]["predictions"] for key in keys], ignore_index=True
        ),
        cv_results={key[1:]: results[key]["cv_results"] for key in keys},
        estimators={key[1:]: results[key]["estimator"] for key in keys},
    )


def run_configurations_cached(
    tasks: Sequence[EvaluationTask],
    *,
    cache_dir: Path,
    n_jobs: int | None = None,
    force: bool = False,
    verbose: int = 0,
) -> dict[str, CachedEvaluation]:
    """Evaluate several tasks with one pool of workers, loading cached folds.

    The folds of all tasks share the pool, so the workers stay busy across
    configurations, and each fold is timed under the same load as the others.
    Folds whose file is missing, unreadable or has another signature are fitted;
    the rest are loaded. A fold whose fit fails does not stop the others: the
    error is raised at the end, after every other fold has been saved, and a new
    call fits only the folds still missing.
    """
    names = [task.name for task in tasks]
    if len(set(names)) != len(names):
        raise ValueError("Task names must be unique.")
    loaded, pending = _plan_folds(tasks, cache_dir, force)
    computed, failures = _fit_pending(tasks, pending, n_jobs, verbose)
    if failures:
        first_key, first_error = next(iter(failures.items()))
        raise RuntimeError(
            f"{len(failures)} folds failed and were not saved: {sorted(failures)}. "
            f"First error, in {first_key}:\n{first_error}"
        )
    results = {**loaded, **computed}
    status = {
        **dict.fromkeys(loaded, "loaded"),
        **dict.fromkeys(computed, "computed"),
    }
    return {task.name: _assemble(task, results, status) for task in tasks}


def run_nested_cv_cached(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: ArrayLike,
    splits: list[Split],
    *,
    name: str,
    cache_dir: Path,
    fields: Mapping,
    n_jobs: int | None = None,
    force: bool = False,
    keep_estimators: bool = False,
) -> CachedEvaluation:
    """Run run_nested_cv's evaluation, loading the folds already on disk.

    A single-task run_configurations_cached: only the folds whose file is
    missing, unreadable or has another signature are fitted, in parallel with
    n_jobs, and each worker saves its fold as soon as it finishes. With
    keep_estimators, the fitted best estimator of each fold is kept, and folds
    cached without it are refitted.
    """
    task = EvaluationTask(name, estimator, X, y, splits, fields, keep_estimators)
    return run_configurations_cached(
        [task], cache_dir=cache_dir, n_jobs=n_jobs, force=force
    )[name]


def directory_size(path: Path) -> int:
    """Return the total size in bytes of the files under path (0 if missing)."""
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def clear_pipeline_memory(path: Path) -> None:
    """Delete the preprocessing cache of Pipeline(memory=...) at path.

    Refuses any folder that is not a pipeline_memory folder or inside one, so
    the cache of evaluation results can never be removed by mistake.
    """
    path = Path(path)
    if PIPELINE_MEMORY_FOLDER not in path.resolve().parts:
        raise ValueError(f"{path} is not inside a {PIPELINE_MEMORY_FOLDER!r} folder.")
    if path.exists():
        shutil.rmtree(path)


def cached_size(task: EvaluationTask, cache_dir: Path) -> int:
    """Return the size in bytes of the cached folds of one task (0 if missing).

    Unlike directory_size, it measures only the files of this task, so it does
    not change when other evaluations are added to the same folder.
    """
    paths = (_fold_path(Path(cache_dir), task.name, split) for split in task.splits)
    return sum(path.stat().st_size for path in paths if path.is_file())
