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
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
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
    hash_indices,
)

# Increase by hand whenever the code of the estimators or the preprocessing
# changes in a way that can change results; every cached file is then refitted.
CACHE_VERSION = 1
SKLEARN_VERSION = sklearn.__version__
PIPELINE_MEMORY_FOLDER = "pipeline_memory"
IGNORED_PARAMETERS = frozenset({"memory", "n_jobs", "verbose", "pre_dispatch"})
STORED_KEYS = frozenset({"fields", "fields_hash", "result", "metadata"})


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


def _fold_path(cache_dir: Path, name: str, split: Split) -> Path:
    """Return the cache file of one outer fold of an evaluation."""
    return cache_dir / f"{name}_r{split.repeat}_f{split.fold}.joblib"


def _compute_fold(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    levels: np.ndarray,
    split: Split,
    path: Path,
    fields: Mapping,
    keep_estimator: bool,
) -> tuple[dict, str | None]:
    """Evaluate one fold and save it from the worker; return any save error."""
    start = time.perf_counter()
    result = evaluate_fold(estimator, X, levels, split, keep_estimator)
    error = write_cached(path, fields, result, (time.perf_counter() - start) / 60)
    return result, error


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

    Only the folds whose file is missing, unreadable or has another signature
    are fitted, in parallel with n_jobs; each worker saves its fold as soon as
    it finishes. fields comes from evaluation_fields. With keep_estimators, the
    fitted best estimator of each fold is kept, and folds cached without it are
    refitted.
    """
    levels = as_levels(y)
    require = (
        (lambda result: result["estimator"] is not None) if keep_estimators else None
    )
    results: dict[tuple[int, int], dict] = {}
    status: dict[tuple[int, int], str] = {}
    pending = []
    for split in splits:
        key = (split.repeat, split.fold)
        path = _fold_path(cache_dir, name, split)
        cached, state = (
            (None, "forced")
            if force
            else read_cached(path, fold_fields(fields, split), require)
        )
        if state == "loaded":
            results[key], status[key] = cached, "loaded"
        else:
            pending.append((split, path))
    computed = Parallel(n_jobs=n_jobs)(
        delayed(_compute_fold)(
            estimator,
            X,
            levels,
            split,
            path,
            fold_fields(fields, split),
            keep_estimators,
        )
        for split, path in pending
    )
    for (split, _), (result, error) in zip(pending, computed, strict=True):
        if error:
            warnings.warn(error, CacheWarning, stacklevel=2)
        key = (split.repeat, split.fold)
        results[key], status[key] = result, "computed"
    keys = [(split.repeat, split.fold) for split in splits]
    folds = pd.DataFrame(
        [{**results[key]["row"], "status": status[key]} for key in keys]
    )
    extra_columns = [c for c in folds.columns if c not in FOLD_COLUMNS]
    return CachedEvaluation(
        folds=folds[[*FOLD_COLUMNS, *extra_columns]],
        predictions=pd.concat(
            [results[key]["predictions"] for key in keys], ignore_index=True
        ),
        cv_results={key: results[key]["cv_results"] for key in keys},
        estimators={key: results[key]["estimator"] for key in keys},
    )


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