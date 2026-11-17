"""Record of the selected configuration, shared with the later notebooks.

Each evaluation in the record keeps what make_search_task needs to rebuild it
(its configuration and name suffix), its cache name and the hash of its
signature. A later notebook rebuilds the task with its own data and splits,
checks that name and signature match, and loads the folds from the cache
instead of fitting them again. The record holds no data: only names,
parameters, hashes and aggregate results.
"""

import json
import os
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
from numpy.typing import ArrayLike

from piu_severity.config import MODELS_DIR
from piu_severity.evaluation.persistence import (
    CachedEvaluation,
    EvaluationTask,
    canonical,
    signature_hash,
)
from piu_severity.evaluation.validation import Split
from piu_severity.models.pipelines import make_search_task

SELECTED_CONFIGURATION_FILE = MODELS_DIR / "selected_configuration.json"
# Largest difference in mean QWK accepted when an evaluation is fitted again
# from its record, for example on another machine without the cache: the
# versions and seeds are fixed, but the numerical libraries of the operating
# system can still change a few predictions.
QWK_REPRODUCTION_TOLERANCE = 0.005


def task_entry(
    task: EvaluationTask,
    name_suffix: str = "",
    result: CachedEvaluation | None = None,
) -> dict:
    """Return the record of one cached evaluation built with make_search_task.

    With result, the record also keeps its number of outer folds and its mean
    QWK, which check_result compares with a later run.
    """
    if not task.name.endswith(name_suffix):
        raise ValueError(f"{task.name!r} does not end with {name_suffix!r}.")
    entry = {
        "name": task.name,
        "signature": signature_hash(task.fields),
        "config": canonical(task.fields["config"]),
        "name_suffix": name_suffix,
    }
    if result is not None:
        if len(result.folds) != len(task.splits):
            raise ValueError(f"{task.name!r}: the result does not cover its splits.")
        entry["n_folds"] = len(result.folds)
        entry["mean_qwk"] = float(result.folds["qwk"].mean())
    return entry


def task_arguments(entry: Mapping) -> dict:
    """Return the keyword arguments of make_search_task that rebuild an entry.

    The data, the splits and the execution options (memory, keep_estimators)
    are passed apart, because they belong to the notebook that rebuilds it.
    """
    return {**entry["config"], "name_suffix": entry["name_suffix"]}


def check_task(task: EvaluationTask, entry: Mapping) -> None:
    """Raise if a rebuilt task is not the evaluation recorded in entry.

    A different signature means that the data, the splits' seed, the grid or a
    library version changed, so the cached folds would not be reused.
    """
    if task.name != entry["name"]:
        raise ValueError(f"Task {task.name!r} is not {entry['name']!r}.")
    if signature_hash(task.fields) != entry["signature"]:
        raise ValueError(f"{task.name!r} has another signature than the record.")


def rebuild_task(
    entry: Mapping,
    X: pd.DataFrame,
    y: ArrayLike,
    splits: list[Split],
    *,
    feature_set: str | None = None,
    memory: str | None = None,
    keep_estimators: bool = False,
) -> EvaluationTask:
    """Return the task of a recorded evaluation, checked against its entry.

    X, y and splits must be built as in the notebook that recorded the entry.
    With feature_set, the task is the same evaluation on another feature set,
    with the same name suffix and fixed parameters; it is not in the record, so
    it is not checked, and X must hold the columns of that feature set.
    """
    arguments = task_arguments(entry)
    recorded_set = arguments["feature_set"]
    if feature_set is not None:
        arguments["feature_set"] = feature_set
    task = make_search_task(
        X=X,
        y=y,
        splits=splits,
        memory=memory,
        keep_estimators=keep_estimators,
        **arguments,
    )
    if arguments["feature_set"] == recorded_set:
        check_task(task, entry)
    return task


def check_result(
    result: CachedEvaluation,
    entry: Mapping,
    tolerance: float = QWK_REPRODUCTION_TOLERANCE,
) -> float:
    """Raise if a result does not reproduce the recorded one; return the difference.

    The number of outer folds must match and the mean QWK must lie within
    tolerance of the recorded value. Loaded from the same cache, the difference
    is zero.
    """
    if "mean_qwk" not in entry:
        raise ValueError(f"{entry['name']!r} was recorded without its result.")
    if len(result.folds) != entry["n_folds"]:
        raise ValueError(
            f"{entry['name']!r}: {len(result.folds)} folds instead of "
            f"{entry['n_folds']}."
        )
    difference = float(result.folds["qwk"].mean()) - entry["mean_qwk"]
    if abs(difference) > tolerance:
        raise ValueError(
            f"{entry['name']!r}: mean QWK differs by {difference:+.4f} from the "
            f"record, more than {tolerance}."
        )
    return difference


def save_record(record: Mapping, path: Path = SELECTED_CONFIGURATION_FILE) -> Path:
    """Write the record as JSON, replacing any previous file atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(canonical(record), indent=2, ensure_ascii=False)
    temporary.write_text(text + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def load_record(path: Path = SELECTED_CONFIGURATION_FILE) -> dict:
    """Read a record written by save_record."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
