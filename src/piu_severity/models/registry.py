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

from piu_severity.config import MODELS_DIR
from piu_severity.evaluation.persistence import (
    EvaluationTask,
    canonical,
    signature_hash,
)

SELECTED_CONFIGURATION_FILE = MODELS_DIR / "selected_configuration.json"


def task_entry(task: EvaluationTask, name_suffix: str = "") -> dict:
    """Return the record of one cached evaluation built with make_search_task."""
    if not task.name.endswith(name_suffix):
        raise ValueError(f"{task.name!r} does not end with {name_suffix!r}.")
    return {
        "name": task.name,
        "signature": signature_hash(task.fields),
        "config": canonical(task.fields["config"]),
        "name_suffix": name_suffix,
    }


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
