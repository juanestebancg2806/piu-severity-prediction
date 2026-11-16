import json

import numpy as np
import pandas as pd
import pytest

from piu_severity.evaluation.persistence import run_configurations_cached
from piu_severity.evaluation.validation import outer_splits
from piu_severity.models.pipelines import make_search_task
from piu_severity.models.registry import (
    check_task,
    load_record,
    save_record,
    task_arguments,
    task_entry,
)

FIXED = {"preprocess__strategy": "simple", "model__weight_power": np.float64(0.5)}


def _data(seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.permutation(np.repeat(np.arange(4), (60, 30, 20, 10)))
    X = pd.DataFrame(
        {
            "Basic_Demos-Enroll_Season": rng.choice(["Fall", "Spring"], size=len(y)),
            "Basic_Demos-Age": 8 + 3 * y + rng.normal(size=len(y)),
            "Basic_Demos-Sex": rng.integers(0, 2, size=len(y)).astype(float),
        }
    )
    return X, y


def _task(X, y, splits, **options):
    return make_search_task(
        "linear",
        "threshold_regression",
        "reference",
        X,
        y,
        splits,
        fixed_params=FIXED,
        name_suffix="__fixed",
        **options,
    )


def test_task_entry_rebuilds_the_same_task():
    X, y = _data()
    splits = outer_splits(y)
    task = _task(X, y, splits)
    entry = task_entry(task, "__fixed")
    assert entry["name"] == task.name
    assert entry["config"]["fixed_params"] == {
        "model__weight_power": 0.5,
        "preprocess__strategy": "simple",
    }
    rebuilt = make_search_task(X=X, y=y, splits=splits, **task_arguments(entry))
    check_task(rebuilt, entry)


def test_check_task_rejects_other_data_and_names():
    X, y = _data()
    splits = outer_splits(y)
    entry = task_entry(_task(X, y, splits), "__fixed")
    X_other = X.assign(**{"Basic_Demos-Age": X["Basic_Demos-Age"] + 1})
    with pytest.raises(ValueError, match="signature"):
        check_task(_task(X_other, y, splits), entry)
    with pytest.raises(ValueError, match="is not"):
        check_task(
            make_search_task("linear", "multiclass", "reference", X, y, splits), entry
        )
    with pytest.raises(ValueError, match="does not end"):
        task_entry(_task(X, y, splits), "__other")


def test_record_round_trip_is_plain_json(tmp_path):
    X, y = _data()
    entry = task_entry(_task(X, y, outer_splits(y)), "__fixed")
    record = {"selected": entry, "mean_qwk": np.float64(0.365), "repeats": np.int64(5)}
    path = save_record(record, tmp_path / "models" / "selected_configuration.json")
    assert load_record(path) == json.loads(path.read_text(encoding="utf-8"))
    loaded = load_record(path)
    assert loaded["selected"] == entry
    assert loaded["mean_qwk"] == pytest.approx(0.365)
    assert loaded["repeats"] == 5
    assert not list(path.parent.glob("*.tmp"))


def test_rebuilt_task_loads_the_cached_folds(tmp_path):
    X, y = _data()
    splits = outer_splits(y)[:2]
    task = _task(X, y, splits)
    run_configurations_cached([task], cache_dir=tmp_path)
    entry = task_entry(task, "__fixed")
    rebuilt = make_search_task(X=X, y=y, splits=splits, **task_arguments(entry))
    check_task(rebuilt, entry)
    result = run_configurations_cached([rebuilt], cache_dir=tmp_path)[rebuilt.name]
    assert result.counts() == {"loaded": 2, "computed": 0}
