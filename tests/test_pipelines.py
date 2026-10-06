import numpy as np
import pandas as pd
import pytest

from piu_severity.evaluation.validation import outer_splits, run_nested_cv
from piu_severity.models.pipelines import (
    BOOSTING_GRID,
    LOGISTIC_GRID,
    make_boosting_search,
    make_logistic_search,
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