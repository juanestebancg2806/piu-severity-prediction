import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline

from piu_severity.evaluation.interpretation import (
    MISSING_PREFIX,
    aggregate_contributions,
    aggregation_map,
    coefficient_signs,
    explain_folds,
    fold_contributions,
    importance_by_repeat,
    order_is_stable,
    original_variable,
    ranking_stability,
    stack_contributions,
    top_targets,
)
from piu_severity.evaluation.validation import outer_splits
from piu_severity.features.preprocessing import (
    InstrumentMissingIndicator,
    MissingDataPreprocessor,
)
from piu_severity.models.ordinal import ThresholdRegressor

SEASON = "Basic_Demos-Enroll_Season"


def _data(seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 200
    y = rng.permutation(np.repeat(np.arange(4), (100, 50, 35, 15)))
    X = pd.DataFrame(
        {
            SEASON: rng.choice(["Fall", "Spring", "Summer", "Winter"], n),
            "Basic_Demos-Age": (8 + 2 * y + rng.integers(0, 3, n)).astype(float),
            "Basic_Demos-Sex": rng.integers(0, 2, n).astype(float),
            "Physical-BMI": 18 + y + rng.normal(size=n),
            "Physical-Height": 120 + 5 * y + rng.normal(size=n),
            "Actigraphy-ENMO_Day": rng.normal(size=n),
        }
    )
    X.loc[rng.random(n) < 0.3, ["Physical-BMI", "Physical-Height"]] = np.nan
    X.loc[rng.random(n) < 0.6, "Actigraphy-ENMO_Day"] = np.nan
    return X, y


def _pipeline(columns: list[str], strategy: str) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", MissingDataPreprocessor(columns, "linear", strategy)),
            (
                "model",
                ThresholdRegressor(ElasticNet(alpha=0.01), weight_power=0.5),
            ),
        ]
    )


def _fitted_folds(X, y, n_repeats: int = 2):
    """Fit one model per outer fold, alternating the missing-data strategy."""
    splits = outer_splits(y, n_repeats=n_repeats)
    estimators = {}
    for index, split in enumerate(splits):
        strategy = "simple" if index % 2 == 0 else "iterative"
        estimators[(split.repeat, split.fold)] = _pipeline(
            list(X.columns), strategy
        ).fit(X.iloc[split.train], y[split.train])
    return splits, estimators


def test_original_variable_of_kept_one_hot_and_indicator_names():
    columns = ["A-x", "A-x_y", "B-z"]
    assert original_variable("B-z", columns) == "B-z"
    assert original_variable("A-x_y_Fall", columns) == "A-x_y"
    assert original_variable("A-x_Fall", columns) == "A-x"
    assert original_variable("missing_B", columns) == "missing_B"
    with pytest.raises(KeyError, match="No input column"):
        original_variable("C-w", columns)


def test_missing_prefix_matches_the_indicator_names():
    X, _ = _data()
    names = InstrumentMissingIndicator().fit(X).get_feature_names_out()
    assert all(name.startswith(MISSING_PREFIX) for name in names)


def test_aggregation_map_by_level():
    columns = [SEASON, "Physical-BMI", "Actigraphy-ENMO_Day"]
    names = [
        f"{SEASON}_Fall",
        "Physical-BMI",
        "Actigraphy-ENMO_Day",
        "missing_Physical",
    ]
    assert aggregation_map(names, columns, "variable") == {
        f"{SEASON}_Fall": SEASON,
        "Physical-BMI": "Physical-BMI",
        "Actigraphy-ENMO_Day": "Actigraphy-ENMO_Day",
        "missing_Physical": "missing_Physical",
    }
    assert aggregation_map(names, columns, "instrument")["missing_Physical"] == (
        "Physical"
    )
    assert aggregation_map(names, columns, "group") == {
        f"{SEASON}_Fall": "demographic",
        "Physical-BMI": "physical",
        "Actigraphy-ENMO_Day": "actigraphy",
        "missing_Physical": "physical",
    }
    with pytest.raises(ValueError, match="Unknown aggregation level"):
        aggregation_map(names, columns, "fold")


def test_aggregate_contributions_sums_with_sign_in_order():
    contributions = pd.DataFrame(
        {"b1": [1.0, -2.0], "a": [0.5, 0.5], "b2": [-1.0, 1.0]}
    )
    summed = aggregate_contributions(contributions, {"b1": "b", "a": "a", "b2": "b"})
    assert list(summed.columns) == ["b", "a"]
    assert summed["b"].tolist() == [0.0, -1.0]
    with pytest.raises(KeyError, match="without a target"):
        aggregate_contributions(contributions, {"a": "a"})


@pytest.mark.parametrize("strategy", ["simple", "iterative"])
def test_fold_contributions_are_exact_and_use_the_whole_background(strategy):
    X, y = _data()
    train, test = np.arange(150), np.arange(150, 200)
    pipeline = _pipeline(list(X.columns), strategy).fit(X.iloc[train], y[train])
    contributions, values, base = fold_contributions(
        pipeline, X.iloc[train], X.iloc[test]
    )
    regressor = pipeline.named_steps["model"].estimator_
    background = pipeline.named_steps["preprocess"].transform(X.iloc[train])
    expected = regressor.coef_ * (values - background.mean())
    np.testing.assert_allclose(contributions.to_numpy(), expected.to_numpy())
    assert base == pytest.approx(
        regressor.intercept_ + regressor.coef_ @ background.mean()
    )
    continuous = pipeline.named_steps["model"].predict_continuous(values)
    np.testing.assert_allclose(contributions.sum(axis=1) + base, continuous)


def test_stacked_contributions_cover_each_participant_once_per_repeat():
    X, y = _data()
    splits, estimators = _fitted_folds(X, y)
    explanations = explain_folds(estimators, splits, X)
    variables = stack_contributions(explanations, "variable")
    for _, rows in variables.groupby(level="repeat"):
        positions = rows.index.get_level_values("position")
        assert sorted(positions) == list(range(len(X)))
    assert SEASON in variables.columns
    assert "missing_Physical" in variables.columns
    groups = stack_contributions(explanations, "group")
    np.testing.assert_allclose(groups.sum(axis=1), variables.sum(axis=1))
    assert set(groups.columns) == {"demographic", "physical", "actigraphy"}
    with pytest.raises(KeyError, match="No fitted model"):
        explain_folds({}, splits, X)


def test_importance_is_the_mean_absolute_contribution_per_repeat():
    index = pd.MultiIndex.from_tuples(
        [(0, 0, 0), (0, 1, 1), (1, 0, 0), (1, 1, 1)],
        names=["repeat", "fold", "position"],
    )
    stacked = pd.DataFrame(
        {"a": [1.0, -3.0, 2.0, 2.0], "b": [0.0, 1.0, -1.0, 0.0]}, index
    )
    importance = importance_by_repeat(stacked)
    assert importance.loc[0].to_dict() == {"a": 2.0, "b": 0.5}
    assert importance.loc[1].to_dict() == {"a": 2.0, "b": 0.5}


def _importance(rows: list[list[float]]) -> pd.DataFrame:
    columns = [f"v{i}" for i in range(len(rows[0]))]
    return pd.DataFrame(rows, columns=columns).rename_axis("repeat")


def test_identical_rankings_are_stable_and_reversed_are_not():
    base = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    same = _importance([base, base, base])
    assert ranking_stability(same)["spearman"].tolist() == pytest.approx([1.0] * 3)
    assert order_is_stable(same)["stable"]
    reversed_ = _importance([base, base[::-1]])
    assert ranking_stability(reversed_)["spearman"].iloc[0] == pytest.approx(-1.0)
    assert not order_is_stable(reversed_)["stable"]


def test_stability_requires_the_same_top_k():
    first = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    second = [10.0, 9.0, 8.0, 7.0, 5.0, 6.0, 4.0, 3.0, 2.0, 1.0]
    result = order_is_stable(_importance([first, second]))
    assert result["min_spearman"] >= 0.9
    assert not result["same_top_k"]
    assert not result["stable"]
    assert top_targets(_importance([first]), 2).iloc[0].sum() == 2


def test_coefficient_signs_count_the_folds():
    X, y = _data()
    _, estimators = _fitted_folds(X, y, n_repeats=1)
    signs = coefficient_signs(estimators)
    assert (signs[["positive", "negative", "zero"]].sum(axis=1) == signs["folds"]).all()
    assert signs.loc["Basic_Demos-Age", "folds"] == 5
    assert signs.loc["missing_Physical", "folds"] == 3
