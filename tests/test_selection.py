import numpy as np
import pandas as pd
import pytest

from piu_severity.evaluation.selection import (
    EDGE_RULE_MIN_FOLDS,
    apply_selection_rule,
    chosen_value_counts,
    compare_with_best,
    edge_review,
    extension_value,
    grid_extension,
    most_frequent_value,
    parameter_short_name,
    selected_configuration,
)
from piu_severity.evaluation.validation import outer_splits
from piu_severity.models.pipelines import complexity_key, search_grid

N_FOLDS = 25


@pytest.mark.parametrize(
    ("name", "values", "edge", "expected"),
    [
        ("model__estimator__C", [1e-4, 1e-3, 1e-2], "lower", 1e-5),
        ("model__estimator__C", [1e-2, 1e-1, 1.0, 10.0], "upper", 100.0),
        ("model__estimator__gamma", [1e-4, 1e-3, 1.0], "upper", 1_000.0),
        ("model__estimator__max_leaf_nodes", [7, 15, 31], "upper", 64),
        ("model__estimator__max_leaf_nodes", [7, 15, 31], "lower", 3),
        ("model__estimator__min_samples_leaf", [10, 30, 80], "upper", 213),
        ("model__estimator__min_samples_leaf", [10, 30, 80], "lower", 3),
        ("model__estimator__min_samples_leaf", [1, 5, 20, 50], "upper", 125),
        ("model__estimator__l2_regularization", [0.0, 1.0, 10.0], "upper", 100.0),
    ],
)
def test_extension_repeats_the_step_of_the_grid(name, values, edge, expected):
    result = extension_value(name, values, edge)
    assert result == pytest.approx(expected)
    assert type(result) is type(expected)


@pytest.mark.parametrize(
    ("name", "values", "edge"),
    [
        ("model__estimator__l1_ratio", [0.0, 0.5, 1.0], "lower"),
        ("model__estimator__l1_ratio", [0.0, 0.5, 1.0], "upper"),
        ("model__estimator__max_features", [0.33, 0.66, 1.0], "upper"),
        ("model__estimator__min_samples_leaf", [1, 5, 20, 50], "lower"),
        ("model__estimator__l2_regularization", [0.0, 1.0, 10.0], "lower"),
        ("model__estimator__max_features", ["sqrt", 0.33, 1.0], "lower"),
    ],
)
def test_edges_on_their_natural_limit_or_not_numeric_have_no_extension(
    name, values, edge
):
    assert extension_value(name, values, edge) is None


def test_extension_is_clipped_to_the_natural_limit():
    assert extension_value("model__estimator__l1_ratio", [0.2, 0.5, 0.8], "upper") == (
        pytest.approx(1.0)
    )
    assert extension_value("model__estimator__max_leaf_nodes", [3, 7, 15], "lower") == 2


def test_extension_rejects_an_unknown_edge():
    with pytest.raises(ValueError):
        extension_value("model__estimator__C", [1.0, 10.0], "middle")


def _params(name: str, chosen: list) -> list[dict]:
    return [{name: value, "preprocess__strategy": "simple"} for value in chosen]


def test_edge_review_counts_folds_and_applies_the_threshold():
    name = "model__estimator__C"
    grid = {name: [0.01, 0.1, 1.0, 10.0], "preprocess__strategy": ["simple"]}
    chosen = [10.0] * EDGE_RULE_MIN_FOLDS + [0.1] * (N_FOLDS - EDGE_RULE_MIN_FOLDS - 1)
    review = edge_review(_params(name, [*chosen, 0.01]), grid)
    upper = review.set_index("edge").loc["upper"]
    lower = review.set_index("edge").loc["lower"]
    assert upper["folds_at_edge"] == EDGE_RULE_MIN_FOLDS
    assert upper["n_folds"] == N_FOLDS
    assert bool(upper["extend"])
    assert upper["extension_value"] == pytest.approx(100.0)
    assert lower["folds_at_edge"] == 1
    assert not bool(lower["extend"])
    assert grid_extension(review) == {name: [100.0]}


def test_integer_extensions_stay_integer():
    name = "model__estimator__min_samples_leaf"
    grid = {name: [1, 5, 20, 50], "model__estimator__C": [0.1, 1.0, 10.0]}
    best = [{name: 50, "model__estimator__C": 10.0}] * N_FOLDS
    review = edge_review(best, grid)
    extension = grid_extension(review)
    assert extension == {name: [125], "model__estimator__C": [100.0]}
    assert type(extension[name][0]) is int
    assert type(extension["model__estimator__C"][0]) is float
    assert (
        review.set_index(["parameter", "edge"]).loc[(name, "lower"), "edge_value"] == 1
    )


def test_edge_review_skips_categorical_and_two_value_grids():
    grid = {
        "preprocess__strategy": ["simple", "iterative"],
        "model__weight_power": [0.5, 1.0],
        "model__estimator__learning_rate": [0.03, 0.1],
    }
    best = [
        {
            "preprocess__strategy": "iterative",
            "model__weight_power": 1.0,
            "model__estimator__learning_rate": 0.1,
        }
    ] * N_FOLDS
    review = edge_review(best, grid)
    assert review.empty
    assert grid_extension(review) == {}


def test_edge_on_a_natural_limit_is_reported_but_not_extended():
    name = "model__estimator__l1_ratio"
    review = edge_review(_params(name, [1.0] * N_FOLDS), {name: [0.0, 0.5, 1.0]})
    upper = review.set_index("edge").loc["upper"]
    assert upper["folds_at_edge"] == N_FOLDS
    assert upper["extension_value"] is None
    assert not bool(upper["extend"])


def test_edge_review_works_on_every_grid_of_the_search():
    for algorithm in ("linear", "svm", "random_forest", "gradient_boosting"):
        for formulation in ("threshold_regression", "multiclass", "frank_hall"):
            grid = search_grid(algorithm, formulation)
            best = [{name: values[0] for name, values in grid.items()}] * N_FOLDS
            review = edge_review(best, grid)
            assert set(review["edge"]) <= {"lower", "upper"}
            extension = grid_extension(review)
            for name, values in extension.items():
                assert all(value not in grid[name] for value in values)
                assert all(type(value) is type(grid[name][-1]) for value in values)


def test_chosen_values_and_most_frequent_value():
    name = "preprocess__strategy"
    best = _params("model__estimator__C", [1.0] * 3)
    best = [{**b, name: s} for b, s in zip(best, ["iterative", "simple", "simple"])]
    counts = chosen_value_counts(best, name)
    assert counts.to_dict() == {"iterative": 1, "simple": 2}
    assert most_frequent_value(best, name) == "simple"
    tie = [{name: "iterative"}, {name: "simple"}]
    assert most_frequent_value(tie, name) == "iterative"
    with pytest.raises(ValueError):
        most_frequent_value([], name)


def test_parameter_short_name():
    assert parameter_short_name("model__estimator__estimator__C") == "C"
    assert parameter_short_name("preprocess__strategy") == "strategy"


def _folds(qwk: np.ndarray) -> pd.DataFrame:
    """Fold results of one configuration on the same 25 splits."""
    y = np.repeat(np.arange(4), (60, 30, 20, 10))
    splits = outer_splits(y)
    return pd.DataFrame(
        {
            "repeat": [s.repeat for s in splits],
            "fold": [s.fold for s in splits],
            "n_train": [len(s.train) for s in splits],
            "n_test": [len(s.test) for s in splits],
            "test_hash": [str(s.repeat * 10 + s.fold) for s in splits],
            "qwk": qwk,
        }
    )


def _comparison_data() -> dict[tuple[str, str], pd.DataFrame]:
    rng = np.random.default_rng(0)
    base = 0.36 + rng.normal(scale=0.03, size=N_FOLDS)
    noise = rng.normal(scale=0.02, size=N_FOLDS)
    noise -= noise.mean()
    return {
        ("linear", "threshold_regression"): _folds(base),
        # Slightly lower, with noisy differences: not significant.
        ("svm", "threshold_regression"): _folds(base - 0.004 + noise),
        # Clearly lower on every fold: significant.
        ("random_forest", "frank_hall"): _folds(base - 0.06 + noise / 4),
    }


def test_compare_with_best_marks_significant_differences():
    comparison = compare_with_best(_comparison_data())
    assert comparison.loc[("linear", "threshold_regression"), "is_best"]
    assert not comparison.loc[("linear", "threshold_regression"), "significant"]
    assert not comparison.loc[("svm", "threshold_regression"), "significant"]
    assert comparison.loc[("random_forest", "frank_hall"), "significant"]
    assert comparison.loc[("random_forest", "frank_hall"), "mean_difference"] == (
        pytest.approx(0.06)
    )


def test_constant_differences_have_a_defined_result():
    folds = _comparison_data()[("linear", "threshold_regression")]
    lower = folds.assign(qwk=folds["qwk"] - 0.05)
    comparison = compare_with_best({"a": folds, "b": folds.copy(), "c": lower})
    assert not comparison.loc["b", "significant"]
    assert comparison.loc["b", "p_value_one_sided"] == 1.0
    assert comparison.loc["c", "significant"]
    assert comparison.loc["c", "p_value_one_sided"] == 0.0


def _complexities(keys):
    return {key: complexity_key(*key, "simple") for key in keys}


def test_selection_prefers_lower_underestimation_among_candidates():
    data = _comparison_data()
    comparison = compare_with_best(data)
    underestimation = {
        ("linear", "threshold_regression"): 0.68,
        ("svm", "threshold_regression"): 0.60,
        # The lowest rate, but significantly worse in QWK: not a candidate.
        ("random_forest", "frank_hall"): 0.40,
    }
    table = apply_selection_rule(comparison, underestimation, _complexities(data))
    assert selected_configuration(table) == ("svm", "threshold_regression")
    assert not table.loc[("random_forest", "frank_hall"), "candidate"]


def test_selection_breaks_underestimation_ties_by_complexity():
    data = _comparison_data()
    comparison = compare_with_best(data)
    underestimation = {
        ("linear", "threshold_regression"): 0.605,
        ("svm", "threshold_regression"): 0.60,
        ("random_forest", "frank_hall"): 0.40,
    }
    table = apply_selection_rule(comparison, underestimation, _complexities(data))
    # The two rates differ by less than the tolerance, so the simpler one wins.
    assert table["tied_underestimation"].sum() == 2
    assert selected_configuration(table) == ("linear", "threshold_regression")


def test_selection_requires_rates_and_complexities_for_every_configuration():
    data = _comparison_data()
    comparison = compare_with_best(data)
    with pytest.raises(ValueError):
        apply_selection_rule(
            comparison, {("linear", "threshold_regression"): 0.6}, _complexities(data)
        )


def test_selected_configuration_requires_exactly_one():
    with pytest.raises(ValueError):
        selected_configuration(pd.DataFrame({"selected": [True, True]}))
