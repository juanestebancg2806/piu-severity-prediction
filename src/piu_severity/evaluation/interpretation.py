"""SHAP contributions of the selected configuration and their stability.

The explained output is the continuous estimate of the threshold regression,
before the thresholds. Each outer fold is explained with its own model, on its
evaluation rows, against its own training rows as background, so every
participant is explained once per repeat by a model that did not see it. For a
linear model with an independent masker the contribution of a transformed
predictor is coef * (x - background mean), which is exact.

The transformed predictors differ between folds (the simple strategy adds one
missingness indicator per instrument; one-hot encoding splits the enrollment
season), so contributions are summed with their sign into original variables,
instruments or groups before they are compared across folds. The importance of
a variable in a repeat is the mean absolute value of its summed contribution
over the participants of that repeat.
"""

from collections.abc import Mapping, Sequence
from itertools import combinations

import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.base import BaseEstimator

from piu_severity.evaluation.validation import Split
from piu_severity.features.groups import (
    ACTIGRAPHY_INSTRUMENT,
    INSTRUMENT_GROUPS,
    instrument_of,
)

# Name prefix of the per-instrument indicators of InstrumentMissingIndicator.
MISSING_PREFIX = "missing_"
ACTIGRAPHY_GROUP = "actigraphy"
AGGREGATION_LEVELS: tuple[str, ...] = ("variable", "instrument", "group")
# Criterion fixed before the contributions are computed: the order of
# importance is kept across repeats when every pair of repeats ranks the
# variables with a Spearman correlation of at least STABILITY_MIN_SPEARMAN and
# the STABILITY_TOP_K most important variables are the same in every repeat.
STABILITY_MIN_SPEARMAN = 0.9
STABILITY_TOP_K = 5
# Descriptive: in how many repeats each variable is among the most important.
DESCRIPTIVE_TOP_K = 10


def original_variable(name: str, input_columns: Sequence[str]) -> str:
    """Return the original variable of a transformed predictor.

    A predictor kept by name is its own variable, a one-hot column belongs to
    the longest input column it extends, and a missingness indicator is kept
    as its own variable.
    """
    if name in input_columns or name.startswith(MISSING_PREFIX):
        return name
    matches = [c for c in input_columns if name.startswith(f"{c}_")]
    if not matches:
        raise KeyError(f"No input column produces {name!r}.")
    return max(matches, key=len)


def variable_instrument(variable: str) -> str:
    """Return the instrument of an original variable or missingness indicator."""
    if variable.startswith(MISSING_PREFIX):
        return variable.removeprefix(MISSING_PREFIX)
    return instrument_of(variable)


def instrument_group(instrument: str) -> str:
    """Return the group of an instrument, with actigraphy apart from the physical."""
    if instrument == ACTIGRAPHY_INSTRUMENT:
        return ACTIGRAPHY_GROUP
    if instrument not in INSTRUMENT_GROUPS:
        raise KeyError(f"Unknown instrument {instrument!r}.")
    return INSTRUMENT_GROUPS[instrument]


def aggregation_map(
    names: Sequence[str], input_columns: Sequence[str], level: str
) -> dict[str, str]:
    """Return the variable, instrument or group of each transformed predictor."""
    if level not in AGGREGATION_LEVELS:
        raise ValueError(f"Unknown aggregation level {level!r}.")
    mapping = {}
    for name in names:
        target = original_variable(name, input_columns)
        if level in ("instrument", "group"):
            target = variable_instrument(target)
        if level == "group":
            target = instrument_group(target)
        mapping[name] = target
    return mapping


def aggregate_contributions(
    contributions: pd.DataFrame, mapping: Mapping[str, str]
) -> pd.DataFrame:
    """Sum signed contributions into the targets of mapping, in order of appearance."""
    missing = [c for c in contributions.columns if c not in mapping]
    if missing:
        raise KeyError(f"Columns without a target: {missing}")
    targets = list(dict.fromkeys(mapping[c] for c in contributions.columns))
    summed = contributions.T.groupby(
        contributions.columns.map(mapping), sort=False
    ).sum()
    return summed.T[targets]


def fold_contributions(
    pipeline: BaseEstimator, X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Return the SHAP values, transformed predictors and base value of one fold.

    The background is the whole transformed training fold; shap would otherwise
    subsample it to 100 rows, which changes the background mean.
    """
    preprocess = pipeline.named_steps["preprocess"]
    regressor = pipeline.named_steps["model"].estimator_
    background = preprocess.transform(X_train)
    values = preprocess.transform(X_test)
    masker = shap.maskers.Independent(background, max_samples=len(background))
    explainer = shap.LinearExplainer(regressor, masker)
    contributions = pd.DataFrame(
        explainer.shap_values(values), index=values.index, columns=values.columns
    )
    return contributions, values, float(np.ravel(explainer.expected_value)[0])


def explain_folds(
    estimators: Mapping[tuple[int, int], BaseEstimator],
    splits: Sequence[Split],
    X: pd.DataFrame,
) -> dict[tuple[int, int], dict]:
    """Explain every outer fold with its model, on its evaluation rows.

    The rows of X are positions, as in the predictions of the evaluation; each
    fold returns its contributions, transformed predictors, base value and the
    input columns of its model.
    """
    explanations = {}
    for split in splits:
        key = (split.repeat, split.fold)
        pipeline = estimators.get(key)
        if pipeline is None:
            raise KeyError(f"No fitted model for fold {key}.")
        X_test = X.iloc[split.test].set_axis(split.test)
        contributions, values, base = fold_contributions(
            pipeline, X.iloc[split.train], X_test
        )
        explanations[key] = {
            "contributions": contributions,
            "values": values,
            "base_value": base,
            "input_columns": list(X.columns),
        }
    return explanations


def stack_contributions(
    explanations: Mapping[tuple[int, int], Mapping], level: str = "variable"
) -> pd.DataFrame:
    """Return the contributions of every fold aggregated at one level.

    Rows are indexed by repeat, fold and position. A target that a fold does
    not produce (an indicator of the simple strategy in an iterative fold)
    contributes zero there.
    """
    frames = []
    for (repeat, fold), explanation in explanations.items():
        contributions = explanation["contributions"]
        mapping = aggregation_map(
            contributions.columns, explanation["input_columns"], level
        )
        aggregated = aggregate_contributions(contributions, mapping)
        aggregated.index = pd.MultiIndex.from_arrays(
            [
                np.full(len(aggregated), repeat),
                np.full(len(aggregated), fold),
                aggregated.index,
            ],
            names=["repeat", "fold", "position"],
        )
        frames.append(aggregated)
    stacked = pd.concat(frames)
    return stacked.fillna(0.0)


def importance_by_repeat(stacked: pd.DataFrame) -> pd.DataFrame:
    """Return the mean absolute contribution of each target in each repeat."""
    return stacked.abs().groupby(level="repeat").mean()


def ranking_stability(importance: pd.DataFrame) -> pd.DataFrame:
    """Return the Spearman correlation of the importances of every pair of repeats."""
    rows = []
    for first, second in combinations(importance.index, 2):
        rho = stats.spearmanr(importance.loc[first], importance.loc[second]).statistic
        rows.append({"repeat_a": first, "repeat_b": second, "spearman": float(rho)})
    return pd.DataFrame(rows)


def top_targets(importance: pd.DataFrame, k: int) -> pd.DataFrame:
    """Return, per repeat, whether each target is among the k most important."""
    ranks = importance.rank(axis="columns", ascending=False, method="first")
    return ranks <= k


def order_is_stable(
    importance: pd.DataFrame,
    min_spearman: float = STABILITY_MIN_SPEARMAN,
    top_k: int = STABILITY_TOP_K,
) -> dict[str, float | bool]:
    """Apply the stability criterion fixed before the contributions were seen."""
    pairs = ranking_stability(importance)
    top = top_targets(importance, top_k)
    same_top = bool((top.eq(top.iloc[0], axis="columns")).all().all())
    lowest = float(pairs["spearman"].min())
    return {
        "min_spearman": lowest,
        "median_spearman": float(pairs["spearman"].median()),
        "same_top_k": same_top,
        "stable": bool(lowest >= min_spearman and same_top),
    }


def coefficient_signs(
    estimators: Mapping[tuple[int, int], BaseEstimator],
) -> pd.DataFrame:
    """Count, for each transformed predictor, the folds with each coefficient sign.

    The direction of a contribution in the transformed scale is the sign of its
    coefficient; ElasticNet can set a coefficient to zero.
    """
    signs = {}
    for key, pipeline in estimators.items():
        names = pipeline.named_steps["preprocess"].get_feature_names_out()
        coef = pipeline.named_steps["model"].estimator_.coef_
        signs[key] = pd.Series(np.sign(coef), index=names)
    table = pd.DataFrame(signs).T
    return pd.DataFrame(
        {
            "positive": (table > 0).sum(),
            "negative": (table < 0).sum(),
            "zero": (table == 0).sum(),
            "folds": table.notna().sum(),
        }
    ).rename_axis("predictor")
