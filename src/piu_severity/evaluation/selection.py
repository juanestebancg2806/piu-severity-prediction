"""Model selection rule, grid-edge review and the decisions fixed before the search.

The constants below are fixed before the search is run, so that no threshold of
the selection or of the hypothesis tests depends on the results. The functions
apply the edge rule to the hyperparameters chosen in the inner validation and
the selection rule to the outer-fold results; neither looks at anything else.
"""

from collections import Counter
from collections.abc import Hashable, Mapping, Sequence

import numpy as np
import pandas as pd

from piu_severity.evaluation.validation import paired_differences, summarize_paired

SELECTION_ALPHA = 0.05
# Smallest QWK difference considered relevant; a contribution is negligible
# when the one-sided corrected upper bound of the difference lies below it.
MINIMUM_RELEVANT_QWK_DIFFERENCE = 0.03
# Two under-estimation rates closer than this are a tie in step 2 of the rule.
UNDERESTIMATION_TIE_TOLERANCE = 0.01
# A grid is extended past an edge chosen in at least this many outer folds.
EDGE_RULE_MIN_FOLDS = 5
# Families of tests whose p-values are adjusted together with Holm's method.
HOLM_FAMILIES: dict[str, tuple[str, ...]] = {
    "superiority": ("H2", "H3"),
    "negligibility": ("H2", "H3"),
    "actigraphy_subset": (
        "reference_with_actigraphy_vs_reference",
        "main_vs_main_without_actigraphy",
    ),
    "severe_paired": ("selected_vs_best_of_each_other_algorithm",),
}

# Edge rule. A grid needs an interior for an edge to suggest that the optimum
# lies outside it, so only numeric grids with at least this many values are
# reviewed; with two values every choice is an edge.
EDGE_RULE_MIN_GRID_VALUES = 3
# Natural limits of the hyperparameters, by their name in the estimator. An
# edge that already sits on its limit cannot be extended: l1_ratio 0 and 1 are
# ridge and lasso, max_features 1.0 uses every predictor, a leaf needs at
# least one observation and a regularization strength cannot be negative.
PARAMETER_LIMITS: dict[str, tuple[float | None, float | None]] = {
    "l1_ratio": (0.0, 1.0),
    "max_features": (None, 1.0),
    "min_samples_leaf": (1, None),
    "max_leaf_nodes": (2, None),
    "l2_regularization": (0.0, None),
}
EDGES: tuple[str, ...] = ("lower", "upper")


def parameter_short_name(name: str) -> str:
    """Return the estimator name of a pipeline parameter (its last segment)."""
    return name.split("__")[-1]


def _is_number(value: object) -> bool:
    """Return True for int and float values, but not for booleans."""
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, bool
    )


def _numeric_values(values: Sequence) -> list[float] | None:
    """Return the sorted numeric values of a grid, or None if any is not a number."""
    if not all(_is_number(value) for value in values):
        return None
    return sorted(values)


def extension_value(name: str, values: Sequence, edge: str) -> float | int | None:
    """Return the value one step past an edge of a grid, or None if there is none.

    The step repeats the ratio between the two values nearest to the edge when
    both are positive (the grids are geometric) and their difference otherwise.
    Integer grids stay integer. The new value is clipped to the natural limit of
    the hyperparameter; an edge already on its limit has no extension.
    """
    if edge not in EDGES:
        raise ValueError(f"edge must be one of {EDGES}.")
    numeric = _numeric_values(values)
    if numeric is None or len(numeric) < 2:
        return None
    if edge == "upper":
        inner, outer = numeric[-2], numeric[-1]
    else:
        inner, outer = numeric[1], numeric[0]
    new = outer * outer / inner if inner > 0 and outer > 0 else 2 * outer - inner
    lower, upper = PARAMETER_LIMITS.get(parameter_short_name(name), (None, None))
    limit = upper if edge == "upper" else lower
    if limit is not None:
        if outer == limit:
            return None
        new = min(new, limit) if edge == "upper" else max(new, limit)
    if all(isinstance(v, (int, np.integer)) for v in numeric):
        rounded = round(new)
        step = 1 if edge == "upper" else -1
        if rounded == outer:
            rounded = outer + step
        if limit is not None and (rounded - limit) * step > 0:
            return None
        return int(rounded)
    return float(f"{new:.6g}")


def edge_review(
    best_params: Sequence[Mapping],
    grid: Mapping[str, Sequence],
    min_folds: int = EDGE_RULE_MIN_FOLDS,
) -> pd.DataFrame:
    """Count how often each reviewed hyperparameter chose an edge of its grid.

    best_params holds the parameters chosen in each outer fold. Only numeric
    grids with at least EDGE_RULE_MIN_GRID_VALUES values are reviewed. An edge
    is extended when it was chosen in at least min_folds folds and it has an
    extension value. Returns one row per parameter and edge.
    """
    rows = []
    for name, values in grid.items():
        numeric = _numeric_values(values)
        if numeric is None or len(numeric) < EDGE_RULE_MIN_GRID_VALUES:
            continue
        chosen = [params[name] for params in best_params if name in params]
        for edge in EDGES:
            edge_value = numeric[0] if edge == "lower" else numeric[-1]
            folds = sum(value == edge_value for value in chosen)
            extension = extension_value(name, numeric, edge)
            rows.append(
                {
                    "parameter": name,
                    "edge": edge,
                    "edge_value": edge_value,
                    "folds_at_edge": folds,
                    "n_folds": len(chosen),
                    "extension_value": extension,
                    "extend": folds >= min_folds and extension is not None,
                }
            )
    columns = [
        "parameter",
        "edge",
        "edge_value",
        "folds_at_edge",
        "n_folds",
        "extension_value",
        "extend",
    ]
    return pd.DataFrame(rows, columns=columns)


def grid_extension(review: pd.DataFrame) -> dict[str, list]:
    """Return the values to add to each grid parameter, from an edge review."""
    extension: dict[str, list] = {}
    for row in review.loc[review["extend"]].itertuples():
        extension.setdefault(row.parameter, []).append(row.extension_value)
    return extension


def chosen_value_counts(best_params: Sequence[Mapping], name: str) -> pd.Series:
    """Return how many outer folds chose each value of one parameter."""
    counts = Counter(params[name] for params in best_params if name in params)
    return pd.Series(dict(counts), name="folds", dtype=int).sort_index(
        key=lambda index: index.map(str)
    )


def most_frequent_value(best_params: Sequence[Mapping], name: str) -> object:
    """Return the value of a parameter chosen in most outer folds.

    Ties are broken by the first value in the order of the folds, so the result
    does not depend on how the values sort.
    """
    chosen = [params[name] for params in best_params if name in params]
    if not chosen:
        raise ValueError(f"No fold chose a value for {name!r}.")
    counts = Counter(chosen)
    top = max(counts.values())
    return next(value for value in chosen if counts[value] == top)


def compare_with_best(
    folds: Mapping[Hashable, pd.DataFrame], alpha: float = SELECTION_ALPHA
) -> pd.DataFrame:
    """Compare every configuration with the one of highest mean QWK.

    Each comparison uses the per-fold QWK differences best − other on the same
    splits and the one-sided Nadeau–Bengio corrected t test. No multiplicity
    adjustment is applied, as the selection rule states. When the differences
    do not vary, the t statistic is undefined: a constant positive difference
    is significant (p = 0) and a null one is not (p = 1).
    """
    mean_qwk = pd.Series({key: frame["qwk"].mean() for key, frame in folds.items()})
    best = mean_qwk.idxmax()
    rows = {}
    for key, frame in folds.items():
        row = {"mean_qwk": mean_qwk[key], "is_best": key == best}
        if key == best:
            row.update(
                mean_difference=0.0,
                corrected_t=np.nan,
                p_value_one_sided=np.nan,
                significant=False,
            )
        else:
            summary = summarize_paired(paired_differences(folds[best], frame))
            p_value = summary["p_value_one_sided"]
            if summary["std_difference"] == 0:
                p_value = 0.0 if summary["mean_difference"] > 0 else 1.0
            row.update(
                mean_difference=summary["mean_difference"],
                corrected_t=summary["corrected_t"],
                p_value_one_sided=p_value,
                significant=bool(p_value < alpha),
            )
        rows[key] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def apply_selection_rule(
    comparison: pd.DataFrame,
    underestimation: Mapping[Hashable, float],
    complexity: Mapping[Hashable, tuple],
    tolerance: float = UNDERESTIMATION_TIE_TOLERANCE,
) -> pd.DataFrame:
    """Apply the three steps of the selection rule and mark the selected configuration.

    1. The configuration of highest mean QWK and those whose difference with it
       is not significant are the candidates.
    2. Among them, those whose under-estimation rate is less than tolerance
       above the lowest are tied.
    3. The tie is broken by the lowest complexity.
    Returns the comparison with the columns of each step.
    """
    table = comparison.copy()
    table["underestimation"] = pd.Series(underestimation)
    table["complexity"] = pd.Series(complexity)
    if table[["underestimation", "complexity"]].isna().any().any():
        raise ValueError(
            "Every configuration needs an under-estimation rate and a complexity."
        )
    table["candidate"] = table["is_best"] | ~table["significant"]
    lowest = table.loc[table["candidate"], "underestimation"].min()
    table["tied_underestimation"] = table["candidate"] & (
        table["underestimation"] - lowest < tolerance
    )
    tied = table.loc[table["tied_underestimation"], "complexity"]
    selected = min(tied.items(), key=lambda item: item[1])[0]
    table["selected"] = [key == selected for key in table.index]
    return table


def selected_configuration(table: pd.DataFrame) -> Hashable:
    """Return the key of the configuration marked as selected."""
    selected = table.index[table["selected"]]
    if len(selected) != 1:
        raise ValueError("Exactly one configuration must be selected.")
    return selected[0]
