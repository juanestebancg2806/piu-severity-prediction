"""Errors of the selected configuration by sex and by age group.

The comparison is descriptive: each subgroup reports its size, the QWK, the
recall of each level, the under-estimation rate and the screening recall and
referred share. Subgroups are small (the oldest group has 84 participants), so
every metric is computed on the predictions of each repeat, where every
participant is predicted once, and not per outer fold.
"""

import numpy as np
import pandas as pd

from piu_severity.evaluation.metrics import (
    SII_LEVELS,
    UNDERESTIMATION_LEVELS,
    quadratic_weighted_kappa,
    recall_by_level,
    underestimation_rate,
)
from piu_severity.evaluation.screening import REFERRAL_LEVEL, referral_metrics

# Age groups of the formulation, in years, both limits included.
AGE_GROUPS: dict[str, tuple[int, int]] = {
    "5-12": (5, 12),
    "13-17": (13, 17),
    "18-22": (18, 22),
}
ALL_PARTICIPANTS = "all"
SIZE_COLUMNS: tuple[str, ...] = ("n", "n_moderate_or_severe", "n_severe")


def age_group(ages: pd.Series) -> pd.Series:
    """Return the ordered age group of each age.

    Ages must be whole years within the groups; any other value raises, so
    that no participant is assigned to a group silently.
    """
    values = ages.to_numpy(dtype=float)
    if np.isnan(values).any() or not np.array_equal(values, np.round(values)):
        raise ValueError("Ages must be whole years without missing values.")
    labels = pd.Series(pd.NA, index=ages.index, dtype="object")
    for label, (low, high) in AGE_GROUPS.items():
        labels[(values >= low) & (values <= high)] = label
    if labels.isna().any():
        raise ValueError(
            f"Ages outside the groups: {sorted(set(values[labels.isna()]))}"
        )
    return pd.Series(
        pd.Categorical(labels, categories=list(AGE_GROUPS), ordered=True),
        index=ages.index,
        name="age_group",
    )


def _group_metrics(
    y_true: pd.Series, y_pred: pd.Series, referral_level: int
) -> dict[str, float]:
    """Return the sizes and metrics of one subgroup in one repeat."""
    recall = recall_by_level(y_true, y_pred)
    screening = referral_metrics(y_true, y_pred, referral_level)
    return {
        "n": len(y_true),
        "n_moderate_or_severe": int(np.isin(y_true, UNDERESTIMATION_LEVELS).sum()),
        "n_severe": int((y_true == SII_LEVELS[-1]).sum()),
        "qwk": quadratic_weighted_kappa(y_true, y_pred),
        **{f"recall_{level}": float(value) for level, value in recall.items()},
        "underestimation": underestimation_rate(y_true, y_pred),
        "screening_recall": screening["recall"],
        "referred_share": screening["referred_share"],
    }


def subgroup_metrics_by_repeat(
    predictions: pd.DataFrame,
    groups: pd.Series,
    referral_level: int = REFERRAL_LEVEL,
) -> pd.DataFrame:
    """Return the metrics of each subgroup in each repeat, plus all participants.

    groups gives the subgroup of each participant, indexed by the position in
    the predictions. The rows of all participants are computed the same way, so
    that every subgroup is compared on the same scale.
    """
    labels = predictions["position"].map(groups)
    if labels.isna().any():
        raise ValueError("Every predicted participant needs a subgroup.")
    order = (
        list(groups.cat.categories)
        if isinstance(groups.dtype, pd.CategoricalDtype)
        else sorted(labels.unique())
    )
    labelled = predictions.assign(subgroup=labels.to_numpy())
    rows = []
    for repeat, group in labelled.groupby("repeat"):
        subsets = [(ALL_PARTICIPANTS, group)] + [
            (label, group[group["subgroup"] == label]) for label in order
        ]
        for label, subset in subsets:
            if subset.empty:
                continue
            rows.append(
                {
                    "subgroup": label,
                    "repeat": repeat,
                    **_group_metrics(
                        subset["y_true"], subset["y_pred"], referral_level
                    ),
                }
            )
    table = pd.DataFrame(rows).set_index(["subgroup", "repeat"])
    return table.reindex([ALL_PARTICIPANTS, *order], level="subgroup")


def subgroup_summary(by_repeat: pd.DataFrame) -> pd.DataFrame:
    """Return the sizes and the mean and deviation across repeats of each metric.

    The sizes do not change between repeats; the columns of the metrics hold
    (metric, "mean") and (metric, "std") pairs.
    """
    grouped = by_repeat.groupby(level="subgroup", sort=False)
    sizes = grouped[list(SIZE_COLUMNS)]
    if (sizes.nunique() > 1).any().any():
        raise ValueError("Subgroup sizes differ between repeats.")
    metrics = by_repeat.drop(columns=list(SIZE_COLUMNS))
    summary = metrics.groupby(level="subgroup", sort=False).agg(["mean", "std"])
    sizes_first = sizes.first()
    sizes_first.columns = pd.MultiIndex.from_tuples(
        [(column, "") for column in sizes_first.columns]
    )
    return pd.concat([sizes_first, summary], axis="columns")
