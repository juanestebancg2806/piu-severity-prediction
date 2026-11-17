import numpy as np
import pandas as pd
import pytest

from piu_severity.evaluation.metrics import (
    quadratic_weighted_kappa,
    underestimation_rate,
)
from piu_severity.evaluation.screening import referral_metrics
from piu_severity.evaluation.subgroups import (
    ALL_PARTICIPANTS,
    age_group,
    subgroup_metrics_by_repeat,
    subgroup_summary,
)


def test_age_group_limits_are_included():
    ages = pd.Series([5, 12, 13, 17, 18, 22])
    assert list(age_group(ages).astype(str)) == [
        "5-12",
        "5-12",
        "13-17",
        "13-17",
        "18-22",
        "18-22",
    ]
    assert age_group(ages).cat.ordered


@pytest.mark.parametrize("age", [4, 23, 12.5, np.nan])
def test_age_group_rejects_other_ages(age):
    with pytest.raises(ValueError):
        age_group(pd.Series([10, age]))


def _predictions() -> tuple[pd.DataFrame, pd.Series]:
    """Two repeats of eight participants in subgroups a and b."""
    y_true = [0, 1, 2, 3, 0, 0, 1, 2]
    y_pred_by_repeat = [[0, 1, 1, 2, 0, 1, 1, 2], [0, 0, 2, 2, 0, 0, 2, 1]]
    rows = [
        {"repeat": r, "fold": 0, "position": p, "y_true": t, "y_pred": y}
        for r, y_pred in enumerate(y_pred_by_repeat)
        for p, (t, y) in enumerate(zip(y_true, y_pred, strict=True))
    ]
    groups = pd.Series(
        pd.Categorical(["a"] * 4 + ["b"] * 4, categories=["a", "b"], ordered=True)
    )
    return pd.DataFrame(rows), groups


def test_metrics_match_the_rows_of_each_subgroup():
    predictions, groups = _predictions()
    table = subgroup_metrics_by_repeat(predictions, groups)
    assert list(table.index.get_level_values("subgroup").unique()) == [
        ALL_PARTICIPANTS,
        "a",
        "b",
    ]
    rows = predictions[(predictions["repeat"] == 0) & (predictions["position"] >= 4)]
    row = table.loc[("b", 0)]
    assert row["n"] == 4
    assert row["n_moderate_or_severe"] == 1
    assert row["n_severe"] == 0
    assert row["qwk"] == pytest.approx(
        quadratic_weighted_kappa(rows["y_true"], rows["y_pred"])
    )
    assert np.isnan(row["recall_3"])
    assert row["underestimation"] == pytest.approx(
        underestimation_rate(rows["y_true"], rows["y_pred"])
    )
    screening = referral_metrics(rows["y_true"], rows["y_pred"])
    assert row["screening_recall"] == pytest.approx(screening["recall"])
    assert row["referred_share"] == pytest.approx(screening["referred_share"])
    everyone = predictions[predictions["repeat"] == 1]
    assert table.loc[(ALL_PARTICIPANTS, 1), "qwk"] == pytest.approx(
        quadratic_weighted_kappa(everyone["y_true"], everyone["y_pred"])
    )


def test_every_participant_needs_a_subgroup():
    predictions, groups = _predictions()
    with pytest.raises(ValueError, match="needs a subgroup"):
        subgroup_metrics_by_repeat(predictions, groups.iloc[:6])


def test_summary_keeps_sizes_and_aggregates_repeats():
    predictions, groups = _predictions()
    by_repeat = subgroup_metrics_by_repeat(predictions, groups)
    summary = subgroup_summary(by_repeat)
    assert list(summary.index) == [ALL_PARTICIPANTS, "a", "b"]
    assert summary.loc["a", ("n", "")] == 4
    qwk = by_repeat.xs("a", level="subgroup")["qwk"]
    assert summary.loc["a", ("qwk", "mean")] == pytest.approx(qwk.mean())
    assert summary.loc["a", ("qwk", "std")] == pytest.approx(qwk.std())
    changed = by_repeat.copy()
    changed.loc[("a", 1), "n"] = 3
    with pytest.raises(ValueError, match="sizes differ"):
        subgroup_summary(changed)
