import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from piu_severity.evaluation.metrics import (
    SEVERE_LEVELS,
    SII_LEVELS,
    as_levels,
    level_confusion_matrix,
    quadratic_weighted_kappa,
    qwk_scorer,
    recall_by_level,
    severity_metrics,
    underestimation_rate,
)


def test_qwk_perfect_agreement():
    y = [0, 1, 2, 3, 1, 0]
    assert quadratic_weighted_kappa(y, y) == pytest.approx(1.0)


def test_qwk_majority_class_is_zero():
    y_true = [0, 0, 0, 1, 2, 3]
    y_pred = [0] * len(y_true)
    assert quadratic_weighted_kappa(y_true, y_pred) == pytest.approx(0.0, abs=1e-12)


def test_qwk_hand_computed():
    # Observed weighted disagreement: 1. Expected under independence: 12.
    assert quadratic_weighted_kappa([0, 1, 2, 3], [0, 1, 3, 3]) == pytest.approx(
        11 / 12
    )


def test_float_labels_are_accepted():
    y_true = pd.Series([0.0, 1.0, 2.0, 3.0])
    assert quadratic_weighted_kappa(y_true, [0, 1, 3, 3]) == pytest.approx(11 / 12)


@pytest.mark.parametrize("values", [[0, 1, np.nan], [0, 1.5, 2]])
def test_as_levels_rejects_missing_and_non_integer(values):
    with pytest.raises(ValueError):
        as_levels(values)


def test_confusion_matrix_keeps_all_levels():
    matrix = level_confusion_matrix([0, 0, 1], [0, 1, 1])
    assert matrix.shape == (len(SII_LEVELS), len(SII_LEVELS))
    assert matrix.loc[0, 0] == 1
    assert matrix.loc[0, 1] == 1
    assert matrix.loc[1, 1] == 1
    assert matrix.loc[3].sum() == 0
    assert matrix.to_numpy().sum() == 3


def test_recall_by_level_with_absent_level():
    recall = recall_by_level([0, 0, 1, 1, 2], [0, 1, 1, 1, 1])
    assert recall.loc[0] == pytest.approx(0.5)
    assert recall.loc[1] == pytest.approx(1.0)
    assert recall.loc[2] == pytest.approx(0.0)
    assert np.isnan(recall.loc[3])


def test_underestimation_counts_only_lower_predictions():
    y_true = [2, 2, 2, 3, 3, 1]
    y_pred = [2, 1, 3, 3, 0, 0]
    assert underestimation_rate(y_true, y_pred) == pytest.approx(2 / 5)
    assert underestimation_rate(y_true, y_pred, SEVERE_LEVELS) == pytest.approx(0.5)


def test_underestimation_is_nan_without_cases():
    assert np.isnan(underestimation_rate([0, 1, 1], [0, 0, 1]))


def test_severity_metrics_keys():
    metrics = severity_metrics([0, 1, 2, 3], [0, 1, 3, 3])
    assert set(metrics) == {
        "qwk",
        "recall_0",
        "recall_1",
        "recall_2",
        "recall_3",
        "underestimation",
        "underestimation_severe",
    }
    assert metrics["qwk"] == pytest.approx(11 / 12)
    assert metrics["underestimation"] == pytest.approx(0.0)


def test_qwk_scorer_with_majority_classifier():
    y = np.array([0, 0, 0, 1, 2, 3])
    X = np.zeros((len(y), 1))
    model = DummyClassifier(strategy="most_frequent").fit(X, y)
    assert qwk_scorer(model, X, y) == pytest.approx(0.0, abs=1e-12)
