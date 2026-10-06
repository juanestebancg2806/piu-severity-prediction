import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from piu_severity.evaluation.validation import (
    aggregate_repeats,
    mean_confusion_matrix,
    outer_splits,
    paired_differences,
    permutation_null,
    permutation_p_value,
    run_nested_cv,
    split_level_counts,
    summarize_by_repeat,
    summarize_paired,
)

CLASS_SIZES = (60, 30, 15, 10)


def _labels(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.permutation(np.repeat(np.arange(len(CLASS_SIZES)), CLASS_SIZES))


def _features(y: np.ndarray, noise: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"x": y + rng.normal(scale=noise, size=len(y))})


def test_each_participant_is_evaluated_once_per_repeat():
    y = _labels()
    splits = outer_splits(y, n_splits=5, n_repeats=3)
    assert len(splits) == 15
    for repeat in range(3):
        tested = np.concatenate([s.test for s in splits if s.repeat == repeat])
        assert np.array_equal(np.sort(tested), np.arange(len(y)))


def test_train_and_test_are_disjoint():
    for split in outer_splits(_labels(), n_splits=5, n_repeats=2):
        assert not set(split.train) & set(split.test)


def test_folds_are_stratified():
    y = _labels()
    counts = split_level_counts(outer_splits(y, n_splits=5, n_repeats=2), y)
    assert counts.shape == (10, 4)
    for level, size in enumerate(CLASS_SIZES):
        assert counts[level].between(size // 5, -(-size // 5)).all()


def test_first_repeats_do_not_depend_on_the_number_of_repeats():
    y = _labels()
    short = outer_splits(y, n_repeats=3)
    long = outer_splits(y, n_repeats=5)
    for a, b in zip(short, long[: len(short)], strict=True):
        assert np.array_equal(a.test, b.test)


def test_run_nested_cv_with_majority_classifier():
    y = _labels()
    X = _features(y, noise=1.0)
    splits = outer_splits(y, n_splits=5, n_repeats=2)
    folds, predictions = run_nested_cv(
        DummyClassifier(strategy="most_frequent"), X, y, splits
    )
    assert len(folds) == 10
    assert folds["qwk"].abs().max() == pytest.approx(0.0, abs=1e-12)
    assert len(predictions) == 2 * len(y)
    assert (predictions["y_pred"] == 0).all()

    summary = summarize_by_repeat(folds, predictions)
    assert summary["recall_0"].tolist() == [1.0, 1.0]
    assert summary["recall_3"].tolist() == [0.0, 0.0]
    assert summary["underestimation"].tolist() == [1.0, 1.0]
    assert set(aggregate_repeats(summary).columns) == {"mean", "std"}


def test_run_nested_cv_with_informative_feature():
    y = _labels()
    X = _features(y, noise=0.2)
    splits = outer_splits(y, n_splits=5, n_repeats=1)
    folds, _ = run_nested_cv(LogisticRegression(), X, y, splits)
    assert folds["qwk"].mean() > 0.8


def test_mean_confusion_matrix_rows_are_sensitivities():
    y = _labels()
    X = _features(y, noise=0.5)
    splits = outer_splits(y, n_splits=5, n_repeats=2)
    folds, predictions = run_nested_cv(LogisticRegression(), X, y, splits)
    matrix = mean_confusion_matrix(predictions)
    assert matrix.sum(axis="columns").to_numpy() == pytest.approx(np.ones(4))
    summary = summarize_by_repeat(folds, predictions)
    assert np.diag(matrix) == pytest.approx(
        summary[[f"recall_{level}" for level in range(4)]].mean().to_numpy()
    )


def test_paired_differences_reject_different_splits():
    y = _labels()
    X = _features(y, noise=1.0)
    model = DummyClassifier(strategy="most_frequent")
    folds_a, _ = run_nested_cv(model, X, y, outer_splits(y, n_repeats=1, seed=0))
    folds_b, _ = run_nested_cv(model, X, y, outer_splits(y, n_repeats=1, seed=1))
    with pytest.raises(ValueError):
        paired_differences(folds_a, folds_b)


def test_paired_differences_on_same_splits():
    y = _labels()
    X = _features(y, noise=0.5)
    splits = outer_splits(y, n_repeats=1)
    folds_a, _ = run_nested_cv(LogisticRegression(), X, y, splits)
    folds_b, _ = run_nested_cv(DummyClassifier(), X, y, splits)
    differences = paired_differences(folds_a, folds_b)
    assert differences["difference"].to_numpy() == pytest.approx(
        folds_a["qwk"].to_numpy() - folds_b["qwk"].to_numpy()
    )


def test_summarize_paired_applies_nadeau_bengio_correction():
    differences = pd.DataFrame(
        {
            "difference": [0.1, 0.2, 0.3, 0.4, 0.5],
            "n_train": [80] * 5,
            "n_test": [20] * 5,
        }
    )
    result = summarize_paired(differences)
    assert result["mean_difference"] == pytest.approx(0.3)
    assert result["folds_improved"] == 5
    # Variance factor 1/5 + 20/80 = 0.45; t = 0.3 / sqrt(0.45 * 0.025) = 2 * sqrt(2).
    assert result["corrected_t"] == pytest.approx(2 * np.sqrt(2))
    assert 0 < result["p_value_one_sided"] < 0.05


def test_permutation_null_is_centered_near_zero():
    y = _labels()
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"x": rng.normal(size=len(y))})
    null_scores = permutation_null(LogisticRegression(), X, y, n_permutations=5)
    assert null_scores.shape == (5,)
    assert abs(null_scores.mean()) < 0.2


def test_permutation_p_value():
    assert permutation_p_value(0.5, np.array([0.1, 0.6])) == pytest.approx(2 / 3)
