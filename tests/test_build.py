import numpy as np
import pandas as pd
import pytest

from piu_severity.config import ACTIGRAPHY_SUMMARY_FILE, RAW_DATA_DIR, TRAIN_FILE
from piu_severity.features.actigraphy import (
    ACTIGRAPHY_FEATURES,
    actigraphy_features,
    is_usable,
    load_actigraphy_summary,
    usable_actigraphy_mask,
)
from piu_severity.features.build import add_paq_total, build_analytic_dataset
from piu_severity.features.groups import PAQ_TOTAL_COLUMN, TARGET_COLUMN


def _summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "valid_days": [10, 3, 4],
            "enmo_day_wear_mean": [0.05, 0.02, 0.07],
            "enmo_night_wear_mean": [0.01, 0.00, np.nan],
            "wear_hours_per_day": [20.0, 5.0, 12.0],
            "enmo_wear_mean": [0.04, 0.01, 0.05],
        },
        index=pd.Index(["a", "b", "c"], name="id"),
    )


def _write_series(series_dir, participant_id):
    hours = np.arange(0, 48, 0.5)
    series = pd.DataFrame(
        {
            "step": np.arange(len(hours), dtype="uint32"),
            "enmo": np.full(len(hours), 0.03, dtype="float32"),
            "non-wear_flag": np.zeros(len(hours), dtype="float32"),
            "time_of_day": ((hours % 24) * 3_600 * 10**9).astype("int64"),
            "relative_date_PCIAT": (hours // 24).astype("float32"),
        }
    )
    path = series_dir / f"id={participant_id}"
    path.mkdir(parents=True)
    series.to_parquet(path / "part-0.parquet")


def test_is_usable_uses_minimum_valid_days():
    assert is_usable(_summary()).tolist() == [True, False, True]


def test_actigraphy_features_keep_usable_series_only():
    features = actigraphy_features(_summary())
    assert features.index.tolist() == ["a", "c"]
    assert features.columns.tolist() == list(ACTIGRAPHY_FEATURES.values())
    assert features.at["a", "Actigraphy-ENMO_Day"] == pytest.approx(0.05)


def test_add_paq_total_averages_available_totals():
    data = pd.DataFrame(
        {
            "PAQ_A-PAQ_A_Total": [2.0, np.nan, np.nan],
            "PAQ_C-PAQ_C_Total": [4.0, 3.0, np.nan],
        }
    )
    result = add_paq_total(data)
    assert result[PAQ_TOTAL_COLUMN].iloc[0] == pytest.approx(3.0)
    assert result[PAQ_TOTAL_COLUMN].iloc[1] == pytest.approx(3.0)
    assert np.isnan(result[PAQ_TOTAL_COLUMN].iloc[2])
    assert PAQ_TOTAL_COLUMN not in data.columns


def test_build_analytic_dataset_marks_unusable_series_as_missing():
    cleaned = pd.DataFrame(
        {
            "id": ["c", "b", "z", "a"],
            "PAQ_A-PAQ_A_Total": [1.0, 2.0, 3.0, 4.0],
            "PAQ_C-PAQ_C_Total": [np.nan] * 4,
        }
    )
    result = build_analytic_dataset(cleaned, _summary())
    assert result["id"].tolist() == ["c", "b", "z", "a"]
    assert len(result) == len(cleaned)
    assert result["Actigraphy-ENMO_Day"].tolist()[0] == pytest.approx(0.07)
    assert result["Actigraphy-ENMO_Day"].isna().tolist() == [False, True, True, False]


def test_load_actigraphy_summary_reuses_matching_cache(tmp_path):
    series_dir = tmp_path / "series"
    for participant_id in ("p1", "p2"):
        _write_series(series_dir, participant_id)
    cache_path = tmp_path / "interim" / "summary.parquet"

    summary = load_actigraphy_summary(["p2", "p1"], cache_path, series_dir)
    assert summary.index.tolist() == ["p1", "p2"]
    assert summary.at["p1", "recorded_days"] == 2
    assert cache_path.is_file()

    _write_series(series_dir, "p3")
    cached = load_actigraphy_summary(["p1", "p2"], cache_path, series_dir)
    pd.testing.assert_frame_equal(cached, summary)

    extended = load_actigraphy_summary(["p1", "p2", "p3"], cache_path, series_dir)
    assert extended.index.tolist() == ["p1", "p2", "p3"]


def test_usable_actigraphy_mask_accepts_a_partially_observed_series():
    data = pd.DataFrame(
        {
            "Basic_Demos-Age": [10.0, 11.0, 12.0],
            "Actigraphy-ENMO_Day": [0.05, np.nan, 0.07],
            "Actigraphy-ENMO_Night": [0.01, np.nan, np.nan],
            "Actigraphy-Wear_Hours": [20.0, np.nan, 12.0],
        }
    )
    assert usable_actigraphy_mask(data).tolist() == [True, False, True]


@pytest.mark.skipif(
    not (RAW_DATA_DIR / TRAIN_FILE).is_file() or not ACTIGRAPHY_SUMMARY_FILE.is_file(),
    reason="raw data or actigraphy summary not available",
)
def test_actigraphy_subpopulation_has_823_participants_and_6_severe():
    from piu_severity.data.clean import clean_values
    from piu_severity.data.load import list_actigraphy_ids, load_train

    train = load_train()
    labeled = train[train[TARGET_COLUMN].notna()].reset_index(drop=True)
    data = build_analytic_dataset(
        clean_values(labeled), load_actigraphy_summary(list_actigraphy_ids())
    )
    mask = usable_actigraphy_mask(data)
    assert int(mask.sum()) == 823
    assert int((data.loc[mask, TARGET_COLUMN] == 3).sum()) == 6