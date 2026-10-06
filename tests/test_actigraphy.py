import numpy as np
import pandas as pd
import pytest

from piu_severity.data.load import list_actigraphy_ids, load_actigraphy
from piu_severity.features.actigraphy import summarize_actigraphy

INTERVAL_NS = 5 * 10**9
ROWS_PER_HOUR = 720


def _synthetic_series():
    """Two days of 12 hours each; the second day is worn only 6 hours."""
    hours_12 = 12 * ROWS_PER_HOUR
    time_of_day = np.arange(hours_12) * INTERVAL_NS
    day_0 = pd.DataFrame(
        {
            "time_of_day": time_of_day,
            "relative_date_PCIAT": 0,
            "non-wear_flag": 0,
            "enmo": 0.02,
        }
    )
    day_1 = day_0.assign(relative_date_PCIAT=1)
    day_1.loc[day_1.index >= 6 * ROWS_PER_HOUR, "non-wear_flag"] = 1
    series = pd.concat([day_0, day_1], ignore_index=True)
    series["step"] = np.arange(len(series))
    return series


def test_summarize_actigraphy():
    summary = summarize_actigraphy(_synthetic_series())
    assert summary["sampling_interval_s"] == pytest.approx(5)
    assert summary["recorded_days"] == 2
    assert summary["recorded_hours"] == pytest.approx(24)
    assert summary["nonwear_fraction"] == pytest.approx(0.25)
    assert summary["valid_days"] == 1
    assert summary["wear_hours_per_day"] == pytest.approx(9)
    assert summary["enmo_wear_mean"] == pytest.approx(0.02)
    assert summary["enmo_day_wear_mean"] == pytest.approx(0.02)


def test_summarize_actigraphy_too_short():
    assert summarize_actigraphy(_synthetic_series().head(1)) == {}


def test_list_and_load_actigraphy(tmp_path):
    series_dir = tmp_path / "series_train.parquet"
    participant_dir = series_dir / "id=abc123"
    participant_dir.mkdir(parents=True)
    _synthetic_series().to_parquet(participant_dir / "part-0.parquet")
    (series_dir / "otra_carpeta").mkdir()

    assert list_actigraphy_ids(series_dir) == ["abc123"]
    loaded = load_actigraphy("abc123", columns=["enmo"], series_dir=series_dir)
    assert list(loaded.columns) == ["enmo"]
    with pytest.raises(FileNotFoundError):
        load_actigraphy("no_existe", series_dir=series_dir)
