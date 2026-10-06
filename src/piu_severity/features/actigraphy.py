"""Per-participant summaries of actigraphy series and the features derived from them."""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from piu_severity.config import ACTIGRAPHY_SUMMARY_FILE, ACTIGRAPHY_TRAIN_DIR
from piu_severity.data.load import load_actigraphy

ACTIGRAPHY_COLUMNS: list[str] = [
    "step",
    "enmo",
    "non-wear_flag",
    "time_of_day",
    "relative_date_PCIAT",
]
NANOSECONDS_PER_HOUR = 3_600 * 10**9
MIN_WEAR_HOURS_PER_DAY = 10
MIN_VALID_DAYS = 4
NIGHT_HOURS = (0, 6)
DAY_HOURS = (8, 21)

SUMMARY_KEYS: list[str] = [
    "sampling_interval_s",
    "recorded_days",
    "recorded_hours",
    "nonwear_fraction",
    "valid_days",
    "wear_hours_per_day",
    "enmo_wear_mean",
    "enmo_day_wear_mean",
    "enmo_night_wear_mean",
    "median_days_from_pciat",
]

ACTIGRAPHY_FEATURES: dict[str, str] = {
    "enmo_day_wear_mean": "Actigraphy-ENMO_Day",
    "enmo_night_wear_mean": "Actigraphy-ENMO_Night",
    "wear_hours_per_day": "Actigraphy-Wear_Hours",
}


def _hour_window(hour: pd.Series, window: tuple[int, int]) -> pd.Series:
    """Return True for the records whose hour falls inside the window [start, end)."""
    return (hour >= window[0]) & (hour < window[1])


def summarize_actigraphy(series: pd.DataFrame) -> dict[str, float]:
    """Summarize recording length, wear time and movement intensity of a series."""
    if len(series) < 2:
        return {}
    series = series.sort_values("step")
    interval_seconds = series["time_of_day"].diff().median() / 1e9
    wearing = series["non-wear_flag"] == 0
    day = series["relative_date_PCIAT"]
    wear_hours_per_day = wearing.groupby(day).sum() * interval_seconds / 3_600
    hour = series["time_of_day"] / NANOSECONDS_PER_HOUR
    night = _hour_window(hour, NIGHT_HOURS)
    daytime = _hour_window(hour, DAY_HOURS)
    return {
        "sampling_interval_s": float(interval_seconds),
        "recorded_days": int(day.nunique()),
        "recorded_hours": float(len(series) * interval_seconds / 3_600),
        "nonwear_fraction": float(1 - wearing.mean()),
        "valid_days": int((wear_hours_per_day >= MIN_WEAR_HOURS_PER_DAY).sum()),
        "wear_hours_per_day": float(wear_hours_per_day.mean()),
        "enmo_wear_mean": float(series.loc[wearing, "enmo"].mean()),
        "enmo_day_wear_mean": float(series.loc[wearing & daytime, "enmo"].mean()),
        "enmo_night_wear_mean": float(series.loc[wearing & night, "enmo"].mean()),
        "median_days_from_pciat": float(np.abs(day.median())),
    }


def summarize_participants(
    participant_ids: Iterable[str], series_dir: Path = ACTIGRAPHY_TRAIN_DIR
) -> pd.DataFrame:
    """Summarize the series of several participants, one row per participant id."""
    rows = [
        {
            "id": participant_id,
            **summarize_actigraphy(
                load_actigraphy(participant_id, ACTIGRAPHY_COLUMNS, series_dir)
            ),
        }
        for participant_id in sorted(set(participant_ids))
    ]
    return pd.DataFrame(rows, columns=["id", *SUMMARY_KEYS]).set_index("id")


def load_actigraphy_summary(
    participant_ids: Iterable[str],
    cache_path: Path = ACTIGRAPHY_SUMMARY_FILE,
    series_dir: Path = ACTIGRAPHY_TRAIN_DIR,
) -> pd.DataFrame:
    """Return the participants' summaries, reusing the cache when it matches them."""
    expected_ids = set(participant_ids)
    if cache_path.is_file():
        cached = pd.read_parquet(cache_path)
        same_ids = set(cached.index) == expected_ids
        if same_ids and set(SUMMARY_KEYS) <= set(cached.columns):
            return cached
    summary = summarize_participants(expected_ids, series_dir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(cache_path)
    return summary


def is_usable(summary: pd.DataFrame, min_valid_days: int = MIN_VALID_DAYS) -> pd.Series:
    """Return True for the series with enough days of sufficient wear time."""
    return summary["valid_days"] >= min_valid_days


def actigraphy_features(
    summary: pd.DataFrame, min_valid_days: int = MIN_VALID_DAYS
) -> pd.DataFrame:
    """Return the actigraphy features of the participants with a usable series."""
    usable = is_usable(summary, min_valid_days)
    return summary.loc[usable, list(ACTIGRAPHY_FEATURES)].rename(
        columns=ACTIGRAPHY_FEATURES
    )
