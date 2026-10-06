"""Per-participant summaries of actigraphy series."""

import numpy as np
import pandas as pd

ACTIGRAPHY_COLUMNS: list[str] = [
    "step",
    "enmo",
    "non-wear_flag",
    "time_of_day",
    "relative_date_PCIAT",
]
NANOSECONDS_PER_HOUR = 3_600 * 10**9
MIN_WEAR_HOURS_PER_DAY = 10
NIGHT_HOURS = (0, 6)
DAY_HOURS = (8, 21)


def _hour_window(hour: pd.Series, window: tuple[int, int]) -> pd.Series:
    """Return True for the records whose hour falls inside the window [start, end)."""
    return (hour >= window[0]) & (hour < window[1])


def summarize_actigraphy(series: pd.DataFrame) -> dict[str, float]:
    """Summarize recording length, wear time and movement intensity of one participant."""
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