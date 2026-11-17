"""Aggregate summary tables of the results that support the recommendations.

Each metric is reported with its mean across repeats, its standard deviation
across repeats (stability against the partition of the data), an optional
interval (uncertainty, for example a bootstrap over participants) and the
number of participants it is computed on. The tables hold no participant data.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd

SUMMARY_COLUMNS: tuple[str, ...] = (
    "mean",
    "std_between_repeats",
    "interval_low",
    "interval_high",
    "n",
)


def metric_summary(
    by_repeat: pd.DataFrame,
    intervals: Mapping[str, tuple[float, float]] | None = None,
    sizes: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    """Return one row per metric with its mean, deviation, interval and size.

    by_repeat holds one row per repeat and one column per metric. Metrics
    without an interval or a size get NaN or <NA> in those columns.
    """
    intervals = dict(intervals or {})
    sizes = dict(sizes or {})
    unknown = (set(intervals) | set(sizes)) - set(by_repeat.columns)
    if unknown:
        raise KeyError(f"Metrics not in by_repeat: {sorted(unknown)}")
    for metric, (low, high) in intervals.items():
        if low > high:
            raise ValueError(f"{metric!r}: interval limits are reversed.")
    metrics = list(by_repeat.columns)
    table = pd.DataFrame(
        {
            "mean": by_repeat.mean(),
            "std_between_repeats": by_repeat.std(ddof=1),
            "interval_low": [intervals.get(m, (np.nan, np.nan))[0] for m in metrics],
            "interval_high": [intervals.get(m, (np.nan, np.nan))[1] for m in metrics],
            "n": pd.array([sizes.get(m) for m in metrics], dtype="Int64"),
        },
        index=pd.Index(metrics, name="metric"),
    )
    return table[list(SUMMARY_COLUMNS)]
