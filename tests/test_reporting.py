import numpy as np
import pandas as pd
import pytest

from piu_severity.evaluation.reporting import SUMMARY_COLUMNS, metric_summary


def _by_repeat() -> pd.DataFrame:
    return pd.DataFrame(
        {"qwk": [0.30, 0.32, 0.34], "recall": [0.5, 0.6, 0.7]},
        index=pd.Index([0, 1, 2], name="repeat"),
    )


def test_summary_has_one_row_per_metric_in_order():
    table = metric_summary(_by_repeat())
    assert list(table.index) == ["qwk", "recall"]
    assert tuple(table.columns) == SUMMARY_COLUMNS
    assert table.loc["qwk", "mean"] == pytest.approx(0.32)
    assert table.loc["recall", "std_between_repeats"] == pytest.approx(0.1)


def test_missing_intervals_and_sizes_are_empty():
    table = metric_summary(
        _by_repeat(), intervals={"recall": (0.4, 0.8)}, sizes={"recall": 412}
    )
    assert table.loc["recall", "interval_low"] == pytest.approx(0.4)
    assert table.loc["recall", "interval_high"] == pytest.approx(0.8)
    assert table.loc["recall", "n"] == 412
    assert np.isnan(table.loc["qwk", "interval_low"])
    assert pd.isna(table.loc["qwk", "n"])


def test_rejects_unknown_metrics_and_reversed_intervals():
    with pytest.raises(KeyError, match="not in by_repeat"):
        metric_summary(_by_repeat(), sizes={"precision": 10})
    with pytest.raises(ValueError, match="reversed"):
        metric_summary(_by_repeat(), intervals={"qwk": (0.4, 0.3)})
