"""Assemble the analytic dataset from the cleaned tabular data and the actigraphy."""

import pandas as pd

from piu_severity.features.actigraphy import MIN_VALID_DAYS, actigraphy_features
from piu_severity.features.groups import ID_COLUMN, PAQ_SOURCE_COLUMNS, PAQ_TOTAL_COLUMN


def add_paq_total(data: pd.DataFrame) -> pd.DataFrame:
    """Add the PAQ total, the mean of the PAQ-A and PAQ-C totals that are present."""
    result = data.copy()
    result[PAQ_TOTAL_COLUMN] = data[list(PAQ_SOURCE_COLUMNS)].mean(axis="columns")
    return result


def build_analytic_dataset(
    cleaned: pd.DataFrame,
    actigraphy_summary: pd.DataFrame,
    min_valid_days: int = MIN_VALID_DAYS,
) -> pd.DataFrame:
    """Add the PAQ total and the actigraphy features to data processed by clean_values.

    Participants without a usable actigraphy series get missing actigraphy features.
    """
    features = actigraphy_features(actigraphy_summary, min_valid_days).rename_axis(
        ID_COLUMN
    )
    return add_paq_total(cleaned).join(features, on=ID_COLUMN)
