import numpy as np
import pandas as pd

from piu_severity.data.clean import (
    cap_protocol_maxima,
    clean_values,
    invalid_value_mask,
    mask_invalid_values,
)


def _frame():
    return pd.DataFrame(
        {
            "BIA-BIA_Fat": [20.0, -5.0, 75.0, np.nan],
            "BIA-BIA_BMC": [3.0, 4000.0, 0.0, 2.5],
            "FGC-FGC_TL": [8.0, 12.0, 15.0, -1.0],
            "Physical-Diastolic_BP": [70.0, 90.0, 60.0, np.nan],
            "Physical-Systolic_BP": [110.0, 85.0, 100.0, 120.0],
            "Basic_Demos-Enroll_Season": ["Fall", "Spring", "Summer", "Winter"],
        }
    )


def _bia_frame():
    return pd.DataFrame(
        {
            "BIA-Season": ["Fall", "Fall", "Spring"],
            "BIA-BIA_Fat": [20.0, -5.0, 15.0],
            "BIA-BIA_FFM": [60.0, 8000.0, 55.0],
            "BIA-BIA_TBW": [45.0, np.nan, 40.0],
            "Physical-Weight": [90.0, 95.0, 80.0],
        }
    )


def test_positive_and_range_rules():
    mask = invalid_value_mask(_frame())
    assert mask["BIA-BIA_Fat"].tolist() == [False, True, True, False]
    assert mask["BIA-BIA_BMC"].tolist() == [False, True, True, False]


def test_values_above_protocol_maximum_are_not_invalid():
    mask = invalid_value_mask(_frame())
    assert mask["FGC-FGC_TL"].tolist() == [False, False, False, True]


def test_inverted_blood_pressure_marks_both_columns():
    mask = invalid_value_mask(_frame())
    assert mask["Physical-Diastolic_BP"].tolist() == [False, True, False, False]
    assert mask["Physical-Systolic_BP"].tolist() == [False, True, False, False]


def test_columns_without_rules_are_untouched():
    mask = invalid_value_mask(_frame())
    assert not mask["Basic_Demos-Enroll_Season"].any()


def test_missing_rule_columns_are_ignored():
    data = pd.DataFrame({"Basic_Demos-Age": [10, 12]})
    assert not invalid_value_mask(data).any().any()


def test_invalid_bia_value_invalidates_the_whole_record():
    mask = invalid_value_mask(_bia_frame())
    assert mask["BIA-BIA_FFM"].tolist() == [False, True, False]
    assert mask["BIA-BIA_Fat"].tolist() == [False, True, False]
    assert mask["BIA-BIA_TBW"].tolist() == [False, False, False]
    assert not mask["BIA-Season"].any()
    assert not mask["Physical-Weight"].any()


def test_record_propagation_can_be_disabled():
    mask = invalid_value_mask(_bia_frame(), propagate_records=False)
    assert mask["BIA-BIA_Fat"].tolist() == [False, True, False]
    assert mask["BIA-BIA_FFM"].tolist() == [False, False, False]


def test_mask_invalid_values_returns_copy():
    data = _frame()
    cleaned = mask_invalid_values(data)
    assert np.isnan(cleaned.loc[1, "BIA-BIA_Fat"])
    assert data.loc[1, "BIA-BIA_Fat"] == -5.0
    assert cleaned["Basic_Demos-Enroll_Season"].equals(data["Basic_Demos-Enroll_Season"])


def test_cap_protocol_maxima_clips_without_modifying_input():
    data = _frame()
    capped = cap_protocol_maxima(data)
    assert capped["FGC-FGC_TL"].tolist() == [8.0, 12.0, 12.0, -1.0]
    assert data.loc[2, "FGC-FGC_TL"] == 15.0


def test_clean_values_clips_then_masks():
    cleaned = clean_values(_frame())
    assert cleaned.loc[2, "FGC-FGC_TL"] == 12.0
    assert np.isnan(cleaned.loc[3, "FGC-FGC_TL"])