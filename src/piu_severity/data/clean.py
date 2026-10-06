"""Rules for implausible values and protocol maxima of the physical measures."""

import math

import pandas as pd

POSITIVE_COLUMNS: tuple[str, ...] = (
    "Physical-Height",
    "Physical-Weight",
    "Physical-BMI",
    "FGC-FGC_GSD",
    "FGC-FGC_GSND",
    "BIA-BIA_BMC",
    "BIA-BIA_BMI",
    "BIA-BIA_BMR",
    "BIA-BIA_DEE",
    "BIA-BIA_ECW",
    "BIA-BIA_FFM",
    "BIA-BIA_FFMI",
    "BIA-BIA_FMI",
    "BIA-BIA_Fat",
    "BIA-BIA_ICW",
    "BIA-BIA_LDM",
    "BIA-BIA_LST",
    "BIA-BIA_SMM",
    "BIA-BIA_TBW",
)

VALUE_RANGES: dict[str, tuple[float, float]] = {
    "Physical-Height": (30, 84),
    "Physical-Weight": (20, 400),
    "Physical-BMI": (10, 70),
    "Physical-Diastolic_BP": (30, 130),
    "Physical-Systolic_BP": (60, 220),
    "Physical-HeartRate": (40, 200),
    "FGC-FGC_CU": (0, math.inf),
    "FGC-FGC_PU": (0, math.inf),
    "FGC-FGC_SRL": (0, math.inf),
    "FGC-FGC_SRR": (0, math.inf),
    "FGC-FGC_TL": (0, math.inf),
    "FGC-FGC_GSD": (0, 100),
    "FGC-FGC_GSND": (0, 100),
    "BIA-BIA_BMC": (0, 20),
    "BIA-BIA_BMI": (10, 70),
    "BIA-BIA_Fat": (0, 60),
    "BIA-BIA_BMR": (500, 4000),
    "BIA-BIA_DEE": (500, 8000),
    "CGAS-CGAS_Score": (1, 100),
    "SDS-SDS_Total_Raw": (26, 130),
    "PAQ_A-PAQ_A_Total": (1, 5),
    "PAQ_C-PAQ_C_Total": (1, 5),
}

PROTOCOL_MAXIMA: dict[str, float] = {
    "FGC-FGC_CU": 75,
    "FGC-FGC_PU": 75,
    "FGC-FGC_SRL": 12,
    "FGC-FGC_SRR": 12,
    "FGC-FGC_TL": 12,
}

BLOOD_PRESSURE_COLUMNS: tuple[str, str] = (
    "Physical-Diastolic_BP",
    "Physical-Systolic_BP",
)

RECORD_LEVEL_PREFIXES: tuple[str, ...] = ("BIA-",)
SEASON_SUFFIX = "-Season"


def invalid_value_mask(
    data: pd.DataFrame, propagate_records: bool = True
) -> pd.DataFrame:
    """Return a boolean frame that marks implausible values; missing values are never marked.

    With propagate_records, one implausible value of an instrument listed in
    RECORD_LEVEL_PREFIXES marks every recorded value of that instrument for the participant.
    """
    mask = pd.DataFrame(False, index=data.index, columns=data.columns)
    for column in POSITIVE_COLUMNS:
        if column in data.columns:
            mask[column] |= data[column] <= 0
    for column, (low, high) in VALUE_RANGES.items():
        if column in data.columns:
            mask[column] |= (data[column] < low) | (data[column] > high)
    diastolic, systolic = BLOOD_PRESSURE_COLUMNS
    if diastolic in data.columns and systolic in data.columns:
        inverted = data[diastolic] >= data[systolic]
        mask[diastolic] |= inverted
        mask[systolic] |= inverted
    if propagate_records:
        for prefix in RECORD_LEVEL_PREFIXES:
            columns = [
                c
                for c in data.columns
                if c.startswith(prefix) and not c.endswith(SEASON_SUFFIX)
            ]
            if columns:
                corrupted = mask[columns].any(axis="columns")
                mask.loc[corrupted, columns] = data.loc[corrupted, columns].notna()
    return mask


def mask_invalid_values(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of data with implausible values replaced by missing values."""
    return data.mask(invalid_value_mask(data))


def cap_protocol_maxima(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of data with FitnessGram scores clipped to the protocol maxima."""
    capped = data.copy()
    for column, maximum in PROTOCOL_MAXIMA.items():
        if column in capped.columns:
            capped[column] = capped[column].clip(upper=maximum)
    return capped


def clean_values(data: pd.DataFrame) -> pd.DataFrame:
    """Clip scores to the protocol maxima, then replace implausible values with missing values."""
    return mask_invalid_values(cap_protocol_maxima(data))
