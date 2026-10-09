import numpy as np
import pandas as pd
import pytest

from piu_severity.features.preprocessing import (
    BINARY_COLUMNS,
    NOMINAL_COLUMNS,
    ORDINAL_COLUMNS,
    InstrumentMissingIndicator,
    build_preprocessor,
    column_types,
    measured_instruments,
)


def _frame(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = pd.DataFrame(
        {
            "Basic_Demos-Enroll_Season": rng.choice(
                ["Fall", "Spring", "Summer", "Winter"], size=n
            ),
            "Basic_Demos-Age": rng.integers(5, 22, size=n).astype(float),
            "Basic_Demos-Sex": rng.integers(0, 2, size=n).astype(float),
            "Physical-Height": rng.normal(55, 8, size=n),
            "Physical-Weight": rng.normal(90, 20, size=n),
            "FGC-FGC_CU_Zone": rng.integers(0, 2, size=n).astype(float),
            "BIA-BIA_Frame_num": rng.integers(1, 4, size=n).astype(float),
            "Actigraphy-ENMO_Day": rng.gamma(2.0, 0.02, size=n),
        }
    )
    data.loc[:4, ["Physical-Height", "Physical-Weight"]] = np.nan
    data.loc[5, "Physical-Weight"] = np.nan
    data.loc[10:19, "Actigraphy-ENMO_Day"] = np.nan
    data.loc[20:24, ["FGC-FGC_CU_Zone", "BIA-BIA_Frame_num"]] = np.nan
    return data


def test_type_lists_match_the_exploratory_analysis():
    assert len(BINARY_COLUMNS) == 6
    assert len(ORDINAL_COLUMNS) == 5
    assert len(NOMINAL_COLUMNS) == 1


def test_column_types_split_and_keep_order():
    types = column_types(_frame().columns)
    assert types["nominal"] == ["Basic_Demos-Enroll_Season"]
    assert types["binary"] == ["Basic_Demos-Sex", "FGC-FGC_CU_Zone"]
    assert types["ordinal"] == ["BIA-BIA_Frame_num"]
    assert types["numeric"] == [
        "Basic_Demos-Age",
        "Physical-Height",
        "Physical-Weight",
        "Actigraphy-ENMO_Day",
    ]


def test_measured_instruments_exclude_demographics():
    assert measured_instruments(_frame().columns) == [
        "Physical",
        "FGC",
        "BIA",
        "Actigraphy",
    ]


def test_indicator_marks_only_fully_missing_instruments():
    data = _frame()
    indicators = InstrumentMissingIndicator().fit_transform(data)
    assert list(indicators.columns) == [
        "missing_Physical",
        "missing_FGC",
        "missing_BIA",
        "missing_Actigraphy",
    ]
    assert indicators["missing_Physical"].iloc[:5].eq(1).all()
    assert indicators.at[5, "missing_Physical"] == 0
    assert indicators["missing_Actigraphy"].sum() == 10


def test_indicator_requires_a_dataframe():
    with pytest.raises(TypeError):
        InstrumentMissingIndicator().fit(_frame().to_numpy())


def test_linear_preprocessor_leaves_no_missing_values():
    data = _frame()
    output = build_preprocessor(data.columns, "linear", "simple").fit_transform(data)
    assert not output.isna().any().any()
    assert {"missing_Physical", "missing_Actigraphy"} <= set(output.columns)
    assert "Basic_Demos-Enroll_Season_Fall" in output.columns
    assert output["Physical-Height"].mean() == pytest.approx(0.0, abs=1e-8)


def test_tree_preprocessor_keeps_missing_values_without_indicators():
    data = _frame()
    output = build_preprocessor(data.columns, "tree", "native").fit_transform(data)
    assert output["Actigraphy-ENMO_Day"].isna().sum() == 10
    assert not any(c.startswith("missing_") for c in output.columns)
    assert "Basic_Demos-Enroll_Season_Fall" in output.columns


def test_reference_columns_get_no_indicators():
    data = _frame()[["Basic_Demos-Enroll_Season", "Basic_Demos-Age", "Basic_Demos-Sex"]]
    output = build_preprocessor(data.columns, "linear", "simple").fit_transform(data)
    assert not any(c.startswith("missing_") for c in output.columns)


def test_preprocessor_is_fitted_on_training_rows_only():
    data = _frame()
    preprocessor = build_preprocessor(data.columns, "linear", "simple")
    preprocessor.fit(data.iloc[:30])
    output = preprocessor.transform(data.iloc[30:])
    assert len(output) == 10
    assert not output.isna().any().any()


def test_unsupported_combination_raises():
    with pytest.raises(ValueError):
        build_preprocessor(_frame().columns, "linear", "native")