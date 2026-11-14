from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from piu_severity.features.preprocessing import (
    BINARY_COLUMNS,
    NOMINAL_COLUMNS,
    ORDINAL_COLUMNS,
    InstrumentMissingIndicator,
    MissingDataPreprocessor,
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


REFERENCE_FILE = Path(__file__).parent / "fixtures" / "preprocessing_reference.npz"
NEW_COMBINATIONS = [("linear", "iterative"), ("tree", "simple"), ("tree", "iterative")]
# Yeo-Johnson estimates its lambda with a numerical optimizer whose last digits
# differ across platforms (for example, macOS ARM and Linux x86), by about 1e-8.
# Any change in the preprocessing logic produces differences of 1e-2 or more.
REFERENCE_TOLERANCE = 1e-6


def _reference() -> tuple[np.lib.npyio.NpzFile, pd.DataFrame, int]:
    """Load the frozen inputs and outputs of the original implementation.

    The file was generated with build_preprocessor as it was at commit 64c7707,
    before the iterative and tree-simple strategies were added.
    """
    stored = np.load(REFERENCE_FILE)
    columns = list(stored["input_columns"])
    data = pd.DataFrame(stored["input_values"], columns=columns[1:])
    data.insert(0, columns[0], stored["input_season"])
    return stored, data, int(stored["n_train"])


@pytest.mark.parametrize(
    ("family", "strategy"), [("linear", "simple"), ("tree", "native")]
)
def test_original_strategies_reproduce_the_frozen_outputs(family, strategy):
    stored, data, n_train = _reference()
    preprocessor = build_preprocessor(data.columns, family, strategy)
    output = preprocessor.fit(data.iloc[:n_train]).transform(data.iloc[n_train:])
    expected = pd.DataFrame(
        stored[f"{family}_{strategy}_values"],
        columns=list(stored[f"{family}_{strategy}_columns"]),
        index=output.index,
    )
    # Columns, their order and their types must match exactly.
    assert [str(t) for t in output.dtypes] == list(
        stored[f"{family}_{strategy}_dtypes"]
    )
    pd.testing.assert_frame_equal(
        output.astype(float),
        expected,
        check_exact=False,
        rtol=REFERENCE_TOLERANCE,
        atol=REFERENCE_TOLERANCE,
    )


@pytest.mark.parametrize(("family", "strategy"), NEW_COMBINATIONS)
def test_new_strategies_leave_no_missing_values(family, strategy):
    _, data, n_train = _reference()
    preprocessor = build_preprocessor(data.columns, family, strategy)
    output = preprocessor.fit(data.iloc[:n_train]).transform(data.iloc[n_train:])
    assert not output.isna().any().any()
    assert len(output) == len(data) - n_train


def test_tree_simple_imputes_without_rescaling():
    _, data, _ = _reference()
    output = build_preprocessor(data.columns, "tree", "simple").fit_transform(data)
    observed = data["Physical-Height"].dropna()
    imputed = output.loc[data["Physical-Height"].isna(), "Physical-Height"]
    assert np.allclose(output.loc[observed.index, "Physical-Height"], observed)
    assert imputed.eq(observed.median()).all()


@pytest.mark.parametrize("family", ["linear", "tree"])
def test_iterative_imputation_keeps_discrete_codes_valid(family):
    _, data, _ = _reference()
    output = build_preprocessor(data.columns, family, "iterative").fit_transform(data)
    for column in ["Basic_Demos-Sex", "FGC-FGC_CU_Zone"]:
        assert set(output[column].unique()) <= {0.0, 1.0}
    if family == "tree":
        for column in ["FGC-FGC_GSND_Zone", "BIA-BIA_Frame_num"]:
            values = output[column]
            assert np.array_equal(values, values.round())
            assert values.min() >= data[column].min()
            assert values.max() <= data[column].max()


def test_iterative_imputation_never_modifies_observed_cells():
    _, data, n_train = _reference()
    preprocessor = build_preprocessor(data.columns, "tree", "iterative")
    test = data.iloc[n_train:].copy()
    # An observed code outside the training range must reach the model unchanged.
    test.iloc[0, test.columns.get_loc("BIA-BIA_Frame_num")] = 9.0
    output = preprocessor.fit(data.iloc[:n_train]).transform(test)
    observed = test.drop(columns="Basic_Demos-Enroll_Season").notna()
    for column in observed.columns:
        rows = observed[column]
        assert np.allclose(output.loc[rows, column], test.loc[rows, column])


def test_iterative_linear_standardizes_numeric_and_ordinal_predictors():
    _, data, _ = _reference()
    output = build_preprocessor(data.columns, "linear", "iterative").fit_transform(
        data
    )
    for column in ["Physical-Height", "FGC-FGC_GSND_Zone"]:
        assert output[column].mean() == pytest.approx(0.0, abs=1e-8)
        assert output[column].std(ddof=0) == pytest.approx(1.0, abs=1e-8)


def test_indicators_are_added_only_with_simple_imputation():
    _, data, _ = _reference()
    for family, strategy in [
        *NEW_COMBINATIONS,
        ("linear", "simple"),
        ("tree", "native"),
    ]:
        output = build_preprocessor(data.columns, family, strategy).fit_transform(data)
        has_indicators = any(c.startswith("missing_") for c in output.columns)
        assert has_indicators == (strategy == "simple")


@pytest.mark.parametrize(("family", "strategy"), NEW_COMBINATIONS)
def test_new_strategies_use_training_parameters_only(family, strategy):
    _, data, n_train = _reference()
    train, test = data.iloc[:n_train], data.iloc[n_train:]
    fitted = build_preprocessor(data.columns, family, strategy).fit(train)
    shifted = data.copy()
    shifted.iloc[n_train:, 1:] = shifted.iloc[n_train:, 1:] + 1_000.0
    refitted = build_preprocessor(data.columns, family, strategy).fit(
        shifted.iloc[:n_train]
    )
    # Changing only the evaluation rows must not change what is learned.
    pd.testing.assert_frame_equal(fitted.transform(test), refitted.transform(test))


def test_iterative_imputation_is_deterministic():
    _, data, _ = _reference()
    first = build_preprocessor(data.columns, "linear", "iterative").fit_transform(data)
    second = build_preprocessor(data.columns, "linear", "iterative").fit_transform(
        data
    )
    pd.testing.assert_frame_equal(first, second)


def test_iterative_imputation_keeps_shape_with_an_empty_column():
    _, data, n_train = _reference()
    train = data.iloc[:n_train].copy()
    train["Actigraphy-ENMO_Day"] = np.nan
    output = build_preprocessor(data.columns, "tree", "iterative").fit(train)
    assert output.transform(data.iloc[n_train:]).shape[1] == 16


def test_missing_data_preprocessor_matches_build_preprocessor():
    _, data, n_train = _reference()
    for family, strategy in [
        *NEW_COMBINATIONS,
        ("linear", "simple"),
        ("tree", "native"),
    ]:
        wrapper = MissingDataPreprocessor(list(data.columns), family, strategy)
        direct = build_preprocessor(data.columns, family, strategy)
        train, test = data.iloc[:n_train], data.iloc[n_train:]
        pd.testing.assert_frame_equal(
            wrapper.fit(train).transform(test), direct.fit(train).transform(test)
        )
        assert list(wrapper.get_feature_names_out()) == list(
            direct.get_feature_names_out()
        )


def test_missing_data_preprocessor_strategy_is_a_tunable_parameter():
    _, data, _ = _reference()
    wrapper = MissingDataPreprocessor(list(data.columns), "tree", "native")
    tuned = clone(wrapper).set_params(strategy="iterative")
    assert tuned.get_params()["strategy"] == "iterative"
    assert wrapper.get_params()["strategy"] == "native"
    assert not tuned.fit_transform(data).isna().any().any()


def test_missing_data_preprocessor_rejects_unknown_columns_and_arrays():
    _, data, _ = _reference()
    with pytest.raises(KeyError):
        MissingDataPreprocessor(["unknown"], "tree", "native").fit(data)
    with pytest.raises(TypeError):
        MissingDataPreprocessor(list(data.columns)).fit(data.to_numpy())


def test_unsupported_strategy_raises():
    with pytest.raises(ValueError):
        build_preprocessor(_frame().columns, "tree", "multiple")