"""Preprocessing pipelines fitted within each training fold, by model family."""

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler
from sklearn.utils.validation import check_is_fitted

from piu_severity.features.groups import instrument_of

BINARY_COLUMNS: tuple[str, ...] = (
    "Basic_Demos-Sex",
    "FGC-FGC_CU_Zone",
    "FGC-FGC_PU_Zone",
    "FGC-FGC_SRL_Zone",
    "FGC-FGC_SRR_Zone",
    "FGC-FGC_TL_Zone",
)
ORDINAL_COLUMNS: tuple[str, ...] = (
    "FGC-FGC_GSND_Zone",
    "FGC-FGC_GSD_Zone",
    "BIA-BIA_Activity_Level_num",
    "BIA-BIA_Frame_num",
    "PreInt_EduHx-computerinternet_hoursday",
)
NOMINAL_COLUMNS: tuple[str, ...] = ("Basic_Demos-Enroll_Season",)
ALWAYS_OBSERVED_INSTRUMENTS: tuple[str, ...] = ("Basic_Demos",)

SUPPORTED_PREPROCESSING: frozenset[tuple[str, str]] = frozenset(
    {("linear", "simple"), ("tree", "native")}
)


def column_types(columns: Iterable[str]) -> dict[str, list[str]]:
    """Split predictors into numeric, binary, ordinal and nominal, keeping their order."""
    columns = list(columns)
    typed = {
        "binary": [c for c in columns if c in BINARY_COLUMNS],
        "ordinal": [c for c in columns if c in ORDINAL_COLUMNS],
        "nominal": [c for c in columns if c in NOMINAL_COLUMNS],
    }
    categorical = {c for group in typed.values() for c in group}
    return {"numeric": [c for c in columns if c not in categorical], **typed}


def measured_instruments(columns: Iterable[str]) -> list[str]:
    """Return the instruments of the columns that can be missing, in order of appearance."""
    instruments = dict.fromkeys(instrument_of(c) for c in columns)
    return [i for i in instruments if i not in ALWAYS_OBSERVED_INSTRUMENTS]


class InstrumentMissingIndicator(TransformerMixin, BaseEstimator):
    """Add one indicator per instrument, equal to 1 when the participant has none of its values.

    The instruments are taken from the column names, not learned from the data.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> "InstrumentMissingIndicator":
        """Record the columns of each instrument."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("InstrumentMissingIndicator requires a pandas DataFrame.")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.instrument_columns_ = {
            instrument: [c for c in X.columns if instrument_of(c) == instrument]
            for instrument in measured_instruments(X.columns)
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return the indicators, one column per instrument."""
        check_is_fitted(self, "instrument_columns_")
        indicators = {
            f"missing_{instrument}": X[columns].isna().all(axis="columns").astype(float)
            for instrument, columns in self.instrument_columns_.items()
        }
        return pd.DataFrame(indicators, index=X.index)

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return the names of the indicator columns."""
        check_is_fitted(self, "instrument_columns_")
        return np.asarray(
            [f"missing_{instrument}" for instrument in self.instrument_columns_],
            dtype=object,
        )


def _linear_transformers(types: dict[str, list[str]]) -> list[tuple]:
    """Return imputation, transformation and encoding steps for linear models."""
    return [
        (
            "numeric",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("yeo_johnson", PowerTransformer(method="yeo-johnson")),
                ]
            ),
            types["numeric"],
        ),
        (
            "ordinal",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("scale", StandardScaler()),
                ]
            ),
            types["ordinal"],
        ),
        ("binary", SimpleImputer(strategy="most_frequent"), types["binary"]),
    ]


def _tree_transformers(types: dict[str, list[str]]) -> list[tuple]:
    """Return pass-through steps for trees, which handle missing values natively."""
    return [
        (
            "passthrough",
            "passthrough",
            [*types["numeric"], *types["ordinal"], *types["binary"]],
        )
    ]


def build_preprocessor(
    columns: Iterable[str], family: str, strategy: str
) -> ColumnTransformer:
    """Return the preprocessing step for a model family and a missing-data strategy.

    linear + simple: median or mode imputation, Yeo-Johnson and standardization of
    numeric predictors, standardization of ordinal codes and per-instrument
    missingness indicators.
    tree + native: predictors pass unchanged, so missing values reach the model.
    Both one-hot encode the nominal predictors. Output is a pandas DataFrame.
    """
    if (family, strategy) not in SUPPORTED_PREPROCESSING:
        raise ValueError(f"Unsupported preprocessing: {family!r} with {strategy!r}.")
    columns = list(columns)
    types = column_types(columns)
    steps = (
        _linear_transformers(types) if family == "linear" else _tree_transformers(types)
    )
    steps.append(
        (
            "nominal",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            types["nominal"],
        )
    )
    if family == "linear" and measured_instruments(columns):
        steps.append(("missing", InstrumentMissingIndicator(), columns))
    steps = [step for step in steps if len(step[2]) > 0]
    return ColumnTransformer(
        steps, remainder="drop", verbose_feature_names_out=False
    ).set_output(transform="pandas")