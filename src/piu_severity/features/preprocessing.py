"""Preprocessing pipelines fitted within each training fold, by model family."""

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler
from sklearn.utils.validation import check_is_fitted

from piu_severity.config import RANDOM_SEED
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
ITERATIVE_IMPUTER_MAX_ITER = 25

SUPPORTED_PREPROCESSING: frozenset[tuple[str, str]] = frozenset(
    {
        ("linear", "simple"),
        ("linear", "iterative"),
        ("tree", "native"),
        ("tree", "simple"),
        ("tree", "iterative"),
    }
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


def _writable_values(X: pd.DataFrame) -> np.ndarray:
    """Return a writable float copy of X.

    pandas copy-on-write returns read-only arrays, which IterativeImputer with
    keep_empty_features=True tries to modify in place.
    """
    return X.to_numpy(dtype=float, copy=True)


class RoundedIterativeImputer(TransformerMixin, BaseEstimator):
    """Multivariate imputation that keeps discrete codes valid.

    IterativeImputer returns continuous values. Imputed cells of the discrete
    columns (binary and ordinal codes) are rounded and clipped to the range
    observed in the training data; observed cells are never modified.
    """

    def __init__(
        self,
        discrete_columns: Iterable[str] = (),
        max_iter: int = ITERATIVE_IMPUTER_MAX_ITER,
        random_state: int = RANDOM_SEED,
    ) -> None:
        self.discrete_columns = discrete_columns
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: object = None) -> "RoundedIterativeImputer":
        """Record the observed range of the discrete columns and fit the imputer."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("RoundedIterativeImputer requires a pandas DataFrame.")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        discrete = [c for c in self.discrete_columns if c in X.columns]
        self.lower_ = X[discrete].min()
        self.upper_ = X[discrete].max()
        # keep_empty_features keeps the output shape when a column has no observed
        # value in a training fold; such a column is imputed with zeros.
        self.imputer_ = IterativeImputer(
            max_iter=self.max_iter,
            random_state=self.random_state,
            keep_empty_features=True,
        ).fit(_writable_values(X))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return X with every missing value imputed."""
        check_is_fitted(self, "imputer_")
        imputed = pd.DataFrame(
            self.imputer_.transform(_writable_values(X)),
            index=X.index,
            columns=X.columns,
        )
        discrete = list(self.lower_.index)
        codes = imputed[discrete].round().clip(self.lower_, self.upper_, axis=1)
        was_missing = X[discrete].isna()
        imputed[discrete] = imputed[discrete].mask(was_missing, codes)
        return imputed

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return the input column names, which are kept unchanged."""
        check_is_fitted(self, "imputer_")
        return self.feature_names_in_


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


def _tree_simple_transformers(types: dict[str, list[str]]) -> list[tuple]:
    """Return median or mode imputation without any rescaling, for trees."""
    return [
        ("numeric", SimpleImputer(strategy="median"), types["numeric"]),
        (
            "discrete",
            SimpleImputer(strategy="most_frequent"),
            [*types["ordinal"], *types["binary"]],
        ),
    ]


def _iterative_transformers(
    types: dict[str, list[str]], family: str, seed: int
) -> list[tuple]:
    """Return one joint multivariate imputation step over all non-nominal columns.

    For linear models, numeric predictors are Yeo-Johnson transformed before the
    imputation, so the linear imputation models work on less skewed scales, and
    numeric and ordinal predictors are standardized afterwards. Imputed binary and
    ordinal codes are rounded and clipped to their training range.
    """
    numeric = types["numeric"]
    discrete = [*types["ordinal"], *types["binary"]]
    steps: list[tuple] = []
    if family == "linear" and numeric:
        steps.append(
            (
                "yeo_johnson",
                ColumnTransformer(
                    [("numeric", PowerTransformer(method="yeo-johnson"), numeric)],
                    remainder="passthrough",
                    verbose_feature_names_out=False,
                ),
            )
        )
    steps.append(("impute", RoundedIterativeImputer(discrete, random_state=seed)))
    scaled = [*numeric, *types["ordinal"]]
    if family == "linear" and scaled:
        steps.append(
            (
                "scale",
                ColumnTransformer(
                    [("scale", StandardScaler(), scaled)],
                    remainder="passthrough",
                    verbose_feature_names_out=False,
                ),
            )
        )
    return [("joint", Pipeline(steps), [*numeric, *discrete])]


def _value_transformers(
    types: dict[str, list[str]], family: str, strategy: str, seed: int
) -> list[tuple]:
    """Return the steps of the non-nominal predictors for a family and strategy."""
    if strategy == "iterative":
        return _iterative_transformers(types, family, seed)
    if family == "linear":
        return _linear_transformers(types)
    if strategy == "simple":
        return _tree_simple_transformers(types)
    return _tree_transformers(types)


def build_preprocessor(
    columns: Iterable[str], family: str, strategy: str, seed: int = RANDOM_SEED
) -> ColumnTransformer:
    """Return the preprocessing step for a model family and a missing-data strategy.

    linear + simple: median or mode imputation, Yeo-Johnson and standardization of
    numeric predictors, standardization of ordinal codes and per-instrument
    missingness indicators.
    linear + iterative: Yeo-Johnson, joint multivariate imputation, rounding of
    discrete codes and standardization of numeric and ordinal predictors.
    tree + native: predictors pass unchanged, so missing values reach the model.
    tree + simple: median or mode imputation and per-instrument indicators.
    tree + iterative: joint multivariate imputation and rounding of discrete codes.
    All one-hot encode the nominal predictors. Output is a pandas DataFrame.
    """
    if (family, strategy) not in SUPPORTED_PREPROCESSING:
        raise ValueError(f"Unsupported preprocessing: {family!r} with {strategy!r}.")
    columns = list(columns)
    types = column_types(columns)
    steps = _value_transformers(types, family, strategy, seed)
    steps.append(
        (
            "nominal",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            types["nominal"],
        )
    )
    if strategy == "simple" and measured_instruments(columns):
        steps.append(("missing", InstrumentMissingIndicator(), columns))
    steps = [step for step in steps if len(step[2]) > 0]
    return ColumnTransformer(
        steps, remainder="drop", verbose_feature_names_out=False
    ).set_output(transform="pandas")


class MissingDataPreprocessor(TransformerMixin, BaseEstimator):
    """Preprocessing whose missing-data strategy is a tunable hyperparameter.

    Wraps build_preprocessor so that a grid search can choose the strategy by
    name (for example, preprocess__strategy) and store it as plain text.
    """

    def __init__(
        self,
        columns: Iterable[str] = (),
        family: str = "linear",
        strategy: str = "simple",
        seed: int = RANDOM_SEED,
    ) -> None:
        self.columns = columns
        self.family = family
        self.strategy = strategy
        self.seed = seed

    def fit(self, X: pd.DataFrame, y: object = None) -> "MissingDataPreprocessor":
        """Build and fit the preprocessing of the chosen family and strategy."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("MissingDataPreprocessor requires a pandas DataFrame.")
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise KeyError(f"Columns not found in X: {missing}")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.preprocessor_ = build_preprocessor(
            self.columns, self.family, self.strategy, self.seed
        ).fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return the preprocessed predictors."""
        check_is_fitted(self, "preprocessor_")
        return self.preprocessor_.transform(X)

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return the names of the output columns."""
        check_is_fitted(self, "preprocessor_")
        return self.preprocessor_.get_feature_names_out()
