"""Variable groups, feature sets and column exclusions of the project."""

from collections.abc import Iterable

ID_COLUMN = "id"
TARGET_COLUMN = "sii"
TARGET_SOURCE_PREFIX = "PCIAT-"
ENROLL_SEASON_COLUMN = "Basic_Demos-Enroll_Season"
SEASON_SUFFIX = "-Season"

INSTRUMENT_GROUPS: dict[str, str] = {
    "Basic_Demos": "demographic",
    "Physical": "physical",
    "Fitness_Endurance": "physical",
    "FGC": "physical",
    "BIA": "physical",
    "PAQ_A": "physical",
    "PAQ_C": "physical",
    "PAQ": "physical",
    "Actigraphy": "physical",
    "SDS": "complementary",
    "CGAS": "complementary",
    "PreInt_EduHx": "complementary",
}

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "reference": ("demographic",),
    "main": ("demographic", "physical"),
    "complementary": ("demographic", "physical", "complementary"),
}

PAQ_SOURCE_COLUMNS: tuple[str, ...] = ("PAQ_A-PAQ_A_Total", "PAQ_C-PAQ_C_Total")
PAQ_TOTAL_COLUMN = "PAQ-PAQ_Total"

LOW_COVERAGE_COLUMNS: tuple[str, ...] = (
    "Fitness_Endurance-Max_Stage",
    "Fitness_Endurance-Time_Mins",
    "Fitness_Endurance-Time_Sec",
    "Physical-Waist_Circumference",
)

REDUNDANT_COLUMNS: tuple[str, ...] = (
    "BIA-BIA_BMR",
    "BIA-BIA_TBW",
    "BIA-BIA_LST",
    "BIA-BIA_LDM",
    "BIA-BIA_BMI",
    "BIA-BIA_DEE",
    "BIA-BIA_BMC",
    "SDS-SDS_Total_Raw",
)

DROPPED_COLUMNS: frozenset[str] = frozenset(
    (*LOW_COVERAGE_COLUMNS, *REDUNDANT_COLUMNS, *PAQ_SOURCE_COLUMNS)
)


def instrument_of(column: str) -> str:
    """Return the instrument prefix of a column name."""
    return column.split("-", 1)[0]


def is_excluded(column: str) -> bool:
    """Return True for the id, the target and the columns that define it."""
    return column in (ID_COLUMN, TARGET_COLUMN) or column.startswith(
        TARGET_SOURCE_PREFIX
    )


def is_instrument_season(column: str) -> bool:
    """Return True for per-instrument participation season columns."""
    return column.endswith(SEASON_SUFFIX) and column != ENROLL_SEASON_COLUMN


def is_dropped(column: str) -> bool:
    """Return True for the predictors removed after the exploratory analysis."""
    return column in DROPPED_COLUMNS


def column_group(column: str) -> str:
    """Return the group of a column: excluded, season or a predictor group."""
    if is_excluded(column):
        return "excluded"
    if is_instrument_season(column):
        return "season"
    instrument = instrument_of(column)
    if instrument not in INSTRUMENT_GROUPS:
        raise KeyError(f"Unknown instrument for column {column!r}")
    return INSTRUMENT_GROUPS[instrument]


def feature_set_columns(columns: Iterable[str], feature_set: str) -> list[str]:
    """Return the columns of a feature set, keeping their original order."""
    if feature_set not in FEATURE_SETS:
        raise KeyError(f"Unknown feature set {feature_set!r}")
    groups = FEATURE_SETS[feature_set]
    return [column for column in columns if column_group(column) in groups]


def model_feature_columns(columns: Iterable[str], feature_set: str) -> list[str]:
    """Return the predictors of a feature set after the exploratory exclusions."""
    return [
        column
        for column in feature_set_columns(columns, feature_set)
        if not is_dropped(column)
    ]