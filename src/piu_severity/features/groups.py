"""Variable groups and feature sets defined in the project formulation."""

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
    "SDS": "complementary",
    "CGAS": "complementary",
    "PreInt_EduHx": "complementary",
}

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "reference": ("demographic",),
    "main": ("demographic", "physical"),
    "complementary": ("demographic", "physical", "complementary"),
}


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