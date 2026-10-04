import pytest

from piu_severity.config import RAW_DATA_DIR, TRAIN_FILE
from piu_severity.features.groups import (
    FEATURE_SETS,
    column_group,
    feature_set_columns,
    is_excluded,
    is_instrument_season,
)

COLUMNS = [
    "id",
    "Basic_Demos-Enroll_Season",
    "Basic_Demos-Age",
    "Physical-Season",
    "Physical-BMI",
    "BIA-BIA_FFM",
    "PAQ_C-PAQ_C_Total",
    "SDS-SDS_Total_T",
    "PCIAT-Season",
    "PCIAT-PCIAT_01",
    "PCIAT-PCIAT_Total",
    "sii",
]


def _assert_nested_and_clean(columns):
    sets = {name: feature_set_columns(columns, name) for name in FEATURE_SETS}
    assert set(sets["reference"]) <= set(sets["main"]) <= set(sets["complementary"])
    for selected in sets.values():
        assert not any(is_excluded(c) or is_instrument_season(c) for c in selected)


@pytest.mark.parametrize(
    "column", ["id", "sii", "PCIAT-PCIAT_01", "PCIAT-PCIAT_Total", "PCIAT-Season"]
)
def test_is_excluded_true(column):
    assert is_excluded(column)


def test_is_excluded_false():
    assert not is_excluded("Physical-BMI")


@pytest.mark.parametrize(
    ("column", "group"),
    [
        ("Basic_Demos-Enroll_Season", "demographic"),
        ("Physical-Season", "season"),
        ("PCIAT-Season", "excluded"),
        ("BIA-BIA_FFM", "physical"),
        ("PAQ_C-PAQ_C_Total", "physical"),
        ("SDS-SDS_Total_T", "complementary"),
    ],
)
def test_column_group(column, group):
    assert column_group(column) == group


def test_column_group_unknown_instrument():
    with pytest.raises(KeyError):
        column_group("Foo-Bar")


def test_feature_sets_are_nested_and_clean():
    _assert_nested_and_clean(COLUMNS)


def test_reference_has_only_demographic_columns():
    selected = feature_set_columns(COLUMNS, "reference")
    assert selected
    assert all(column_group(c) == "demographic" for c in selected)


def test_feature_set_keeps_order():
    selected = feature_set_columns(COLUMNS, "complementary")
    assert selected == [c for c in COLUMNS if c in selected]


def test_unknown_feature_set():
    with pytest.raises(KeyError):
        feature_set_columns(COLUMNS, "unknown")


@pytest.mark.skipif(
    not (RAW_DATA_DIR / TRAIN_FILE).is_file(), reason="raw data not available"
)
def test_train_columns_are_classified():
    from piu_severity.data.load import load_train

    columns = list(load_train().columns)
    for column in columns:
        column_group(column)
    _assert_nested_and_clean(columns)