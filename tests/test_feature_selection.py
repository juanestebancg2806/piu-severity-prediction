from piu_severity.features.actigraphy import ACTIGRAPHY_FEATURES
from piu_severity.features.groups import (
    DROPPED_COLUMNS,
    INSTRUMENT_GROUPS,
    PAQ_SOURCE_COLUMNS,
    PAQ_TOTAL_COLUMN,
    column_group,
    instrument_of,
    is_dropped,
    model_feature_columns,
)

COLUMNS = [
    "id",
    "Basic_Demos-Enroll_Season",
    "Basic_Demos-Age",
    "Physical-Height",
    "Physical-Waist_Circumference",
    "Fitness_Endurance-Max_Stage",
    "BIA-BIA_FFM",
    "BIA-BIA_BMR",
    "PAQ_A-Season",
    "PAQ_A-PAQ_A_Total",
    "PAQ_C-PAQ_C_Total",
    PAQ_TOTAL_COLUMN,
    *ACTIGRAPHY_FEATURES.values(),
    "SDS-SDS_Total_Raw",
    "SDS-SDS_Total_T",
    "PCIAT-PCIAT_Total",
    "sii",
]


def test_dropped_columns_are_known_predictors():
    assert all(instrument_of(column) in INSTRUMENT_GROUPS for column in DROPPED_COLUMNS)


def test_paq_total_replaces_its_sources():
    assert all(is_dropped(column) for column in PAQ_SOURCE_COLUMNS)
    assert column_group(PAQ_TOTAL_COLUMN) == "physical"
    assert not is_dropped(PAQ_TOTAL_COLUMN)


def test_actigraphy_features_are_physical():
    assert all(
        column_group(column) == "physical" for column in ACTIGRAPHY_FEATURES.values()
    )


def test_model_feature_columns_apply_exclusions():
    assert model_feature_columns(COLUMNS, "reference") == [
        "Basic_Demos-Enroll_Season",
        "Basic_Demos-Age",
    ]
    assert model_feature_columns(COLUMNS, "main") == [
        "Basic_Demos-Enroll_Season",
        "Basic_Demos-Age",
        "Physical-Height",
        "BIA-BIA_FFM",
        PAQ_TOTAL_COLUMN,
        *ACTIGRAPHY_FEATURES.values(),
    ]
    complementary = model_feature_columns(COLUMNS, "complementary")
    assert complementary[-1] == "SDS-SDS_Total_T"
    assert "SDS-SDS_Total_Raw" not in complementary


def test_model_feature_columns_never_include_target_or_id():
    for feature_set in ("reference", "main", "complementary"):
        columns = model_feature_columns(COLUMNS, feature_set)
        assert not {"id", "sii", "PCIAT-PCIAT_Total"} & set(columns)
