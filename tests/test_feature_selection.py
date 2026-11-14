from piu_severity.features.actigraphy import ACTIGRAPHY_FEATURES
from piu_severity.features.groups import (
    DROPPED_COLUMNS,
    EXPERIMENT_SETS,
    INSTRUMENT_GROUPS,
    PAQ_SOURCE_COLUMNS,
    PAQ_TOTAL_COLUMN,
    actigraphy_subset_feature_sets,
    column_group,
    drop_instruments,
    experiment_feature_sets,
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


def test_drop_instruments_keeps_order():
    assert drop_instruments(COLUMNS[:4], ["Physical"]) == COLUMNS[:3]


def test_experiment_feature_sets_are_nested():
    sets = experiment_feature_sets(COLUMNS)
    assert tuple(sets) == EXPERIMENT_SETS
    assert (
        set(sets["reference"])
        <= set(sets["main_without_actigraphy"])
        <= set(sets["main"])
        <= set(sets["complementary"])
    )


def test_main_without_actigraphy_removes_only_actigraphy():
    sets = experiment_feature_sets(COLUMNS)
    removed = set(sets["main"]) - set(sets["main_without_actigraphy"])
    assert removed == set(ACTIGRAPHY_FEATURES.values())
    assert sets["main_without_actigraphy"] == [
        column for column in sets["main"] if column not in removed
    ]


def test_experiment_feature_sets_keep_their_four_sets():
    # Notebook 02 names exactly these four sets; new sets live elsewhere.
    assert tuple(experiment_feature_sets(COLUMNS)) == EXPERIMENT_SETS
    assert len(EXPERIMENT_SETS) == 4


def test_actigraphy_subset_sets_isolate_the_actigraphy_features():
    sets = actigraphy_subset_feature_sets(COLUMNS)
    experiment = experiment_feature_sets(COLUMNS)
    assert tuple(sets) == (
        "reference",
        "reference_with_actigraphy",
        "main_without_actigraphy",
        "main",
    )
    for name in ("reference", "main_without_actigraphy", "main"):
        assert sets[name] == experiment[name]
    added = sets["reference_with_actigraphy"][len(sets["reference"]) :]
    assert (
        sets["reference_with_actigraphy"][: len(sets["reference"])] == sets["reference"]
    )
    assert added == list(ACTIGRAPHY_FEATURES.values())