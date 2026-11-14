"""Model selection rule and the decisions fixed before the full search.

The constants below are fixed before the search is run, so that no threshold of
the selection or of the hypothesis tests depends on the results.
"""

SELECTION_ALPHA = 0.05
# Smallest QWK difference considered relevant; a contribution is negligible
# when the one-sided corrected upper bound of the difference lies below it.
MINIMUM_RELEVANT_QWK_DIFFERENCE = 0.03
# Two under-estimation rates closer than this are a tie in step 2 of the rule.
UNDERESTIMATION_TIE_TOLERANCE = 0.01
# A grid is extended past an edge chosen in at least this many outer folds.
EDGE_RULE_MIN_FOLDS = 5
# Families of tests whose p-values are adjusted together with Holm's method.
HOLM_FAMILIES: dict[str, tuple[str, ...]] = {
    "superiority": ("H2", "H3"),
    "negligibility": ("H2", "H3"),
    "actigraphy_subset": (
        "reference_with_actigraphy_vs_reference",
        "main_vs_main_without_actigraphy",
    ),
    "severe_paired": ("selected_vs_best_of_each_other_algorithm",),
}