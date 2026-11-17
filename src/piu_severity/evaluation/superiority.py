"""Magnitude and conclusion of a superiority hypothesis such as H4.

Both are fixed before the complementary feature set is evaluated. The test is
the one-sided Nadeau–Bengio corrected t test of summarize_paired, adjusted with
Holm's method within its family. Its conclusion comes from the main analysis
(treatment fixed across folds); the sensitivity analysis (treatment chosen in
each fold) confirms it when it reaches the same conclusion. The one-sided lower
bound of the mean difference is reported only as a magnitude, not as a test
against the minimum relevant difference.
"""

import numpy as np
import pandas as pd
from scipy import stats

from piu_severity.evaluation.selection import SELECTION_ALPHA

SUPERIORITY_CONFIDENCE = 0.95


def superiority_lower_bound(
    differences: pd.DataFrame, confidence: float = SUPERIORITY_CONFIDENCE
) -> dict[str, float]:
    """Return the one-sided lower confidence bound of a mean paired QWK difference.

    Uses the Nadeau–Bengio corrected standard error, like summarize_paired and
    negligibility_test, so the bound lies above zero exactly when the one-sided
    p-value of summarize_paired is below 1 - confidence.
    """
    diff = differences["difference"].to_numpy()
    n_folds = len(diff)
    if n_folds < 2:
        raise ValueError("At least two folds are needed.")
    mean = float(diff.mean())
    std = float(diff.std(ddof=1))
    ratio = float((differences["n_test"] / differences["n_train"]).mean())
    corrected_se = float(np.sqrt((1 / n_folds + ratio) * std**2))
    critical = float(stats.t.ppf(confidence, df=n_folds - 1))
    return {
        "mean_difference": mean,
        "lower_bound": mean - critical * corrected_se,
        "confidence": confidence,
    }


def superiority_conclusion(
    p_main: float, p_sensitivity: float, alpha: float = SELECTION_ALPHA
) -> dict[str, bool]:
    """Return the conclusion of a superiority hypothesis from Holm-adjusted p-values.

    supported comes from the main analysis; confirmed is True when the
    sensitivity analysis reaches the same conclusion, whichever it is.
    """
    for name, value in (("p_main", p_main), ("p_sensitivity", p_sensitivity)):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must lie in [0, 1].")
    supported = p_main < alpha
    return {
        "supported": supported,
        "sensitivity_supported": p_sensitivity < alpha,
        "confirmed": supported == (p_sensitivity < alpha),
    }
