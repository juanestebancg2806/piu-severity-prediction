"""Model configurations of the initial experiments, each with its own inner search."""

from collections.abc import Iterable

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from piu_severity.config import RANDOM_SEED
from piu_severity.evaluation.metrics import qwk_scorer
from piu_severity.evaluation.validation import inner_cv
from piu_severity.features.preprocessing import build_preprocessor

LOGISTIC_GRID: dict[str, list[float]] = {
    "model__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
}
BOOSTING_GRID: dict[str, list[int]] = {
    "model__max_iter": [100, 300],
    "model__max_leaf_nodes": [7, 15],
}
BOOSTING_LEARNING_RATE = 0.05
BOOSTING_MIN_SAMPLES_LEAF = 20
LOGISTIC_MAX_ITER = 5_000


def make_logistic_search(
    columns: Iterable[str], seed: int = RANDOM_SEED, n_jobs: int | None = None
) -> GridSearchCV:
    """Return multinomial logistic regression with simple imputation and indicators.

    Classes are weighted by n / (K n_k) within each training fold, and the
    regularization strength C is chosen by QWK in the inner cross-validation.
    """
    pipeline = Pipeline(
        [
            ("preprocess", build_preprocessor(columns, "linear", "simple")),
            (
                "model",
                LogisticRegression(class_weight="balanced", max_iter=LOGISTIC_MAX_ITER),
            ),
        ]
    )
    return GridSearchCV(
        pipeline, LOGISTIC_GRID, scoring=qwk_scorer, cv=inner_cv(seed=seed), n_jobs=n_jobs
    )


def make_boosting_search(
    columns: Iterable[str], seed: int = RANDOM_SEED, n_jobs: int | None = None
) -> GridSearchCV:
    """Return histogram gradient boosting with native handling of missing values.

    Classes are weighted by n / (K n_k) within each training fold, and the number
    of iterations and leaves are chosen by QWK in the inner cross-validation.
    """
    pipeline = Pipeline(
        [
            ("preprocess", build_preprocessor(columns, "tree", "native")),
            (
                "model",
                HistGradientBoostingClassifier(
                    class_weight="balanced",
                    learning_rate=BOOSTING_LEARNING_RATE,
                    min_samples_leaf=BOOSTING_MIN_SAMPLES_LEAF,
                    early_stopping=False,
                    random_state=seed,
                ),
            ),
        ]
    )
    return GridSearchCV(
        pipeline, BOOSTING_GRID, scoring=qwk_scorer, cv=inner_cv(seed=seed), n_jobs=n_jobs
    )