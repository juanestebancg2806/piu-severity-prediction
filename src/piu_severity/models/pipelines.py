"""Model configurations with their inner hyperparameter searches.

make_logistic_search and make_boosting_search are the two fixed configurations
of the initial experiments (notebook 02). make_search builds the twelve
configurations of the full search (four algorithms by three formulations of the
SII), whose grids, budget cuts and complexity order are fixed here before the
search is run.
"""

import math
import os
from collections.abc import Iterable, Mapping

import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR

from piu_severity.config import RANDOM_SEED
from piu_severity.evaluation.metrics import qwk_scorer
from piu_severity.evaluation.persistence import EvaluationTask, evaluation_fields
from piu_severity.evaluation.validation import N_OUTER_REPEATS, Split, inner_cv
from piu_severity.features.preprocessing import (
    MissingDataPreprocessor,
    build_preprocessor,
)
from piu_severity.models.ordinal import (
    N_THRESHOLD_SPLITS,
    FrankHallClassifier,
    MulticlassClassifier,
    ThresholdRegressor,
)

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
        pipeline,
        LOGISTIC_GRID,
        scoring=qwk_scorer,
        cv=inner_cv(seed=seed),
        n_jobs=n_jobs,
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
        pipeline,
        BOOSTING_GRID,
        scoring=qwk_scorer,
        cv=inner_cv(seed=seed),
        n_jobs=n_jobs,
    )


# Full search: algorithms, formulations and the options tuned in the inner loop.

ALGORITHMS: tuple[str, ...] = ("linear", "svm", "random_forest", "gradient_boosting")
FORMULATIONS: tuple[str, ...] = ("threshold_regression", "multiclass", "frank_hall")
ALGORITHM_FAMILY: dict[str, str] = {
    "linear": "linear",
    "svm": "linear",
    "random_forest": "tree",
    "gradient_boosting": "tree",
}
MISSING_STRATEGIES: dict[str, tuple[str, ...]] = {
    "linear": ("simple", "iterative"),
    "svm": ("simple", "iterative"),
    "random_forest": ("simple", "iterative"),
    "gradient_boosting": ("native", "simple", "iterative"),
}
WEIGHT_POWERS: tuple[float, ...] = (0.5, 1.0)

LINEAR_MAX_ITER = 10_000
FOREST_N_ESTIMATORS = 500
BOOSTING_MAX_ITER = 1_000
BOOSTING_N_ITER_NO_CHANGE = 20
BOOSTING_VALIDATION_FRACTION = 0.15
CALIBRATION_SPLITS = 3

# Grids of the base estimator of each algorithm; classifier and regressor differ
# only for the linear model (logistic regression or elastic net) and the SVM
# (epsilon of the SVR). Values are plain numbers or text, so the chosen
# hyperparameters can be stored as JSON.
CLASSIFIER_GRIDS: dict[str, dict[str, list]] = {
    "linear": {
        "C": [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
        "l1_ratio": [0.0, 0.5, 1.0],
    },
    "svm": {
        "C": [1e-2, 1e-1, 1.0, 10.0, 100.0, 1_000.0],
        "gamma": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    },
    "random_forest": {
        "max_features": ["sqrt", 0.33, 0.66, 1.0],
        "min_samples_leaf": [1, 5, 20, 50],
    },
    "gradient_boosting": {
        "learning_rate": [0.03, 0.1],
        "max_leaf_nodes": [7, 15, 31],
        "min_samples_leaf": [10, 30, 80],
        "l2_regularization": [0.0, 1.0, 10.0],
    },
}
REGRESSOR_GRIDS: dict[str, dict[str, list]] = {
    **CLASSIFIER_GRIDS,
    "linear": {
        "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0],
        "l1_ratio": [0.0, 0.5, 1.0],
    },
    "svm": {**CLASSIFIER_GRIDS["svm"], "epsilon": [0.1, 0.5]},
}

# Budget cuts, cumulative: level k applies cuts 1..k. The level is chosen by
# choose_cut_level from the times of the first repeat, never from results.
MAX_CUT_LEVEL = 5
CUT_SVM_WEIGHT_POWERS: tuple[float, ...] = (1.0,)
CUT_SVM_GRID: dict[str, list[float]] = {
    "C": [1e-1, 1.0, 10.0, 100.0],
    "gamma": [1e-3, 1e-2, 1e-1, 1.0],
}
CUT_FOREST_N_ESTIMATORS = 300
CUT_BOOSTING_RANDOM_CANDIDATES = 40
SEARCH_BUDGET_HOURS = 10.0
N_FOLDS_FULL_SEARCH = 25

# Parallel workers adapt to the machine: every core but RESERVED_CORES runs one
# outer fold (inner searches stay sequential), except for random forests, which
# hold many deep trees in memory and get one worker per FOREST_CORES_PER_WORKER
# cores. The memory rule can only lower these numbers. The number of workers
# changes the running time, never the results.
RESERVED_CORES = 2
FOREST_CORES_PER_WORKER = 3
# Algorithms whose outer folds share one pool of workers. Random forests run in
# their own pool, with fewer workers, so they never compete for memory with the
# other algorithms.
EXECUTION_POOLS: tuple[tuple[str, ...], ...] = (
    ("linear", "svm", "gradient_boosting"),
    ("random_forest",),
)
MEMORY_BUDGET_FRACTION = 0.75
MEMORY_SAFETY_FACTOR = 1.25

# Convergence rule: a configuration whose model warns in more than this share of
# its fits in the first repeat gets its iteration limits multiplied.
CONVERGENCE_WARNING_MAX_SHARE = 0.05
MAX_ITER_INCREASE = 5
ALGORITHMS_WITH_MAX_ITER: tuple[str, ...] = ("linear",)

# Complexity of the final model, from lowest to highest, used only to break ties
# in the selection rule: algorithm first, then formulation, then the strategy.
COMPLEXITY_ORDER: dict[str, tuple[str, ...]] = {
    "algorithm": ALGORITHMS,
    "formulation": FORMULATIONS,
}
STRATEGY_COMPLEXITY: dict[str, int] = {"native": 0, "simple": 0, "iterative": 1}


def _check_configuration(algorithm: str, formulation: str, cut_level: int) -> None:
    """Reject unknown algorithms, formulations or cut levels."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm {algorithm!r}.")
    if formulation not in FORMULATIONS:
        raise ValueError(f"Unknown formulation {formulation!r}.")
    if not 0 <= cut_level <= MAX_CUT_LEVEL:
        raise ValueError(f"cut_level must be between 0 and {MAX_CUT_LEVEL}.")


def model_grid(algorithm: str, formulation: str, cut_level: int = 0) -> dict[str, list]:
    """Return the grid of the base estimator after the budget cuts of cut_level."""
    _check_configuration(algorithm, formulation, cut_level)
    grids = (
        REGRESSOR_GRIDS if formulation == "threshold_regression" else CLASSIFIER_GRIDS
    )
    grid = {name: list(values) for name, values in grids[algorithm].items()}
    if algorithm == "svm" and cut_level >= 2:
        grid.update({name: list(values) for name, values in CUT_SVM_GRID.items()})
    if algorithm == "gradient_boosting" and cut_level >= 4:
        grid.pop("l2_regularization")
    return grid


def weight_powers(algorithm: str, cut_level: int = 0) -> list[float]:
    """Return the candidate exponents of the level weights."""
    if algorithm == "svm" and cut_level >= 1:
        return list(CUT_SVM_WEIGHT_POWERS)
    return list(WEIGHT_POWERS)


def n_forest_trees(cut_level: int = 0) -> int:
    """Return the number of trees of the random forests."""
    return CUT_FOREST_N_ESTIMATORS if cut_level >= 3 else FOREST_N_ESTIMATORS


def uses_random_search(algorithm: str, cut_level: int = 0) -> bool:
    """Return True when the configuration samples candidates instead of a full grid."""
    return algorithm == "gradient_boosting" and cut_level >= MAX_CUT_LEVEL


def _base_estimator(
    algorithm: str,
    regression: bool,
    cut_level: int,
    max_iter_factor: int,
    seed: int,
) -> BaseEstimator:
    """Return the base estimator of an algorithm, with fixed seeds and limits."""
    if algorithm == "linear":
        max_iter = LINEAR_MAX_ITER * max_iter_factor
        if regression:
            return ElasticNet(max_iter=max_iter)
        return LogisticRegression(solver="saga", max_iter=max_iter, random_state=seed)
    if algorithm == "svm":
        return SVR(kernel="rbf") if regression else SVC(kernel="rbf")
    if algorithm == "random_forest":
        forest = RandomForestRegressor if regression else RandomForestClassifier
        return forest(
            n_estimators=n_forest_trees(cut_level), random_state=seed, n_jobs=1
        )
    boosting = (
        HistGradientBoostingRegressor if regression else HistGradientBoostingClassifier
    )
    return boosting(
        max_iter=BOOSTING_MAX_ITER,
        early_stopping=True,
        n_iter_no_change=BOOSTING_N_ITER_NO_CHANGE,
        validation_fraction=BOOSTING_VALIDATION_FRACTION,
        random_state=seed,
    )


def _calibrated(estimator: BaseEstimator, seed: int) -> CalibratedClassifierCV:
    """Return the estimator with sigmoid calibration, for models without probabilities."""
    return CalibratedClassifierCV(
        estimator,
        method="sigmoid",
        cv=StratifiedKFold(CALIBRATION_SPLITS, shuffle=True, random_state=seed),
        ensemble=False,
    )


def _formulation_estimator(
    algorithm: str,
    formulation: str,
    cut_level: int,
    max_iter_factor: int,
    seed: int,
) -> tuple[BaseEstimator, str]:
    """Return the formulation wrapper and the parameter prefix of its base estimator."""
    regression = formulation == "threshold_regression"
    base = _base_estimator(algorithm, regression, cut_level, max_iter_factor, seed)
    if regression:
        return ThresholdRegressor(base, random_state=seed), "model__estimator__"
    if formulation == "multiclass":
        return MulticlassClassifier(base), "model__estimator__"
    if algorithm == "svm":
        # Frank and Hall needs P(SII > k); the SVM provides it through calibration.
        return FrankHallClassifier(
            _calibrated(base, seed)
        ), "model__estimator__estimator__"
    return FrankHallClassifier(base), "model__estimator__"


def search_grid(
    algorithm: str, formulation: str, cut_level: int = 0
) -> dict[str, list]:
    """Return the full inner-search grid: strategy, weights and model hyperparameters."""
    _, prefix = _formulation_estimator(
        algorithm, formulation, cut_level, 1, RANDOM_SEED
    )
    return {
        "preprocess__strategy": list(MISSING_STRATEGIES[algorithm]),
        "model__weight_power": weight_powers(algorithm, cut_level),
        **{
            f"{prefix}{name}": values
            for name, values in model_grid(algorithm, formulation, cut_level).items()
        },
    }


def n_candidates(algorithm: str, formulation: str, cut_level: int = 0) -> int:
    """Return the number of candidates evaluated by the inner search."""
    if uses_random_search(algorithm, cut_level):
        return CUT_BOOSTING_RANDOM_CANDIDATES
    grid = search_grid(algorithm, formulation, cut_level)
    return math.prod(len(values) for values in grid.values())


def make_search(
    algorithm: str,
    formulation: str,
    columns: Iterable[str],
    *,
    cut_level: int = 0,
    max_iter_factor: int = 1,
    memory: str | None = None,
    seed: int = RANDOM_SEED,
    n_jobs: int = 1,
) -> GridSearchCV | RandomizedSearchCV:
    """Return the inner search of one of the twelve configurations of the full search.

    The pipeline preprocesses the predictors with the strategy chosen in the
    inner loop and fits the formulation wrapper, which computes the level weights
    in each fit. memory caches the fitted preprocessing across candidates; it is
    a folder path, ignored by version control.
    """
    _check_configuration(algorithm, formulation, cut_level)
    model, _ = _formulation_estimator(
        algorithm, formulation, cut_level, max_iter_factor, seed
    )
    preprocess = MissingDataPreprocessor(
        list(columns),
        ALGORITHM_FAMILY[algorithm],
        MISSING_STRATEGIES[algorithm][0],
        seed,
    )
    pipeline = Pipeline([("preprocess", preprocess), ("model", model)], memory=memory)
    grid = search_grid(algorithm, formulation, cut_level)
    if uses_random_search(algorithm, cut_level):
        return RandomizedSearchCV(
            pipeline,
            grid,
            n_iter=CUT_BOOSTING_RANDOM_CANDIDATES,
            scoring=qwk_scorer,
            cv=inner_cv(seed=seed),
            n_jobs=n_jobs,
            random_state=seed,
        )
    return GridSearchCV(
        pipeline, grid, scoring=qwk_scorer, cv=inner_cv(seed=seed), n_jobs=n_jobs
    )


def complexity_key(algorithm: str, formulation: str, strategy: str) -> tuple[int, ...]:
    """Return the position of a configuration in the complexity order (lower is simpler)."""
    return (
        ALGORITHMS.index(algorithm),
        FORMULATIONS.index(formulation),
        STRATEGY_COMPLEXITY[strategy],
    )


def available_workers(algorithm: str, cpu_count: int) -> int:
    """Return the workers of an algorithm on a machine with cpu_count cores."""
    usable = max(1, cpu_count - RESERVED_CORES)
    if algorithm == "random_forest":
        return max(1, usable // FOREST_CORES_PER_WORKER)
    return usable


def total_memory_gb() -> float:
    """Return the physical memory of this machine, in gigabytes."""
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3


def memory_budget_gb(total_gb: float) -> float:
    """Return the memory the parallel workers may use: a share of the total."""
    return MEMORY_BUDGET_FRACTION * total_gb


def workers_within_memory(
    workers: int, peak_mb_per_worker: float, total_gb: float
) -> int:
    """Lower the workers so their projected peak memory fits the memory budget."""
    per_worker = peak_mb_per_worker * MEMORY_SAFETY_FACTOR
    allowed = math.floor(memory_budget_gb(total_gb) * 1024 / per_worker)
    return max(1, min(workers, allowed))


# Base-model fits in one pipeline fit of each formulation: one model, one per cut
# point, or one per internal fold of the thresholds plus the final refit.
MODELS_PER_PIPELINE_FIT: dict[str, int] = {
    "multiclass": 1,
    "frank_hall": 3,
    "threshold_regression": N_THRESHOLD_SPLITS + 1,
}


def convergence_warning_share(
    n_warnings: int, n_pipeline_fits: int, formulation: str
) -> float:
    """Return the share of base-model fits that raised a convergence warning."""
    return n_warnings / (n_pipeline_fits * MODELS_PER_PIPELINE_FIT[formulation])


def max_iter_factor_for(warning_share: float) -> int:
    """Return the iteration multiplier required by the convergence rule."""
    return MAX_ITER_INCREASE if warning_share > CONVERGENCE_WARNING_MAX_SHARE else 1


def cost_ratio(algorithm: str, formulation: str, cut_level: int) -> float:
    """Return the cost of a configuration at cut_level relative to level 0.

    Cost is taken as proportional to the number of candidates and, for random
    forests, to the number of trees.
    """
    ratio = n_candidates(algorithm, formulation, cut_level) / n_candidates(
        algorithm, formulation, 0
    )
    if algorithm == "random_forest":
        ratio *= n_forest_trees(cut_level) / n_forest_trees(0)
    return ratio


def pool_workers(pool: Iterable[str], workers: Mapping[str, int]) -> int:
    """Return the workers of a pool: the fewest allowed to any of its algorithms."""
    return min(workers[algorithm] for algorithm in pool)


def project_search_hours(
    fold_seconds: Mapping[tuple[str, str], float],
    workers: Mapping[str, int],
    cut_level: int,
    n_folds: int = N_FOLDS_FULL_SEARCH,
) -> float:
    """Return the projected wall-clock hours of the full search at cut_level.

    fold_seconds holds the mean seconds of one outer fold of each configuration
    at level 0, measured in the first repeat. The folds of each pool share its
    workers and the pools run one after the other. A pool takes its total work
    divided by its workers, plus its longest fold, which may finish alone.
    """
    total = 0.0
    for pool in EXECUTION_POOLS:
        costs = [
            seconds * cost_ratio(algorithm, formulation, cut_level)
            for (algorithm, formulation), seconds in fold_seconds.items()
            if algorithm in pool
        ]
        if costs:
            total += n_folds * sum(costs) / pool_workers(pool, workers) + max(costs)
    return total / 3_600


def choose_cut_level(
    fold_seconds: Mapping[tuple[str, str], float],
    workers: Mapping[str, int],
    budget_hours: float = SEARCH_BUDGET_HOURS,
) -> int:
    """Return the lowest cut level whose projected search fits the budget.

    When no level fits, the highest level is returned; the projection shows by
    how much the budget is exceeded.
    """
    for level in range(MAX_CUT_LEVEL + 1):
        if project_search_hours(fold_seconds, workers, level) <= budget_hours:
            return level
    return MAX_CUT_LEVEL


def effective_cut_level(algorithm: str, formulation: str, cut_level: int) -> int:
    """Return the lowest cut level whose search is identical to the one at cut_level.

    Most cuts touch only some algorithms; the linear model, for example, has the
    same search at every level. Naming the evaluation by this level lets a later
    run at a higher level reuse the folds already fitted when nothing changed.
    """
    _check_configuration(algorithm, formulation, cut_level)

    def search_at(level: int) -> tuple:
        trees = n_forest_trees(level) if algorithm == "random_forest" else None
        grid = search_grid(algorithm, formulation, level)
        return grid, uses_random_search(algorithm, level), trees

    target = search_at(cut_level)
    return next(level for level in range(cut_level + 1) if search_at(level) == target)


def effective_max_iter_factor(algorithm: str, max_iter_factor: int) -> int:
    """Return the iteration multiplier that actually changes the estimator.

    Only the linear models have an iteration limit that the convergence rule
    raises; for the other algorithms the multiplier is 1.
    """
    return max_iter_factor if algorithm in ALGORITHMS_WITH_MAX_ITER else 1


def make_search_task(
    algorithm: str,
    formulation: str,
    feature_set: str,
    X: pd.DataFrame,
    y: ArrayLike,
    splits: list[Split],
    *,
    cut_level: int = 0,
    max_iter_factor: int = 1,
    memory: str | None = None,
    evaluation: str = "search",
    n_repeats: int = N_OUTER_REPEATS,
    extra: Mapping | None = None,
    keep_estimators: bool = False,
    name_suffix: str = "",
) -> EvaluationTask:
    """Return the cached evaluation of one configuration on one feature set.

    The cut level and the iteration multiplier are reduced to the ones that
    change this configuration, and both enter the cache name and signature: a
    run with another search never overwrites the folds of an earlier one, and a
    run with the same search reuses them. n_repeats must be the number of
    repeats of the full evaluation, also when only some of its splits are
    passed, so that the folds fitted first are reused later.
    """
    cut_level = effective_cut_level(algorithm, formulation, cut_level)
    max_iter_factor = effective_max_iter_factor(algorithm, max_iter_factor)
    search = make_search(
        algorithm,
        formulation,
        X.columns,
        cut_level=cut_level,
        max_iter_factor=max_iter_factor,
        memory=memory,
    )
    config = {
        "algorithm": algorithm,
        "formulation": formulation,
        "feature_set": feature_set,
        "cut_level": cut_level,
        "max_iter_factor": max_iter_factor,
    }
    fields = evaluation_fields(
        search,
        X,
        y,
        config=config,
        evaluation=evaluation,
        n_repeats=n_repeats,
        extra=extra,
    )
    name = (
        f"{algorithm}__{formulation}__{feature_set}"
        f"__c{cut_level}__i{max_iter_factor}{name_suffix}"
    )
    return EvaluationTask(name, search, X, y, splits, fields, keep_estimators)