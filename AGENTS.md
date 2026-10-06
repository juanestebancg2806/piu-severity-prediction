# Project rules for AI assistants

Prediction of PIU severity (SII, ordinal 0–3) from physical and demographic data. Kaggle Child Mind Institute dataset. Course project, Universidad Icesi.

## Data privacy (non-negotiable)
- Data belongs to minors and the repository is public.
- Never commit anything under `data/` except `.gitkeep` and `README.md`.
- Inspecting rows while working is fine, but row-level outputs (`head()`, `sample()`, `tail()`, row-level `print`) must be cleared before committing. Committed notebook outputs must be aggregated: counts, percentages, statistics, plots, `info()`, `describe()`.
- Never hardcode values copied from the data.

## Leakage prevention (non-negotiable)
- Exclude every `PCIAT-*` column and the participant `id` from features. They exist only in train and define the target.
- Every fitted transformation (imputation, scaling, encoding, resampling, threshold optimization, feature selection) must happen inside a scikit-learn `Pipeline` fitted only on the training fold.
- Hyperparameters, missing-data strategy, and thresholds are selected in the inner loop of nested cross-validation. Evaluation folds are never used for any decision.
- Exploratory analysis of feature–target relationships is descriptive only. It must not drive feature selection outside cross-validation.

## Reproducibility
- Import paths and `RANDOM_SEED` from `piu_severity.config`. No absolute or relative file paths in code or notebooks, and no `sys.path` manipulation.
- Pass `random_state=RANDOM_SEED` explicitly. Use `np.random.default_rng`, never `np.random.seed`.
- All model comparisons use the same participants and the same cross-validation splits.

## Architecture
- `src/piu_severity/`: reusable code. `data/` loading and cleaning, `features/` variable groups and feature engineering, `models/` pipelines, training, and evaluation, `visualization/` reusable plots.
- `notebooks/`: narrative and exploration only. When a function is used twice, move it to `src/`.
- `data/raw/` is immutable. Write intermediate data to `data/interim/` and model-ready data to `data/processed/`.
- Save report figures to a subfolder named after the notebook stem, `reports/figures/<notebook_stem>/` (for example, `FIGURES_DIR / "01_eda"`), so figures from different notebooks do not mix.

## Code style
- Python 3.13, dependencies managed only with `uv add` / `uv sync`. Never use pip or edit `uv.lock` by hand.
- Run `uv run ruff check` and `uv run ruff format` before finishing.
- Small functions with one responsibility, type hints, and short English docstrings.
- Pass dependencies as parameters instead of reading globals inside functions.
- Prefer plain functions and scikit-learn-compatible estimators. Create classes only when a scikit-learn estimator is needed (custom transformers, ordinal wrappers).
- Do not duplicate logic across notebooks and `src/`. Do not add abstractions that are not used yet.
- Add or update tests in `tests/` for functions in `src/`. Tests that need the data must skip when `data/raw/` is empty.

## Language
- Code, docstrings, commit messages, and READMEs in English.
- Notebook markdown in Spanish, written as the team's own technical prose: an introduction and a conclusion per section, no notes addressed to the reader, and no claims about results not yet shown.

## Git
- All team members are collaborators: clone the repository directly. Never suggest forking.
- `main` is protected. Work on a `feature/*` branch, push it to this repository, and open a pull request to `main`.
- Never push directly to `main`, rewrite shared history (`push --force`, rebasing pushed branches), or commit data.
- Avoid editing the same notebook on two branches at once; notebook merge conflicts are hard to resolve.
- Do not commit unless explicitly asked.
