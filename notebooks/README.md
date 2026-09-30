# Notebooks

Naming convention: `NN-short-description.ipynb` (for example, `01-eda-tabular.ipynb`).

Notebooks are for exploration and narrative. Reusable logic must be moved to `src/piu_severity/` and imported (`from piu_severity.features import ...`). Do not modify `sys.path` to import project code: `uv sync` installs the package in editable mode.

Select the project's `.venv` as the notebook kernel. It contains all project dependencies and the `piu_severity` package.

Clear large outputs before committing.
