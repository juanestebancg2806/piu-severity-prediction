# Notebooks

Naming convention: `NN_short_description.ipynb`, with a two-digit prefix that sets the reading order and a short description in Spanish (for example, `01_eda.ipynb`).

| Notebook | Content |
| --- | --- |
| `01_eda.ipynb` | Exploratory data analysis and analytic dataset |
| `02_modelo_referencia.ipynb` | Validation scheme, baseline models, and initial hypothesis tests |

Notebooks are for exploration and narrative. Reusable logic must be moved to `src/piu_severity/` and imported (`from piu_severity.features import ...`). Do not modify `sys.path` to import project code: `uv sync` installs the package in editable mode.

Select the project's `.venv` as the notebook kernel. It contains all project dependencies and the `piu_severity` package.

Save figures with `save_figure` in a subfolder named after the notebook stem, `FIGURES_DIR / "<notebook_stem>"`.

Clear large outputs and any row-level output before committing.