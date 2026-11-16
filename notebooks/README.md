# Notebooks

Naming convention: `NN_short_description.ipynb`, with a two-digit prefix that sets the reading order and a short description in Spanish (for example, `01_eda.ipynb`).

| Notebook | Content |
| --- | --- |
| `01_eda.ipynb` | Exploratory data analysis and analytic dataset |
| `02_experimentos_iniciales.ipynb` | Validation scheme, baseline models, and initial hypothesis tests |
| `03_busqueda_modelos.ipynb` | Full model search, selection of the configuration, per-level analysis of the selected configuration, H2 and H3, and the actigraphy subpopulation |

Notebooks are for exploration and narrative. Reusable logic must be moved to `src/piu_severity/` and imported (`from piu_severity.features import ...`). Do not modify `sys.path` to import project code: `uv sync` installs the package in editable mode.

Select the project's `.venv` as the notebook kernel. It contains all project dependencies and the `piu_severity` package.

Save figures with `save_figure` in a subfolder named after the notebook stem, `FIGURES_DIR / "<notebook_stem>"`, and aggregate tables as CSV in `TABLES_DIR / "<notebook_stem>"`.

`03_busqueda_modelos.ipynb` writes `models/selected_configuration.json`, the record of the selected configuration. Later notebooks rebuild its evaluations with `piu_severity.models.registry` (`task_arguments` and `check_task`) and load their folds and fitted models from the evaluation cache under `data/interim/`, which is not versioned.

Clear large outputs and any row-level output before committing.