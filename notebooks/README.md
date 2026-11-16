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

Clear large outputs and any row-level output before committing.

## Reusing the evaluations of `03_busqueda_modelos.ipynb`

`03_busqueda_modelos.ipynb` writes `models/selected_configuration.json`, the record of the selected configuration and of its evaluations. For each evaluation it stores the cache name, the configuration, the hash of the signature, the number of outer folds and the mean QWK. It holds no participant data.

A later notebook rebuilds an evaluation with `piu_severity.models.registry`:

1. `make_search_task(X=..., y=..., splits=..., **task_arguments(entry))`, with the same analytic dataset and `outer_splits`;
2. `check_task(task, entry)`, which fails if the name or the signature differ (other data, grid, seeds or library versions);
3. `run_configurations_cached([task], ...)`, which loads the folds from the evaluation cache under `data/interim/` or fits the missing ones;
4. `check_result(result, entry)`, which fails if the mean QWK differs from the record by more than `QWK_REPRODUCTION_TOLERANCE`.

The evaluation cache is not versioned and, because it holds per-participant predictions of minors, it is not shared outside the project team. Without it, step 3 fits the evaluation again (minutes for the selected configuration) and step 4 checks that the result reproduces the recorded one.