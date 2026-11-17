# Notebooks

Naming convention: `NN_short_description.ipynb`, with a two-digit prefix that sets the reading order and a short description in Spanish (for example, `01_eda.ipynb`).

| Notebook | Content |
| --- | --- |
| `01_eda.ipynb` | Exploratory data analysis and analytic dataset (objective 1) |
| `02_experimentos_iniciales.ipynb` | Validation scheme, baseline models, and initial hypothesis tests |
| `03_busqueda_modelos.ipynb` | Full model search, selection of the configuration, per-level analysis of the selected configuration, H2 and H3, and the actigraphy subpopulation (objective 2) |
| `04_interpretabilidad_errores.ipynb` | Per-level performance and severe cases, screening scenario, SHAP contributions, and errors by sex and age group of the selected configuration (objective 3) |
| `05_comparacion_complementario.ipynb` | H4: the selected configuration on the complementary set, per-level and screening differences, comparison within age groups, the Entrega 2 configurations, and the inputs of objective 5 (objective 4) |

Notebooks are for exploration and narrative. Reusable logic must be moved to `src/piu_severity/` and imported (`from piu_severity.features import ...`). Do not modify `sys.path` to import project code: `uv sync` installs the package in editable mode.

Select the project's `.venv` as the notebook kernel. It contains all project dependencies and the `piu_severity` package.

Save figures with `save_figure` in a subfolder named after the notebook stem, `FIGURES_DIR / "<notebook_stem>"`, and aggregate tables as CSV in `TABLES_DIR / "<notebook_stem>"`. Exported tables must not contain participant identifiers, positions or one row per participant; notebooks 04 and 05 check this before writing them.

Clear large outputs and any row-level output before committing, and commit notebooks after a full run from a fresh kernel, so that every output matches its code.

## Inputs of objective 5

The recommendations of objective 5 are written in the report, not in a notebook. Their inputs are aggregate tables:

- `reports/tables/04_interpretabilidad_errores/resumen_objetivo_5.csv`: central metrics of the selected configuration (objective 3), each with its mean, deviation between repeats, interval and number of participants;
- `reports/tables/05_comparacion_complementario/resumen_objetivo_4.csv`: differences between the complementary and main sets (objective 4);
- `reports/tables/05_comparacion_complementario/insumos_objetivo_5.csv`: the metrics of objectives 2 to 4 in one table, with the notebook each comes from. Notebook 05 checks the tables it reads against the model record and against its own evaluations.

## Reusing the evaluations of `03_busqueda_modelos.ipynb`

`03_busqueda_modelos.ipynb` writes `models/selected_configuration.json`, the record of the selected configuration and of its evaluations. For each evaluation it stores the cache name, the configuration, the hash of the signature, the number of outer folds and the mean QWK. It also stores the H4 protocol, fixed before the complementary set was evaluated. It holds no participant data.

A later notebook rebuilds an evaluation with `piu_severity.models.registry`:

1. `rebuild_task(entry, X, y, splits, memory=...)`, with the same analytic dataset and `outer_splits`. It builds the task with `make_search_task(**task_arguments(entry))` and runs `check_task`, which fails if the name or the signature differ (other data, grid, seeds or library versions). With `feature_set=...`, it builds the same configuration on another feature set, as notebook 05 does with the complementary set; the signature check is then skipped, because the record holds no entry for that set.
2. `run_configurations_cached([task], ...)`, which loads the folds from the evaluation cache under `data/interim/` or fits the missing ones.
3. `check_result(result, entry)`, which fails if the mean QWK differs from the record by more than `QWK_REPRODUCTION_TOLERANCE`.

Notebooks 04 and 05 load the recorded evaluations from the cache of notebook 03 and keep their new evaluations (the complementary set, the demographic reference and the Entrega 2 configurations) in their own cache, `data/interim/<notebook_stem>/cache`.

The evaluation caches are not versioned and, because they hold per-participant predictions of minors, they are not shared outside the project team. Without them, step 2 fits the evaluations again (minutes for the selected configuration) and step 3 checks that the result reproduces the recorded one.