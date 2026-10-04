# PIU Severity Prediction

Part of the "Proyecto I de Innovación Tecnológica en Inteligencia Artificial" course, Applied Artificial Intelligence Master, Universidad Icesi, Cali, Colombia.

## Project Status

Active

## Contributing Members

**Instructor:** [Milton Orlando Sarria](https://github.com/miltonsarria)

| Name |
| --- |
| Katherin Adriana Camargo Cetina |
| Juan Esteban Cardona García |
| Juan David Martínez Legarda |
| Daniel Velasco López |

## Project Intro/Objective

Problematic internet use (PIU) in children and adolescents is associated with depression, anxiety, and sleep disturbances, yet it is often detected late: its assessment depends on specialized clinical evaluations that are costly and not accessible to many families. Physical activity and fitness measurements, in contrast, are easy to obtain and widely collected.

The objective of this project is to evaluate how well physical activity and fitness indicators, together with basic demographic variables, can estimate the severity of PIU, measured by the Severity Impairment Index (SII), in children and adolescents from the Healthy Brain Network, using supervised machine learning. The goal is to determine whether these indicators could serve as the basis for an early and accessible screening tool. The project assesses feasibility; it does not build a diagnostic tool.

## Methods Used

- CRISP-DM process model
- Exploratory data analysis
- Ordinal classification: multiclass baseline, Frank and Hall decomposition, and regression with optimized thresholds
- Regularized logistic regression, random forests, gradient boosting, and support vector machines
- Nested, repeated, stratified cross-validation
- Quadratic weighted kappa (QWK), per-level sensitivity, and under-estimation rate
- Model interpretability with SHAP values
- Error analysis by sex and age group

## Technologies

- Python
- uv
- pandas, NumPy, PyArrow
- scikit-learn
- SHAP
- Matplotlib, seaborn

## Project Description

**Data.** The project uses the [Child Mind Institute — Problematic Internet Use](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use) dataset from Kaggle, derived from the Healthy Brain Network. The training set contains 3,960 participants aged 5 to 22; 2,736 of them have an SII label. The data is not redistributed in this repository; see [data/README.md](data/README.md) to obtain it.

**Target.** The SII is an ordinal variable with four levels (none, mild, moderate, severe), derived from the Parent-Child Internet Addiction Test. The classes are highly imbalanced: only 34 labeled participants fall in the severe level.

**Feature sets.** Three nested feature sets are compared on the same participants and cross-validation splits:

| Set | Variables |
| --- | --- |
| Reference | Demographics (age, sex, enrollment season) |
| Main | Demographics + physical measures (anthropometrics, vital signs, FitnessGram, bioelectrical impedance, physical activity questionnaire) |
| Complementary | Main + sleep disturbance, global functioning, and internet use hours |

**Approach.** Models are searched on the main set. The selected configuration is then trained on the reference and complementary sets, so that performance differences reflect the information carried by the variables rather than the algorithm. Missing-data strategy and hyperparameters are selected within the inner cross-validation loop to prevent data leakage, and the PCIAT columns that define the target are excluded.

**Evaluation.** Because QWK is symmetric, it is complemented with per-level sensitivity and the rate at which moderate and severe cases are under-estimated, which is the most costly error in a screening context.

**Scope and ethics.** The data comes from minors in a clinical, non-representative sample from the New York area, and the target relies on parent report. Results are intended for academic analysis only and do not constitute a diagnostic tool.

## Getting Started

1. Clone the repository.
2. Install [uv](https://docs.astral.sh/uv/).
3. Run `uv sync`. This installs dependencies and the `piu_severity` package in editable mode.
4. Obtain the data as described in [data/README.md](data/README.md).

Anyone not using uv can generate a `requirements.txt` with:

```bash
uv export --format requirements-txt > requirements.txt
```

## Project Structure

```
piu-severity-prediction/
├── README.md                 # Project overview
├── .gitignore
├── .python-version           # Python version pinned for uv
├── pyproject.toml            # Project metadata and dependencies
├── uv.lock                   # Locked dependency versions (reproducibility)
├── LICENSE                   # MIT license (code only; data has its own Kaggle license)
├── deliverables/             # Documents submitted for each course milestone
│   ├── README.md
│   ├── deliverable-1/        # Milestone 1 documents
│   ├── deliverable-2/        # Milestone 2 documents
│   └── deliverable-3/        # Milestone 3 documents
├── docs/                     # Cross-cutting technical documentation
│   └── README.md
├── references/               # Bibliography and source materials
├── data/                     # Local datasets (not versioned)
│   ├── README.md
│   ├── raw/                  # Immutable original data
│   ├── interim/              # Intermediate transformations
│   └── processed/            # Final datasets ready for modeling
├── notebooks/                # Exploration and narrative
│   └── README.md
├── src/
│   └── piu_severity/         # Installable package
│       ├── __init__.py
│       ├── config.py         # Centralized paths and settings
│       ├── data/             # Loading, downloading, and cleaning
│       ├── features/         # Feature engineering by domain
│       ├── models/           # Pipelines, training, and evaluation
│       └── visualization/    # Reusable plotting functions
├── models/                   # Trained model artifacts
├── reports/
│   └── figures/              # Generated figures
└── tests/                    # Automated tests
```

## Featured Deliverables

- [Deliverable 1](deliverables/deliverable-1/) — Project formulation: problem analysis, state of the art, problem tree, objectives, proposed methodology, and semester plan (October 19, 2026).
- [Deliverable 2](deliverables/deliverable-2/) — Data understanding and initial experiments: exploratory data analysis, data treatment, validation pipeline, baseline model, and initial hypothesis tests (November 14, 2026, planned).
- [Deliverable 3](deliverables/deliverable-3/) — Final report, video, and oral presentation: modeling, evaluation, interpretability, and feasibility recommendations (December 1, 2026, planned).