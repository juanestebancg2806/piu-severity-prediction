# PIU Severity Prediction

Part of the "Proyecto I de Innovación Tecnológica" course, Applied Artificial Intelligence Master, Universidad Icesi, Cali, Colombia.

## Project Status

Active

## Contributing Members

**Instructor:** [Milton Orlando Sarria](https://github.com/miltonsarria)

| Name |
| --- |
| Juan David Martinez Legarda |
| Katherin Adriana Camargo Cetina |
| Daniel Velasco López |
| Juan Esteban Cardona Garcia |

## Project Intro/Objective

[placeholder]

## Methods Used

[placeholder]

## Technologies

- Python
- uv
- pandas, NumPy, PyArrow
- scikit-learn
- Matplotlib, seaborn

## Project Description

[placeholder]

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

- [Deliverable 1](deliverables/deliverable-1/) — [placeholder]
- [Deliverable 2](deliverables/deliverable-2/) — [placeholder]
- [Deliverable 3](deliverables/deliverable-3/) — [placeholder]
