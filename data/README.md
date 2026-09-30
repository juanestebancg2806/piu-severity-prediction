# Data

Data comes from the Kaggle competition [Child Mind Institute — Problematic Internet Use](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use) (Healthy Brain Network study, de-identified data of minors).

Data is not versioned in this repository. The competition rules forbid redistribution, and the accelerometer parquet files are several GB.

## How to download

1. Create a Kaggle account and **accept the competition rules** on the competition page (downloads fail with a 403 error otherwise).
2. Create an API token in Kaggle → Settings → API, and place `kaggle.json` in `~/.kaggle/` (never inside this repository).
3. From the repository root:

```bash
uv run kaggle competitions download -c child-mind-institute-problematic-internet-use -p data/raw
unzip data/raw/child-mind-institute-problematic-internet-use.zip -d data/raw
```

## Layout

- `raw/`: immutable original data, as downloaded.
- `interim/`: intermediate transformations.
- `processed/`: final datasets ready for modeling.

## Responsible use

This data belongs to minors. Do not attempt re-identification. Do not share it outside the team.
