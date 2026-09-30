# Data

Data comes from the Kaggle competition [Child Mind Institute — Problematic Internet Use](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use) (Healthy Brain Network study, de-identified data of minors).

Data is not versioned in this repository. The competition rules forbid redistribution, and the accelerometer parquet files are several GB.

## How to download

TODO: Kaggle API command (the team will fill this in).

## Layout

- `raw/`: immutable original data, as downloaded.
- `interim/`: intermediate transformations.
- `processed/`: final datasets ready for modeling.

## Responsible use

This data belongs to minors. Do not attempt re-identification. Do not share it outside the team.
