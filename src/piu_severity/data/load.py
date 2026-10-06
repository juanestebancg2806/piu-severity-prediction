"""Load raw competition files without transformation."""

from pathlib import Path

import pandas as pd

from piu_severity.config import (
    ACTIGRAPHY_TRAIN_DIR,
    DATA_DICTIONARY_FILE,
    RAW_DATA_DIR,
    TEST_FILE,
    TRAIN_FILE,
)

_DOWNLOAD_HINT = "See data/README.md for download instructions."
_ID_PREFIX = "id="


def _load_raw_csv(filename: str) -> pd.DataFrame:
    """Read a single CSV from the raw data directory."""
    path = RAW_DATA_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Raw data file not found: data/raw/{filename}. {_DOWNLOAD_HINT}"
        )
    return pd.read_csv(path)


def load_train() -> pd.DataFrame:
    """Load the training set as provided in the raw data directory."""
    return _load_raw_csv(TRAIN_FILE)


def load_test() -> pd.DataFrame:
    """Load the test set as provided in the raw data directory."""
    return _load_raw_csv(TEST_FILE)


def load_data_dictionary() -> pd.DataFrame:
    """Load the data dictionary as provided in the raw data directory."""
    return _load_raw_csv(DATA_DICTIONARY_FILE)


def list_actigraphy_ids(series_dir: Path = ACTIGRAPHY_TRAIN_DIR) -> list[str]:
    """Return the ids of the participants that have an actigraphy series."""
    if not series_dir.is_dir():
        raise FileNotFoundError(f"Actigraphy directory not found: {series_dir}. {_DOWNLOAD_HINT}")
    return sorted(
        path.name.removeprefix(_ID_PREFIX)
        for path in series_dir.iterdir()
        if path.is_dir() and path.name.startswith(_ID_PREFIX)
    )


def load_actigraphy(
    participant_id: str,
    columns: list[str] | None = None,
    series_dir: Path = ACTIGRAPHY_TRAIN_DIR,
) -> pd.DataFrame:
    """Load the actigraphy series of one participant."""
    path = series_dir / f"{_ID_PREFIX}{participant_id}"
    if not path.is_dir():
        raise FileNotFoundError(f"No actigraphy series for participant {participant_id!r}.")
    return pd.read_parquet(path, columns=columns)