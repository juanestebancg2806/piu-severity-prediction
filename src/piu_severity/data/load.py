"""Load raw competition files without transformation."""

import pandas as pd

from piu_severity.config import (
    DATA_DICTIONARY_FILE,
    RAW_DATA_DIR,
    TEST_FILE,
    TRAIN_FILE,
)

_DOWNLOAD_HINT = "See data/README.md for download instructions."


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
