"""Input/output helpers for the data pipeline.

Provides validated loading of configured CSV inputs and creation of
pipeline output directories.
"""
import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)


def load_csv(path: Path) -> pd.DataFrame:
    """Loads a CSV file."""
    LOGGER.info("Loading %s", path)
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    df = pd.read_csv(path)
    LOGGER.info("Loaded %s rows x %s columns", *df.shape)
    return df

def load_inputs(data_dir: Path, files: dict[str, str]) -> dict[str, pd.DataFrame]:
    """Loads input files from a directory."""
    return {
        name: load_csv(data_dir / filename)
        for name, filename in files.items()
    }

def ensure_dir(path: Path) -> Path:
    """Ensures a directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path

def export_to_csv(df: pd.DataFrame, path: Path) -> None:
    """Exports a pandas DataFrame to a CSV file."""
    df.to_csv(path, index=False)
    LOGGER.info("Saved data into %s", path)