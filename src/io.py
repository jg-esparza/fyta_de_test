import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)


def load_csv(path: Path) -> pd.DataFrame:
    LOGGER.info("Loading %s", path)
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    df = pd.read_csv(path)
    LOGGER.info("Loaded %s rows x %s columns", *df.shape)
    return df

def load_inputs(data_dir: Path, files: dict[str, str]) -> dict[str, pd.DataFrame]:
    return {
        name: load_csv(data_dir / filename)
        for name, filename in files.items()
    }

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path