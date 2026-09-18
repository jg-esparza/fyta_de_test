"""Run the read-only Test exploratory data-quality analysis.

Loads the raw datasets, executes validation checks, creates a human-readable
quality report.
"""
from pathlib import Path

import hydra
from omegaconf import DictConfig

from .logging_utils import configure_logging
from .io import ensure_dir, load_inputs
from .validation import run_validation


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run validation pipeline and create validation quality report."""
    configure_logging(str(cfg.logging.level))
    data = load_inputs(Path(str(cfg.data_dir)), dict(cfg.files))
    report = run_validation(data, cfg)
    output = ensure_dir(Path(str(cfg.output_dir)) / "validation")
    report.to_csv(output / "data_quality_report.csv", index=False)



if __name__ == "__main__":
    main()
