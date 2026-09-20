"""Run the read-only Test exploratory data-quality analysis.

Loads the raw datasets, executes validation checks, creates a human-readable
quality report.
"""
from pathlib import Path

import hydra
from omegaconf import DictConfig


from src.logging_utils import configure_logging
from src.io import ensure_dir, load_inputs, export_to_csv
from src.profile import profile_datasets
from src.validation import run_validation


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run validation pipeline and create validation quality report."""
    configure_logging(str(cfg.logging.level))
    data = load_inputs(Path(str(cfg.data_dir)), dict(cfg.files))

    profiles = profile_datasets(data)
    profiling_dir = ensure_dir(Path(str(cfg.output_dir)) / "profiling")
    for name, df in profiles.items():
        export_to_csv(df ,profiling_dir / f"{name}.csv")

    report = run_validation(data, cfg)
    output = ensure_dir(Path(str(cfg.output_dir)) / "validation")
    export_to_csv(report, output / "data_quality_report.csv")


if __name__ == "__main__":
    main()
