"""pipeline.py. Hydra entry point for the complete FYTA processing pipeline.

Coordinates loading, validation, cleaning, feature construction, and
cross-modal analysis. Runs the same `run_validation` report analyze.py
produces (rather than a second, divergent copy inside cleaning), then persists
each cleaning stage's action log alongside it, so `outputs/validation`
contains both what was wrong with the raw data and what cleaning did about it.
"""
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from .logging_utils import configure_logging
from .io import ensure_dir, load_inputs
from .validation import run_validation
from .cleaning import clean_sensor_data, parse_context_logs, parse_images
from .features import build_unified_plant_table


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run the full pipeline: validate, clean, build features."""
    configure_logging(str(cfg.logging.level))
    data_dir = Path(str(cfg.data_dir))
    output_dir = ensure_dir(Path(str(cfg.output_dir)))
    for subdir in ["validation", "cleaned", "features", "triangulation"]:
        ensure_dir(output_dir / subdir)

    data = load_inputs(data_dir, dict(cfg.files))

    # Single source of truth for "what's wrong with the raw data" -- same
    # report analyze.py produces, not re-derived inside cleaning.py.
    quality_report = run_validation(data, cfg)
    quality_report.to_csv(output_dir / "validation" / "data_quality_report.csv", index=False)

    cleaned_sensor, sensor_actions = clean_sensor_data(data["sensor"], data["mapping"], cfg)
    parsed_context, context_actions = parse_context_logs(data["contextual"], cfg)
    parsed_images, image_actions = parse_images(data["images"], cfg)

    cleaned_sensor.to_csv(output_dir / "cleaned" / "sensor_cleaned.csv", index=False)
    parsed_context.to_csv(output_dir / "cleaned" / "context_cleaned.csv", index=False)
    parsed_images.to_csv(output_dir / "cleaned" / "images_cleaned.csv", index=False)

    cleaning_actions = pd.concat([sensor_actions, context_actions, image_actions], ignore_index=True)
    cleaning_actions.to_csv(output_dir / "validation" / "cleaning_actions.csv", index=False)

    plant_features = build_unified_plant_table(
        cleaned_sensor, parsed_context, parsed_images,
        tolerance_hours=int(cfg.triangulation.nearest_image_tolerance_hours),
    )
    plant_features.to_csv(output_dir / "features" / "plant_features_15min.csv", index=False)


if __name__ == "__main__":
    main()