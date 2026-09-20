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

from src.logging_utils import configure_logging
from src.io import ensure_dir, load_inputs, export_to_csv
from src.validation import run_validation
from src.cleaning import clean_sensor_data, parse_context_logs, parse_images
from src.features import build_unified_plant_table
from src.triangulation import run_triangulation


@hydra.main(version_base=None, config_path="conf", config_name="config")
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
    export_to_csv(quality_report, output_dir / "validation" / "data_quality_report.csv")

    cleaned_sensor, sensor_actions = clean_sensor_data(data["sensor"], data["mapping"], cfg)
    parsed_context, context_actions = parse_context_logs(data["contextual"], cfg)
    parsed_images, image_actions = parse_images(data["images"], cfg)

    export_to_csv(cleaned_sensor, output_dir / "cleaned" / "sensor_cleaned.csv")
    export_to_csv(parsed_context, output_dir / "cleaned" / "context_cleaned.csv")
    export_to_csv(parsed_images, output_dir / "cleaned" / "images_cleaned.csv")

    cleaning_actions = pd.concat([sensor_actions, context_actions, image_actions], ignore_index=True)
    export_to_csv(cleaning_actions, output_dir / "validation" / "cleaning_actions.csv")

    plant_features = build_unified_plant_table(
        cleaned_sensor, parsed_context, parsed_images,
        tolerance_hours=int(cfg.triangulation.nearest_image_tolerance_hours),
    )
    export_to_csv(plant_features, output_dir / "features" / "plant_features_15min.csv")

    triangulation_results = run_triangulation(cleaned_sensor, parsed_context, parsed_images, data["mapping"], cfg)
    tri_dir = output_dir / "triangulation"
    for name, df in triangulation_results.items():
        export_to_csv(df, tri_dir / f"{name}.csv")

if __name__ == "__main__":
    main()