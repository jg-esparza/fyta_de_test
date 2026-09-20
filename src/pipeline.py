"""pipeline.py. Hydra entry point for the complete FYTA processing pipeline.

Coordinates loading, validation, cleaning, feature construction,
and cross-modal analysis.
"""
from pathlib import Path

import hydra
from omegaconf import DictConfig

from .logging_utils import configure_logging
from .io import ensure_dir, load_inputs
from .cleaning import clean_sensor_data, parse_context_logs, parse_images
from .features import build_unified_plant_table

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run validation pipeline and create validation quality report."""
    configure_logging(str(cfg.logging.level))
    data_dir = Path(str(cfg.data_dir))
    output_dir = ensure_dir(Path(str(cfg.output_dir)))

    data = load_inputs(data_dir, dict(cfg.files))
    for subdir in ["validation", "cleaned", "features", "triangulation"]:
        ensure_dir(output_dir / subdir)

    cleaned_sensor = clean_sensor_data(data["sensor"], data["mapping"], cfg)
    cleaned_sensor.to_csv(output_dir / "cleaned" / "sensor_cleaned.csv", index=False)

    parsed_context = parse_context_logs(data["contextual"])
    parsed_context.to_csv(output_dir / "cleaned" / "context_cleaned.csv", index=False)

    parsed_images = parse_images(data["images"])
    parsed_images.to_csv(output_dir / "cleaned" / "images_cleaned.csv", index=False)

    plant_features = build_unified_plant_table(
        cleaned_sensor,
        parsed_context,
        parsed_images,
        tolerance_hours=int(cfg.triangulation.nearest_image_tolerance_hours),
    )
    plant_features.to_csv(output_dir / "features" / "plant_features_15min.csv", index=False)

if __name__ == "__main__":
    main()