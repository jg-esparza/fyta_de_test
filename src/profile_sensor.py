"""profiling.py

Read-only exploratory profiling of the raw FYTA datasets -- the "look at the
data before anything else" step (assignment point 1). Produces compact
per-entity summary tables rather than a full `.describe()` dump, so the
output is something a person actually reads before diving into
validation.py's issue-by-issue report. Never mutates input data.
"""
import logging
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)


def profile_sensor(sensor: pd.DataFrame) -> pd.DataFrame:
    """Per-device overview: row coverage, substrate, and summary stats for
    each physical measurement column. This is what first surfaced SENS-06's
    near-zero std (flatline) and SENS-07's out-of-family temp range, well
    before any formal validation check ran.
    """
    numeric_cols = ["soil_moisture_vwc", "soil_temp_c", "ec_us_cm", "light_par", "air_humidity_pct"]
    rows = []
    for device_id, group in sensor.groupby("device_id"):
        row: dict[str, Any] = {
            "device_id": device_id,
            "n_rows": len(group),
            "substrate_label": group["substrate_label"].iloc[0],
            "timestamp_min": group["timestamp"].min(),
            "timestamp_max": group["timestamp"].max(),
        }
        for col in numeric_cols:
            values = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}_min"] = values.min()
            row[f"{col}_mean"] = values.mean()
            row[f"{col}_max"] = values.max()
            row[f"{col}_std"] = values.std()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("device_id")


def profile_contextual(context: pd.DataFrame) -> pd.DataFrame:
    """Per-plant log count, date coverage, and log_type distribution."""
    rows = []
    for plant_id, group in context.groupby("user_plant_id"):
        row = {
            "user_plant_id": plant_id,
            "n_logs": len(group),
            "created_at_min": group["created_at"].min(),
            "created_at_max": group["created_at"].max(),
        }
        row.update(group["log_type"].value_counts().to_dict())
        rows.append(row)
    return pd.DataFrame(rows).fillna(0).sort_values("user_plant_id")


def profile_images(images: pd.DataFrame) -> pd.DataFrame:
    """Per-plant image count and date coverage. If predictions have already
    been parsed (an `image_top_condition` column is present), also breaks
    down the distribution of top predicted conditions per plant.
    """
    rows = []
    for plant_id, group in images.groupby("user_plant_id"):
        row = {
            "user_plant_id": plant_id,
            "n_images": len(group),
            "captured_at_min": group["captured_at"].min(),
            "captured_at_max": group["captured_at"].max(),
        }
        if "image_top_condition" in group.columns:
            row.update(group["image_top_condition"].value_counts().to_dict())
        rows.append(row)
    return pd.DataFrame(rows).fillna(0).sort_values("user_plant_id")


def profile_mapping(mapping: pd.DataFrame) -> pd.DataFrame:
    """Devices and species per plant -- surfaces the multi-device-per-plant
    cases (UP-1001, UP-1003) at a glance, ahead of
    validate_mapping_integrity's formal check.
    """
    rows = []
    for plant_id, group in mapping.groupby("user_plant_id"):
        rows.append({
            "user_plant_id": plant_id,
            "n_devices": group["device_id"].nunique(),
            "device_ids": ", ".join(sorted(group["device_id"])),
            "species": ", ".join(sorted(group["species"].unique())),
        })
    return pd.DataFrame(rows).sort_values("user_plant_id")


def profile_datasets(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Run all dataset profiles and return them as a dict for the caller to
    persist individually (one CSV per entity, easier to skim than one wide
    table).

    Expects sensor.timestamp / contextual.created_at / images.captured_at to
    already be parsed to datetime -- run this after `validate_timestamps` (or
    accept that min/max will be lexicographic-string-sorted, not
    chronological, if called on raw strings).
    """
    profiles = {"sensor_overview": profile_sensor(data["sensor"])}
    if "contextual" in data:
        profiles["contextual_overview"] = profile_contextual(data["contextual"])
    if "images" in data:
        profiles["images_overview"] = profile_images(data["images"])
    if "mapping" in data:
        profiles["mapping_overview"] = profile_mapping(data["mapping"])
    LOGGER.info("Profiled %d datasets", len(profiles))
    return profiles