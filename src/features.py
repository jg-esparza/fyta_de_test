from __future__ import annotations

import logging

import pandas as pd

LOGGER = logging.getLogger(__name__)

IMAGE_FEATURES = [
    "image_is_plant_probability",
    "image_healthy_probability",
    "image_abiotic_probability",
    "image_water_related_probability",
    "image_water_deficiency_probability",
    "image_water_excess_probability",
    "image_nutrient_deficiency_probability",
    "image_fungi_probability",
    "image_animalia_probability",
]

def _aggregate_sensor(sensors: pd.DataFrame) -> pd.DataFrame:
    s = sensors.sort_values("timestamp").copy()
    group_cols = ["user_plant_id", "timestamp"]
    numeric = ["soil_moisture_vwc", "soil_temp_c", "ec_us_cm", "light_par", "air_humidity_pct"]
    agg = s.groupby(group_cols)[numeric].agg(["mean", "min", "max", "std"]).reset_index()
    agg.columns = ["_".join(x).strip("_") if isinstance(x, tuple) else x for x in agg.columns]
    counts = s.groupby(group_cols).agg(
        sensor_count=("device_id", "nunique")
    ).reset_index()
    return agg.merge(counts, on=group_cols, how="left")


def _nearest_image_features(plant_index: pd.DataFrame, images: pd.DataFrame, tolerance_hours: int) -> pd.DataFrame:
    """Nearest image join performed per plant to guarantee merge_asof sort order."""
    outputs = []
    for plant_id, left in plant_index.groupby("user_plant_id", sort=False):
        right = images.loc[images["user_plant_id"].eq(plant_id)].copy()
        left = left.sort_values("timestamp")
        right = right.sort_values("captured_at")
        cols = ["captured_at", *IMAGE_FEATURES, "image_top_condition"]
        outputs.append(pd.merge_asof(
            left, right[cols],
            left_on="timestamp", right_on="captured_at",
            direction="nearest",
            tolerance=pd.Timedelta(hours=tolerance_hours),
        ))
    return pd.concat(outputs, ignore_index=True)


def build_unified_plant_table(
    sensors: pd.DataFrame,
    context: pd.DataFrame,
    images: pd.DataFrame,
    tolerance_hours: int = 24,
) -> pd.DataFrame:
    """Build unified time-indexed plant-level features.

    Aggregates sensor measurements by plant and aligns contextual events and
    image predictions onto the common 15-minute timeline."""
    sensors = sensors.copy()
    context = context.copy()
    images = images.copy()

    sensor_features = _aggregate_sensor(sensors)
    start = sensor_features["timestamp"].min()
    end = sensor_features["timestamp"].max()

    # Reindex each plant to a complete 15-minute grid. Missingness is represented explicitly.
    plants = sensors[["user_plant_id", "species"]].drop_duplicates()
    grids = []
    for _, row in plants.iterrows():
        idx = pd.date_range(start, end, freq="15min")
        grids.append(pd.DataFrame({
            "user_plant_id": row.user_plant_id,
            "species": row.species,
            "timestamp": idx,
        }))
    grid = pd.concat(grids, ignore_index=True)
    result = grid.merge(sensor_features, on=["user_plant_id", "timestamp"], how="left")

    context = context.copy()
    context["created_at"] = pd.to_datetime(context["created_at"], errors="coerce", format="mixed")
    for event in ["watering", "fertilising", "repotting", "light"]:
        events = context.loc[context["log_type"].eq(event), ["user_plant_id", "created_at"]].copy()
        events["event"] = 1
        joined = []
        for plant_id, left in result.groupby("user_plant_id", sort=False):
            right = events.loc[events["user_plant_id"].eq(plant_id)]
            joined.append(pd.merge_asof(
                left.sort_values("timestamp"),
                right[["created_at", "event"]].sort_values("created_at"),
                left_on="timestamp", right_on="created_at",
                direction="backward",
                tolerance=pd.Timedelta(hours=24),
            ))
        result = pd.concat(joined, ignore_index=True).rename(
            columns={"event": f"{event}_last_24h"}
        ).drop(columns=["created_at"], errors="ignore")

    result = _nearest_image_features(result, images, tolerance_hours)
    LOGGER.info("Built unified table with %s rows", len(result))
    return result.sort_values(["timestamp", "user_plant_id"]).reset_index(drop=True)
