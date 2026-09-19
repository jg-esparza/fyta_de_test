"""Conservative cleaning and normalization of FYTA sensor data.

Preserves raw measurements while creating cleaned values and explicit
quality flags for invalid or corrected observations.
"""
import json
import logging
from typing import Any
from omegaconf import DictConfig

import numpy as np
import pandas as pd

from .validation import validate_schema, validate_duplicates, validate_nulls, validate_timestamps, validate_ranges

LOGGER = logging.getLogger(__name__)

DISEASE_COLUMNS = {
    "Abiotic": "image_abiotic_probability",
    "Water-related issue": "image_water_related_probability",
    "Water deficiency": "image_water_deficiency_probability",
    "Nutrient deficiency": "image_nutrient_deficiency_probability",
    "Fungi": "image_fungi_probability",
    "Animalia": "image_animalia_probability",
    "Water excess and/or uneven watering": "image_water_excess_probability",
}

def handle_duplicates(df: pd.DataFrame, dataset:str, deduplicate_exact_rows: bool = False) -> pd.DataFrame:
    """Handle exact duplicate values."""
    LOGGER.info("[%s] Handling duplicates, deduplicate_exact_rows=%s", dataset, deduplicate_exact_rows)
    df["duplicate_exact_flag"] = df.duplicated(keep="first")
    if deduplicate_exact_rows:
        df = df.loc[~df["duplicate_exact_flag"]].copy()
    return df

def normalize_datetime(series: pd.Series) -> pd.Series:
    """Return a timezone-naive datetime64[ns] Series.

    All datetime keys used in merge_asof must share the same dtype.
    """
    result = pd.to_datetime(series, errors="coerce")

    if result.dt.tz is not None:
        result = result.dt.tz_convert("UTC").dt.tz_localize(None)
    return result.astype("datetime64[ns]")

def parse_mixed_timestamp(series: pd.Series, dataset:str) -> pd.Series:
    """Parse known timestamp into iso_8601 format."""
    LOGGER.info("[%s] Standardize timestamp into target format", dataset)
    raw = series.astype("string")
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    slash = raw.str.contains("/", regex=False, na=False)
    parsed.loc[~slash] = pd.to_datetime(raw.loc[~slash], errors="coerce", format="mixed")
    parsed.loc[slash] = pd.to_datetime(raw.loc[slash], errors="coerce", dayfirst=True, format="mixed")
    return parsed

def converts_temp_units_f_to_c(df: pd.DataFrame, correct_temperature_f_to_c: bool, device_id) -> pd.DataFrame:
    """Converts temperature from Fahrenheit to Celsius."""
    LOGGER.info("[%s] Deal with wrong soil_temp_c in sensor %s", "sensor", device_id)
    df["temperature_unit_corrected_flag"] = False
    if correct_temperature_f_to_c:
        mask = df["device_id"].eq(device_id)
        df.loc[mask, "soil_temp_c"] = (df.loc[mask, "soil_temp_c"] - 32.0) * 5.0 / 9.0
        df.loc[mask, "temperature_unit_corrected_flag"] = True
    return df

def flag_out_of_ranges_values(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Flag values out of defined ranges."""
    ranges = dict(cfg.sensor.physical_ranges)
    action = str(cfg.cleaning.invalid_value_action)

    for col, bounds in ranges.items():
        lo, hi = bounds
        values = pd.to_numeric(df[col], errors="coerce")
        flag = pd.Series(False, index=df.index)
        if lo is not None:
            flag |= values < float(lo)
        if hi is not None:
            flag |= values > float(hi)
        df[f"{col}_invalid_flag"] = flag
        if action == "nan_and_flag":
            df.loc[flag, col] = np.nan
    return df

def clean_sensor_data(sensor: pd.DataFrame, mapping: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Normalize sensor data while preserving raw values."""
    dataset_name = "sensor"
    df = sensor.copy()
    issues: list[dict[str, Any]] = []
    # Validation
    validate_schema(df, dataset_name, issues)
    validate_nulls(df, dataset_name, issues)
    # Handle duplicates
    duplicates = validate_duplicates(df, dataset_name, issues)
    if duplicates:
        df = handle_duplicates(df, dataset_name, bool(cfg.cleaning.deduplicate_exact_rows))
    # Handle timestamps
    wrong_format = validate_timestamps(df, dataset_name, "timestamp", str(cfg.sensor.timestamp_format.target_format), issues)
    if wrong_format:
        df["timestamp"] = parse_mixed_timestamp(df["timestamp"], dataset_name)

    df["timestamp"] = normalize_datetime(df["timestamp"])
    # Preserve raw values before any transformation.
    measurements = ["soil_moisture_vwc", "soil_temp_c", "ec_us_cm", "light_par", "air_humidity_pct"]
    for col in measurements:
        df[f"{col}_raw"] = df[col]

    validate_ranges(df, dict(cfg.sensor.physical_ranges), issues)
    # TODO: Detect incorrect temperature
    # Device-level unit anomaly: SENS-07 reports Fahrenheit despite the C-labelled field.
    df = converts_temp_units_f_to_c(df, bool(cfg.cleaning.correct_temperature_f_to_c), device_id="SENS-07")
    # Handle physical incorrect values or out of defined ranges
    df = flag_out_of_ranges_values(df, cfg)
    # TODO: SENS-03 shows a sustained EC increase compared to other sensores Max=2266.0 mean = 1093.937500, ask domain expert
    # TODO: SENS-08 high light readings max = 589 high compared to other sensores, ask domain expert
    df = df.merge(
        mapping[["device_id", "user_plant_id", "species"]],
        on="device_id",
        how="left",
        validate="many_to_one",
    )
    return df

def parse_context_logs(context: pd.DataFrame) -> pd.DataFrame:
    """Parse contextual user logs into structured plant-care events."""
    dataset_name = "contextual"
    df = context.copy()
    issues: list[dict[str, Any]] = []
    # Validation
    validate_schema(df, dataset_name, issues)
    validate_nulls(df, dataset_name, issues)
    # Handle duplicates
    validate_duplicates(df, dataset_name, issues)
    LOGGER.info("[%s] Parsing contextual user logs into structured plant-care events", dataset_name)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", format="mixed")
    df["created_at"] = normalize_datetime(df["created_at"])

    df["event_watering"] = df["log_type"].eq("watering")
    df["event_fertilising"] = df["log_type"].eq("fertilising")
    df["event_repotting"] = df["log_type"].eq("repotting")
    df["event_light"] = df["log_type"].eq("light")
    return df

def _parse_prediction(value: Any) -> dict[str, Any]:
    """Parse a prediction value and return it as a dict."""
    if isinstance(value, dict):
        return value
    return json.loads(value)

def parse_images(images: pd.DataFrame) -> pd.DataFrame:
    """Parse and normalize image-model predictions.

    Extracts probabilistic plant-health signals from the image prediction JSON
    without treating model predictions as ground-truth labels."""

    dataset_name = "images"
    df = images.copy()
    issues: list[dict[str, Any]] = []
    # Validation
    validate_schema(df, dataset_name, issues)
    validate_nulls(df, dataset_name, issues)
    # Handle duplicates
    validate_duplicates(df, dataset_name, issues)

    df["captured_at"] = pd.to_datetime(df["captured_at"], errors="coerce", format="mixed")
    df["captured_at"] = normalize_datetime(df["captured_at"])

    parsed = df["prediction"].map(_parse_prediction)
    df["image_is_plant_probability"] = parsed.map(lambda x: x.get("is_plant_probability"))
    health = parsed.map(lambda x: x.get("health_assessment", {}))
    df["image_healthy_probability"] = health.map(lambda x: x.get("is_healthy_probability"))
    df["image_model_version"] = parsed.map(lambda x: x.get("model_version"))

    def disease_dict(item: dict[str, Any]) -> dict[str, float]:
        return {
            str(d.get("name")): float(d.get("probability", 0.0))
            for d in item.get("diseases", [])
            if d.get("name") is not None
        }

    disease_maps = health.map(disease_dict)
    for disease, column in DISEASE_COLUMNS.items():
        df[column] = disease_maps.map(lambda x, name=disease: x.get(name, 0.0))

    df["image_top_condition"] = disease_maps.map(
        lambda x: max(x, key=x.get) if x else None
    )
    LOGGER.info("[%s] Parsed %s image predictions", dataset_name,len(df))
    return df
