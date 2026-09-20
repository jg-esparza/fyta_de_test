"""Conservative cleaning and normalization of FYTA sensor data.

Preserves raw measurements while creating cleaned values and explicit
quality flags for invalid or corrected observations. Detection lives in
validation.py; this module only acts on issues already characterized there,
via the same shared parsing/range helpers, so the two can never disagree.
Every cleaning action taken is recorded to a returned action log rather than
discarded, so a "what did cleaning actually change" audit trail survives.
"""
import json
import logging
from typing import Any
from omegaconf import DictConfig

import numpy as np
import pandas as pd

from .validation import validate_schema, compute_range_flags, parse_timestamps

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
# Categories with no parent (top-level umbrellas in the image model's taxonomy).
# Excluded when picking image_top_condition so we don't just report "Abiotic"
# for nearly every row -- see parse_images.
UMBRELLA_CONDITIONS = {"Abiotic"}


def _log_action(actions: list[dict[str, Any]], dataset: str, action: str, message: str, count: int = 0) -> None:
    """Record a cleaning action taken (or explicitly not taken), distinct from
    validation issues: this is an audit trail of what cleaning *did*, not a
    re-statement of what's wrong with the raw data.
    """
    actions.append({"dataset": dataset, "action": action, "count": count, "message": message})
    LOGGER.info("[%s] %s: %s (count=%s)", dataset, action, message, count)


def handle_duplicates(df: pd.DataFrame, dataset: str, deduplicate_exact_rows: bool, actions: list[dict[str, Any]]) -> pd.DataFrame:
    """Flag exact duplicate rows and optionally drop them (keep-first)."""
    df["duplicate_exact_flag"] = df.duplicated(keep="first")
    n = int(df["duplicate_exact_flag"].sum())
    if deduplicate_exact_rows:
        df = df.loc[~df["duplicate_exact_flag"]].copy()
    _log_action(actions, dataset, "handle_duplicates",
                f"{'Dropped' if deduplicate_exact_rows else 'Flagged'} {n} exact duplicate rows", n)
    return df


def normalize_datetime(series: pd.Series) -> pd.Series:
    """Return a timezone-naive datetime64[ns] Series.

    All datetime keys used in merge_asof must share the same dtype.
    """
    result = pd.to_datetime(series, errors="coerce")
    if result.dt.tz is not None:
        result = result.dt.tz_convert("UTC").dt.tz_localize(None)
    return result.astype("datetime64[ns]")


def parse_and_normalize_timestamps(
    series: pd.Series, dataset: str, formats: list[str], actions: list[dict[str, Any]],
) -> pd.Series:
    """Parse a timestamp column using the same explicit-format parser as
    validation.py (`parse_timestamps`), then normalize to a tz-naive dtype.

    Using the shared parser -- rather than each dataset's own
    `format="mixed"` call -- means cleaning can never silently disagree with
    what validation already reported about this column.
    """
    parsed, unparsed = parse_timestamps(series, formats)
    if unparsed:
        _log_action(actions, dataset, "parse_timestamps",
                     f"{unparsed} values did not match any configured format and became NaT", unparsed)
    return normalize_datetime(parsed)


def correct_temperature_units(df: pd.DataFrame, device_ids: list[str], enabled: bool, actions: list[dict[str, Any]]) -> pd.DataFrame:
    """Correct devices known to report Fahrenheit in the Celsius-labelled field.

    `device_ids` is config-driven (`cleaning.f_to_c_device_ids`) rather than
    hardcoded, since new mislabeled devices are expected to show up as the
    fleet grows and shouldn't require a code change. Confirmed for SENS-07 in
    this sample: raw range 61.9-78.0 converts to 16.6-25.6C, matching every
    other device's normal range almost exactly -- strong evidence of a unit
    mislabel rather than a real reading, but still worth confirming with
    whoever owns firmware/device provisioning before trusting it at fleet scale.
    """
    df["temperature_unit_corrected_flag"] = False
    if not enabled or not device_ids:
        return df
    mask = df["device_id"].isin(device_ids)
    df.loc[mask, "soil_temp_c"] = (df.loc[mask, "soil_temp_c"] - 32.0) * 5.0 / 9.0
    df.loc[mask, "temperature_unit_corrected_flag"] = True
    _log_action(actions, "sensor", "correct_temperature_units",
                f"Applied F->C correction to devices {device_ids}", int(mask.sum()))
    return df


def flag_out_of_ranges_values(df: pd.DataFrame, cfg: DictConfig, actions: list[dict[str, Any]]) -> pd.DataFrame:
    """Flag (and optionally null out) physically-implausible values.

    Uses `compute_range_flags`, the same pure function `validate_ranges` uses
    for reporting -- there is exactly one implementation of "is this value
    out of range" in the codebase.
    """
    ranges = dict(cfg.sensor.physical_ranges)
    action = str(cfg.cleaning.invalid_value_action)
    masks = compute_range_flags(df, ranges)

    for column, flag in masks.items():
        df[f"{column}_invalid_flag"] = flag
        if action == "nan_and_flag":
            df.loc[flag, column] = np.nan
        _log_action(actions, "sensor", f"flag_out_of_range:{column}",
                    f"{'Flagged and nulled' if action == 'nan_and_flag' else 'Flagged'} out-of-range values in {column}",
                    int(flag.sum()))
    return df


def mark_unreliable_devices(df: pd.DataFrame, unreliable_device_ids: list[str], actions: list[dict[str, Any]]) -> pd.DataFrame:
    """Carry a `sensor_unreliable_flag` for devices validation identified as
    stuck/flatlined (e.g. SENS-06) for a large share of the study period.

    Deliberately does not drop or interpolate these rows: interpolating over
    ~1,300 stuck readings would fabricate a trend that never happened, and
    dropping them removes a device-health signal a modelling team may still
    want (e.g. to exclude a plant from training while still reporting on it).
    The decision of *which* devices are unreliable is made from
    `validate_flatline`'s report, not re-derived here, to keep detection and
    action from diverging -- pass the device list in from the validation
    report at the call site.
    """
    df["sensor_unreliable_flag"] = df["device_id"].isin(unreliable_device_ids)
    if unreliable_device_ids:
        _log_action(actions, "sensor", "mark_unreliable_devices",
                    f"Marked devices as unreliable (not dropped/interpolated): {unreliable_device_ids}",
                    int(df["sensor_unreliable_flag"].sum()))
    return df


def clean_sensor_data(
    sensor: pd.DataFrame, mapping: pd.DataFrame, cfg: DictConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize sensor data while preserving raw values.

    Returns `(cleaned_df, actions_df)` -- the actions log is no longer
    discarded, so `pipeline.py` can persist it as an audit trail of what
    cleaning changed.
    """
    dataset_name = "sensor"
    df = sensor.copy()
    actions: list[dict[str, Any]] = []

    validate_schema(df, dataset_name, [])  # fail fast; report discarded, cleaning assumes analyze.py already ran the full report

    df = handle_duplicates(df, dataset_name, bool(cfg.cleaning.deduplicate_exact_rows), actions)

    formats = list(cfg.sensor.timestamp_format.formats)
    df["timestamp"] = parse_and_normalize_timestamps(df["timestamp"], dataset_name, formats, actions)

    measurements = ["soil_moisture_vwc", "soil_temp_c", "ec_us_cm", "light_par", "air_humidity_pct"]
    for col in measurements:
        df[f"{col}_raw"] = df[col]  # snapshot before any correction below

    df = correct_temperature_units(
        df, list(cfg.cleaning.f_to_c_device_ids), bool(cfg.cleaning.correct_temperature_f_to_c), actions,
    )
    df = flag_out_of_ranges_values(df, cfg, actions)
    df = mark_unreliable_devices(df, list(cfg.cleaning.unreliable_device_ids), actions)

    df = df.merge(
        mapping[["device_id", "user_plant_id", "species"]], on="device_id", how="left", validate="many_to_one",
    )
    return df, pd.DataFrame(actions)


def parse_context_logs(context: pd.DataFrame, cfg: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse contextual user logs into structured plant-care events."""
    dataset_name = "contextual"
    df = context.copy()
    actions: list[dict[str, Any]] = []

    validate_schema(df, dataset_name, [])
    df = handle_duplicates(df, dataset_name, bool(cfg.cleaning.deduplicate_exact_rows), actions)

    formats = list(cfg.sensor.timestamp_format.formats)
    df["created_at"] = parse_and_normalize_timestamps(df["created_at"], dataset_name, formats, actions)

    for event in ("watering", "fertilising", "repotting", "light"):
        df[f"event_{event}"] = df["log_type"].eq(event)

    _log_action(actions, dataset_name, "parse_context_logs", f"Parsed {len(df)} contextual logs", len(df))
    return df, pd.DataFrame(actions)


def _parse_prediction(value: Any) -> tuple[dict[str, Any], bool]:
    """Parse a prediction value, returning (parsed_dict, ok).

    Never raises: a single malformed row must not crash the whole pipeline.
    On failure returns `({}, False)` so the caller can flag it and continue.
    """
    if isinstance(value, dict):
        return value, True
    try:
        return json.loads(value), True
    except (TypeError, ValueError):
        return {}, False


def _pick_top_condition(disease_probs: dict[str, float], classifications: dict[str, list[str]]) -> str | None:
    """Pick the most likely *specific* condition, preferring child conditions
    over top-level umbrella categories.

    The image model's `diseases`/`diseases_simple` payload mixes parent
    categories (e.g. "Abiotic", `classification: []`) with their children
    (e.g. "Water deficiency", `classification: ["Abiotic", ...]`). A plain
    argmax over probabilities returns the umbrella almost every time, since
    it aggregates its children's probability mass -- which is true but not
    actionable. We argmax over children only (non-empty `classification`),
    falling back to the umbrella only if no child conditions were reported.
    """
    if not disease_probs:
        return None
    children = {name: p for name, p in disease_probs.items() if classifications.get(name)}
    pool = children or disease_probs
    return max(pool, key=pool.get)


def parse_images(images: pd.DataFrame, cfg: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse and normalize image-model predictions.

    Extracts probabilistic plant-health signals from the prediction JSON
    without treating model predictions as ground-truth labels.
    """
    dataset_name = "images"
    df = images.copy()
    actions: list[dict[str, Any]] = []

    validate_schema(df, dataset_name, [])
    df = handle_duplicates(df, dataset_name, bool(cfg.cleaning.deduplicate_exact_rows), actions)

    formats = list(cfg.sensor.timestamp_format.formats)
    df["captured_at"] = parse_and_normalize_timestamps(df["captured_at"], dataset_name, formats, actions)

    parse_results = df["prediction"].map(_parse_prediction)
    parsed = parse_results.map(lambda x: x[0])
    df["prediction_parse_error_flag"] = parse_results.map(lambda x: not x[1])
    n_failed = int(df["prediction_parse_error_flag"].sum())
    if n_failed:
        _log_action(actions, dataset_name, "parse_predictions", f"Failed to parse prediction JSON; flagged, not dropped", n_failed)

    df["image_is_plant_probability"] = parsed.map(lambda x: x.get("is_plant_probability"))
    health = parsed.map(lambda x: x.get("health_assessment", {}))
    df["image_healthy_probability"] = health.map(lambda x: x.get("is_healthy_probability"))
    df["image_model_version"] = parsed.map(lambda x: x.get("model_version"))

    def disease_dict(item: dict[str, Any]) -> dict[str, float]:
        return {str(d.get("name")): float(d.get("probability", 0.0)) for d in item.get("diseases", []) if d.get("name") is not None}

    def classification_dict(item: dict[str, Any]) -> dict[str, list[str]]:
        return {str(d.get("name")): d.get("classification", []) for d in item.get("diseases", []) if d.get("name") is not None}

    disease_maps = health.map(disease_dict)
    classification_maps = health.map(classification_dict)

    for disease, column in DISEASE_COLUMNS.items():
        df[column] = disease_maps.map(lambda x, name=disease: x.get(name, 0.0))

    df["image_top_condition"] = [
        _pick_top_condition(d, c) for d, c in zip(disease_maps, classification_maps)
    ]

    _log_action(actions, dataset_name, "parse_images", f"Parsed {len(df)} image predictions", len(df))
    return df, pd.DataFrame(actions)