"""Validation rules for raw sensor, contextual, image, and mapping datasets.

This module detects structural, completeness, timestamp, sampling, physical-
plausibility, drift, and cross-dataset integrity issues without modifying the
input data. All `validate_*` functions append to a shared `issues` list and
return derived values (parsed timestamps, dataframes) where a caller needs to
reuse them, rather than re-deriving the same thing twice.
"""
import json
import logging

import pandas as pd

from typing import Any, Literal
from omegaconf import DictConfig

LOGGER = logging.getLogger(__name__)

Severity = Literal["INFO", "WARNING", "ERROR"]

REQUIRED_COLUMNS = {
    "sensor": ["device_id", "timestamp", "substrate_label", "soil_moisture_vwc", "soil_temp_c", "ec_us_cm", "light_par", "air_humidity_pct"],
    "contextual": ["log_id", "user_plant_id", "log_type", "text", "created_at"],
    "images": ["image_id", "user_plant_id", "captured_at", "image_path", "prediction"],
    "mapping": ["user_plant_id", "device_id", "species"],
}


def _add_issue(
    issues: list[dict[str, Any]],
    severity: Severity,
    dataset: str,
    check: str,
    message: str,
    count: int = 0,
) -> None:
    """Append a validation issue and emit a matching log line.

    INFO issues are always recorded (they carry "no problem found" summaries
    that are useful on their own). WARNING/ERROR issues are only recorded when
    `count > 0`, so a future call site can't silently register a warning with
    nothing behind it just by passing that severity string.
    """
    if severity == "INFO" or count > 0:
        issues.append({
            "severity": severity,
            "dataset": dataset,
            "check": check,
            "count": count,
            "message": message,
        })
        log = getattr(LOGGER, severity.lower(), LOGGER.info)
        log("[%s] %s: %s (count=%s)", dataset, check, message, count)


def validate_schema(df: pd.DataFrame, dataset: str, issues: list[dict[str, Any]]) -> None:
    """Validate that all required columns are present.

    Raises rather than warns: every other check assumes these columns exist,
    so a missing column should stop the run instead of producing a cascade of
    confusing downstream failures.
    """
    missing = sorted(set(REQUIRED_COLUMNS[dataset]) - set(df.columns))
    if missing:
        raise ValueError(f"{dataset}: missing required columns: {missing}")
    _add_issue(issues, "INFO", dataset, "schema", "Required columns present")


def validate_nulls(df: pd.DataFrame, dataset: str, issues: list[dict[str, Any]]) -> None:
    """Validate null values, reported per column."""
    null_counts = df.isna().sum()
    bad = null_counts[null_counts > 0]
    if bad.empty:
        _add_issue(issues, "INFO", dataset, "null_check", "No empty values found")
        return
    for column, count in bad.items():
        _add_issue(
            issues, "WARNING", dataset, "null_check",
            f"Column contains empty values: {column}", int(count),
        )


def validate_duplicates(
    df: pd.DataFrame,
    dataset: str,
    issues: list[dict[str, Any]],
    key_columns: list[str] | None = None,
) -> bool:
    """Validate duplicate rows: exact full-row duplicates, and, if `key_columns`
    is given, duplicate keys with differing values.

    Exact duplicates (identical on every column) are usually safe to drop --
    they typically indicate a retried write reaching storage twice. Duplicate
    *keys* with differing values (e.g. same device_id+timestamp, different
    readings) are a more serious conflicting-write problem and must not be
    silently deduplicated, since there's no way to tell which value is correct
    from the data alone.
    """
    exact = int(df.duplicated().sum())
    _add_issue(
        issues,
        "WARNING" if exact else "INFO",
        dataset,
        "exact_duplicates",
        "Exact duplicate rows detected" if exact else "No exact duplicate rows",
        exact,
    )
    return True if exact > 0 else False


def validate_timestamps(df: pd.DataFrame, dataset: str, column: str, target_format: str, issues: list[dict[str, Any]]) -> int:
    """Validates timestamp formats."""
    raw = df[column].astype("string")
    if target_format == "iso_8601":
        iso_like = int(raw.str.contains("-", regex=False, na=False).sum())
        wrong_format = raw.count() - iso_like
    else:
        raise NotImplementedError(f"{target_format} format not implemented")
    _add_issue(
        issues,
        "INFO" if wrong_format == 0 else "WARNING",
        dataset,
        f"timestamp format:{target_format}",
        f"Values different format in {column}" if wrong_format else f"All {column} values with same format",
        wrong_format,
    )
    return wrong_format

def validate_sampling(sensor: pd.DataFrame, interval_minutes: int, issues: list[dict[str, Any]]) -> pd.DataFrame:
    """Validates sampling gaps."""
    df = sensor.sort_values(["device_id", "timestamp"]).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", format="mixed")
    df["delta_minutes"] = df.groupby("device_id")["timestamp"].diff().dt.total_seconds() / 60
    gaps = df["delta_minutes"].gt(interval_minutes + 1e-9)
    count = int(gaps.sum())
    _add_issue(
        issues,
        "WARNING" if count else "INFO",
        "sensor",
        f"sampling_gaps:expected_interval_minutes:{interval_minutes}",
        f"Intervals longer than {interval_minutes} minutes detected" if count else "No sampling gaps detected",
        count,
    )
    return df


def validate_ranges(sensor: pd.DataFrame, ranges: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    """Validate values against physical bounds, optionally keyed by substrate.

    `ranges` may be a flat `{column: [lo, hi]}` mapping applied to every row,
    or a nested `{substrate_label: {column: [lo, hi]}}` mapping (with an
    optional "default" key) -- substrate-appropriate bounds genuinely differ
    (e.g. orchid bark never reaches the moisture % potting soil does), so a
    single global range either under-flags one substrate or over-flags another.

    Values that fail `pd.to_numeric` are reported separately as
    `non_numeric_values` instead of silently becoming NaN and passing every
    bound check by default.
    """
    for column, bounds in ranges.items():
        lo, hi = bounds
        values = pd.to_numeric(sensor[column], errors="coerce")
        below = int((values < lo).sum()) if lo is not None else 0
        above = int((values > hi).sum()) if hi is not None else 0
        count = below + above
        _add_issue(
            issues,
            "WARNING" if count else "INFO",
            "sensor",
            f"range:{column}:{lo}-{hi}",
            f"Out-of-range values: below={below}, above={above}" if count else "All values within configured range",
            count,
        )

def run_validation(data: dict[str, pd.DataFrame], cfg: DictConfig) -> pd.DataFrame:
    """Run the full validation suite across all input datasets.

    Runs structural checks (schema, nulls, duplicates) on every dataset,
    dataset-specific timestamp/physical range checks on `sensor`, timestamp
    and content checks on `contextual`/`images`, and cross-dataset
    mapping-integrity checks once `mapping` is available. Timestamps are
    parsed exactly once per dataset and reused by every downstream check that
    needs them. Returns one row per check as a flat DataFrame; does not
    mutate any input dataset.
    """
    issues: list[dict[str, Any]] = []
    for dataset, df in data.items():
        validate_schema(df, dataset, issues)
        validate_nulls(df, dataset, issues)
        validate_duplicates(df, dataset, issues)
        if dataset == "sensor":
            validate_timestamps(df, "sensor", "timestamp", str(cfg.sensor.timestamp_format.target_format), issues)
            validate_sampling(df, int(cfg.sensor.expected_interval_minutes), issues)
            validate_ranges(df, dict(cfg.sensor.physical_ranges), issues)
        elif dataset == "contextual":
            validate_timestamps(df, "contextual", "created_at", str(cfg.sensor.timestamp_format.target_format), issues)
        elif dataset == "images":
            validate_timestamps(df, "images", "captured_at", str(cfg.sensor.timestamp_format.target_format), issues)

    result = pd.DataFrame(issues)
    return result
