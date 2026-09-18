"""Validation rules for raw sensor and contextual datasets.

This module detects structural, completeness, timestamp, sampling,
and physical-plausibility issues without modifying the input data.
"""
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "sensor": ["device_id", "timestamp", "substrate_label", "soil_moisture_vwc", "soil_temp_c", "ec_us_cm", "light_par", "air_humidity_pct"],
    "contextual": ["log_id", "user_plant_id", "log_type", "text", "created_at"],
    "images": ["image_id", "user_plant_id", "captured_at", "image_path", "prediction"],
    "mapping": ["user_plant_id", "device_id", "species"],
}


def _add_issue(
    issues: list[dict[str, Any]],
    severity: str,
    dataset: str,
    check: str,
    message: str,
    count: int = 0,
) -> None:
    """Adds issues to validation output."""
    if count > 0 or severity in {"INFO", "WARNING"}:
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
    """Validates schema with required columns."""
    missing = sorted(set(REQUIRED_COLUMNS[dataset]) - set(df.columns))
    if missing:
        raise ValueError(f"{dataset}: missing required columns: {missing}")
    _add_issue(issues, "INFO", dataset, "schema", "Required columns present")


def validate_nulls(df: pd.DataFrame, dataset: str, issues: list[dict[str, Any]]) -> None:
    """Validates null values."""
    null_counts = df.isna().sum()
    bad = null_counts[null_counts > 0]
    if bad.empty:
        _add_issue(issues, "INFO", dataset, "null_check", "No empty values found")
        return
    for column, count in bad.items():
        _add_issue(
            issues, "WARNING", dataset, "null_check",
            f"Column contains empty values: {column}", int(count)
        )


def validate_duplicates(df: pd.DataFrame, dataset: str, issues: list[dict[str, Any]]) -> None:
    """Validates duplicate values."""
    exact = int(df.duplicated().sum())
    _add_issue(
        issues,
        "WARNING" if exact else "INFO",
        dataset,
        "exact_duplicates",
        "Exact duplicate rows detected" if exact else "No exact duplicate rows",
        exact,
    )


def parse_mixed_timestamp(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Parse known timestamp families explicitly."""
    raw = series.astype("string")
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    slash = raw.str.contains("/", regex=False, na=False)
    parsed.loc[~slash] = pd.to_datetime(raw.loc[~slash], errors="coerce", format="mixed")
    parsed.loc[slash] = pd.to_datetime(raw.loc[slash], errors="coerce", dayfirst=True, format="mixed")
    failures = parsed.isna() & series.notna()
    return parsed, failures


def validate_timestamps(df: pd.DataFrame, dataset: str, column: str, issues: list[dict[str, Any]]) -> pd.Series:
    """Validates timestamp formats."""
    raw = df[column].astype("string")
    parsed, failures = parse_mixed_timestamp(raw)
    count = int(failures.sum())
    slash_format = int(raw.str.contains("/", regex=False, na=False).sum())
    iso_like = int(raw.str.contains("-", regex=False, na=False).sum())
    _add_issue(
        issues,
        "ERROR" if count else "INFO",
        dataset,
        "timestamp_parse",
        f"Timestamp parse failures in {column}" if count else f"All {column} values parsed to ISO-like; from slash-format={slash_format}, ISO-like={iso_like}",
        count,
    )
    return parsed


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
        "sampling_gaps",
        f"Intervals longer than {interval_minutes} minutes detected" if count else "No sampling gaps detected",
        count,
    )
    return df


def validate_ranges(sensor: pd.DataFrame, ranges: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    """Validate values inside defined range."""
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
            f"range:{column}",
            f"Out-of-range values {lo}-{hi}: below={below}, above={above}" if count else "All values within configured range",
            count,
        )

def run_validation(data: dict[str, pd.DataFrame], cfg: Any) -> pd.DataFrame:
    issues: list[dict[str, Any]] = []
    """Run data validation pipeline."""
    for dataset, df in data.items():
        validate_schema(df, dataset, issues)
        validate_nulls(df, dataset, issues)
        validate_duplicates(df, dataset, issues)

    sensor = data["sensor"]
    sensor = sensor.copy()
    sensor["timestamp"] = validate_timestamps(sensor, "sensor", "timestamp", issues)
    for dataset, column in [("contextual", "created_at"), ("images", "captured_at")]:
        validate_timestamps(data[dataset], dataset, column, issues)

    sensor = validate_sampling(sensor, int(cfg.sensor.expected_interval_minutes), issues)
    validate_ranges(sensor, dict(cfg.sensor.physical_ranges), issues)

    result = pd.DataFrame(issues)
    return result
