"""Validation rules for raw sensor and contextual datasets.

This module detects structural, completeness, timestamp, sampling,
and physical-plausibility issues without modifying the input data.
"""
import logging

import pandas as pd

from typing import Any
from omegaconf import DictConfig

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


def validate_timestamps(df: pd.DataFrame, dataset: str, column: str, target_format: str, issues: list[dict[str, Any]]):
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
            f"range:{column}:{lo}-{hi}",
            f"Out-of-range values: below={below}, above={above}" if count else "All values within configured range",
            count,
        )

def run_validation(data: dict[str, pd.DataFrame], cfg: DictConfig) -> pd.DataFrame:
    issues: list[dict[str, Any]] = []
    """Run data validation pipeline."""
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
