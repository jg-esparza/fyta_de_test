"""Validation rules for raw sensor, contextual, image, and mapping datasets.

This module detects structural, completeness, timestamp, sampling, physical-
plausibility, and cross-dataset integrity issues without modifying the input
data. All `validate_*` functions append to a shared `issues` list. Range and
timestamp logic used both here and in cleaning.py is factored into pure,
side-effect-free helpers (`compute_range_flags`, `parse_timestamps`) so
detection and correction can never quietly disagree with each other.
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
    `count > 0`, so a call site can't silently register a warning with nothing
    behind it just by passing that severity string.
    """
    if severity == "INFO" or count > 0:
        issues.append({
            "severity": severity, "dataset": dataset, "check": check,
            "count": count, "message": message,
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
        _add_issue(issues, "WARNING", dataset, "null_check", f"Column contains empty values: {column}", int(count))


def validate_duplicates(
    df: pd.DataFrame,
    dataset: str,
    issues: list[dict[str, Any]],
    key_columns: list[str] | None = None,
) -> bool:
    """Validate duplicate rows: exact full-row duplicates, and, if `key_columns`
    is given, duplicate keys with differing values."""
    exact = int(df.duplicated().sum())
    _add_issue(
        issues, "WARNING" if exact else "INFO", dataset, "exact_duplicates",
        "Exact duplicate rows detected" if exact else "No exact duplicate rows", exact,
    )
    if key_columns:
        key_dupes = int(df.duplicated(subset=key_columns).sum())
        conflicting = max(key_dupes - exact, 0)
        _add_issue(
            issues, "WARNING" if conflicting else "INFO", dataset, f"conflicting_duplicate_keys:{key_columns}",
            (f"Rows sharing key {key_columns} but with differing values" if conflicting
             else f"No conflicting duplicates on key {key_columns}"),
            conflicting,
        )
    return exact > 0


def parse_timestamps(series: pd.Series, formats: list[str]) -> tuple[pd.Series, int]:
    """Parse a timestamp column by trying explicit formats in order, first match wins."""
    raw = series.astype("string")
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    remaining = pd.Series(True, index=series.index)

    for fmt in formats:
        candidates = pd.to_datetime(raw[remaining], format=fmt, errors="coerce")
        matched = candidates.notna()
        parsed.loc[candidates.index[matched]] = candidates[matched]
        remaining.loc[candidates.index[matched]] = False
        if not remaining.any():
            break

    return parsed, int(remaining.sum())


def validate_timestamps(
    df: pd.DataFrame, dataset: str, column: str, formats: list[str], issues: list[dict[str, Any]],
) -> pd.Series:
    """Validate a timestamp column against configured formats and return the parse.

    Returning the parsed series lets callers (`validate_sampling`,
    `validate_flatline`, cleaning.py) reuse this exact parse.
    """
    parsed, unparsed = parse_timestamps(df[column], formats)
    _add_issue(
        issues, "INFO" if unparsed == 0 else "WARNING", dataset, f"timestamp_format:{column}",
        (f"Values in {column} that matched none of {formats}" if unparsed
         else f"All {column} values matched a configured format"),
        unparsed,
    )
    return parsed


def validate_sampling(
    sensor: pd.DataFrame, timestamps: pd.Series, interval_minutes: int, tolerance_minutes: float,
    issues: list[dict[str, Any]],
) -> pd.DataFrame:
    """Validate sampling cadence per device using already-parsed timestamps.

    `timestamps` must come from `validate_timestamps`, so gap detection can't
    disagree with the format check on what a raw value parses to.
    `tolerance_minutes` (from config) sets how much jitter around the nominal
    interval is acceptable before a gap is flagged -- previously hardcoded.
    """
    df = sensor.assign(timestamp=timestamps).sort_values(["device_id", "timestamp"])
    df["delta_minutes"] = df.groupby("device_id")["timestamp"].diff().dt.total_seconds() / 60
    gaps = df["delta_minutes"].gt(interval_minutes + tolerance_minutes)
    count = int(gaps.sum())
    _add_issue(
        issues, "WARNING" if count else "INFO", "sensor",
        f"sampling_gaps:expected_interval_minutes:{interval_minutes}",
        f"Intervals longer than {interval_minutes}+{tolerance_minutes} minutes detected" if count else "No sampling gaps detected",
        count,
    )
    return df


def validate_row_counts(
    sensor: pd.DataFrame, timestamps: pd.Series, interval_minutes: int, issues: list[dict[str, Any]],
) -> None:
    """Flag devices whose total row count falls short of what their own time
    span implies, even if no single gap looked dramatic.

    Complements `validate_sampling`: a device quietly under-reporting (e.g. a
    flaky radio link dropping ~5% of packets) can pass a single-gap check
    while still delivering far less data than expected overall.
    """
    df = sensor.assign(timestamp=timestamps)
    for device_id, group in df.groupby("device_id"):
        valid = group["timestamp"].dropna()
        if valid.empty:
            continue
        span_minutes = (valid.max() - valid.min()).total_seconds() / 60
        expected = int(span_minutes / interval_minutes) + 1
        actual = int(valid.shape[0])
        missing_pct = 100 * (1 - actual / expected) if expected else 0.0
        _add_issue(
            issues, "WARNING" if missing_pct > 1.0 else "INFO", "sensor", f"row_count_reconciliation:{device_id}",
            f"{device_id}: {actual}/{expected} expected readings ({missing_pct:.1f}% missing)",
            max(expected - actual, 0),
        )


def compute_range_flags(df: pd.DataFrame, ranges: dict[str, Any]) -> dict[str, pd.Series]:
    """Pure range check: returns {column: boolean out-of-range mask}, aligned to `df`.

    `ranges` may be flat `{column: [lo, hi]}` or nested `{substrate_label:
    {column: [lo, hi]}}` with an optional "default" key. This is the single
    source of truth for "is this value out of range" -- both `validate_ranges`
    (reporting) and cleaning's `flag_out_of_ranges_values` (row-level
    flag/NaN) call it, so they can't diverge.
    """
    if not ranges:
        return {}
    is_nested = isinstance(next(iter(ranges.values())), dict)
    masks: dict[str, pd.Series] = {}

    groups = df.groupby("substrate_label") if is_nested and "substrate_label" in df.columns else [(None, df)]
    for substrate, group in groups:
        column_ranges = ranges.get(substrate, ranges.get("default", {})) if is_nested else ranges
        for column, bounds in column_ranges.items():
            lo, hi = bounds
            values = pd.to_numeric(group[column], errors="coerce")
            flag = pd.Series(False, index=group.index)
            if lo is not None:
                flag |= values < float(lo)
            if hi is not None:
                flag |= values > float(hi)
            masks.setdefault(column, pd.Series(False, index=df.index))
            masks[column].loc[group.index] = flag
    return masks


def validate_ranges(sensor: pd.DataFrame, ranges: dict[str, Any], issues: list[dict[str, Any]]) -> None:
    """Report out-of-range and non-numeric values using `compute_range_flags`.

    Non-numeric values are counted separately: `pd.to_numeric(errors="coerce")`
    turns them into NaN, and NaN never satisfies `< lo` / `> hi`, so without
    this they'd pass every bound check silently instead of surfacing as data
    corruption.
    """
    if not ranges:
        return
    masks = compute_range_flags(sensor, ranges)
    flat_columns = ranges if not isinstance(next(iter(ranges.values())), dict) else {
        col for sub in ranges.values() for col in sub
    }
    for column in flat_columns:
        raw_nulls = int(sensor[column].isna().sum())
        non_numeric = int(pd.to_numeric(sensor[column], errors="coerce").isna().sum()) - raw_nulls
        if non_numeric > 0:
            _add_issue(issues, "WARNING", "sensor", f"non_numeric:{column}", f"Values in {column} that failed numeric coercion", non_numeric)
        count = int(masks.get(column, pd.Series(dtype=bool)).sum())
        _add_issue(
            issues, "WARNING" if count else "INFO", "sensor", f"range:{column}", f"Out-of-range values in {column}" if count else f"All {column} values within configured range", count,
        )


def validate_flatline(
    sensor: pd.DataFrame, timestamps: pd.Series, columns: list[str], min_run: int, issues: list[dict[str, Any]],
) -> None:
    """Flag devices with implausibly long runs of an unchanging value.

    A sensor that dies mid-study often keeps reporting its last value rather
    than going silent, so row-count/gap checks won't catch it. Detected as the
    longest run of exactly-equal consecutive readings per (device, column).
    `count` holds the run length, not an occurrence count.
    """
    df = sensor.assign(timestamp=timestamps).sort_values(["device_id", "timestamp"])
    for column in columns:
        for device_id, group in df.groupby("device_id"):
            same_as_prev = group[column].diff().eq(0)
            run_id = (~same_as_prev).cumsum()
            longest_run = int(same_as_prev.groupby(run_id).cumsum().max() or 0) + 1
            flagged = longest_run >= min_run
            _add_issue(
                issues, "WARNING" if flagged else "INFO", "sensor", f"flatline:{device_id}:{column}",
                f"Longest run of unchanging values: {longest_run} samples (threshold {min_run})", longest_run,
            )


def validate_needs_expert_review(
    sensor: pd.DataFrame, timestamps: pd.Series, columns: list[str], z_threshold: float,
    window: str, issues: list[dict[str, Any]],
) -> None:
    """Surface unusual regimes for columns we've deliberately chosen not to
    hard-clean (e.g. ec_us_cm, light_par with an open upper bound pending
    domain input).

    This is informational only -- always INFO, never blocks or corrects
    anything -- but without it, nothing tells a reviewer where to look. It
    reports, per device, the largest rolling-mean deviation from that
    device's own baseline, so a domain expert reviewing the report has
    concrete evidence (which device, how far off, over what window) instead
    of having to re-explore the raw CSVs themselves.
    """
    df = sensor.assign(timestamp=timestamps).sort_values(["device_id", "timestamp"])
    for column in columns:
        for device_id, group in df.groupby("device_id"):
            series = group.set_index("timestamp")[column]
            baseline_mean, baseline_std = series.mean(), series.std()
            if not baseline_std or pd.isna(baseline_std):
                continue
            rolling_mean = series.rolling(window).mean()
            max_z = float(((rolling_mean - baseline_mean).abs() / baseline_std).max())
            _add_issue(
                issues, "INFO", "sensor", f"needs_expert_review:{device_id}:{column}",
                f"Max rolling-mean deviation from device baseline: {max_z:.2f} std (window {window}); "
                f"left unflagged/uncorrected pending domain review -- see physical_ranges config",
                1 if max_z > z_threshold else 0,
            )


def validate_image_predictions(images: pd.DataFrame, issues: list[dict[str, Any]]) -> None:
    """Validate that each row's `prediction` field is well-formed JSON with the
    expected top-level keys.

    Catches malformed rows at ingestion, rather than deferring the failure to
    whichever downstream consumer first calls `json.loads` on it.
    """
    required_keys = {"is_plant", "health_assessment"}
    bad_json = missing_keys = 0
    for value in images["prediction"]:
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            bad_json += 1
            continue
        if not required_keys.issubset(parsed.keys()):
            missing_keys += 1

    _add_issue(issues, "WARNING" if bad_json else "INFO", "images", "prediction_json_parse",
               "Predictions that failed to parse as JSON" if bad_json else "All predictions parsed as JSON", bad_json)
    _add_issue(issues, "WARNING" if missing_keys else "INFO", "images", "prediction_schema",
               f"Predictions missing expected keys {sorted(required_keys)}" if missing_keys else "All predictions have expected keys", missing_keys)


def validate_mapping_integrity(data: dict[str, pd.DataFrame], issues: list[dict[str, Any]]) -> None:
    """Validate the device/plant mapping table and its joins into the other datasets.

    Checks: orphan device_ids in sensor data absent from mapping; orphan
    user_plant_ids in contextual/images; any device_id mapped to more than one
    user_plant_id (a genuine conflict, would break `clean_sensor_data`'s
    many_to_one merge); and, for visibility only, any user_plant_id mapped to
    more than one device_id (a supported redundant-sensor setup, not
    necessarily an error, but worth surfacing since it changes how per-plant
    features get aggregated).
    """
    mapping = data["mapping"]

    orphan_devices = sorted(set(data["sensor"]["device_id"]) - set(mapping["device_id"]))
    _add_issue(issues, "WARNING" if orphan_devices else "INFO", "mapping", "orphan_devices",
               f"device_ids in sensor data absent from mapping: {orphan_devices}" if orphan_devices else "All sensor device_ids are mapped",
               len(orphan_devices))

    for dataset in ("contextual", "images"):
        orphan_plants = sorted(set(data[dataset]["user_plant_id"]) - set(mapping["user_plant_id"]))
        _add_issue(issues, "WARNING" if orphan_plants else "INFO", dataset, "orphan_plants",
                   f"user_plant_ids absent from mapping: {orphan_plants}" if orphan_plants else "All user_plant_ids are mapped",
                   len(orphan_plants))

    conflicted = mapping.groupby("device_id")["user_plant_id"].nunique()
    conflicted = conflicted[conflicted > 1]
    _add_issue(issues, "WARNING" if len(conflicted) else "INFO", "mapping", "conflicting_device_mapping",
               f"device_ids mapped to more than one user_plant_id: {list(conflicted.index)}" if len(conflicted) else "No device mapped to multiple plants",
               len(conflicted))

    shared = mapping.groupby("user_plant_id")["device_id"].nunique()
    shared = shared[shared > 1]
    _add_issue(issues, "INFO", "mapping", "multi_device_plants",
               (f"user_plant_ids with more than one device -- confirm this is redundant probes, "
                f"not a labeling error, before aggregating: {list(shared.index)}" if len(shared) else "No plant has multiple devices"),
               len(shared))


def run_validation(data: dict[str, pd.DataFrame], cfg: DictConfig) -> pd.DataFrame:
    """Run the full validation suite across all input datasets.

    Structural checks on every dataset; sensor-specific timestamp, sampling,
    range, flatline, and expert-review checks on `sensor`; timestamp and
    content checks on `contextual`/`images`; cross-dataset mapping integrity
    once `mapping` is available. Timestamps are parsed once per dataset and
    reused by every check that needs them. Returns one row per check; never
    mutates an input dataset.
    """
    issues: list[dict[str, Any]] = []
    formats = list(cfg.sensor.timestamp_format.formats)
    key_columns = {"sensor": ["device_id", "timestamp"], "contextual": ["log_id"], "images": ["image_id"]}

    for dataset, df in data.items():
        validate_schema(df, dataset, issues)
        validate_nulls(df, dataset, issues)
        validate_duplicates(df, dataset, issues, key_columns.get(dataset))

        if dataset == "sensor":
            timestamps = validate_timestamps(df, dataset, "timestamp", formats, issues)
            validate_sampling(df, timestamps, int(cfg.sensor.expected_interval_minutes), float(cfg.sensor.expected_tolerance_minutes), issues)
            validate_row_counts(df, timestamps, int(cfg.sensor.expected_interval_minutes), issues)
            validate_ranges(df, dict(cfg.sensor.physical_ranges), issues)
            validate_flatline(df, timestamps, list(cfg.sensor.flatline_columns), int(cfg.sensor.flatline_min_run), issues)
            validate_needs_expert_review(
                df, timestamps, list(cfg.sensor.expert_review_columns),
                float(cfg.sensor.expert_review_z_threshold), str(cfg.sensor.expert_review_window), issues,
            )
        elif dataset == "contextual":
            validate_timestamps(df, dataset, "created_at", formats, issues)
        elif dataset == "images":
            validate_timestamps(df, dataset, "captured_at", formats, issues)
            validate_image_predictions(df, issues)

    if "mapping" in data:
        validate_mapping_integrity(data, issues)

    return pd.DataFrame(issues)