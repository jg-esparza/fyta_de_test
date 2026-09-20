"""triangulation.py

Cross-validates contextual logs, sensor readings, and image-model predictions
against each other. Answers assignment point 4 (label validation / weak
supervision / cross-validation) and point 6 (feature triangulation) together
-- both ask whether one signal's claim about plant state shows up in another
signal's data, just at different grain (a single log event vs. a general
rule).

This module never changes labels, drops rows, or "resolves" a disagreement --
disagreement is itself the useful output: it flags either a broken sensor, an
unreliable label, an under-logged real event, or a genuinely wrong rule. All
of those are worth a human's attention, none of them should be silently
decided by this code.
"""
import logging
from typing import Any

import pandas as pd
from omegaconf import DictConfig

LOGGER = logging.getLogger(__name__)


def _device_ids_for_plant(mapping: pd.DataFrame, user_plant_id: str) -> list[str]:
    return mapping.loc[mapping["user_plant_id"] == user_plant_id, "device_id"].tolist()


def _plant_sensor_series(sensor: pd.DataFrame, mapping: pd.DataFrame, user_plant_id: str, column: str) -> pd.Series:
    """Time-indexed series for a plant's `column`, pooled across its device(s).

    Triangulation asks a coarser question ("did EC rise for this plant?")
    than per-device modelling does, so averaging across a plant's devices is
    acceptable here even where features.py's resolve_analysis_units keeps
    them distinct for modelling -- these are different consumers with
    different granularity needs.
    """
    device_ids = _device_ids_for_plant(mapping, user_plant_id)
    subset = sensor[sensor["device_id"].isin(device_ids)]
    return subset.groupby("timestamp")[column].mean().sort_index()


def evaluate_log_sensor_rules(
    sensor: pd.DataFrame, context: pd.DataFrame, mapping: pd.DataFrame, rules: list[dict[str, Any]],
) -> pd.DataFrame:
    """For each logged care event, check whether the paired sensor column
    moved in the expected direction within a response window.

    Example rule: a `fertilising` log should be followed by a rise in
    `ec_us_cm` within 48h. Baseline = column's mean over
    `baseline_window_hours` immediately before the event; response = its
    mean over `response_window_hours` immediately after. Agreement is a
    direction check only (rose vs. fell), not a magnitude/dose-response
    test -- deliberately coarse, to catch gross mismatches (a fertilising
    log with literally no EC response) rather than model exact curves.
    """
    rows = []
    for rule in rules:
        log_type, column, direction = rule["log_type"], rule["sensor_column"], rule["expected_direction"]
        response_window = pd.Timedelta(hours=rule["response_window_hours"])
        baseline_window = pd.Timedelta(hours=rule["baseline_window_hours"])

        for _, event in context[context["log_type"] == log_type].iterrows():
            plant_id, t = event["user_plant_id"], event["created_at"]
            series = _plant_sensor_series(sensor, mapping, plant_id, column)
            if series.empty:
                continue

            baseline = series[(series.index >= t - baseline_window) & (series.index < t)]
            response = series[(series.index > t) & (series.index <= t + response_window)]
            if baseline.empty or response.empty:
                rows.append({
                    "log_id": event["log_id"], "user_plant_id": plant_id, "log_type": log_type,
                    "sensor_column": column, "created_at": t, "agreement": None,
                    "note": "insufficient sensor coverage around event",
                })
                continue

            delta = response.mean() - baseline.mean()
            observed = "increase" if delta > 0 else "decrease"
            rows.append({
                "log_id": event["log_id"], "user_plant_id": plant_id, "log_type": log_type,
                "sensor_column": column, "created_at": t,
                "baseline_mean": baseline.mean(), "response_mean": response.mean(), "delta": delta,
                "expected_direction": direction, "observed_direction": observed,
                "agreement": observed == direction, "note": None,
            })
    return pd.DataFrame(rows)


def evaluate_image_sensor_agreement(
    images: pd.DataFrame, sensor: pd.DataFrame, mapping: pd.DataFrame,
    rules: list[dict[str, Any]], tolerance_hours: int,
) -> pd.DataFrame:
    """For each image whose top predicted condition is covered by a rule,
    check whether the plant's sensor value near capture time was above or
    below that *plant's own* median for the paired column.

    Compared against the plant's own median, not a fleet-wide or
    config-specified absolute threshold: normal ranges differ enough by
    species/substrate (the orchid_bark case) that an absolute cutoff would be
    wrong for some plants by construction.
    """
    rows = []
    tol = pd.Timedelta(hours=tolerance_hours)
    rule_by_condition = {r["condition"]: r for r in rules}

    for _, img in images.iterrows():
        condition = img.get("image_top_condition")
        rule = rule_by_condition.get(condition)
        if rule is None:
            continue
        plant_id, t = img["user_plant_id"], img["captured_at"]
        column, direction = rule["sensor_column"], rule["expected_direction"]

        series = _plant_sensor_series(sensor, mapping, plant_id, column)
        if series.empty:
            continue
        window = series[(series.index >= t - tol) & (series.index <= t + tol)]
        if window.empty:
            rows.append({
                "image_id": img["image_id"], "user_plant_id": plant_id, "condition": condition,
                "sensor_column": column, "captured_at": t, "agreement": None,
                "note": "no sensor coverage within tolerance window",
            })
            continue

        plant_median = series.median()
        observed_value = window.mean()
        observed = "above_median" if observed_value > plant_median else "below_median"
        rows.append({
            "image_id": img["image_id"], "user_plant_id": plant_id, "condition": condition,
            "sensor_column": column, "captured_at": t, "plant_median": plant_median,
            "observed_value": observed_value, "expected_direction": direction,
            "observed_direction": observed, "agreement": observed == direction, "note": None,
        })
    return pd.DataFrame(rows)


def evaluate_log_image_agreement(context: pd.DataFrame, images: pd.DataFrame, tolerance_hours: int) -> pd.DataFrame:
    """Pair each image with any logged events near its capture time, as
    descriptive evidence rather than a pass/fail rule.

    log_type <-> image_top_condition relationships are looser and more
    context-dependent than the sensor-based rules above -- e.g. a
    `repotting` log near a "Water-related issue" prediction could be cause
    or coincidence -- so this is left for a human reviewer to interpret
    rather than automatically scored.
    """
    tol = pd.Timedelta(hours=tolerance_hours)
    rows = []
    for _, img in images.iterrows():
        nearby = context[
            (context["user_plant_id"] == img["user_plant_id"])
            & (context["created_at"] >= img["captured_at"] - tol)
            & (context["created_at"] <= img["captured_at"] + tol)
        ]
        for _, log in nearby.iterrows():
            rows.append({
                "image_id": img["image_id"], "user_plant_id": img["user_plant_id"],
                "captured_at": img["captured_at"], "image_top_condition": img.get("image_top_condition"),
                "log_id": log["log_id"], "log_type": log["log_type"], "log_created_at": log["created_at"],
            })
    return pd.DataFrame(rows)


def summarize_agreement(*named_results: tuple[str, pd.DataFrame]) -> pd.DataFrame:
    """Roll rule-based checks up into one agreement-rate-per-check summary --
    the headline table for a data-quality dashboard. Rows with `agreement is
    None` (insufficient coverage) are excluded from the rate, not counted as
    disagreement.
    """
    rows = []
    for check_type, df in named_results:
        if df.empty or "agreement" not in df.columns:
            continue
        scored = df[df["agreement"].notna()]
        if scored.empty:
            continue
        key_col = "sensor_column" if "sensor_column" in scored.columns else "condition"
        for key, group in scored.groupby(key_col):
            rows.append({
                "check_type": check_type, "key": key,
                "n": int(len(group)), "agreement_rate": float(group["agreement"].mean()),
            })
    return pd.DataFrame(rows)


def run_triangulation(
    cleaned_sensor: pd.DataFrame, parsed_context: pd.DataFrame, parsed_images: pd.DataFrame,
    mapping: pd.DataFrame, cfg: DictConfig,
) -> dict[str, pd.DataFrame]:
    """Run all triangulation checks and return each result table plus a
    summary, for pipeline.py to persist under outputs/triangulation/.
    """
    log_sensor = evaluate_log_sensor_rules(
        cleaned_sensor, parsed_context, mapping, list(cfg.triangulation.log_sensor_rules),
    )
    image_sensor = evaluate_image_sensor_agreement(
        parsed_images, cleaned_sensor, mapping, list(cfg.triangulation.image_sensor_rules),
        int(cfg.triangulation.nearest_image_tolerance_hours),
    )
    log_image = evaluate_log_image_agreement(
        parsed_context, parsed_images, int(cfg.triangulation.nearest_image_tolerance_hours),
    )
    summary = summarize_agreement(("log_sensor", log_sensor), ("image_sensor", image_sensor))

    LOGGER.info("Triangulation: %d log-sensor checks, %d image-sensor checks, %d log-image pairs",
                len(log_sensor), len(image_sensor), len(log_image))

    return {
        "log_sensor_agreement": log_sensor,
        "image_sensor_agreement": image_sensor,
        "log_image_proximity": log_image,
        "agreement_summary": summary,
    }