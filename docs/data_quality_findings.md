# FYTA Sensor Data — Quality Findings

Synthetic dataset, ~16k readings / 8 devices / 6 plants / 3 weeks. Findings
below are grouped by kind, each classified as a **sensor/pipeline artifact**
or a **possible real physical signal**, with the evidence used to decide and
what's still open.

## 1. Duplicated lines - SENS-8 
- **What**: 15 rows that are exact duplicates (same device_id, timestamp, and all
  values) of another row elsewhere in the file.
- **Detected by**: `validate_duplicates` and `row_count_reconciliation`- 15 exact duplicates.
- **Severity**: Medium.
- **Classification**: Sensor/ingestion, not physical.

- ## 2. Mixed timestamp formats - SENS-07
- **What**: SENS-07 reports `dd/mm/yyyy HH:MM`; every other device reports
  `yyyy-mm-dd HH:MM:SS`.
- **Detected by**: `validate_timestamps` (format-match count per configured
  format list).

- **Severity**: High if unhandled — silently misaligns joins/sorts; low once
  parsed explicitly (see `parse_timestamps`).
- **Classification**: Ingestion/pipeline bug, not physical. Likely a
  different firmware/batch for this device.

## 3. Missing entries - SENS-5
- **What**: 1823/2016 expected readings (9.6% missing).
- **Detected by**: `row_count_reconciliation`.
- **Severity**: Medium-high.
- **Classification**: Sensor/ingestion, needs further analysis.

## 4. Flatlined - SENS-06
- **What**: Reports exactly `25.01` for 1,343 of 2,016 readings (~67% of the
  study period), starting ~36h in.
- **Detected by**: `validate_flatline` — longest run of identical consecutive
  values per (device, column).
- **Severity**: High.
- **Classification**: Dead/stuck sensor, not physical. needs hardware review. 

## 5. Moisture spikes (100–107%) - SENS-04 
- **What**: Moisture jumps to 100–107% (physically impossible) at the start
  of each 2-day cycle, then decays smoothly.
- **Detected by**: `validate_ranges` + manual inspection of the daily
  pattern.
- **Severity**: Medium-high.
- **Classification**: Sensor/calibration, needs hardware review.

## 6. Negative moisture - SENS-08
- **What**: Moisture readings of -1% to -5% on dry days;
- **Detected by**: `validate_ranges` + manual inspection of the daily
- **Severity**: Medium-high.
- **Classification**: Sensor/calibration, needs hardware review.

## 7. Soil_temp reported units - SENS-07
- **What**: Raw values reported in Fahrenheit, not Celsius. Values 61.9–78.0 in the `soil_temp_c` field — implausible
  for a live orchid.
- **Detected by**: range check flags it against the fleet; converting
  temp lands almost exactly inside every other device's 16–26°C range.
- **Severity**: High if uncorrected (unusable/misleading feature); low once
  corrected.
- **Classification**: Sensor/pipeline — unit-labelling bug, not physical.
  Corrected via `correct_temperature_units` for devices listed in
  `cleaning.f_to_c_device_ids` (config-driven so new mislabeled devices don't
  need a code change). Raw value preserved in `soil_temp_c_raw`.
  **Note**: Hard coded, needs temp_units detection.

## 8. EC drift - SENS-03 
- **What**: Gradual, sustained rise, not a step change. (700 → 2,266 µS/cm over the final week) .
- **Detected by**: per-device range/std (`std=488` vs. fleet's `~40`);
  timing checked against UP-1002's one logged fertilising event (6/1),
  which is two weeks before the drift starts.
- **Severity**: Medium-high for anyone modelling EC downstream.
- **Classification**: **Genuinely undetermined from data alone.** Two
  plausible causes: an unlogged additional feeding, or classic EC-probe salt
  fouling (a known maintenance issue that produces exactly this slow-drift
  shape). Left un-auto-corrected in `physical_ranges` (upper bound `null`)
  and instead surfaced via `validate_needs_expert_review` for visibility.
  **Need**: recommend to ask domain expert, probe cleaning/calibration log for SENS-03, or ask the user
  directly whether additional unlogged feeding occurred.

## 9. Elevated light_par - SENS-08
- **What**: SENS-08's max PAR reading is notably higher (max 589) than every other
  device's max (~340).
- **Detected by**: `validate_ranges` + manual inspection of the daily.
- **Severity**: Low.
- **Classification**: Undetermined — could be a real placement difference
  (this plant closer to a window/grow light) or a sensor calibration
  difference. Left flagged, not corrected.
  **Need**: recommend to ask domain expert, ask the user about this plant's location, or check whether
  SENS-08 is a different hardware revision.

## 10. High humidity >100% - SENS-08
- **What**: air humidity up to  131%.
- **Detected by**: per-device range comparison.
- **Severity**: Medium.
- **Classification**: Sensor/calibration, needs hardware review.

## 11. Two devices mapped to one `user_plant_id` (UP-1001, UP-1003)
- **What**: UP-1001 → SENS-01 & SENS-02; UP-1003 → SENS-04 & SENS-06. Both
  device pairs' moisture readings correlate near zero (r≈-0.08, r≈0.10).
- **Detected by**: `validate_mapping_integrity`'s `multi_device_plants`
  check.
- **Severity**: Structural — changes how per-plant features get built.
- **Classification**: Ambiguous — could be two redundant probes in one pot,
  or two distinct plants sharing one `user_plant_id` by data-entry error.
  The near-zero correlation leans toward "distinct plants" but is not proof.
  **Need**: ask whoever owns the FYTA app's plant-creation flow. See
  "Cleaning decision" below for the interim approach.

---

## What I'd check with more time
- Validate with logs on issue #4 on SENS-06
- Add function to detect units before convert.
- Define ranges with domain experts for `ec_us_cm` and `light_par`.
- Confirm with product/ops whether multi-device `user_plant_id`s represent
  redundant sensors or distinct plants — this affects both
  feature construction and any per-plant modeling built on top of it.