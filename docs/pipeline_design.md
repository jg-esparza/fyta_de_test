# Pipeline Design Sketch — 100,000+ devices, 15-min cadence

## Design principle

Never block ingestion on quality checks. Every reading gets written somewhere, always, with whatever quality signal is known about it at that moment — bad data must be visible and queryable, not silently dropped or silently delayed. Checks get cheaper and more contextual the later they run, so cost-sensitive checks run early and context-sensitive checks run late.

Three tiers, increasing context / decreasing frequency
```markdown
Device → Ingest gateway → [Tier 1: per-message]  → raw store (append-only)
                                                         │
                                              [Tier 2: per-device, hourly/daily batch]
                                                         │
                                              [Tier 3: cross-signal, daily/weekly batch]
                                                         │
                                              cleaned store + QC flags → feature store → modelling

```
### Gateway

**Monitors**: ingest rate, device heartbeat/silence, schema-reject rate

**Catches**: dead devices, firmware regressions pushing bad schemas

### Tier 1 — at ingestion, per message (sub-second, stateless)

- Schema validity, timestamp parses against the known format list, physical bounds check against a substrate-keyed config table, duplicate-key detection against a short recent-window cache (device_id+timestamp).
- Failing checks don't reject the message, they attach a flag and the message still lands in the raw store.
- This tier has to be cheap enough to run per-message at fleet scale (millions of readings/day at 15-min × 100k devices), so no cross-row or cross-device logic here.

**Monitors**: % flagged per device/fleet, duplicate rate, timestamp-format mismatch rate

**Catches**: new devices reporting a new format/unit convention, retried-write storms

### Tier 2 — per-device batch (hourly or daily, stateful per device)

- Sampling-gap and row-count reconciliation, flatline detection (stuck-sensor runs), drift detection (rolling mean vs. that device's own baseline).
- Units and datatypes normalization.

**Monitors**: flatline alerts, drift alerts, row-count-vs-expected

**Catches**: stuck sensors, probe fouling, connectivity degradation

### Tier 3 — cross-signal, daily/weekly batch (needs joins across datasets)

- The triangulation checks (log→sensor, image→sensor rule agreement), mapping-integrity checks, fleet-wide anomaly comparisons (is this device an outlier vs its substrate/species peers). 

**Monitors**: rule agreement rate per (log_type, sensor_column) and (image_condition, sensor_column), trending over time

**Catches**: mislabeled logs, image-model regressions, genuinely wrong physical-bounds config

