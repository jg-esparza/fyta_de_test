# Summary Write-up

## Approach. 

1. Profiled all four datasets first (profiling.py / analyze.py) before writing any cleaning logic
2. Built a validation layer that only detects and reports (validation.py)
3. Added cleaning layer that only acts on what validation characterized 
4. Built unified feature table
5. Added triangulation layer that cross-checks logs, sensors, and image predictions against each other.
6. Full findings with severity and sensor-vs-physical classification are in docs/data_quality_findings.md.

### Key findings. 

- Duplicate rows - SENS-08
- Mixed timestamp formats - SENS-07
- Missing rows - SENS-5
- A dead/flatlined sensor - SENS-06
- Moisture out of physical ranges - SENS-04, SENS-08
- An F→C unit mislabel - SENS-07 
- EC drift - SENS-03
- High humidity >100% - SENS-08
- Two devices mapped to one user_plant_id - UP-1001, UP-1003

### Future work

What I'd check with more time:

- Add function to detect units before convert.
- Define ranges with domain experts for `ec_us_cm` and `light_par`.
- Confirm with product/ops whether multi-device `user_plant_id`s represent
  redundant sensors or distinct plants 
- then run triangulation.py's agreement rates against the resolved labels to see whether they change; extend resolve_analysis_units once that's answered; and load-test the Tier 1/2/3 check split from the pipeline design against a synthetic 100k-device fleet to size the drift-detecti