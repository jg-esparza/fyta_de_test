# FYTA Data Engineer - Test 

## Overview

Ingests raw sensor, contextual-log, image-prediction, and device-mapping
data; characterizes data-quality issues; produces a cleaned, flagged dataset
and a unified per-plant feature table; cross-validates logs/sensors/images
against each other.

# Structure

```markdown
fyta_de_test/ 
├── conf/ 
│ └── config.yaml                # file paths, physical ranges, cleaning behavior, triangulation rules
│
├── data/                        # raw data
│ ├── fyta_sensor_sample.csv 
│ ├── fyta_contextual.csv 
│ ├── fyta_images.csv 
│ └── user_plant_device_map.csv
│
├── docs/ 
│ ├── data_quality_findings.md   # key findings
│ └── pipeline_design.md         # production-scale design sketch
│
├── src/
│ ├── io.py                      # inputs / outputs utils
│ ├── logging_utils.py           # configure_logging
│ ├── profile.py                 # profile_datasets -- first-look exploration
│ ├── validation.py              # data validation, detection only
│ ├── cleaning.py                # correction/normalization based on defined rules
│ ├── features.py                # build unified plant table
│ └── triangulation.py           # cross-modal agreement checks
│
├── outputs/                     # outputs per layer
│ ├── validation/                
│ ├── cleaned/ 
│ ├── features/ 
│ └── triangulation/ 
│
├── analyze.py                   # profiling + validation report only
├── pipeline.py                  # validation, cleaning, build unified plant table, triangulation 
├── requirements.txt             
├── README.md   
└── .gitignore
```

## Design reasoning

- **Detect, then act, never both at once.** validation.py only reports;
  cleaning.py only acts on what's already characterized. They share one
  timestamp parser and one range-check function so they can't disagree.
- **Never silently drop or average away a problem.** Every correction is
  flagged and the raw value is preserved. Ambiguous cases (SENS-03's EC
  drift, the two-device-per-plant mapping) are surfaced for a domain expert
  rather than resolved by a heuristic — see data_quality_findings.md.
- **Disagreement is signal, not error.** triangulation.py never overrules a
  label or a sensor reading; it reports agreement rates so a human can
  decide what a mismatch means.

## Run

Create a virtual environment and install the project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Running profiling + validation report only

```bash
python -m analyze
```

Running full pipeline

```bash
python -m pipeline
```