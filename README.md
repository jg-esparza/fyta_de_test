# FYTA Data Engineer - Test 

## Overview

This project implements a small data engineering pipeline plant monitoring data.

The dataset contains:

- Sensor readings collected every 15 minutes 
- Plant/device mappings 
- User contextual logs such as watering and fertilising 
- Image-model predictions describing potential plant health conditions

The pipeline is deliberately conservative: raw observations are not silently overwritten or discarded. 
Validation and cleaning decisions are explicit, configurable through Hydra, and logged/exported so that data-quality issues remain traceable.

## Design reasoning

The main principle is:

Validate first, preserve raw data, clean only when the transformation is justified, and flag uncertainty rather than hiding it.

Sensor data can contain both genuinely bad measurements and unusual but physically meaningful behaviour.
Therefore, I avoid aggressive global outlier removal.

For example:

- Negative soil moisture is physically implausible and is flagged as invalid.
- Relative humidity above 100% is physically implausible and is flagged.
- A long period without sensor readings is preserved as missing.
- A sustained EC increase is not automatically removed because it may represent a real environmental change, fertilisation, substrate effects, sensor drift, or another domain-specific effect. 
- A suspected temperature unit mismatch is corrected only when the evidence supports the conversion, and the correction is explicitly flagged.

The raw dataset remains the source of truth; cleaned values and quality flags are additional outputs.

## Data Validation

Validation performed on all datasets

The following structural checks are applied:

- Schema 
- Null-values 
- Duplicates 
- Timestamp-format, using target_format configured through Hydra 

### Sensor-specific validation

The sensor dataset additionally receives:

- Sampling/completeness validation from expected values
- Expected 15-minute interval checks
- Missing-period detection 
- Physical-range validation using ranges configured through Hydra

The validation report is exported as CSV for further exploration.

### Data pipeline

Performs the actual data preparation and feature construction. Includes:

- Data validation
- Deduplication of exact duplicate rows, configured through Hydra
- Timestamp validation and normalization into predefined format
- Temperature unit normalization. Note: Do not detect different unit
- Physical sensor range validation and flagging, ranges defined in Hydra configuration
  - Optional config with option to flag only or flag and replace with 'nan'
- Contextual table parsed into structured event information
- JSON/model output parsed into separate columns 
- Cleaned data is exported ito csv files
- Create unified plant table with useful aggregate features
  - such as: mean, min, max, std

```markdown
fyta_de_test/ 
├── conf/ 
│ └── config.yaml
│
├── data/ 
│ ├── fyta_sensor_sample.csv 
│ ├── fyta_contextual.csv 
│ ├── fyta_images.csv 
│ └── user_plant_device_map.csv
│
├── src/ 
│ ├── analyze.py 
│ ├── pipeline.py 
│ ├── io.py 
│ ├── logging_utils.py 
│ ├── validation.py 
│ ├── cleaning.py
│ └── features.py
│
├── outputs/ 
│ ├── validation/ 
│ ├── cleaned/ 
│ └── features/ 
│
├── requirements.txt 
├── README.md 
└── .gitignore
```
## Installation

Create a virtual environment and install the project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running analysis

```bash
python -m src.analyze
```

### Running pipeline

```bash
python -m src.pipeline
```
