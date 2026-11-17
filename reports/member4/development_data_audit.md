# Member 4 development-data audit



Development-only audit; no training, preprocessing fitting, clustering or predictions. Existing saved folds are unchanged.

Source: `C:\Users\mssmo\Documents\glucose\Smart_Glucose_Work`. All 252 protected files retain identical SHA-256 hashes.



CSV chunks are filtered by approved participant assignments before measurement analysis. Locked rows are counted only for exclusion accounting. No test measurements or targets are summarized.



| Dataset | Development rows | Participants | Locked rows excluded | Backwards steps | Participants with large gaps |
|---|---:|---:|---:|---:|---:|
| shanghai | 91,829 | 80 | 19,924 | 2 | 13 |
| cgmacros | 534,235 | 36 | 143,835 | 0 | 9 |

## shanghai



```json

{
  "dataset": "shanghai",
  "development_rows": 91829,
  "development_participants": 80,
  "locked_rows_excluded": 19924,
  "invalid_timestamps": 0,
  "invalid_target_timestamps": 0,
  "non_30min_target_timestamps": 0,
  "duplicate_timestamp_extra_rows": 0,
  "backwards_steps": 2,
  "missing_cells": 9411,
  "nonfinite_numeric_cells": 0,
  "participants_with_large_gaps": 13,
  "modal_intervals_minutes": [
    15.0
  ],
  "recording_span_hours_min": 60.0,
  "recording_span_hours_median": 303.0,
  "recording_span_hours_max": 4291.683333333333,
  "median_grid_coverage_fraction": 1.0,
  "current_glucose_quantiles": {
    "0.0": 39.6,
    "0.01": 64.8,
    "0.5": 127.8,
    "0.99": 295.2,
    "1.0": 468.0
  },
  "glucose_units": "UNVERIFIED: no unit field or authoritative unit metadata supplied",
  "quality_rows_excluded": 0
}

```



## cgmacros



```json

{
  "dataset": "cgmacros",
  "development_rows": 534235,
  "development_participants": 36,
  "locked_rows_excluded": 143835,
  "invalid_timestamps": 0,
  "invalid_target_timestamps": 0,
  "non_30min_target_timestamps": 0,
  "duplicate_timestamp_extra_rows": 0,
  "backwards_steps": 0,
  "missing_cells": 616048,
  "nonfinite_numeric_cells": 0,
  "participants_with_large_gaps": 9,
  "modal_intervals_minutes": [
    1.0
  ],
  "recording_span_hours_min": 171.96666666666667,
  "recording_span_hours_median": 249.48333333333332,
  "recording_span_hours_max": 458.18333333333334,
  "median_grid_coverage_fraction": 1.0,
  "current_glucose_quantiles": {
    "0.0": 40.0,
    "0.01": 40.0,
    "0.5": 97.06666666666666,
    "0.99": 252.9333333333333,
    "1.0": 405.0
  },
  "glucose_units": "UNVERIFIED: no unit field or authoritative unit metadata supplied",
  "quality_rows_excluded": 0
}

```



## Interpretation and exclusions



Only locked-test rows were excluded. No development rows were silently dropped, sorted in the returned loader, deduplicated, imputed or converted. The audit sorts unique timestamps only to describe intervals and gaps. Backwards steps refer to original within-participant file order.

Grid coverage is observed unique timestamps divided by the expected grid count across the first-to-last span, using each participantâ€™s modal positive interval, capped at one. It is an approximate recording-density measure, not sensor wear-time or proof of continuous observation. Gaps over twice that interval are flagged; they are not automatically excluded.

Sampling statistics describe ML-ready rows, which can differ from raw sensor cadence after lag/target eligibility filtering. Glucose quantiles cannot establish units. No mg/dLâ†”mmol/L conversion or clinical thresholds were applied; verify authoritative source units before clinical interpretation or unit-dependent clustering features.

Missingness and infinity checks cover retained development rows only. Horizon checks compare recorded timestamps; they do not independently prove target construction. No medical causation or diagnostic claims are made.



Participant coverage files ending `.local.csv` contain development identifiers and remain ignored by Git. Aggregate interval, missingness and fold-check tables are saved alongside the JSON evidence.



Final-test evaluation: NOT STARTED.