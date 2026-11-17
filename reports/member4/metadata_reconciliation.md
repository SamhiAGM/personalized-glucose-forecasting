# Member 4 metadata reconciliation addendum

This addendum records evidence newly available from `main`. It supplements the earlier readiness, data, Ridge, permutation and SHAP reports; their historical statements, output hashes and integrity manifests are preserved. No source measurements were changed or converted.

## Integration provenance

- Imported `main`: **7e75bda108b3abf6f6401bd7620cec9d00eda4ab** (merge of Member 1 preprocessing work).
- Preprocessing contribution: **c2083e6** (reachable through that main commit).
- Feature branch before integration: **b97e223512f789c2d95a55fefe63fd70159a3862**, already committed and pushed.
- Integration uses a merge commit on `feature/shadhir-explainability-clustering`, without rebasing or rewriting shared history. It does not merge a PR into main.
- The single add/add conflict was `.gitignore`. The resolution retains both versions' useful rules: virtual environments, Python caches, processed/raw data, temporary/editor files, `.env`, Member 4 model caches, participant-linked CSVs and local SHAP artifacts.

All imported scripts, `reports/data_audit/` files, README and requirements were checked against the exact main commit and preserved. Existing Member 4 tracked code/reports/results were checked against b97e223 and preserved; only this new addendum, reconciliation utility and evidence are added. Imported previously tracked team audit tables are retained as supplied; this work does not create new participant-linked public outputs.

## Source definitions and paths

The following repository paths are from the imported main commit above:

| Evidence | Shanghai | CGMacros |
|---|---|---|
| Glucose dictionary | `reports/data_audit/shanghai_glucose_feature_dictionary.csv` | `reports/data_audit/cgmacros_glucose_feature_dictionary.csv` |
| Approved feature list | `reports/data_audit/shanghai_final_feature_list.csv` | `reports/data_audit/cgmacros_final_feature_list.csv` |
| Temporal dictionary | `reports/data_audit/shanghai_temporal_feature_dictionary.csv` | `reports/data_audit/cgmacros_temporal_feature_dictionary.csv` |
| Meal dictionary | `reports/data_audit/shanghai_meal_feature_dictionary.csv` | `reports/data_audit/cgmacros_meal_feature_dictionary.csv` |
| Standardization | `scripts/03_standardize_shanghai.py` | `scripts/10_standardize_cgmacros.py` |
| Target construction | `scripts/04_create_shanghai_target.py` | `scripts/11_create_cgmacros_target.py` |
| Glucose history | `scripts/05_create_glucose_features.py` | `scripts/12_create_cgmacros_glucose_features.py` |
| Temporal construction | `scripts/06_create_temporal_features.py` | `scripts/14_create_cgmacros_temporal_features.py` |
| Final ML-ready output | `scripts/18_final_cross_dataset_audit.py` | `scripts/17_finalize_cgmacros_ml_ready.py` |

Local processed inputs remain read-only in `C:/Users/mssmo/Documents/glucose/Smart_Glucose_Work/data/processed/`. Both approved feature lists have identical parsed content, feature order and groups to the corresponding local handoff lists (25 Shanghai / 39 CGMacros inputs). Their text is identical after newline normalization; raw file SHA-256 differs because of representation/line endings. These hashes are recorded separately in `metadata_reconciliation_evidence.json`, rather than replacing old checksums.

## Glucose and target units

Both imported glucose dictionaries explicitly assign **mg/dL** to current glucose, lags, changes and rolling glucose summaries; the slope is **mg/dL/min**. Shanghai standardization names the primary CGM field `cgm_mg_dl`; CGMacros maps the primary Libre signal to `glucose_mg_dl` and keeps Dexcom separately. Glucose-feature code copies those fields into `current_glucose`.

Shanghai target code sets `target_timestamp_30min = timestamp + 30 minutes` and joins the future `cgm_mg_dl` on **session_id + exact target timestamp** (`scripts/04_create_shanghai_target.py`, create_target, lines 101–165). CGMacros uses the same exact horizon and joins future `glucose_mg_dl` on **participant_id + exact target timestamp**, with TARGET_HORIZON_MINUTES=30 (`scripts/11_create_cgmacros_target.py`, lines 161–221). Neither target operation rescales glucose: the target inherits **mg/dL**. These are time-keyed lookups, not a fixed-number-of-rows shift or nearest-time approximation.

Finalization selects the approved inputs plus participant/time metadata and target into the named ML-ready files. Shanghai finalization reads `shanghai_meal_features_ready.csv` and writes `shanghai_ml_ready.csv`; CGMacros finalization reads `cgmacros_activity_features_ready.csv` and writes `cgmacros_ml_ready.csv`. The supplied local files have those expected schemas and remain identical to the pre-existing recorded data hashes.

**Conclusion:** source declarations, target-copy operations, matching approved schemas and the development checks below support labeling these local glucose inputs and target as mg/dL in future analysis. This is a documented unit reconciliation, not an independently rerun raw-data provenance proof. Earlier figures saying “units unverified” remain accurate records of the evidence available when generated; they are not edited here. No numerical conversion or model rerun is warranted by this metadata addition.

## Development-only consistency checks

`notebooks/member4/reconcile_metadata.py` uses the existing loader and original split/fold files. Locked rows are excluded before measurement checks. No original preprocessing scripts were executed, no models were fitted, and no predictions were made.

| Check | Shanghai | CGMacros |
|---|---:|---:|
| Development participants | 80 | 36 |
| Development rows | 91,829 | 534,235 |
| Exact +30-minute target timestamps | All rows | All rows |
| Target values checked against future current glucose remaining in ML-ready data | 91,635 | 532,264 |
| Target value mismatches | 0 | 0 |
| Rows without a future current-glucose counterpart in the ML-ready subset | 194 | 1,971 |
| Change/slope definition mismatches | 0 | 0 |
| Day/night definition mismatches | 0 | 0 |

The checked identities are current minus lag at 15/30 minutes, and 15-minute change divided by 15 for slope. Tolerances: absolute 1e-8 and relative 1e-9; residuals are floating-point roundoff only. Eligible target values match exactly. The unmatched target comparisons are **not target errors or missing targets**: their future timestamp is absent from the filtered ML-ready history used for this limited lookup. No raw or locked-test rows were accessed to fill that evidence gap. Full target lineage, session boundaries and every rolling feature were not independently reconstructed from raw data.

## Definitions relevant to participant clustering

- Existing development audit found modal ML-ready intervals of **15 minutes for Shanghai** and **1 minute for CGMacros**. These describe retained model rows, not guaranteed raw-sensor cadence or uninterrupted sensor coverage. Keep gap/coverage handling and participant-level aggregation explicit.
- Lags are exact timestamp lookups at 15, 30, 45 and 60 minutes, even for the one-minute dataset. Compute new adjacent changes only across justified intervals; never assume all rows are equally adjacent after filtering.
- Rolling summaries use the past 60 minutes through time t (`closed="both"`, minimum two observations). Thus boundaries are inclusive and windows overlap. Mean, SD, extrema, lags and differences are redundant; do not give them repeated implicit weight in a small clustering feature set. pandas rolling SD uses its default sample-SD convention.
- Shanghai history grouping uses sessions; CGMacros history grouping uses participant. Shanghai ML-ready files omit session IDs. Avoid spanning recording gaps or assuming separate Shanghai sessions form one continuous recording; metadata needed for exact raw-session reconstruction is absent here.
- `day_night_indicator=1` means **06:00 through 17:59**, and 0 otherwise. Imported code/dictionaries and all retained development rows agree. This is a recorded-clock definition; timezone interpretation was not established.
- `elapsed_recording_hours` is measured from the start of a Shanghai monitoring session, but from the participant's first recorded timestamp in CGMacros. It is not a comparable physiological measure across datasets and is affected by recording design.
- Meal features count recorded events at/before t and over the previous 120 minutes through t. CGMacros additionally sums recorded calories/macronutrients. These are meal timing/intake summaries, **not measured post-meal glucose responses**. Any response feature needs a separate event alignment, gap/overlap rule and coverage check. Do not use the future prediction target as a clustering feature.
- Use development participants only for clustering, separately for each dataset. Participant IDs/time/target are metadata rather than clustering predictors. Clinical summaries can be used for descriptions after clusters are fixed; clinical values are not uniformly constant in all Shanghai records, as the permutation audit documented.

## Remaining uncertainties and historical integrity

1. The artifact lineage is supported by code, definitions, schema and numerical relationships, not a complete execution log linking raw input hashes to every processed output. Raw preprocessing has intentionally not been rerun.
2. Measurement calibration and clinical validity are not established by a unit declaration. No new medical claims follow from this addendum.
3. Recording gaps, retained-row selection, sessions, meal-event reliability, clinical inconsistency and missingness remain relevant limitations for clustering. The processed files are forecasting-eligible subsets rather than necessarily complete continuous sensor records.
4. The original modeling-notebook checksum discrepancy remains **unresolved**. Asly manifest expected `a59ec8764b87801f581ee65f93964eead1c1d49374d6fb1161fb7aef77c1ea42`; the supplied notebook has `7f3c5cf3a95249d4f487b643055f41a65f02f7391b545a7928dce2a1196768e4`. The copy inside the supplied handoff ZIP matched the latter during readiness auditing. This does not identify the cause; no old manifest was corrected. Independently verified original model hashes still match their inventory.
5. The supplied virtual environment remains broken; the working system Python and package differences are documented in earlier reports. This merge did not install or change dependencies.

## Checks and reproduction

```powershell
python -B notebooks/member4/reconcile_metadata.py --source-root C:\Users\mssmo\Documents\glucose\Smart_Glucose_Work
$env:MEMBER4_SOURCE_ROOT='C:\Users\mssmo\Documents\glucose\Smart_Glucose_Work'
python -B notebooks/member4/test_data_access.py
python -B notebooks/member4/test_ridge_coefficients.py
python -B notebooks/member4/test_permutation_importance.py
python -B notebooks/member4/test_tree_shap.py
```

All **17 focused tests passed** after integration. All **252 protected source files** retain their baseline hashes before and after reconciliation. Earlier Member 4 reports, results and integrity manifests remain unchanged. No test-set measurements were analyzed. Clustering remains unfinished.

**Final-test evaluation: NOT STARTED.**
