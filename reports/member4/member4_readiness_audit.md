# Member 4 readiness audit

Audit date: 2 November 2026 (Asia/Colombo). Member: Shadhir.

Status: ready for development-only data auditing and descriptive model inspection, with a recorded notebook provenance discrepancy. Held-out explainability requires new fold-specific fitted clones; none were found in the inspected extracted artifacts. No model was trained and no predictions were generated.

## Locations and scope

Read-only source: `C:/Users/mssmo/Documents/glucose/Smart_Glucose_Work`.

Member 4 output checkout: `C:/Users/mssmo/Documents/glucose/personalized-glucose-forecasting`, branch `feature/shadhir-explainability-clustering`.

The Git checkout contains the starting explainability/clustering notebooks, not the complete Members 1–3 handoff. Inputs remain in the sibling source folder. Future scripts must accept an explicit source root instead of assuming the repository contains those inputs. Nothing was committed or pushed during this audit.

Read the Member 3 handoff and completion record, Asly modeling handoff, both integrity manifests, modeling configuration/environment records, trained-model inventory, both approved feature lists, both splits and fold files, both Ridge feature-group comparisons, subset factory, Shanghai and CGMacros all-feature Fold 1 scripts, and model-saving cells in the modeling notebook. Machine-readable CSVs were inspected programmatically.

Processed CSVs were checked for existence, file hashes and headers only. Full-file hashing reads bytes for integrity checking; no participant measurements, targets, or test rows were parsed or analyzed. Split metadata containing locked participant IDs was used only to check separation. No raw-data reconstruction or independent performance evaluation was performed.

## Artifact integrity

The audit established SHA-256 baselines for **252 existing files** in source reports, results, models, notebooks and processed data. Every baseline was checked again after inspection: **zero source-file changes**. Existing manifests were not rewritten. Archives, virtual-environment contents and the separate `ml_ready` folder are outside this baseline scope.

| Existing record | Entries checked | Matching hashes | Finding |
|---|---:|---:|---|
| Asly artifact manifest | 49 | 48 | Modeling notebook mismatch |
| Member 3 evaluation/report manifest | 63 | 63 | Passed |
| Member 3 model inventory | 10 | 10 | Passed |

These are 122 manifest-entry comparisons, including repeated artifacts across manifests, not 122 unique artifacts. All referenced files exist. Both processed data hashes and all ten model hashes match their recorded values.

The notebook `notebooks/04_asly_model_development.ipynb` differs from the Asly manifest:

- Recorded: `a59ec8764b87801f581ee65f93964eead1c1d49374d6fb1161fb7aef77c1ea42`
- Current: `7f3c5cf3a95249d4f487b643055f41a65f02f7391b545a7928dce2a1196768e4`

The notebook inside `Asly_to_Samhi_Modeling_Handoff_2026-10-09.zip` has the same hash as the current extracted notebook. This narrows the discrepancy to the supplied handoff/manifest rather than this audit; it does not establish why the manifest differs. Preserve both and seek an authoritative explanation before claiming complete Member 2 provenance verification. This discrepancy does not invalidate the independently matching ten model hashes or the Member 3 checks.

Detailed evidence: `member4_protected_checksums.csv`, `member4_manifest_verification.csv`, and `member4_readiness_evidence.json`.

## Participant assignments and features

| Dataset | Development | Locked test | Approved inputs | Validation sizes, folds 1–5 | Saved feature-group fold results |
|---|---:|---:|---:|---|---:|
| Shanghai | 80 | 20 | 25 | 16, 17, 16, 16, 15 | 25 |
| CGMacros | 36 | 9 | 39 | 7, 7, 7, 7, 8 | 30 |

Both split tables have unique nonempty participant IDs, exactly the expected development/test counts, and no overlap. Each fold table contains every development participant exactly once, no locked-test IDs, and exactly five fold labels. Training membership is the complement of the selected validation fold within development participants. The saved result metadata agrees with these fold-specific participant counts.

Feature numbering is complete and ordered; feature names are unique. All approved entries declare `input_X` and no future information. Identifiers, timestamps and target columns are excluded. All approved input names exist in the processed CSV headers. This checks metadata/schema consistency, not independent proof that feature construction never used future information.

The saved modeling configuration specifies `target_glucose_30min`, a 30-minute horizon, participant split seed 42, final-test fraction 0.2, five development folds, and final-test evaluation false.

## Existing development findings

Shanghai has five feature configurations. Glucose + meal has lowest mean MAE (9.54045387997914); all features has MAE 9.542209753562327, RMSE 13.47434015201876 and R² 0.9218473591581035. The MAE difference is about 0.001756; no statistical significance is established by these comparisons.

CGMacros has six configurations. All features has lowest mean MAE (8.233110009745781), lowest RMSE (11.896356112684977) and highest R² (0.9008604371087816). Comparison tables declare development-only scope. This audit does not recompute model metrics.

## Model provenance and reuse

All ten existing joblib pipelines loaded in the available Python environment. Every pipeline's `feature_names_in_` exactly matches its dataset's approved feature order. There are five estimators per dataset: Ridge, KNN, SVR, Random Forest and Gradient Boosting. Persistence is a rule-based baseline, not an additional saved trained pipeline.

Ridge alpha is 10.0 for Shanghai and 0.1 for CGMacros. Shanghai Ridge uses median numeric imputation and standard scaling, plus most-frequent categorical imputation and one-hot encoding of participant sex. Its 25 original inputs become 26 transformed inputs. The saved encoder does not drop a reference category; later coefficient reporting must describe the actual coding. CGMacros Ridge uses median imputation and standard scaling for all 39 inputs. Tree pipelines bypass scaling. See `member4_model_inspection.json`.

The modeling notebook saves `GridSearchCV.best_estimator_` after searches configured with `refit="MAE"` on development data. The exception is CGMacros SVR: its search disables refitting and a separate final pipeline is fitted on a participant-balanced development sample. The handoff reports a 2,000-row-per-participant cap and 71,841 final fitting rows. These are development-fitted artifacts, not models for individual validation folds. This provenance conclusion is supported by code and handoff records; fitted objects alone cannot independently prove training participant identities, and the notebook hash discrepancy remains disclosed.

The subset factory clones the saved pipeline into an unfitted model while preserving settings and retaining requested preprocessing columns. The representative Member 3 scripts fit these clones on fold training participants, predict validation participants in memory, and save one-row metric CSVs. They do not persist those fitted clones or row-level predictions.

No separately saved fold-fitted models or out-of-fold prediction artifacts were found in the inspected extracted model/results/report/notebook directories. The 55 feature-group fold CSVs are metric summaries, not predictions. The modeling notebook's ten model-save calls correspond to the ten inventoried pipelines. Archives were not exhaustively searched for additional artifacts; the handoff archive notebook was inspected only to investigate its hash.

Reuse saved development-fitted models for explicitly descriptive coefficient/tree inspection. For held-out permutation importance, fold-stability coefficients and held-out SHAP, first search any additional team-supplied artifacts, then create only the needed clones using the exact saved folds and existing hyperparameters. Cache new models under Member 4 paths. No such fitting was performed in this audit.

## Environment

Original environment record: Python 3.14.3; pandas 3.0.6; NumPy 2.5.3; scikit-learn 1.9.1; joblib 1.6.0.

The supplied `.venv/Scripts/python.exe` cannot start: it references a missing interpreter at `C:/Users/User/AppData/Roaming/uv/python/cpython-3.14.3-windows-x86_64-none/python.exe`.

Available system Python is 3.13.15. Successful imports: pandas 3.0.6, NumPy 2.5.3, scikit-learn 1.9.1, joblib 1.6.0, matplotlib 3.11.2, SciPy 1.18.1 and SHAP 0.52.0. Seaborn, nbformat and the jupyter distribution are absent from this interpreter. They are optional for script-based analysis; notebook authoring/execution needs a separate runtime check. No package was installed or upgraded. Successful SHAP import does not prove explainer compatibility or numerical additivity; test those during the explanation phase.

The matching core package versions and successful deserialization support reuse, but the changed Python minor version means exact original-environment reproduction is not established. Full import evidence is in `member4_environment_check.csv`.

## Missing evidence and next actions

1. **Provenance issue:** reconcile the notebook hash discrepancy with an authoritative original or owner explanation before marking Member 2 provenance fully verified. Do not repair the old manifest automatically.
2. **Runtime issue:** the supplied virtual environment is unusable. Use an explicitly documented working interpreter; create an isolated environment later if required. Current model inspection succeeded without changing dependencies.
3. **Held-out explanation prerequisite:** no persisted fold models/predictions were found. Plan necessary fold refitting only, not hyperparameter searches or a rerun of Members 1–3's complete experiments.
4. **Next data audit:** verify development-only recording coverage, actual participant rows, missingness, timestamp gaps, glucose units and meal definitions before selecting clustering features. Header checks do not establish these properties.
5. **Repository portability:** source artifacts live outside the checkout. Record paths/configuration and do not copy datasets or trained models into the public repository as part of this step. The repository README currently mentions only CGMacros; the Member 4 deliverables should document both datasets when implementation begins.

Reproduction from the repository root: `python -B reports/member4/run_readiness_audit.py`. The script writes Member 4 evidence only and rechecks protected hashes. The sibling source-folder layout is currently required. The narrative report and proposed inventory are reviewed documents, not automatically regenerated by the script.

Model explanations will describe associations in fitted models, not medical causation. Participant clusters will be exploratory patterns, not diagnoses.

Member 4 readiness audit: COMPLETE, with the qualifications above. Final-test evaluation: NOT STARTED.
