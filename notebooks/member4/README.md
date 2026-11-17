# Member 4 development data utilities

Run from the repository root with the working system Python 3.13.15. Core dependencies are pandas 3.0.6 and NumPy 2.5.3. The supplied source virtual environment is broken; see the readiness report. No dependencies were changed for this step.

```powershell
python -B notebooks/member4/audit_development_data.py --source-root C:\Users\mssmo\Documents\glucose\Smart_Glucose_Work
$env:MEMBER4_SOURCE_ROOT='C:\Users\mssmo\Documents\glucose\Smart_Glucose_Work'
python -B notebooks/member4/test_data_access.py
```

Import from a script in this folder:

```python
from data_access import load_development

data = load_development(source_root, 'shanghai')  # or 'cgmacros'
training, validation = data.fold(1)  # existing fold membership
X_train, y_train = data.xy(training)
X_valid, y_valid = data.xy(validation)
```

The loader preserves approved feature order, string participant IDs and original development file order. It validates metadata and all five folds, fails on unknown source IDs, and filters chunks before returning or analyzing measurement data. It retains missing values and irregular timestamps for explicit downstream decisions. It does not fit imputers/scalers or create new splits. Fit predictive preprocessing only on the returned training subset. Keep participant ID and timestamp metadata available separately for grouped sampling and time-aware audits.

Returned objects are mutable pandas data frames; callers must not append locked rows or alter approved metadata. `xy()` validates participant membership again. The five focused tests cover exact fold membership and X columns, locked-row exclusion across chunk boundaries, duplicate assignments, unknown source IDs, target-column injection, and rejecting a locked participant in an X/y request.

Audit outputs: `results/member4/audit/` and `reports/member4/development_data_audit.md`. Participant-linked coverage files end in `.local.csv` and are ignored by Git. Other files contain aggregate diagnostics. No development records are removed for quality issues in this step. Source files are hashed before and after the audit; all remain unchanged.

The audit records units as unverified rather than inferring them from numeric ranges. Coverage describes ML-ready records, not raw sensor uptime. Do not use recording-span endpoints across large gaps as continuous coverage.

## Ridge coefficient analysis

```powershell
python -B notebooks/member4/test_ridge_coefficients.py
python -B notebooks/member4/ridge_coefficients.py --source-root C:\Users\mssmo\Documents\glucose\Smart_Glucose_Work
python -B notebooks/member4/report_ridge_coefficients.py
```

Uses the original saved all-feature Ridge pipelines descriptively, plus training-only clones on the five existing folds per dataset. Hyperparameters and preprocessing are preserved. New fitted caches and hashed provenance sidecars reside in `models/member4/`, which is ignored by Git. The analysis checks source/model/data/split/features/package versions before reusing a cache and refuses incomplete or incompatible caches instead of overwriting them.

Aggregate coefficient tables and correlation diagnostics are in `results/member4/explainability/`; four signed/rank stability figures are in `results/member4/figures/`. The report is `reports/member4/ridge_coefficient_interpretation.md`. The JSON evidence records training scope, cache actions, packages and protected-source integrity. Reference-SD rescaling is a post-fitting reporting step, never a fit/tuning input. Numeric and one-hot categorical magnitudes have separate comparison groups. Final-test rows are excluded before model work.

## Held-out permutation importance

```powershell
python -B notebooks/member4/test_permutation_importance.py
python -B notebooks/member4/permutation_importance.py --source-root C:\Users\mssmo\Documents\glucose\Smart_Glucose_Work
# Inspect reports/member4/permutation_pilot.json before scaling up.
python -B notebooks/member4/permutation_importance.py --source-root C:\Users\mssmo\Documents\glucose\Smart_Glucose_Work --run-all
python -B notebooks/member4/verify_permutation_results.py --source-root C:\Users\mssmo\Documents\glucose\Smart_Glucose_Work
python -B notebooks/member4/report_permutation_importance.py
```

The first analysis command runs Shanghai Fold 1 as a complete timed pilot. `--run-all` requires the same pilot configuration/script hash, reuses its results, and processes the remaining folds. Existing result files are reused only after identity/hash checks. This analysis never calls `fit()` and verifies the model/split/data hashes against the prior training-fold sidecars.

All approved individual inputs and feature groups are evaluated with 10 repeats of global row permutation and within-participant 60-minute block permutation. The master seed is 42; selected blocks and exact repeat seeds are recorded. Each group shares one donor-row mapping across every column. The response target and participant/timestamp metadata are never permuted as predictors.

Validation samples have equal complete-block row counts per participant within each dataset. The planned maximum is 480 rows; Shanghai is limited to 240 rows by the shortest eligible development recording. CGMacros uses 480 rows. All 80/36 development participants appear once across the held-out folds. Only sampled validation rows are predicted, with the corresponding fold-trained model. No source rows are deleted.

Aggregate results are in `results/member4/permutation/`; exact selected-row/coverage files end in `.local.csv` and remain ignored by Git. Report: `reports/member4/permutation_importance_report.md`. Results use **permuted MAE minus original sample MAE**, retaining all negative values. Fold means are aggregated with equal fold weights; fold SDs are descriptive, and repeat/row counts are not independent participant sample sizes.

Within-participant block permutations cannot identify participant-constant feature reliance. Global permutations can generate implausible combinations of correlated histories and other inputs. Read the report's limitations before interpreting rankings or comparing schemes. Glucose units remain unverified.

The saved-result verifier reads assignment metadata and Member 4 outputs only. It checks exact held-out membership, source-row uniqueness, block continuity, sample/output hashes, seeds, signs, repeat counts and reconstruction of fold/aggregate means. Its record is `reports/member4/permutation_verification.json`.
