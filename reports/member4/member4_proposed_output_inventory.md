# Proposed Member 4 output inventory

All paths are relative to the `personalized-glucose-forecasting` checkout. Original inputs remain read-only in the sibling `Smart_Glucose_Work` folder. Dataset-dependent filenames use `shanghai` or `cgmacros`. This is a plan, not a claim that future analyses are complete.

| Phase | Proposed outputs | Acceptance evidence |
|---|---|---|
| Readiness — completed | `reports/member4/member4_readiness_audit.md`, this inventory, checksum baseline CSV, manifest-verification CSV, environment CSV, model-inspection JSON, readiness-evidence JSON, provenance-evidence text, reproducible audit script | Source hashes unchanged; discrepancy disclosed; no fitting/prediction |
| Development data audit | `notebooks/member4/data_access.py`, `reports/member4/development_data_audit.md`, schema/coverage summaries in `results/member4/audit/` | Exact saved ID/fold membership; test rows excluded before calculations; units/gaps documented |
| Fold model preparation | `notebooks/member4/prepare_fold_models.py`, local `models/member4/{dataset}_ridge_fold{fold}.joblib`, local training-provenance sidecars | Fixed existing hyperparameters; training IDs disjoint from validation/test; transformed feature mapping |
| Ridge coefficients | `results/member4/explainability/{dataset}_ridge_coefficients.csv`, coefficient stability tables and plots | Original versus transformed names, scaling and categorical encoding recorded |
| Permutation importance | Individual and feature-group tables in `results/member4/explainability/`, sampling configuration, fold/repeat summaries and plots | Held-out development-only scores; positive MAE degradation convention; temporal/correlation limitations |
| Tree importance | `{dataset}_{model}_tree_importance.csv` and comparison figures | Correct feature mapping; descriptive fitting scope labeled |
| SHAP | Global importance tables/plots and selected local explanations, explainer configuration and additivity checks | Training-only background; held-out explanations; documented compatibility or fallback |
| Participant feature construction | Local participant feature tables, `reports/member4/clustering_feature_dictionary.md`, coverage/exclusion summaries | One row per eligible development participant; no target/ID/error clustering predictors; justified compact features |
| Cluster comparison | `results/member4/clustering/{dataset}_candidate_scores.csv`, stability summaries, local assignments, saved configuration | K-Means and Agglomerative; silhouette/DBI/cluster sizes; weak structure allowed |
| Cluster visualization | PCA plots, score curves, heatmaps, distributions in `results/member4/figures/` | PCA variance shown; original units used for interpretation; neutral labels |
| Final handoff | `reports/member4/member4_final_report.md`, `member4_reproducibility.md`, `member4_integrity_manifest.csv`, `member4_to_team_handoff.md` | All claims traceable; original hashes preserved; final test remains untouched |

Implementation scripts and approved aggregate reports/figures are candidates for Git. Keep source datasets, participant-linked tables, sample identifiers, fitted pipelines and row-level explanations local pending the team's publication decisions. Do not stage this entire inventory blindly. Commit/push is outside this audit step.
