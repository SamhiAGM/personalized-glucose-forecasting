"""Render the short interpretation report from saved aggregate evidence."""
import json
from pathlib import Path
import pandas as pd

REPO=Path(__file__).resolve().parents[2]


def render():
    report_dir=REPO/'reports/member4'
    evidence=json.loads((report_dir/'ridge_coefficient_analysis_evidence.json').read_text())
    parts=['# Member 4: Ridge coefficient interpretation',
        f"Analysis completed {evidence['run_date'][:10]} (Asia/Colombo). Both datasets use all approved features. This report distinguishes the original saved development-fitted models from five training-only fold fits per dataset.",
        '## Scope and provenance',
        'Shanghai uses 91,829 development rows from 80 participants; CGMacros uses 534,235 rows from 36 participants. The saved participant assignments were reused exactly. Shanghai validation-fold sizes are 16/17/16/16/15; CGMacros sizes are 7/7/7/7/8. Every training set is disjoint from its validation participants and locked-test participants.',
        f"Protected files checked: **{evidence['protected_files_checked']}**; changes: **{evidence['protected_changes']}**. Saved Ridge hashes match the Member 3 inventory. No hyperparameter search or model-selection experiment was repeated. No final-test predictions were made. Predictions on up to 32 development fitting rows per model were used only for algebraic reconstruction checks, not performance reporting.",
        'The readiness audit found no compatible fold models. Ten cloned pipelines were fitted once on the existing training partitions and cached under `models/member4/`; subsequent compatible runs reuse them. Cache sidecars record source/model/data/split/feature hashes, unchanged hyperparameters, package versions and hashed participant membership. The models and sidecars stay local and are ignored by Git. The aggregate analysis evidence records how each cache was obtained.',
        'The original modeling-notebook checksum discrepancy recorded in the readiness audit remains unresolved; the ten original model hashes independently match their inventory. The available Python 3.13.15 differs from the original Python 3.14.3, while the recorded core package versions match. No original pipeline was overwritten.',
        '## Coefficient scales and categorical coding',
        'For each numeric input, the fitted transformation is z = (imputed x − training mean) / training scale. The signed transformed coefficient b describes a one-training-SD change. The original-unit slope is b / training scale. Its numeric-system intercept is the transformed intercept minus the sum of these slopes × training means. Both intercepts and per-model imputation/scale statistics are saved.',
        'Fold scales differ. To compare coefficients across folds, the original-unit slope is multiplied by one fixed reference SD from the saved all-development scaler. Thus common-reference coefficient = (fold b / fold scale) × saved reference SD. This rescaling occurs only after fitting and never affects model inputs, regularization or tuning. It is descriptive and uses the development-wide reference solely as a common ruler. Constant reference inputs are excluded from numeric magnitude ranks.',
        'Absolute numeric ranks use this reference-SD scale; 1 means largest magnitude among numeric inputs within that dataset. The tables also preserve raw transformed coefficients and ranks within transform kind. Categorical indicators are unscaled 0/1 columns and are ranked separately; their raw magnitude must not be compared with standardized numeric magnitudes as if the units matched. A one-SD comparison is a model sensitivity convention, not a clinical effect or an independent feature-importance measure. Binary/indicator-like numeric features have SDs that depend on prevalence.',
        'Shanghai has 24 standardized numeric inputs and participant_sex expanded into Female and Male indicator columns, for 26 transformed columns from 25 inputs. The encoder uses `drop=None`: **there is no dropped reference category**. Individual dummy coefficients are offsets in this full coding, not effects relative to Female. The interpretable coding contrast is b_Male − b_Female with other predictors fixed. Unknown categories use `handle_unknown="ignore"` (all-zero coding); that is not a deliberately fitted clinical reference group.',
        'CGMacros has 39 standardized numeric columns. `participant_sex_code` remains the approved numeric-coded input; there is no one-hot reference category. Its original-unit slope describes a one-code increment, but the supplied artifacts do not establish the meaning of the codes, so no sex-specific biological conclusion is assigned.',
        '## Dataset findings']
    for dataset in evidence['datasets']:
        name=dataset['dataset']
        stability=pd.read_csv(REPO/f'results/member4/explainability/{name}_ridge_coefficient_stability.csv')
        saved=pd.read_csv(REPO/f'results/member4/explainability/{name}_ridge_saved_coefficients.csv')
        numeric=stability.loc[stability.kind.eq('numeric_standardized')]
        stable_count=int(((numeric.positive_folds==5)|(numeric.negative_folds==5)).sum())
        parts += [f"### {name}",
            f"Ridge alpha = {dataset['alpha']:g}; {dataset['features']} original inputs and {dataset['transformed']} transformed columns. {stable_count}/{len(numeric)} numeric inputs retain the same nonzero sign in all five fits. Mean pairwise Spearman agreement of absolute numeric ranks is {dataset['rank_spearman_mean']:.3f}."]
        lines=['| Numeric input | Saved signed coefficient | Fold mean ± SD | Fold ranks, min–max | Same nonzero sign in all folds? |',
               '|---|---:|---:|---:|:---:|']
        for r in dataset['top_numeric']:
            same=r['positive_folds']==5 or r['negative_folds']==5
            lines.append(f"| {r['original_feature']} | {r['saved_coefficient']:+.4f} | {r['fold_mean']:+.4f} ± {r['fold_std']:.4f} | {r['rank_min']:g}–{r['rank_max']:g} | {'Yes' if same else 'No'} |")
        parts += ['Coefficients below are in target-glucose units per fixed reference SD. Glucose measurement units remain unverified in the supplied metadata.', '\n'.join(lines)]
        sign_changes=dataset['sign_changing']
        parts.append('Features with fold-to-fold sign changes: '+(', '.join(sign_changes) if sign_changes else 'none')+'. Near-zero is defined as absolute coefficient ≤ 1e-8 on the reported comparison scale; this is a numerical rule, not a threshold of clinical relevance.')
        if dataset['categorical_contrasts']:
            contrasts=dataset['categorical_contrasts']
            original=[r for r in contrasts if r['validation_fold']==0][0]
            values=[r['coefficient_contrast'] for r in contrasts if r['validation_fold']!=0]
            parts.append(f"Shanghai's saved **{original['contrast']}** contrast is {original['coefficient_contrast']:+.4f} target-glucose units. The five training-fold contrasts range from {min(values):+.4f} to {max(values):+.4f}. These are conditional fitted-model offsets, not causal or clinical effects.")
        if name=='cgmacros':
            row=saved.loc[saved.original_feature.eq('participant_sex_code')].iloc[0]
            parts.append(f"The saved participant_sex_code slope is {row.coefficient_per_original_unit:+.4f} target-glucose units per code increment (code meanings unverified).")
        pairs=dataset['highest_correlation_pairs']
        parts.append('Largest observed absolute glucose-history correlations include: '+ '; '.join(f"{r['feature_a']} versus {r['feature_b']}: r={r['pearson_r']:.6f}" for r in pairs)+'. Correlations are descriptive, row-weighted within development data, and no p-values are inferred from time-series rows.')
        parts += [f'![Signed coefficient stability](../../results/member4/figures/{name}_ridge_signed_stability.png)',
                  f'![Absolute-rank stability](../../results/member4/figures/{name}_ridge_rank_stability.png)']
    parts += ['## Interpretation limits',
        'Current glucose, lagged glucose, changes, slopes and overlapping rolling summaries encode much of the same signal. In both datasets, glucose_change_15min and glucose_slope_15min are almost perfectly correlated. Consequently, a large negative lag coefficient does not mean that lower past glucose causes higher future glucose. It is one conditional weight in a redundant representation. Holding every derived feature fixed while changing only one underlying glucose value may be physically impossible.',
        'Ridge regularization stabilizes the joint prediction but does not identify a unique medical contribution from each correlated feature. Signals can be shared or offset across coefficients. In CGMacros, the large negative rolling-mean coefficient coexists with large positive current/lag coefficients; these are compensating weights in a redundant representation. Stable signs and ranks across these five overlapping training sets show sensitivity to the chosen participant omissions; they do not establish causation, independent importance, statistical significance or robustness to a different feature representation. Fold SDs are descriptive, not confidence intervals or standard errors. A negative sign is not a protective factor. Tiny numeric differences between almost equal change/slope coefficients can swap adjacent ranks; these swaps should not be presented as meaningful importance differences.',
        'Shanghai glucose_plus_meal had mean validation MAE 9.54045388 versus 9.54220975 for all features, a difference of approximately 0.001756. This coefficient analysis retains all approved features as its primary scope and makes no superiority claim or new selection decision. No secondary feature-set models were fitted.',
        'Units remain unverified, and no mg/dL or mmol/L label is inferred from value ranges. Numeric coefficient magnitudes are not compared across datasets because reference distributions and feature definitions differ. Missing inputs retain the original fitted imputation policy; this report does not independently audit missingness mechanisms. Fits retain the existing row-weighted training protocol, so participants with longer recordings contribute more rows; the fold stability summary does not make training participant-balanced.',
        '## Outputs and reproduction',
        'Tables in `results/member4/explainability/`: saved coefficients, fold coefficients, coefficient stability, category contrasts (Shanghai), fold rank correlations, glucose-history correlation matrices/pairs, intercepts and reconstruction checks. All numeric and categorical coefficients are included in tables; the figures show the 15 largest saved numeric magnitudes for readability.',
        'The coefficient tables include transformed names, original features, feature groups, signed and absolute coefficients, original-unit slopes, imputation statistics, scale/centering, reference SD, separate ranks and categorical coding. Stability tables contain positive/negative/zero fold counts, sign agreement with the saved fit, coefficient/rank mean, SD and range, and numeric top-five frequency.',
        '```powershell\npython -B notebooks/member4/test_ridge_coefficients.py\npython -B notebooks/member4/ridge_coefficients.py --source-root C:\\Users\\mssmo\\Documents\\glucose\\Smart_Glucose_Work\npython -B notebooks/member4/report_ridge_coefficients.py\n```',
        'The three synthetic tests check output-position mapping with reordered preprocessing branches and underscored category names, missing values and a constant feature; agreement of a category contrast with a prediction difference; and separation of reference rescaling from fitted coefficients. All 12 real fitted pipelines (2 original + 10 fold clones) passed transformed and original-scale prediction reconstruction on development fitting rows.',
        'Method references: [scikit-learn coefficient interpretation](https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html), [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html), and [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html).',
        '**Member 4 Ridge coefficient analysis: COMPLETE. Final-test evaluation: NOT STARTED.**']
    destination=report_dir/'ridge_coefficient_interpretation.md'
    destination.write_text('\n\n'.join(parts)+'\n',encoding='utf-8')
    print(destination)


if __name__=='__main__':
    render()
