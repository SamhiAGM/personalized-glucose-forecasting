"""Cross-method ranking comparisons, plots and interpretation report."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tree_shap_analysis import REPO,OUT,TREE,REPORT


def main():
    evidence=json.loads((REPORT/'tree_shap_evidence.json').read_text())
    pilot=json.loads((REPORT/'tree_shap_pilot.json').read_text())
    shap_summary=pd.read_csv(OUT/'ridge_shap_aggregate.csv')
    figures=REPO/'results/member4/figures'; figures.mkdir(parents=True,exist_ok=True)
    comparisons=[]; associations=[]; findings=[]
    for dataset in ['shanghai','cgmacros']:
        s=shap_summary.loc[shap_summary.dataset.eq(dataset)&shap_summary.level.eq('original')].set_index('name')
        coefficients=pd.read_csv(REPO/f'results/member4/explainability/{dataset}_ridge_saved_coefficients.csv')
        numeric=coefficients.loc[coefficients.kind.eq('numeric_standardized')&coefficients.rank_eligible]
        permutation=pd.read_csv(REPO/'results/member4/permutation/permutation_aggregate_summary.csv')
        p=permutation.loc[permutation.dataset.eq(dataset)&permutation.level.eq('individual')&permutation.method.eq('global_row')].set_index('name')
        rf=pd.read_csv(TREE/f'{dataset}_random_forest_original.csv').set_index('original_feature')
        gb=pd.read_csv(TREE/f'{dataset}_gradient_boosting_original.csv').set_index('original_feature')
        comparison=pd.DataFrame(index=s.index)
        comparison['ridge_shap_mean_abs']=s.mean_abs_shap; comparison['ridge_shap_rank']=s['rank']
        comparison['heldout_permutation_mae_loss']=p.importance_mean
        comparison['heldout_permutation_rank']=p.rank_by_mean
        comparison['random_forest_mdi']=rf.importance; comparison['random_forest_rank']=rf['rank']
        comparison['gradient_boosting_mdi']=gb.importance; comparison['gradient_boosting_rank']=gb['rank']
        comparison['ridge_numeric_abs_reference_sd']=numeric.set_index('original_feature').coefficient_common_reference_sd.abs()
        comparison['ridge_numeric_rank']=numeric.set_index('original_feature').absolute_rank_common_reference_sd
        comparison.index.name='original_feature'; comparison.insert(0,'dataset',dataset)
        comparison.reset_index().to_csv(OUT/f'{dataset}_method_comparison.csv',index=False)
        comparisons.append(comparison.reset_index())
        columns=['ridge_shap_rank','heldout_permutation_rank','random_forest_rank','gradient_boosting_rank','ridge_numeric_rank']
        for i,a in enumerate(columns):
            for b in columns[i+1:]:
                paired=comparison[[a,b]].dropna()
                associations.append(dict(dataset=dataset,method_a=a,method_b=b,features=len(paired),
                    spearman_rank_association=paired[a].corr(paired[b],method='spearman')))
        top=s.nlargest(15,'mean_abs_shap').sort_values('mean_abs_shap')
        fig,ax=plt.subplots(figsize=(10,7))
        ax.barh(top.index,top.mean_abs_shap,color='#386cb0',xerr=top.fold_sd,capsize=2)
        ax.set_title(f'{dataset}: held-out Ridge mean |SHAP|\nMean of five participant-balanced fold summaries; bars show fold SD')
        ax.set_xlabel('Mean absolute contribution (target glucose units; units unverified)')
        ax.grid(axis='x',alpha=.2); fig.tight_layout()
        fig.savefig(figures/f'{dataset}_ridge_shap_global.png',dpi=180); plt.close(fig)
        group=shap_summary.loc[shap_summary.dataset.eq(dataset)&shap_summary.level.eq('group')].sort_values('mean_abs_shap')
        fig,ax=plt.subplots(figsize=(10,5))
        ax.barh(group.name,group.mean_abs_shap,xerr=group.fold_sd,color='#1b9e77',capsize=2)
        ax.set_title(f'{dataset}: glucose-history and other group SHAP\nSum signed contributions inside each group before absolute value')
        ax.set_xlabel('Mean |group contribution| (target units unverified)'); fig.tight_layout()
        fig.savefig(figures/f'{dataset}_ridge_shap_groups.png',dpi=180); plt.close(fig)
        names=comparison[['random_forest_mdi','gradient_boosting_mdi']].max(axis=1).nlargest(12).index.tolist()[::-1]
        fig,ax=plt.subplots(figsize=(10,6.5)); y=np.arange(len(names))
        ax.barh(y-.18,rf.loc[names,'importance'],height=.34,label='Random Forest',color='#386cb0')
        ax.barh(y+.18,gb.loc[names,'importance'],height=.34,label='Gradient Boosting',color='#d95f02')
        ax.set_yticks(y,names); ax.set_title(f'{dataset}: saved development-fitted tree importance\nTop 12 original features; descriptive normalized impurity reduction')
        ax.set_xlabel('Normalized impurity importance (unitless)'); ax.legend(); fig.tight_layout()
        fig.savefig(figures/f'{dataset}_tree_importance.png',dpi=180); plt.close(fig)
        findings.append(dict(dataset=dataset,shap_top=s.nlargest(5,'mean_abs_shap').rename_axis('name').reset_index().to_dict('records'),
            rf_top=rf.nlargest(5,'importance').reset_index().to_dict('records'),gb_top=gb.nlargest(5,'importance').reset_index().to_dict('records')))
    pd.concat(comparisons).to_csv(OUT/'cross_method_comparison.csv',index=False)
    pd.DataFrame(associations).to_csv(OUT/'cross_method_rank_associations.csv',index=False)
    (REPORT/'tree_shap_findings.json').write_text(json.dumps(findings,indent=2),encoding='utf-8')
    errors=[r['max_additivity_error'] for r in evidence['runs']]
    parts=['# Member 4: tree importance and SHAP explanations',
        f"Completed {evidence['date'][:10]} (Asia/Colombo). All approved features; Shanghai and CGMacros analyzed separately. Tree importance is descriptive of the original saved development-fitted models. Ridge SHAP explains only each cached model’s held-out development participants, using training-participant backgrounds.",
        '## Methods and validation',
        'Verified all four Random Forest/Gradient Boosting hashes against Member 3’s model inventory. The fitted preprocessor output positions, branch selectors and actual encoder categories map importance to original inputs. Shanghai sex expands into Female/Male indicators with no dropped reference. The transformed tree importances are summed for each original input; totals remain one. CGMacros retains participant_sex_code as an approved numeric feature, with code meanings unverified. Trees were not retrained.',
        f"SHAP {evidence['versions']['shap']} imported and ran without dependency changes. A bounded Shanghai Fold 1 pilot used {pilot['background_rows']} training-background rows and {pilot['explained_rows']} held-out rows; it finished in {pilot['elapsed_seconds']:.2f} seconds including preparation, SHAP checks and local plots. The pilot was inspected before the remaining nine folds and reused in the full analysis.",
        'Each background contains **8 rows per training participant**. Each explanation sample contains **24 rows per held-out participant**, deterministically subsampled from the prior permutation-importance sample. Sampling is target-blind; SHA-derived seeds use master seed 42. IDs, timestamps, exact development-row indices and selection seeds remain in ignored local CSVs. All training and validation participants are covered in their relevant roles, and background/explanation participants and source rows are disjoint. Locked-test participants are excluded. No predictive metadata or target columns are supplied to the model.',
        'Both background and explanation inputs pass through the corresponding cached, training-fitted preprocessor. `shap.LinearExplainer` uses `shap.maskers.Independent` with an explicit `max_samples` equal to the entire background size, preventing default thinning of the balanced sample. Identity output/link, no target transform. Expected value is the fitted Ridge prediction averaged over that background. For this independent linear explanation, SHAP_j = coefficient_j × (transformed input_j − background mean_j).',
        'This independent/interventional assumption breaks feature dependence; it is not a causal intervention or a correlation-conditional explanation. Correlated lags, changes, slopes and rolling features receive model-weight-based credits that can be large and cancel. One-hot columns are also treated as separate transformed variables; their signed attributions are then summed into the original sex feature. This aggregation preserves reconstruction, but is not a newly estimated conditional/cooperative game on original features.',
        f"For all ten folds, expected value + sum(SHAP) matched pipeline predictions: maximum absolute residual **{max(errors):.3g}**, using atol=1e-7 and rtol=1e-9. Predictions from the transformed estimator also matched the pipeline; SHAP values matched the analytical linear formula. No model fitting or dependency changes occurred. Four synthetic tests covered mapping with reordered branches/underscored categories, exact linear/additive checks, cancellation before absolute value, and deterministic balanced sampling/participant weighting.",
        'The global summaries first average absolute contributions within each participant, then equally across held-out participants in that fold. The main aggregate equally weights the five fold summaries and reports descriptive fold SD/range. A separate participant-weighted aggregate weights folds by held-out participant count. It is not identical when folds differ in size. No repeated explanations are treated as independent participants; there are no p-values or confidence intervals.',
        'Original-feature and group summaries sum **signed** transformed SHAP values per row before taking absolute values. Thus the group mean absolute contribution can be much smaller than the sum of individual mean absolutes when correlated features offset. All global units are target-glucose units, which remain unverified; no mg/dL/mmol/L label or conversion is applied.',
        '## Findings']
    for f in findings:
        dataset=f['dataset']; runs=[r for r in evidence['runs'] if r['dataset']==dataset]
        parts += [f'### {dataset}',f"Across five folds, {sum(r['explained_participants'] for r in runs)} distinct held-out participants and {sum(r['explained_rows'] for r in runs):,} rows were explained. Training background sizes vary with fold membership: "+', '.join(str(r['background_rows']) for r in runs)+'.',
            'Largest mean-absolute Ridge SHAP contributions: '+ '; '.join(f"{r['name']}: {r['mean_abs_shap']:.4f}" for r in f['shap_top'])+'.',
            'Random Forest top original inputs: '+', '.join(r['original_feature'] for r in f['rf_top'])+'.',
            'Gradient Boosting top original inputs: '+', '.join(r['original_feature'] for r in f['gb_top'])+'.',
            f'![Tree importance](../../results/member4/figures/{dataset}_tree_importance.png)',
            f'![Ridge SHAP](../../results/member4/figures/{dataset}_ridge_shap_global.png)',
            f'![Grouped Ridge SHAP](../../results/member4/figures/{dataset}_ridge_shap_groups.png)']
        assoc=pd.DataFrame(associations); subset=assoc.loc[assoc.dataset.eq(dataset)]
        parts.append('Descriptive Spearman ranking associations (not significance tests): '+ '; '.join(f"{r.method_a} / {r.method_b}: {r.spearman_rank_association:.3f} ({r.features} shared features)" for r in subset.itertuples())+'.')
    parts += ['## Why methods differ',
        'Tree impurity reduction measures how the saved nonlinear model allocated splits during development fitting. Ridge coefficient magnitudes reflect conditional weights and preprocessing scale. Mean |SHAP| reflects fitted weight × distance from a particular training-background mean. Held-out permutation importance measures MAE change after disrupting inputs. The estimators, fitting/explanation scopes, samples and units differ, so only descriptive rank comparisons are made; raw numeric magnitudes are not compared across methods or datasets. Cross-method rank tables retain every approved original feature; Ridge coefficient rank is missing for categorical indicators to avoid mixing scales.',
        'Impurity importance can favor continuous/high-cardinality features and can reflect training overfit. Correlated predictors can share or monopolize split credit. Summing dummy importance avoids treating one input as several distinct variables but does not eliminate selection bias. No held-out tree performance or tree-specific permutation importance is inferred from these saved MDI values.',
        'The earlier Ridge analysis found nearly perfectly correlated glucose-change/slope features and strong rolling-mean/lag relationships. Negative or large SHAP contributions show model deviations from a reference prediction, not protective/risk factors. Expected values vary by fold/background. These independent SHAP values do not resolve collinearity or establish medical causation. The current sampling is row-based within each participant; temporal observations remain dependent, and results describe the selected windows. Model training itself remains the original row-weighted protocol.',
        '## Local examples and scope limits',
        'Local-example rule was declared before SHAP: Fold 1 only, first/middle/last sorted participant IDs, earliest timestamp in each participant’s selected explanation sample. Six waterfall PNGs, original-feature contribution CSVs and private example manifests are saved in `results/member4/shap/local/`. They reconstruct each corresponding prediction and are illustrative, not selected for errors, glucose levels or visually favorable SHAP. Row-level arrays, transformed inputs/backgrounds, predictions and selected IDs also stay there; this whole folder is ignored by Git.',
        'No optional tree SHAP example was added: the four supplied trees are all-development fits, so explaining their development rows would be in-sample/descriptive and would not supply held-out tree evidence. The requested ten held-out Ridge explanations and tree MDI provide distinct, explicitly labeled scopes without additional tree fitting. SHAP was compatible; no fallback or environment modification was needed.',
        'The prior original modeling-notebook manifest discrepancy and unverified glucose units remain unresolved, as documented in readiness reports. They were not silently repaired. All original model hashes used here match their inventory. No clustering or diagnostic interpretation was performed.',
        '## Reproduction and outputs',
        '```powershell\npython -B notebooks/member4/test_tree_shap.py\npython -B notebooks/member4/tree_shap_analysis.py --source-root C:\\Users\\mssmo\\Documents\\glucose\\Smart_Glucose_Work\n# Inspect reports/member4/tree_shap_pilot.json before scaling.\npython -B notebooks/member4/tree_shap_analysis.py --source-root C:\\Users\\mssmo\\Documents\\glucose\\Smart_Glucose_Work --all-folds\npython -B notebooks/member4/report_tree_shap.py\npython -B notebooks/member4/verify_tree_shap.py --source-root C:\\Users\\mssmo\\Documents\\glucose\\Smart_Glucose_Work\n```',
        'Results: eight transformed/original tree-importance CSVs; ten feature maps, per-fold SHAP tables at transformed/original/group levels, combined fold and aggregate summaries, method comparison/rank-association tables, six global plots, and six local waterfall examples. Evidence includes model hashes, package versions, explainer settings, seeds, coverage, runtime, base values, error tolerances and protected-file checks. Local files must be preserved privately to reproduce identical selected-row explanations; the scripts also deterministically regenerate them from unchanged sources.',
        f"Protected source files checked: **{evidence['protected_files']}**, changed: **0**. Cached model/sidecar files checked: **{evidence['cached_files']}**, changed: **0**. Model fits: **0**. Final-test predictions: **0**. No commit or push was performed.",
        'References: [SHAP LinearExplainer](https://shap.readthedocs.io/en/latest/generated/shap.LinearExplainer.html) and [scikit-learn tree-importance example and limitations](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html).',
        '**Final-test evaluation: NOT STARTED.**']
    (REPORT/'tree_shap_interpretation.md').write_text('\n\n'.join(parts)+'\n',encoding='utf-8')
    print(REPORT/'tree_shap_interpretation.md')


if __name__=='__main__': main()
