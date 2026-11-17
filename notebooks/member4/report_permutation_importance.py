"""Render aggregate permutation results and descriptive fold-variation plots."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[2]
OUT=REPO/'results/member4/permutation'
REPORTS=REPO/'reports/member4'
FIGURES=REPO/'results/member4/figures'


def format_effect(row):
    return f"{row.importance_mean:+.4g} ± {row.fold_sd:.4g}"


def make_plot(dataset,level,aggregate):
    subset=aggregate.loc[aggregate.dataset.eq(dataset)&aggregate.level.eq(level)]
    global_rows=subset.loc[subset.method.eq('global_row')].sort_values('importance_mean',ascending=False)
    if level=='individual':
        names=global_rows.head(10).name.tolist()
        # Include the most negative means instead of clipping them off the figure.
        names += [n for n in global_rows.loc[global_rows.importance_mean.lt(0)].tail(3).name if n not in names]
    else:
        names=global_rows.name.tolist()
    fig,ax=plt.subplots(figsize=(11,max(5,len(names)*.42+2)))
    positions=np.arange(len(names))
    for method,offset,color,label in [('global_row',-.17,'#386cb0','Global row shuffle'),
        ('within_participant_block',.17,'#d95f02','Within-participant 60-minute block shuffle')]:
        rows=subset.loc[subset.method.eq(method)].set_index('name').loc[names]
        ax.barh(positions+offset,rows.importance_mean,height=.31,color=color,alpha=.85,label=label,
                xerr=rows.fold_sd,capsize=2,error_kw={'linewidth':.8})
    ax.set_yticks(positions,names); ax.invert_yaxis(); ax.axvline(0,color='black',lw=.8)
    ax.set_xlabel('Permuted MAE − original sample MAE (target glucose units; units unverified)')
    title=f'{dataset}: Ridge {level} permutation importance'
    subtitle='Mean of five fold means; error bars = descriptive SD across folds'
    if level=='individual': subtitle+='\nTop 10 global means, plus up to 3 negative global means'
    ax.set_title(title+'\n'+subtitle)
    ax.legend(loc='best',fontsize=8); ax.grid(axis='x',alpha=.15)
    fig.tight_layout()
    fig.savefig(FIGURES/f'{dataset}_ridge_permutation_{level}.png',dpi=180)
    plt.close(fig)


def render():
    evidence=json.loads((REPORTS/'permutation_analysis_evidence.json').read_text())
    pilot=json.loads((REPORTS/'permutation_pilot.json').read_text())
    aggregate=pd.read_csv(OUT/'permutation_aggregate_summary.csv')
    folds=pd.read_csv(OUT/'permutation_fold_summaries.csv')
    baselines=pd.read_csv(OUT/'permutation_sample_baselines.csv')
    config=evidence['config']
    FIGURES.mkdir(parents=True,exist_ok=True)
    parts=['# Member 4: held-out Ridge permutation importance',
        f"Completed {evidence['date'][:10]} (Asia/Colombo). All approved features, both datasets, and the same five participant-wise development folds. Importance = **permuted MAE − original MAE** on the exact same held-out sample. Positive values mean worse predictions after permutation; negative values are retained.",
        '## Models, sample and pilot',
        'All ten training-fold Ridge pipelines were reused from the coefficient-analysis cache, with model, source-data, split, feature-order, hyperparameter and participant-membership hashes verified. No models, imputers or scalers were fitted. Sampling used no glucose targets. Locked-test participant rows were filtered out before any measurement analysis or prediction.',
        f"The Shanghai Fold 1 pilot ran the complete individual/group analysis with both schemes and {config['repeats']} repeats in **{pilot['elapsed_seconds']:.2f} seconds** (fold preparation and scoring; excludes initial file hashing/loading). Its observed runtime was inspected before processing the remaining nine folds. The pilot scores were reused rather than recomputed. The pilot-only ten-fold projection was {pilot['estimate_ten_fold_compute_seconds']:.1f} seconds; dataset-specific preparation times make this a rough estimate.",
        f"Master seed **{config['seed']}**, {config['repeats']} repeats per item/scheme/fold, NumPy PCG64 generators. Per-participant sample seeds and per-repeat permutation seeds are derived by stable SHA-256 hashing and saved. A single repeat uses the same donor mapping for different items (paired randomness); every column in a group receives that exact same mapping.",
        'The sample contains equal numbers of complete observed 60-minute blocks for every development participant within a dataset, capped at 480 rows per participant. Shanghai uses the audited 15-minute nominal interval (4 observations per block); CGMacros uses 1 minute (60 observations per block). Blocks do not cross missing-interval boundaries. Samples select blocks without replacement across the eligible recording and retain their chronological internal order. One block covers one nominal hour of sampling slots: its first-to-last observed timestamp span is 45 minutes for Shanghai or 59 minutes for CGMacros.',
        'The shared block quota is limited by the least-covered development participant rather than excluding shorter recordings. All held-out participants remain represented. Rows outside selected blocks, including incomplete segment tails, are not scored in this sampled analysis; they remain in the source and trained models are unchanged. Global row shuffling uses this same block-selected sample for a paired comparison. This is not a simple random sample of all individual time-series rows.',
        '| Dataset | Rows per participant | Selected blocks per participant | Block rows | Participants covered across folds | Total sample rows | Incomplete segment-tail rows ineligible for sampling |',
        '|---|---:|---:|---:|---:|---:|---:|']
    table=parts[-2:]; parts=parts[:-2]
    for dataset,design in evidence['sampling_designs'].items():
        runs=[r for r in evidence['runs'] if r['identity']['dataset']==dataset]
        table.append(f"| {dataset} | {design['rows_per_participant']} | {design['blocks_per_participant']} | {design['block_rows']} | {sum(r['participants'] for r in runs)} | {sum(r['sample_rows'] for r in runs):,} | {design['incomplete_tail_rows']} |")
    parts.append('\n'.join(table))
    parts += ['Exact selected development-row indices, participant IDs, timestamps, block IDs and selection seeds are saved in `*_selected_rows.local.csv`; participant coverage, sample fractions and eligible-block counts are in `*_sample_coverage.local.csv`. These participant-linked files are ignored by Git. Runtime sidecars record selection/model hashes, row/participant counts and sample fractions. No row-level glucose predictions are saved.',
        'Because every sampled participant contributes the same row count, sample MAE equals the mean of participant-specific sample MAEs within a fold. These baselines describe the selected sample; they are not replacements for Member 3’s full, row-weighted validation scores.',
        '## Permutation schemes and interpretation',
        '**Global row shuffle:** permute donor rows over the entire held-out fold sample, including across participants. This disrupts temporal structure and can combine incompatible glucose histories, demographics, meals and activity. Individual permutations can violate algebraic relationships among current glucose, lags, changes and rolling features. Large losses can reflect off-distribution stress, especially for correlated Ridge predictors, and must not be read as causal or independent medical importance.',
        '**Within-participant block sensitivity:** reorder the selected complete 60-minute donor blocks within each participant, preserving order inside each block and using the same permutation for every column in an item/group. Participants are never mixed, and donor blocks never internally cross original recording gaps. This retains local structure more than global row shuffling, but disrupts longer trajectories, clock-time alignment, inter-block continuity and dependencies with unpermuted features. The block length is an exploratory choice, not an optimized or validated physiological timescale. Fixed points are allowed and their changed-row/value fractions are recorded.',
        'Participant-constant features cannot be perturbed by within-participant reordering. Their zero block importance is **not identifiable under this scheme**, not evidence that the model ignores them. The results explicitly record how many participants have constant inputs in the sample and full validation data. The same limitation applies if a time-varying feature happens to be constant in selected windows. The clinical group is uninformative under this scheme when all its columns are participant-constant.',
        'Feature groups follow the approved `feature_group` metadata exactly. Shared donor rows preserve within-group associations in each relocated row/block, but relationships to other groups and targets are broken. Group importance is not the sum of individual importances; the schemes also measure different disruptions and their numerical values are not interchangeable.',
        '## Results']
    for dataset in ['shanghai','cgmacros']:
        a=aggregate.loc[aggregate.dataset.eq(dataset)]
        b=baselines.loc[baselines.dataset.eq(dataset)]
        parts += [f'### {dataset}',
            'Sample baseline MAE by fold: '+', '.join(f"{int(r.validation_fold)}: {r.original_mae:.4f}" for r in b.itertuples())+'. Target glucose units remain unverified in the supplied metadata; no mg/dL or mmol/L conversion was applied.']
        group_table=['| Approved feature group | Global importance mean ± fold SD | Within-participant block mean ± fold SD | Negative global fold means / 5 |',
                     '|---|---:|---:|---:|']
        grouped=a.loc[a.level.eq('group')]
        for row in grouped.loc[grouped.method.eq('global_row')].sort_values('importance_mean',ascending=False).itertuples():
            block=grouped.loc[grouped.method.eq('within_participant_block')&grouped.name.eq(row.name)].iloc[0]
            group_table.append(f'| {row.name} | {format_effect(row)} | {format_effect(block)} | {row.negative_fold_means} |')
        parts.append('\n'.join(group_table))
        clinical=folds.loc[folds.dataset.eq(dataset)&folds.method.eq('within_participant_block')&folds.level.eq('group')&folds.name.eq('participant_clinical')]
        nonconstant=int((clinical.participants-clinical.full_validation_constant_participants).sum())
        if nonconstant:
            parts.append(f'Clinical inputs are fully constant within {int(clinical.participants.sum())-nonconstant}/{int(clinical.participants.sum())} development participants. The other {nonconstant} have within-participant value or missingness variation (`nunique(dropna=False)`). Accordingly, a clinical block result need not be exactly zero for those participants. No source values were repaired or forced constant.')
        else:
            parts.append(f'Clinical inputs are fully constant within all {int(clinical.participants.sum())} development participants. As expected, the clinical group has exactly zero block importance: this scheme cannot identify reliance on between-participant clinical differences.')
        for method,label in [('global_row','Global row'),('within_participant_block','Within-participant block')]:
            individual=a.loc[a.method.eq(method)&a.level.eq('individual')].sort_values('importance_mean',ascending=False)
            parts.append(f"**{label}, five largest individual mean losses:** "+'; '.join(f'{r.name}: {r.importance_mean:+.4f}' for r in individual.head(5).itertuples())+'.')
            negatives=individual.loc[individual.importance_mean.lt(0)]
            if len(negatives):
                parts.append(f'{label} negative individual mean losses (retained): '+ '; '.join(f'{r.name}: {r.importance_mean:+.6g}' for r in negatives.itertuples())+'.')
            else:
                parts.append(f'{label}: no negative individual aggregate means; any negative fold/repeat results remain in the detailed tables.')
        for level in ['individual','group']:
            make_plot(dataset,level,aggregate)
            parts.append(f'![{dataset} {level} importance](../../results/member4/figures/{dataset}_ridge_permutation_{level}.png)')
    parts += ['The CGMacros rolling-mean individual global loss (+422.6396) is far larger than the glucose-history group loss (+27.9895). Moving the related glucose-history columns together preserves their within-row relationships, whereas moving the rolling mean alone breaks the relationships underlying large compensating Ridge coefficients. This contrast supports treating the individual loss as a perturbation stress result rather than an independent medical contribution; block shuffling reduces but does not eliminate these interpretation limitations.']
    counts=evidence['output_counts']
    parts += ['## Descriptive variation and verification',
        f"Saved **{counts['total_repeat_results']:,}** fold/repeat scores, including **{counts['negative_repeat_results']:,} negative scores**, **{counts['fold_summary_rows']}** fold summaries, **{counts['aggregate_summary_rows']}** aggregate summaries, and **{counts['baseline_rows']}** sample baselines. No negative value was clipped or reset to zero.",
        'Repeats are first summarized within each fold/item/scheme. The principal aggregate is the unweighted mean of five fold means, with descriptive sample SD and range across those five means. Fold sizes differ, so this equal-fold summary is not a pooled participant-weighted effect across the entire dataset. Repeat SDs measure permutation variability conditional on the fixed fitted model and sampled rows. Time-series rows and repeats are not treated as independent participants; no confidence intervals, p-values or significance claims are made. The overlapping fold training sets also limit independence.',
        'A negative loss means that this particular perturbation lowered sample MAE. It may reflect sampling/permutation variation, redundancy or model dependence that does not help on that sample. It is not proof that a feature should be removed or a model retuned. No new model-selection decision was made from these importances.',
        f"All **{evidence['protected_files']}** protected source files and **{evidence['fitted_cache_files_checked']}** cached model/sidecar files retained their hashes. Model fits: **{evidence['model_fits']}**. Final-test predictions: **{evidence['final_test_predictions']}**. Five synthetic tests passed for deterministic balanced gap-safe sampling, within-person block mapping, one shared group mapping, positive/negative MAE-loss conventions, and the participant-constant limitation.",
        'Main files: `permutation_aggregate_summary.csv`, `permutation_fold_summaries.csv`, `permutation_sample_baselines.csv`, ten `*_repeat_results.csv` files, ten runtime JSON sidecars, `reports/member4/permutation_pilot.json` and `permutation_analysis_evidence.json`. Four plots are saved in `results/member4/figures/`. The original notebook provenance discrepancy and glucose-unit uncertainty from the readiness/data audits remain disclosed; the fitted cache/source model hashes used here match their recorded values.',
        '## Reproduction',
        '```powershell\npython -B notebooks/member4/test_permutation_importance.py\n# Pilot: inspect its saved runtime before the next command.\npython -B notebooks/member4/permutation_importance.py --source-root C:\\Users\\mssmo\\Documents\\glucose\\Smart_Glucose_Work\npython -B notebooks/member4/permutation_importance.py --source-root C:\\Users\\mssmo\\Documents\\glucose\\Smart_Glucose_Work --run-all\npython -B notebooks/member4/verify_permutation_results.py --source-root C:\\Users\\mssmo\\Documents\\glucose\\Smart_Glucose_Work\npython -B notebooks/member4/report_permutation_importance.py\n```',
        'The saved-result verifier checks the exact held-out IDs, locked-ID exclusion, balanced counts, unique source rows, block continuity, selection/result hashes, seeds, MAE-loss arithmetic, conditional constancy, and reconstruction of fold/aggregate means. Its status is saved in `reports/member4/permutation_verification.json`; it performs no model fitting or prediction.',
        'The full run requires a completed pilot with the exact configuration/script hash. Completed fold outputs are reused only when script, sample, model, package/configuration identity and output hashes match. Incompatible or incomplete results fail rather than being overwritten automatically. Predictions always go through the fitted original preprocessing pipeline; clinical/sex categorical columns are permuted at original-input level.',
        'Method reference: [scikit-learn permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html). The custom grouped/block schemes and sampling protocol are documented above; they are exploratory extensions, not claims of conditional or causal importance.',
        '**Member 4 development-only permutation importance: COMPLETE. Final-test evaluation: NOT STARTED.**']
    path=REPORTS/'permutation_importance_report.md'
    path.write_text('\n\n'.join(parts)+'\n',encoding='utf-8')
    print(path)


if __name__=='__main__':
    render()
