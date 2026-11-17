"""Run development-only quality audit; retain irregular rows and report them."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from data_access import load_development, require

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def run(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    protected = [p for folder in ['reports', 'results', 'models', 'notebooks', 'data/processed']
                 for p in (source / folder).rglob('*') if p.is_file()]
    before = {p: sha(p) for p in protected}
    summaries = []
    for dataset in ['shanghai', 'cgmacros']:
        data = load_development(source, dataset)
        df = data.frame
        t = pd.to_datetime(df.timestamp, errors='coerce')
        target_t = pd.to_datetime(df.target_timestamp_30min, errors='coerce')
        missing = pd.DataFrame({'column': df.columns, 'missing_rows': df.isna().sum().values})
        missing['missing_fraction'] = missing.missing_rows / len(df)
        missing.to_csv(output / f'{dataset}_missingness.csv', index=False)
        coverage = []
        interval_counts = {}
        for pid, g in df.groupby('participant_id', sort=True):
            times = t.loc[g.index]
            original_deltas = times.diff().dt.total_seconds() / 60
            unique = times.dropna().sort_values().drop_duplicates()
            gaps = unique.diff().dt.total_seconds().div(60).dropna()
            positive = gaps[gaps > 0]
            mode = float(positive.mode().iloc[0]) if len(positive) else None
            span = (unique.iloc[-1] - unique.iloc[0]).total_seconds() / 3600 if len(unique) else None
            for delta, count in positive.value_counts().items():
                interval_counts[float(delta)] = interval_counts.get(float(delta), 0) + int(count)
            coverage.append(dict(participant_id=pid, rows=len(g), invalid_timestamps=int(times.isna().sum()),
                duplicate_timestamp_extra_rows=int(times.dropna().duplicated().sum()),
                backwards_steps=int(original_deltas.lt(0).sum()), unique_timestamps=len(unique),
                first_timestamp=str(unique.iloc[0]) if len(unique) else '',
                last_timestamp=str(unique.iloc[-1]) if len(unique) else '', recording_span_hours=span,
                modal_interval_minutes=mode, median_interval_minutes=float(positive.median()) if len(positive) else None,
                maximum_gap_minutes=float(positive.max()) if len(positive) else None,
                gaps_over_twice_modal=int(positive.gt(2*mode).sum()) if mode else 0,
                grid_coverage_fraction=min(1.,len(unique)/(span*60/mode+1)) if mode and span is not None else None))
        cov = pd.DataFrame(coverage)
        # Participant-linked diagnostics stay local. Only aggregate summaries belong in the report.
        cov.to_csv(output / f'{dataset}_participant_coverage.local.csv', index=False)
        pd.DataFrame([dict(interval_minutes=k,count=v) for k,v in sorted(interval_counts.items())]).to_csv(output / f'{dataset}_interval_counts.csv', index=False)
        fold_records=[]
        for number in range(1,6):
            train, valid = data.fold(number)
            X, y = data.xy(valid)
            require(list(X.columns)==data.features, 'X ordering failed')
            fold_records.append(dict(validation_fold=number,train_participants=train.participant_id.nunique(),
                validation_participants=valid.participant_id.nunique(),train_rows=len(train),validation_rows=len(valid)))
        pd.DataFrame(fold_records).to_csv(output/f'{dataset}_fold_checks.csv',index=False)
        numeric=df[data.features+[ 'target_glucose_30min']].select_dtypes(include='number')
        nonfinite=int(np.isinf(numeric.to_numpy(dtype=float)).sum())
        horizons=(target_t-t).dt.total_seconds()/60
        summaries.append(dict(dataset=dataset,development_rows=len(df),development_participants=len(cov),
            locked_rows_excluded=data.excluded_rows,invalid_timestamps=int(t.isna().sum()),
            invalid_target_timestamps=int(target_t.isna().sum()),
            non_30min_target_timestamps=int((horizons.notna() & ~np.isclose(horizons,30)).sum()),
            duplicate_timestamp_extra_rows=int(cov.duplicate_timestamp_extra_rows.sum()),
            backwards_steps=int(cov.backwards_steps.sum()),missing_cells=int(df.isna().sum().sum()),
            nonfinite_numeric_cells=nonfinite,participants_with_large_gaps=int(cov.gaps_over_twice_modal.gt(0).sum()),
            modal_intervals_minutes=sorted(cov.modal_interval_minutes.dropna().unique().tolist()),
            recording_span_hours_min=float(cov.recording_span_hours.min()),
            recording_span_hours_median=float(cov.recording_span_hours.median()),
            recording_span_hours_max=float(cov.recording_span_hours.max()),
            median_grid_coverage_fraction=float(cov.grid_coverage_fraction.median()),
            current_glucose_quantiles=df.current_glucose.quantile([0,.01,.5,.99,1]).to_dict(),
            glucose_units='UNVERIFIED: no unit field or authoritative unit metadata supplied',
            quality_rows_excluded=0))
    changed=[str(p) for p,h in before.items() if sha(p)!=h]
    require(not changed,'Protected source files changed')
    (output/'development_data_audit.json').write_text(json.dumps(dict(datasets=summaries,protected_files=len(before),protected_changes=changed),indent=2),encoding='utf-8')
    report=['# Member 4 development-data audit','',
        'Development-only audit; no training, preprocessing fitting, clustering or predictions. Existing saved folds are unchanged.',
        'Source: `'+str(source)+'`. All '+str(len(before))+' protected files retain identical SHA-256 hashes.', '',
        'CSV chunks are filtered by approved participant assignments before measurement analysis. Locked rows are counted only for exclusion accounting. No test measurements or targets are summarized.', '']
    table = ['| Dataset | Development rows | Participants | Locked rows excluded | Backwards steps | Participants with large gaps |',
        '|---|---:|---:|---:|---:|---:|']
    for s in summaries:
        table.append(f"| {s['dataset']} | {s['development_rows']:,} | {s['development_participants']} | {s['locked_rows_excluded']:,} | {s['backwards_steps']} | {s['participants_with_large_gaps']} |")
    report.append('\n'.join(table))
    for s in summaries:
        report += ['## '+s['dataset'],'','```json',json.dumps(s,indent=2),'```','']
    report += ['## Interpretation and exclusions','',
        'Only locked-test rows were excluded. No development rows were silently dropped, sorted in the returned loader, deduplicated, imputed or converted. The audit sorts unique timestamps only to describe intervals and gaps. Backwards steps refer to original within-participant file order.',
        'Grid coverage is observed unique timestamps divided by the expected grid count across the first-to-last span, using each participant’s modal positive interval, capped at one. It is an approximate recording-density measure, not sensor wear-time or proof of continuous observation. Gaps over twice that interval are flagged; they are not automatically excluded.',
        'Sampling statistics describe ML-ready rows, which can differ from raw sensor cadence after lag/target eligibility filtering. Glucose quantiles cannot establish units. No mg/dL↔mmol/L conversion or clinical thresholds were applied; verify authoritative source units before clinical interpretation or unit-dependent clustering features.',
        'Missingness and infinity checks cover retained development rows only. Horizon checks compare recorded timestamps; they do not independently prove target construction. No medical causation or diagnostic claims are made.', '',
        'Participant coverage files ending `.local.csv` contain development identifiers and remain ignored by Git. Aggregate interval, missingness and fold-check tables are saved alongside the JSON evidence.', '',
        'Final-test evaluation: NOT STARTED.']
    report_path=Path(__file__).resolve().parents[2]/'reports/member4/development_data_audit.md'
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text('\n\n'.join(report),encoding='utf-8')
    print(json.dumps(summaries,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[2]/'results/member4/audit')
    args=parser.parse_args()
    run(args.source_root,args.output)
