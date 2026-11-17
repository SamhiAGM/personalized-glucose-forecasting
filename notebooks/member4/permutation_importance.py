"""Held-out participant-balanced Ridge permutation importance; never fits models.

Pilot one complete fold first, inspect its runtime, then use --run-all. Each group
gets a shared donor-row mapping. The block sensitivity stays within participant.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from data_access import load_development, require

REPO=Path(__file__).resolve().parents[2]
OUT=REPO/'results/member4/permutation'
REPORTS=REPO/'reports/member4'
METHODS=['global_row','within_participant_block']
CONFIG=dict(seed=42,repeats=10,requested_max_rows_per_participant=480,block_minutes=60,
            sampling='equal complete contiguous blocks per participant within each dataset',
            permutation='shared row mapping across all columns in each feature/group',
            metric='permuted MAE minus original MAE',threadpool_limit=1)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def seed_for(*parts):
    text=json.dumps([CONFIG['seed'],*parts],sort_keys=True)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4],'little')


def sample_development(data):
    """Select full observed blocks, with exact equal row counts per participant.

    A gap or cadence change starts a new segment. Incomplete segment tails are
    ineligible. The shared block quota is capped by the least-covered participant,
    avoiding exclusions of short recordings. No target is consulted for sampling.
    """
    step={'shanghai':15.,'cgmacros':1.}[data.dataset]
    block_rows=int(CONFIG['block_minutes']/step)
    require(block_rows*step==CONFIG['block_minutes'],'Block duration/cadence mismatch')
    candidates={}; coverage=[]
    for pid,group in data.frame.groupby('participant_id',sort=True):
        times=pd.to_datetime(group.timestamp,errors='coerce')
        require(times.notna().all() and times.is_unique,'Sampling needs valid unique participant timestamps')
        ordered=group.assign(_time=times).sort_values('_time',kind='stable')
        delta=ordered._time.diff().dt.total_seconds()/60
        segment=(~np.isclose(delta,step,rtol=0,atol=1e-6)).cumsum()
        blocks=[]; tails=0
        for _,seg in ordered.groupby(segment,sort=False):
            indices=seg.index.to_numpy()
            end=(len(indices)//block_rows)*block_rows
            blocks.extend(indices[:end].reshape(-1,block_rows))
            tails+=len(indices)-end
        require(len(blocks)>=2,'At least two complete blocks per development participant required')
        candidates[str(pid)]=blocks
        coverage.append(dict(participant_id=str(pid),available_rows=len(group),eligible_blocks=len(blocks),
            incomplete_segment_tail_rows=tails,nominal_step_minutes=step,block_rows=block_rows))
    quota=min(CONFIG['requested_max_rows_per_participant']//block_rows,min(len(b) for b in candidates.values()))
    require(quota>=2,'Block permutation requires two sampled blocks per participant')
    records=[]
    assignment=dict(zip(data.assignments.participant_id,data.assignments.validation_fold))
    for pid,blocks in candidates.items():
        selection_seed=seed_for(data.dataset,'sample',pid)
        chosen=np.sort(np.random.default_rng(selection_seed).choice(len(blocks),quota,replace=False))
        for block_id in chosen:
            for offset,row in enumerate(blocks[block_id]):
                records.append(dict(participant_id=pid,validation_fold=int(assignment[pid]),
                    development_row_index=int(row),selected_block_id=int(block_id),within_block_position=offset,
                    timestamp=str(data.frame.loc[row,'timestamp']),selection_seed=selection_seed))
    selection=pd.DataFrame(records)
    require(selection.development_row_index.is_unique,'Sample repeats a source row')
    require(selection.groupby('participant_id').size().nunique()==1,'Unequal participant sampling')
    coverage=pd.DataFrame(coverage)
    coverage['selected_blocks']=quota
    coverage['selected_rows']=quota*block_rows
    coverage['sampling_fraction']=coverage.selected_rows/coverage.available_rows
    return selection,coverage,dict(nominal_step_minutes=step,block_rows=block_rows,
                                  blocks_per_participant=quota,rows_per_participant=quota*block_rows,
                                  incomplete_tail_rows=int(coverage.incomplete_segment_tail_rows.sum()))


def row_mapping(selection,method,seed):
    rng=np.random.default_rng(seed)
    if method=='global_row':
        return rng.permutation(len(selection))
    require(method=='within_participant_block','Unknown permutation method')
    mapping=np.arange(len(selection))
    for _,group in selection.groupby('participant_id',sort=True):
        blocks=[g.index.to_numpy() for _,g in group.groupby('selected_block_id',sort=True)]
        require(len(blocks)>=2 and len({len(b) for b in blocks})==1,'Malformed sampled blocks')
        order=rng.permutation(len(blocks))
        for destination,donor in zip(blocks,[blocks[j] for j in order]):
            mapping[destination]=donor
    require(np.array_equal(np.sort(mapping),np.arange(len(selection))),'Mapping is not a permutation')
    require(np.array_equal(selection.participant_id.to_numpy(),selection.participant_id.to_numpy()[mapping]),
            'Block permutation crossed participant boundary')
    require(np.array_equal(selection.within_block_position.to_numpy(),selection.within_block_position.to_numpy()[mapping]),
            'Block permutation changed within-block order')
    return mapping


def permuted_inputs(X,columns,mapping):
    permuted=X.copy()
    # All columns share the exact same mapping, including missingness/category values.
    permuted.loc[:,columns]=X.iloc[mapping][columns].to_numpy()
    return permuted


def importance_score(model,X,y,columns,mapping,baseline_mae):
    perturbed=permuted_inputs(X,columns,mapping)
    pred=model.predict(perturbed)
    mae=float(np.mean(np.abs(y-pred)))
    a=X[columns]; b=perturbed[columns]
    same=a.eq(b)|(a.isna()&b.isna())
    return dict(original_mae=baseline_mae,permuted_mae=mae,importance=mae-baseline_mae,
                feature_values_changed_fraction=float((~same.all(axis=1)).mean()))


def items_for(metadata):
    items=[dict(level='individual',name=r.feature,features=[r.feature]) for r in metadata.itertuples()]
    items += [dict(level='group',name=str(name),features=group.feature.tolist())
              for name,group in metadata.groupby('feature_group',sort=False)]
    return items


def checked_fold_model(data,fold,source_hashes,versions):
    path=REPO/f'models/member4/{data.dataset}_ridge_fold{fold}.joblib'
    require(path.is_file() and path.with_suffix('.json').is_file(),'Missing previously fitted fold cache')
    sidecar=json.loads(path.with_suffix('.json').read_text())
    identity=sidecar['identity']
    require(json_sha(identity)==sidecar['identity_sha256'],'Cache identity hash mismatch')
    require(sha(path)==sidecar['model_sha256'],'Cache model hash mismatch')
    for relative,expected in identity['source_sha256'].items():
        require(source_hashes[source_hashes['root']/relative]==expected,'Source differs from fitted cache: '+relative)
    require(identity['versions']==versions,'Runtime versions differ from coefficient analysis cache')
    train,valid=data.fold(fold)
    training=set(train.participant_id); validation=set(valid.participant_id)
    require(identity['dataset']==data.dataset and identity['validation_fold']==fold and identity['configuration']=='all_features','Incorrect cached fold identity')
    require(identity['features']==data.features,'Cache feature order mismatch')
    require(identity['training_ids_sha256']==json_sha(sorted(training)) and identity['validation_ids_sha256']==json_sha(sorted(validation)),
            'Cache participant assignment mismatch')
    require(identity['train_rows']==len(train) and identity['validation_rows']==len(valid),'Cache row count mismatch')
    require(training.isdisjoint(validation|set(data.locked_ids)) and identity['locked_participants_used']==0,'Cache leakage')
    model=joblib.load(path)
    require(list(model.feature_names_in_)==data.features,'Fitted feature names mismatch')
    require(joblib.hash(clone(model).get_params(deep=True))==identity['hyperparameter_signature'],'Cached model hyperparameters differ')
    return model,valid,sidecar['model_sha256']


def evaluate_fold(data,fold,selection,coverage,design,source_hashes,versions,phase):
    start=time.perf_counter()
    model,validation,model_sha=checked_fold_model(data,fold,source_hashes,versions)
    chosen=selection.loc[selection.validation_fold==fold].reset_index(drop=True)
    require(set(chosen.participant_id)==set(validation.participant_id),'Validation sample lacks participants')
    require(set(chosen.participant_id).isdisjoint(data.locked_ids),'Locked sample participant')
    sample=data.frame.loc[chosen.development_row_index].reset_index(drop=True)
    require(sample.participant_id.tolist()==chosen.participant_id.tolist(),'Sample identity/order mismatch')
    X,y=data.xy(sample); y=y.to_numpy(dtype=float)
    require(np.isfinite(y).all(),'Invalid validation targets')
    local=OUT/f'{data.dataset}_fold{fold}_selected_rows.local.csv'
    chosen.to_csv(local,index=False)
    identity=dict(dataset=data.dataset,validation_fold=fold,config=CONFIG,sampling_design=design,
                  selection_sha256=sha(local),model_sha256=model_sha,versions=versions,
                  script_sha256=sha(Path(__file__)))
    result_path=OUT/f'{data.dataset}_fold{fold}_repeat_results.csv'
    record_path=OUT/f'{data.dataset}_fold{fold}_runtime.json'
    if result_path.exists() or record_path.exists():
        require(result_path.exists() and record_path.exists(),'Incomplete permutation result cache')
        existing=json.loads(record_path.read_text())
        require(existing['identity']==identity and existing['result_sha256']==sha(result_path),'Incompatible existing permutation results; inspect before replacing')
        print(f'{data.dataset} fold {fold}: reusing verified results',flush=True)
        return existing
    pred=model.predict(X)
    baseline=float(np.mean(np.abs(y-pred)))
    participant_mae=pd.DataFrame({'participant_id':sample.participant_id,'absolute_error':np.abs(y-pred)}).groupby('participant_id').absolute_error.mean()
    np.testing.assert_allclose(baseline,participant_mae.mean(),rtol=1e-12,atol=1e-12)
    items=items_for(data.feature_metadata)
    repeats=[]; timings=[]
    participant_groups=list(sample.groupby('participant_id').groups.values())
    validation_groups=list(validation.groupby('participant_id').groups.values())
    constants={}
    for item in items:
        columns=item['features']
        sample_constant=sum(sample.loc[idx,columns].nunique(dropna=False).le(1).all() for idx in participant_groups)
        full_constant=sum(validation.loc[idx,columns].nunique(dropna=False).le(1).all() for idx in validation_groups)
        constants[(item['level'],item['name'])]=(int(sample_constant),int(full_constant))
    for method in METHODS:
        method_start=time.perf_counter()
        for repeat in range(CONFIG['repeats']):
            permutation_seed=seed_for(data.dataset,fold,method,repeat)
            mapping=row_mapping(chosen,method,permutation_seed)
            for item in items:
                score=importance_score(model,X,y,item['features'],mapping,baseline)
                sample_constant,full_constant=constants[(item['level'],item['name'])]
                repeats.append(dict(dataset=data.dataset,validation_fold=fold,method=method,
                    level=item['level'],name=item['name'],features='|'.join(item['features']),
                    repeat=repeat+1,permutation_seed=permutation_seed,sample_rows=len(sample),
                    participants=len(participant_mae),rows_per_participant=design['rows_per_participant'],
                    sample_constant_participants=sample_constant,full_validation_constant_participants=full_constant,
                    mapping_changed_fraction=float((mapping!=np.arange(len(mapping))).mean()),**score))
        timings.append(dict(method=method,seconds=time.perf_counter()-method_start))
    results=pd.DataFrame(repeats)
    results.to_csv(result_path,index=False)
    runtime=dict(identity=identity,phase=phase,result_sha256=sha(result_path),elapsed_seconds=time.perf_counter()-start,
        method_timings=timings,baseline_sample_mae=baseline,participants=len(participant_mae),sample_rows=len(sample),
        rows_per_participant=design['rows_per_participant'],heldout_rows_available=len(validation),
        sample_fraction=len(sample)/len(validation),repeat_result_rows=len(results),locked_participants_used=0,
        repeated_predictions=len(results)+1,model_fitted=False)
    record_path.write_text(json.dumps(runtime,indent=2),encoding='utf-8')
    print(f'{data.dataset} fold {fold}: {len(results)} scores, {runtime["elapsed_seconds"]:.2f}s, baseline MAE {baseline:.4f}',flush=True)
    return runtime


def summarize():
    files=[OUT/f'{d}_fold{f}_repeat_results.csv' for d in ['shanghai','cgmacros'] for f in range(1,6)]
    require(all(p.is_file() for p in files),'Cannot aggregate incomplete five-fold results')
    results=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
    keys=['dataset','validation_fold','method','level','name']
    folds=results.groupby(keys,sort=False).agg(importance_mean=('importance','mean'),
        permutation_repeat_sd=('importance','std'),importance_min=('importance','min'),importance_max=('importance','max'),
        negative_repeats=('importance',lambda s:int(s.lt(0).sum())),repeats=('importance','size'),
        original_mae=('original_mae','first'),sample_rows=('sample_rows','first'),participants=('participants','first'),
        mean_values_changed_fraction=('feature_values_changed_fraction','mean'),
        sample_constant_participants=('sample_constant_participants','first'),
        full_validation_constant_participants=('full_validation_constant_participants','first')).reset_index()
    folds['rank']=folds.groupby(['dataset','validation_fold','method','level']).importance_mean.rank(ascending=False,method='average')
    folds.to_csv(OUT/'permutation_fold_summaries.csv',index=False)
    aggregate=folds.groupby(['dataset','method','level','name'],sort=False).agg(
        importance_mean=('importance_mean','mean'),fold_sd=('importance_mean','std'),
        fold_min=('importance_mean','min'),fold_max=('importance_mean','max'),folds=('importance_mean','size'),
        negative_fold_means=('importance_mean',lambda s:int(s.lt(0).sum())),negative_repeats_total=('negative_repeats','sum'),
        mean_repeat_sd=('permutation_repeat_sd','mean'),rank_mean=('rank','mean'),rank_min=('rank','min'),rank_max=('rank','max'),
        mean_values_changed_fraction=('mean_values_changed_fraction','mean')).reset_index()
    aggregate['rank_by_mean']=aggregate.groupby(['dataset','method','level']).importance_mean.rank(ascending=False,method='average')
    aggregate.to_csv(OUT/'permutation_aggregate_summary.csv',index=False)
    baseline=folds[['dataset','validation_fold','original_mae','sample_rows','participants']].drop_duplicates()
    baseline.to_csv(OUT/'permutation_sample_baselines.csv',index=False)
    return dict(total_repeat_results=len(results),negative_repeat_results=int(results.importance.lt(0).sum()),
        fold_summary_rows=len(folds),aggregate_summary_rows=len(aggregate),baseline_rows=len(baseline))


def main(source_root,run_all):
    OUT.mkdir(parents=True,exist_ok=True); REPORTS.mkdir(parents=True,exist_ok=True)
    source=Path(source_root).resolve()
    protected={p:sha(p) for folder in ['reports','results','models','notebooks','data/processed']
               for p in (source/folder).rglob('*') if p.is_file()}
    for r in pd.read_csv(REPORTS/'member4_protected_checksums.csv').itertuples():
        require(protected[source/r.source_relative_path]==r.sha256,'Protected baseline changed: '+r.source_relative_path)
    source_hashes=dict(protected); source_hashes['root']=source
    cache_before={p:sha(p) for p in (REPO/'models/member4').glob('*') if p.is_file()}
    versions={p:importlib.metadata.version(p) for p in ['pandas','numpy','scikit-learn','joblib','matplotlib','scipy']}
    versions['Python']=platform.python_version()
    pilot_path=REPORTS/'permutation_pilot.json'
    if run_all:
        require(pilot_path.exists(),'Run and inspect the pilot first')
        pilot=json.loads(pilot_path.read_text())
        require(pilot['config']==CONFIG and pilot['script_sha256']==sha(Path(__file__)),'Pilot configuration/script changed')
        require(pilot['pilot_completed'] and pilot['elapsed_seconds']>0,'Pilot not completed')
        print(f'Pilot verified: {pilot["elapsed_seconds"]:.2f}s for one complete fold; scaling to remaining folds.',flush=True)
    runs=[]; designs={}
    with threadpool_limits(limits=CONFIG['threadpool_limit']):
        for dataset in (['shanghai','cgmacros'] if run_all else ['shanghai']):
            print(f'{dataset}: loading approved development data and selecting blocks',flush=True)
            data=load_development(source,dataset)
            selection,coverage,design=sample_development(data)
            designs[dataset]=design
            coverage.to_csv(OUT/f'{dataset}_sample_coverage.local.csv',index=False)
            for fold in (range(1,6) if run_all else [1]):
                runs.append(evaluate_fold(data,fold,selection,coverage,design,source_hashes,versions,'full' if run_all else 'pilot'))
    require(all(sha(p)==h for p,h in protected.items()),'Protected source changed during permutation analysis')
    require(all(sha(p)==h for p,h in cache_before.items()),'Fitted model cache changed during permutation analysis')
    evidence=dict(date=datetime.now(ZoneInfo('Asia/Colombo')).isoformat(),config=CONFIG,script_sha256=sha(Path(__file__)),
        versions=versions,sampling_designs=designs,protected_files=len(protected),protected_changes=0,
        fitted_cache_files_checked=len(cache_before),fitted_cache_changes=0,model_fits=0,final_test_predictions=0,runs=runs)
    if not run_all:
        pilot=dict(config=CONFIG,script_sha256=sha(Path(__file__)),pilot_completed=True,
            elapsed_seconds=runs[0]['elapsed_seconds'],method_timings=runs[0]['method_timings'],
            estimate_ten_fold_compute_seconds=runs[0]['elapsed_seconds']*10,
            estimate_note='Rough pilot-only projection; different datasets, sample sizes and I/O change runtime',evidence=evidence)
        pilot_path.write_text(json.dumps(pilot,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in pilot.items() if k!='evidence'},indent=2),flush=True)
    else:
        evidence['output_counts']=summarize()
        (REPORTS/'permutation_analysis_evidence.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
        print(json.dumps(evidence['output_counts'],indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-root',required=True,type=Path)
    parser.add_argument('--run-all',action='store_true',help='Requires a completed pilot with the exact same script/configuration')
    args=parser.parse_args()
    main(args.source_root,args.run_all)
