"""Descriptive tree MDI and held-out, training-background Ridge SHAP. No fit()."""
import argparse
import importlib.metadata
import json
import platform
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from data_access import load_development,require
from permutation_importance import checked_fold_model,sha,seed_for

REPO=Path(__file__).resolve().parents[2]
REPORT=REPO/'reports/member4'
OUT=REPO/'results/member4/shap'
TREE=REPO/'results/member4/tree_importance'
CONFIG=dict(seed=42,background_rows_per_participant=8,explanation_rows_per_participant=24,
    explainer='LinearExplainer',masker='Independent',link='identity',atol=1e-7,rtol=1e-9,
    local_rule='Fold 1: first, middle, last sorted participant IDs; earliest sampled explanation timestamp per person',
    grouping='sum signed transformed attributions per original feature BEFORE absolute value',
    repeats=1)


def feature_mapping(pipeline,metadata):
    """Map positions using fitted branch selectors/categories, not string splitting."""
    pre=pipeline.named_steps['preprocessor']; names=list(pre.get_feature_names_out())
    require(list(pipeline.feature_names_in_)==metadata.feature.tolist(),'Predictive input order mismatch')
    group=dict(zip(metadata.feature,metadata.feature_group)); records=[]
    if isinstance(pre,ColumnTransformer):
        for label,branch,columns in pre.transformers_:
            if label=='remainder':
                require(branch=='drop','Unexpected remainder'); continue
            positions=list(range(len(names)))[pre.output_indices_[label]]
            if label=='numeric': originals=list(columns); categories=['']*len(columns)
            elif label=='categorical':
                encoder=branch.named_steps['encoder']
                require(encoder.drop is None and encoder.min_frequency is None and encoder.max_categories is None,'Review non-full categorical coding')
                originals=[col for col,cats in zip(columns,encoder.categories_) for _ in cats]
                categories=[str(cat) for cats in encoder.categories_ for cat in cats]
            else: raise ValueError('Unsupported branch '+label)
            require(len(positions)==len(originals),'Unmapped transformed columns')
            for pos,col,category in zip(positions,originals,categories):
                records.append(dict(position=pos,transformed_feature=names[pos],original_feature=col,
                    feature_group=group[col],category=category))
    elif isinstance(pre,Pipeline):
        require(names==metadata.feature.tolist(),'Unexpected numeric transformation names')
        records=[dict(position=i,transformed_feature=n,original_feature=n,feature_group=group[n],category='') for i,n in enumerate(names)]
    else: raise TypeError(type(pre))
    mapping=pd.DataFrame(records).sort_values('position').reset_index(drop=True)
    require(mapping.position.tolist()==list(range(len(names))) and set(mapping.original_feature)==set(metadata.feature),'Incomplete mapping')
    return mapping


def collapse(values,mapping,level,ordered):
    result=np.column_stack([values[:,mapping.index[mapping[level].eq(n)]].sum(axis=1) for n in ordered])
    np.testing.assert_allclose(result.sum(axis=1),values.sum(axis=1),rtol=1e-10,atol=1e-8)
    return result


def select_rows(frame,quota,dataset,fold,purpose):
    selected=[]
    for pid,group in frame.groupby('participant_id',sort=True):
        require(len(group)>=quota,'Insufficient participant rows for fixed quota')
        seed=seed_for('SHAP',dataset,fold,purpose,str(pid))
        positions=np.sort(np.random.default_rng(seed).choice(group.index.to_numpy(),quota,replace=False))
        for position in positions:
            selected.append(dict(participant_id=str(pid),development_row_index=int(position),
                timestamp=str(frame.loc[position,'timestamp']),selection_seed=seed))
    return pd.DataFrame(selected)


def linear_explanation(model,background,inputs):
    # max_samples prevents the default masker silently thinning a balanced background.
    masker=shap.maskers.Independent(background,max_samples=len(background))
    explainer=shap.LinearExplainer(model,masker)
    explanation=explainer(inputs)
    values=np.asarray(explanation.values,dtype=float)
    bases=np.broadcast_to(np.asarray(explanation.base_values,dtype=float),len(inputs))
    analytic=(inputs-background.mean(axis=0))*model.coef_
    np.testing.assert_allclose(values,analytic,atol=CONFIG['atol'],rtol=CONFIG['rtol'])
    return explanation,values,bases


def summaries(values,names,participants):
    # Average within participant first; sample quotas make this equal to row means.
    signed=pd.DataFrame(values,columns=names).assign(participant_id=list(participants))
    absolute=pd.DataFrame(np.abs(values),columns=names).assign(participant_id=list(participants))
    means=absolute.groupby('participant_id').mean().mean()
    signed_means=signed.groupby('participant_id').mean().mean()
    result=pd.DataFrame({'name':names,'mean_abs_shap':means.reindex(names).to_numpy(),
                         'mean_signed_shap':signed_means.reindex(names).to_numpy()})
    result['rank']=result.mean_abs_shap.rank(ascending=False,method='average')
    return result


def tree_importance(source,data,inventory):
    provenance=[]
    for model_name in ['random_forest','gradient_boosting']:
        relative=f'models/{data.dataset}_{model_name}_pipeline.joblib'; path=source/relative
        expected=inventory.loc[inventory.model_file.eq(relative),'sha256'].item()
        require(sha(path)==expected,'Tree inventory hash mismatch')
        pipeline=joblib.load(path); mapping=feature_mapping(pipeline,data.feature_metadata)
        importance=pipeline.named_steps['model'].feature_importances_
        require(len(importance)==len(mapping) and np.isfinite(importance).all(),'Invalid tree importance')
        np.testing.assert_allclose(importance.sum(),1,atol=1e-10)
        table=mapping.copy(); table['importance']=importance
        table['rank']=table.importance.rank(ascending=False,method='average')
        table['scope']='saved_all_development_descriptive'; table['model']=model_name; table['dataset']=data.dataset
        table.to_csv(TREE/f'{data.dataset}_{model_name}_transformed.csv',index=False)
        grouped=table.groupby(['original_feature','feature_group'],sort=False).importance.sum().reset_index()
        require(set(grouped.original_feature)==set(data.features),'Missing original tree features')
        np.testing.assert_allclose(grouped.importance.sum(),importance.sum(),atol=1e-12)
        grouped['rank']=grouped.importance.rank(ascending=False,method='average')
        grouped['scope']='saved_all_development_descriptive'; grouped['model']=model_name
        grouped.to_csv(TREE/f'{data.dataset}_{model_name}_original.csv',index=False)
        provenance.append(dict(dataset=data.dataset,model=model_name,model_sha256=expected,
            estimator=type(pipeline.named_steps['model']).__name__,inputs=len(data.features),transformed=len(mapping),
            importance_sum=float(importance.sum()),scope='saved_all_development_descriptive',retrained=False))
    return provenance


def explain_fold(data,fold,hashes,versions):
    start=time.perf_counter(); dataset=data.dataset
    model,validation,model_hash=checked_fold_model(data,fold,hashes,versions)
    training,_=data.fold(fold)
    background=select_rows(training,CONFIG['background_rows_per_participant'],dataset,fold,'background')
    prior=REPO/f'results/member4/permutation/{dataset}_fold{fold}_selected_rows.local.csv'
    old=pd.read_csv(prior,dtype={'participant_id':'string'})
    runtime=json.loads(prior.with_name(f'{dataset}_fold{fold}_runtime.json').read_text())
    require(sha(prior)==runtime['identity']['selection_sha256'],'Permutation selection changed')
    require(runtime['identity']['model_sha256']==model_hash,'Permutation used a different model')
    require(old.development_row_index.is_unique and set(old.participant_id)==set(validation.participant_id),'Invalid prior selection')
    candidates=data.frame.loc[old.development_row_index]
    require(candidates.participant_id.tolist()==old.participant_id.tolist(),'Prior row identities mismatch')
    explained=select_rows(candidates,CONFIG['explanation_rows_per_participant'],dataset,fold,'explain')
    require(set(background.participant_id)==set(training.participant_id),'Background coverage incomplete')
    require(set(explained.participant_id)==set(validation.participant_id),'Explanation coverage incomplete')
    require(set(background.participant_id).isdisjoint(set(explained.participant_id)|set(data.locked_ids)),'Background leakage')
    require(set(explained.participant_id).isdisjoint(data.locked_ids),'Test explanations')
    require(set(background.development_row_index).isdisjoint(explained.development_row_index),'Row overlap')
    local=OUT/'local'; local.mkdir(parents=True,exist_ok=True)
    stem=f'{dataset}_fold{fold}'
    background.to_csv(local/f'{stem}_background.local.csv',index=False)
    explained.to_csv(local/f'{stem}_explained.local.csv',index=False)
    mapping=feature_mapping(model,data.feature_metadata)
    mapping.to_csv(OUT/f'{stem}_feature_mapping.csv',index=False)
    B,_=data.xy(data.frame.loc[background.development_row_index]); X,_=data.xy(data.frame.loc[explained.development_row_index])
    pre=model.named_steps['preprocessor']; estimator=model.named_steps['model']
    Zb=np.asarray(pre.transform(B),dtype=float); Zx=np.asarray(pre.transform(X),dtype=float)
    # Predeclare local examples solely from IDs/timestamps before obtaining SHAP.
    chosen=[]
    if fold==1:
        ids=sorted(explained.participant_id.unique())
        for pid in [ids[0],ids[len(ids)//2],ids[-1]]:
            sub=explained.loc[explained.participant_id.eq(pid)]
            chosen.append(int(pd.to_datetime(sub.timestamp).idxmin()))
    try:
        explanation,values,bases=linear_explanation(estimator,Zb,Zx)
    except Exception as exc:
        # Preserve the actual exception and stop rather than substituting a result.
        (REPORT/f'{stem}_shap_error.json').write_text(json.dumps(dict(error_type=type(exc).__name__,message=str(exc),
            shap_version=shap.__version__,fallback='none applied; inspect compatibility before choosing fallback'),indent=2),encoding='utf-8')
        raise
    predicted=model.predict(X)
    reconstructed=bases+values.sum(axis=1)
    np.testing.assert_allclose(reconstructed,predicted,atol=CONFIG['atol'],rtol=CONFIG['rtol'])
    np.testing.assert_allclose(estimator.predict(Zx),predicted,atol=CONFIG['atol'],rtol=CONFIG['rtol'])
    original=collapse(values,mapping,'original_feature',data.features)
    groups=data.feature_metadata.feature_group.drop_duplicates().tolist()
    grouped=collapse(values,mapping,'feature_group',groups)
    np.savez_compressed(local/f'{stem}_row_explanations.npz',values=values,base_values=bases,
        predictions=predicted,transformed_inputs=Zx,background_transformed=Zb,
        original_feature_values=original,group_values=grouped)
    for level,v,names in [('transformed',values,mapping.transformed_feature.tolist()),('original',original,data.features),('group',grouped,groups)]:
        table=summaries(v,names,explained.participant_id)
        table.insert(0,'dataset',dataset); table.insert(1,'validation_fold',fold); table.insert(2,'level',level)
        table['participants']=explained.participant_id.nunique(); table['explained_rows']=len(explained)
        table.to_csv(OUT/f'{stem}_{level}_summary.csv',index=False)
    locals_manifest=[]
    for number,index in enumerate(chosen,1):
        # Local charts omit identifiers in their title but remain local/private.
        item=shap.Explanation(values=original[index],base_values=bases[index],data=None,feature_names=data.features)
        shap.plots.waterfall(item,max_display=12,show=False)
        plt.title(f'{dataset}: preselected local example {number} (fold 1)\nTarget glucose units unverified')
        plt.tight_layout(); plt.savefig(local/f'{stem}_example{number}.png',dpi=160,bbox_inches='tight'); plt.close('all')
        pd.DataFrame({'feature':data.features,'shap_value':original[index]}).to_csv(local/f'{stem}_example{number}.local.csv',index=False)
        locals_manifest.append(dict(example=number,explained_row_position=index,**explained.iloc[index].to_dict(),
            base_value=float(bases[index]),prediction=float(predicted[index])))
    if locals_manifest:
        (local/f'{stem}_local_examples.json').write_text(json.dumps(locals_manifest,indent=2),encoding='utf-8')
    record=dict(dataset=dataset,validation_fold=fold,model_sha256=model_hash,
        background_participants=background.participant_id.nunique(),background_rows=len(background),
        explained_participants=explained.participant_id.nunique(),explained_rows=len(explained),
        background_selection_sha256=sha(local/f'{stem}_background.local.csv'),
        explanation_selection_sha256=sha(local/f'{stem}_explained.local.csv'),prior_selection_sha256=sha(prior),
        background_rows_per_participant=CONFIG['background_rows_per_participant'],explanation_rows_per_participant=CONFIG['explanation_rows_per_participant'],
        masker_max_samples=len(Zb),base_value=float(bases[0]),max_additivity_error=float(np.max(np.abs(reconstructed-predicted))),
        analytic_formula_verified=True,atol=CONFIG['atol'],rtol=CONFIG['rtol'],elapsed_seconds=time.perf_counter()-start,
        local_examples=len(chosen),locked_participants_used=0,model_fitted=False,
        sample_file=f'local/{stem}_explained.local.csv',background_file=f'local/{stem}_background.local.csv')
    (OUT/f'{stem}_runtime.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(f'{stem}: {len(Zb)} background / {len(Zx)} held-out explanations; {record["elapsed_seconds"]:.2f}s; max residual {record["max_additivity_error"]:.3g}',flush=True)
    return record


def aggregate():
    frames=[pd.read_csv(OUT/f'{d}_fold{f}_{level}_summary.csv') for d in ['shanghai','cgmacros'] for f in range(1,6) for level in ['transformed','original','group']]
    folds=pd.concat(frames,ignore_index=True); folds.to_csv(OUT/'ridge_shap_fold_summaries.csv',index=False)
    rows=[]
    for (dataset,level,name),g in folds.groupby(['dataset','level','name'],sort=False):
        require(len(g)==5,'Incomplete SHAP folds')
        rows.append(dict(dataset=dataset,level=level,name=name,mean_abs_shap=g.mean_abs_shap.mean(),
            fold_sd=g.mean_abs_shap.std(ddof=1),fold_min=g.mean_abs_shap.min(),fold_max=g.mean_abs_shap.max(),
            participant_weighted_mean_abs_shap=np.average(g.mean_abs_shap,weights=g.participants),
            mean_signed_shap=g.mean_signed_shap.mean(),folds=5,rank_mean=g['rank'].mean(),rank_min=g['rank'].min(),rank_max=g['rank'].max()))
    summary=pd.DataFrame(rows); summary['rank']=summary.groupby(['dataset','level']).mean_abs_shap.rank(ascending=False,method='average')
    summary.to_csv(OUT/'ridge_shap_aggregate.csv',index=False)


def main(source_root,all_folds):
    for p in [OUT,TREE,REPORT]: p.mkdir(parents=True,exist_ok=True)
    source=Path(source_root).resolve(); protected={p:sha(p) for folder in ['reports','results','models','notebooks','data/processed'] for p in (source/folder).rglob('*') if p.is_file()}
    for r in pd.read_csv(REPORT/'member4_protected_checksums.csv').itertuples():
        require(protected[source/r.source_relative_path]==r.sha256,'Source baseline changed')
    cache={p:sha(p) for p in (REPO/'models/member4').glob('*') if p.is_file()}
    hashes=dict(protected); hashes['root']=source
    versions={p:importlib.metadata.version(p) for p in ['pandas','numpy','scikit-learn','joblib','matplotlib','scipy']}; versions['Python']=platform.python_version()
    pilot_path=REPORT/'tree_shap_pilot.json'
    if all_folds:
        pilot=json.loads(pilot_path.read_text())
        require(pilot['config']==CONFIG and pilot['script_sha256']==sha(Path(__file__)),'Pilot script/config changed')
        require(pilot['max_additivity_error']<CONFIG['atol'],'Pilot additivity failed')
        print(f'Completed pilot verified: {pilot["elapsed_seconds"]:.2f}s. Scaling to all folds.',flush=True)
    runs=[]; trees=[]; inventory=pd.read_csv(source/'reports/evaluation/member3_trained_model_inventory.csv')
    with threadpool_limits(limits=1):
        for dataset in (['shanghai','cgmacros'] if all_folds else ['shanghai']):
            print(dataset+': development-only loading',flush=True)
            data=load_development(source,dataset)
            trees+=tree_importance(source,data,inventory)
            for fold in (range(1,6) if all_folds else [1]):
                if all_folds and dataset=='shanghai' and fold==1:
                    prior=json.loads((OUT/'shanghai_fold1_runtime.json').read_text())
                    for suffix,key in [('background','background_selection_sha256'),('explained','explanation_selection_sha256')]:
                        require(sha(OUT/f'local/shanghai_fold1_{suffix}.local.csv')==prior[key],'Pilot selections changed')
                    runs.append(prior); print('Reusing completed Shanghai Fold 1 pilot',flush=True)
                else: runs.append(explain_fold(data,fold,hashes,versions))
    require(all(sha(p)==h for p,h in protected.items()),'Protected sources changed')
    require(all(sha(p)==h for p,h in cache.items()),'Cached models changed')
    evidence=dict(date=datetime.now(ZoneInfo('Asia/Colombo')).isoformat(),config=CONFIG,versions={**versions,'shap':shap.__version__},
        script_sha256=sha(Path(__file__)),protected_files=len(protected),protected_changes=0,cached_files=len(cache),cached_changes=0,
        model_fits=0,final_test_predictions=0,tree_models=trees,runs=runs)
    if all_folds:
        aggregate(); (REPORT/'tree_shap_evidence.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    else:
        pilot=dict(config=CONFIG,script_sha256=sha(Path(__file__)),shap_version=shap.__version__,**runs[0],evidence=evidence)
        pilot_path.write_text(json.dumps(pilot,indent=2),encoding='utf-8')
    print('Pilot complete' if not all_folds else 'All ten folds complete',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--source-root',required=True,type=Path)
    parser.add_argument('--all-folds',action='store_true'); args=parser.parse_args()
    main(args.source_root,args.all_folds)
