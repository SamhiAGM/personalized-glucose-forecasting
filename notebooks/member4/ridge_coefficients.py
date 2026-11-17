"""Ridge coefficient mapping and stability on the five existing development folds.

No tuning or final-test predictions. Cached pipelines contain fitted statistics and
are kept local. All ranks used for interpretation stay within comparable scales.
"""
import argparse
import hashlib
import importlib.metadata
import itertools
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
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_access import load_development, require

REPO = Path(__file__).resolve().parents[2]
ZERO_TOL = 1e-8


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def extract_coefficients(pipeline, metadata, reference=None):
    """Map exact output positions, never parse underscores in feature names.

    reference is a saved all-development mapping, used only AFTER fitting to
    express fold coefficients on one descriptive scale. It never enters fit().
    """
    pre = pipeline.named_steps['preprocessor']
    est = pipeline.named_steps['model']
    require(isinstance(est, Ridge), 'Expected a Ridge estimator')
    require(est.fit_intercept, 'This mapping expects an intercept')
    coef = np.asarray(est.coef_)
    require(coef.ndim == 1, 'Expected a single regression target')
    names = list(pre.get_feature_names_out())
    require(len(names) == len(coef), 'Transformed feature/coefficient mismatch')
    approved = metadata.feature.tolist()
    require(list(pipeline.feature_names_in_) == approved, 'Input feature order mismatch')
    groups = dict(zip(metadata.feature, metadata.feature_group))
    records = []
    numeric_intercept = float(est.intercept_)

    def numeric(branch, columns, positions):
        nonlocal numeric_intercept
        scaler = branch.named_steps['scaler']
        imputer = branch.named_steps['imputer']
        require(isinstance(scaler, StandardScaler) and scaler.with_mean and scaler.with_std,
                'Expected centered, standardized numeric preprocessing')
        require(len(columns) == len(positions) == len(scaler.scale_) == len(imputer.statistics_),
                'Numeric output count changed during preprocessing')
        for j, (feature, pos) in enumerate(zip(columns, positions)):
            scale = float(scaler.scale_[j])
            center = float(scaler.mean_[j])
            var = float(scaler.var_[j])
            slope = float(coef[pos]) / scale
            numeric_intercept -= slope * center
            # sqrt(var_) rather than scale_ marks constant references correctly.
            reference_sd = float(reference.loc[feature, 'reference_sd']) if reference is not None else float(np.sqrt(var))
            records.append(dict(transformed_feature=names[pos], original_feature=feature,
                feature_group=groups[feature], kind='numeric_standardized', category='',
                coefficient_transformed=float(coef[pos]), coefficient_per_original_unit=slope,
                imputation_strategy=imputer.strategy, imputation_value=float(imputer.statistics_[j]),
                scaler_mean=center, scaler_scale=scale, scaler_variance=var,
                reference_sd=reference_sd, coefficient_common_reference_sd=slope*reference_sd,
                comparison_group='numeric_per_reference_sd', rank_eligible=reference_sd>0,
                drop_reference='not_applicable', handle_unknown='not_applicable'))

    if isinstance(pre, ColumnTransformer):
        require(pre.remainder == 'drop', 'Unexpected remainder transformer')
        for branch_name, branch, columns in pre.transformers_:
            if branch_name == 'remainder':
                continue
            indices = list(range(len(names)))[pre.output_indices_[branch_name]]
            if branch_name == 'numeric':
                numeric(branch, list(columns), indices)
            elif branch_name == 'categorical':
                encoder = branch.named_steps['encoder']
                imputer = branch.named_steps['imputer']
                require(encoder.drop is None and encoder.drop_idx_ is None,
                        'Expected full dummy coding; review mapping if coding changes')
                require(encoder.min_frequency is None and encoder.max_categories is None,
                        'Infrequent category grouping requires a different mapping')
                pos_iter = iter(indices)
                for j, (feature, categories) in enumerate(zip(columns, encoder.categories_)):
                    for category in categories:
                        pos = next(pos_iter)
                        records.append(dict(transformed_feature=names[pos], original_feature=feature,
                            feature_group=groups[feature], kind='categorical_indicator', category=str(category),
                            coefficient_transformed=float(coef[pos]), coefficient_per_original_unit=float(coef[pos]),
                            imputation_strategy=imputer.strategy, imputation_value=str(imputer.statistics_[j]),
                            scaler_mean=0., scaler_scale=1., scaler_variance=None, reference_sd=None,
                            coefficient_common_reference_sd=None, comparison_group='indicator_unscaled',
                            rank_eligible=False, drop_reference='none_full_dummy_coding', handle_unknown=encoder.handle_unknown))
                require(next(pos_iter, None) is None, 'Unmapped categorical output')
            else:
                raise ValueError(f'Unexpected preprocessing branch: {branch_name}')
    elif isinstance(pre, Pipeline):
        numeric(pre, approved, list(range(len(names))))
    else:
        raise TypeError(f'Unsupported preprocessor {type(pre)}')
    result = pd.DataFrame(records).set_index('transformed_feature').loc[names].reset_index()
    require(set(result.original_feature) == set(approved), 'Unmapped approved feature')
    result['absolute_coefficient_transformed'] = result.coefficient_transformed.abs()
    # This rank describes storage values ONLY: dummy and standardized columns differ.
    result['absolute_rank_within_transform_kind'] = result.groupby('kind').absolute_coefficient_transformed.rank(ascending=False, method='average')
    eligible = result.rank_eligible
    result['absolute_rank_common_reference_sd'] = np.nan
    result.loc[eligible, 'absolute_rank_common_reference_sd'] = result.loc[eligible, 'coefficient_common_reference_sd'].abs().rank(ascending=False, method='average')
    return result, dict(intercept_transformed=float(est.intercept_), intercept_original_numeric_system=numeric_intercept,
                        alpha=float(est.alpha), n_inputs=len(approved), n_transformed=len(names))


def verify_reconstruction(model, table, intercept, X):
    """Check both transformed positions and unscaled numeric slopes on dev rows."""
    sample = X.iloc[np.linspace(0, len(X)-1, min(32, len(X)), dtype=int)]
    z = np.asarray(model.named_steps['preprocessor'].transform(sample))
    predicted = model.predict(sample)
    direct = z @ table.coefficient_transformed.to_numpy() + intercept['intercept_transformed']
    raw = np.full(len(sample), intercept['intercept_original_numeric_system'])
    for i, row in enumerate(table.itertuples()):
        if row.kind == 'numeric_standardized':
            imputed = sample[row.original_feature].fillna(row.imputation_value).to_numpy(dtype=float)
            raw += imputed * row.coefficient_per_original_unit
        else:
            raw += z[:, i] * row.coefficient_transformed
    np.testing.assert_allclose(direct, predicted, rtol=1e-10, atol=1e-7)
    np.testing.assert_allclose(raw, predicted, rtol=1e-10, atol=1e-7)
    return dict(rows=len(sample), transformed_max_absolute_error=float(np.max(np.abs(direct-predicted))),
                original_scale_max_absolute_error=float(np.max(np.abs(raw-predicted))))


def categorical_contrasts(table):
    output=[]
    cats=table.loc[table.kind.eq('categorical_indicator')]
    for feature, group in cats.groupby('original_feature'):
        for (_, a), (_, b) in itertools.combinations(group.iterrows(), 2):
            output.append(dict(original_feature=feature, contrast=f'{b.category} minus {a.category}',
                               coefficient_contrast=float(b.coefficient_transformed-a.coefficient_transformed)))
    return output


def summarize_stability(folds, saved):
    records=[]
    for feature, group in folds.groupby('transformed_feature',sort=False):
        original=saved.set_index('transformed_feature').loc[feature]
        numeric=original.kind=='numeric_standardized'
        column='coefficient_common_reference_sd' if numeric else 'coefficient_transformed'
        values=group[column].to_numpy(dtype=float)
        signs=np.where(np.abs(values)<=ZERO_TOL,0,np.sign(values))
        saved_value=float(original[column])
        saved_sign=0 if abs(saved_value)<=ZERO_TOL else np.sign(saved_value)
        ranks=group['absolute_rank_common_reference_sd' if numeric else 'absolute_rank_within_transform_kind']
        records.append(dict(transformed_feature=feature,original_feature=original.original_feature,
            feature_group=original.feature_group,kind=original.kind,comparison_scale=column,
            saved_coefficient=saved_value,fold_mean=float(values.mean()),fold_std=float(values.std(ddof=1)),
            fold_min=float(values.min()),fold_max=float(values.max()),
            positive_folds=int((signs>0).sum()),negative_folds=int((signs<0).sum()),zero_folds=int((signs==0).sum()),
            dominant_nonzero_sign_fraction=float(max((signs>0).sum(),(signs<0).sum())/len(signs)),
            sign_agreement_with_saved_fraction=float((signs==saved_sign).mean()),
            rank_mean=float(ranks.mean()),rank_std=float(ranks.std(ddof=1)),rank_min=float(ranks.min()),rank_max=float(ranks.max()),
            numeric_top5_fraction=float(ranks.le(5).mean()) if numeric and bool(original.rank_eligible) else None))
    return pd.DataFrame(records)


def make_plots(dataset, saved, folds, stability, folder):
    num=stability.loc[stability.kind.eq('numeric_standardized')].copy()
    selected=num.assign(magnitude=num.saved_coefficient.abs()).nlargest(15,'magnitude').sort_values('magnitude')
    indices=np.arange(len(selected))
    fig,ax=plt.subplots(figsize=(11,7.8))
    for i,row in enumerate(selected.itertuples()):
        values=folds.loc[folds.transformed_feature.eq(row.transformed_feature),'coefficient_common_reference_sd'].to_numpy()
        ax.scatter(values,np.full(len(values),i)+np.linspace(-.13,.13,len(values)),s=30,alpha=.75,color='#386cb0',label='Five training-fold fits' if i==0 else None)
        ax.scatter(row.saved_coefficient,i,marker='D',s=44,color='#d95f02',label='Saved all-development fit' if i==0 else None,zorder=3)
    ax.axvline(0,color='gray',lw=.8)
    ax.set_yticks(indices,selected.original_feature)
    ax.set_xlabel('Coefficient × one fixed reference SD (target glucose units; units unverified)')
    ax.set_title(f'{dataset}: Ridge signed coefficients and fold variation\nTop 15 numeric inputs by saved absolute coefficient; categorical indicators excluded')
    ax.legend(loc='best'); ax.grid(axis='x',alpha=.2)
    fig.tight_layout(); fig.savefig(folder/f'{dataset}_ridge_signed_stability.png',dpi=180); plt.close(fig)

    top=num.assign(magnitude=num.saved_coefficient.abs()).nlargest(15,'magnitude').original_feature.tolist()
    pivot=folds.loc[folds.original_feature.isin(top)&folds.kind.eq('numeric_standardized')].pivot(index='original_feature',columns='validation_fold',values='absolute_rank_common_reference_sd').loc[top]
    fig,ax=plt.subplots(figsize=(8,8))
    image=ax.imshow(pivot.to_numpy(),cmap='viridis_r',vmin=1,vmax=len(num),aspect='auto')
    for i in range(len(pivot)):
        for j in range(5):
            ax.text(j,i,f'{pivot.iloc[i,j]:g}',ha='center',va='center',fontsize=9,
                    color='black' if pivot.iloc[i,j]<len(num)*.55 else 'white')
    ax.set_yticks(range(len(top)),top); ax.set_xticks(range(5),[f'Fold {i}' for i in range(1,6)])
    ax.set_title(f'{dataset}: absolute-rank stability\nNumeric inputs on fixed reference SD scale; 1 = largest')
    fig.colorbar(image,ax=ax,label='Rank among numeric inputs')
    fig.tight_layout(); fig.savefig(folder/f'{dataset}_ridge_rank_stability.png',dpi=180); plt.close(fig)


def main(source_root):
    start=time.perf_counter()
    source=Path(source_root).resolve()
    tables=REPO/'results/member4/explainability'; figures=REPO/'results/member4/figures'
    cache=REPO/'models/member4'; reports=REPO/'reports/member4'
    for path in [tables,figures,cache,reports]: path.mkdir(parents=True,exist_ok=True)
    protected={p:sha(p) for f in ['reports','results','models','notebooks','data/processed']
               for p in (source/f).rglob('*') if p.is_file()}
    # Check the prior Member 4 baseline as well as before/after equality.
    prior=pd.read_csv(reports/'member4_protected_checksums.csv')
    for row in prior.itertuples():
        require(protected[source/row.source_relative_path]==row.sha256,'Source changed since readiness audit: '+row.source_relative_path)
    inventory=pd.read_csv(source/'reports/evaluation/member3_trained_model_inventory.csv')
    versions={p:importlib.metadata.version(p) for p in ['pandas','numpy','scikit-learn','joblib','matplotlib','scipy']}
    versions['Python']=platform.python_version()
    provenance=[]; report_bits=[]; all_intercepts=[]; reconstructions=[]
    for dataset in ['shanghai','cgmacros']:
        print(f'{dataset}: loading development rows',flush=True)
        data=load_development(source,dataset)
        model_path=source/f'models/{dataset}_ridge_pipeline.joblib'
        expected=inventory.loc[inventory.model_file.eq(f'models/{dataset}_ridge_pipeline.joblib'),'sha256'].item()
        require(sha(model_path)==expected,'Saved Ridge hash mismatch')
        saved_model=joblib.load(model_path)
        saved,intercept=extract_coefficients(saved_model,data.feature_metadata)
        saved.insert(0,'dataset',dataset); saved.insert(1,'scope','saved_all_development_descriptive')
        saved.to_csv(tables/f'{dataset}_ridge_saved_coefficients.csv',index=False)
        X,_=data.xy()
        reconstructions.append(dict(dataset=dataset,scope='saved_all_development_descriptive',**verify_reconstruction(saved_model,saved,intercept,X)))
        all_intercepts.append(dict(dataset=dataset,scope='saved_all_development_descriptive',validation_fold=0,**intercept))
        reference=saved.loc[saved.kind.eq('numeric_standardized')].set_index('original_feature')
        config={p:sha(source/p) for p in [f'models/{dataset}_ridge_pipeline.joblib',f'data/processed/{dataset}_ml_ready.csv',
                f'reports/modeling/{dataset}_split.csv',f'reports/modeling/{dataset}_development_folds.csv',
                f'reports/data_audit/{dataset}_final_feature_list.csv']}
        signature=joblib.hash(clone(saved_model).get_params(deep=True))
        fold_tables=[]; contrasts=[dict(dataset=dataset,scope='saved_all_development_descriptive',validation_fold=0,**r) for r in categorical_contrasts(saved)]
        for number in range(1,6):
            training,validation=data.fold(number)
            train_ids=set(training.participant_id); val_ids=set(validation.participant_id)
            require(train_ids.isdisjoint(val_ids|set(data.locked_ids)),'Fold leakage')
            identity=dict(dataset=dataset,configuration='all_features',validation_fold=number,
                source_sha256=config,hyperparameter_signature=signature,versions=versions,
                training_ids_sha256=json_sha(sorted(train_ids)),validation_ids_sha256=json_sha(sorted(val_ids)),
                train_participants=len(train_ids),validation_participants=len(val_ids),train_rows=len(training),
                validation_rows=len(validation),locked_participants_used=0,features=data.features)
            identity_sha=json_sha(identity)
            model_file=cache/f'{dataset}_ridge_fold{number}.joblib'
            sidecar=cache/f'{dataset}_ridge_fold{number}.json'
            if model_file.exists() or sidecar.exists():
                require(model_file.exists() and sidecar.exists(),'Incomplete cache; inspect before rerunning')
                cached=json.loads(sidecar.read_text())
                require(cached['identity_sha256']==identity_sha and cached['model_sha256']==sha(model_file),
                        'Incompatible/tampered cache; inspect before replacing')
                model=joblib.load(model_file); action='reused_compatible_cache'
            else:
                model=clone(saved_model)
                require(not hasattr(model,'feature_names_in_'),'Clone unexpectedly fitted')
                require(joblib.hash(model.get_params(deep=True))==signature,'Clone hyperparameters changed')
                X_train,y_train=data.xy(training)
                require(y_train.notna().all() and np.isfinite(y_train).all(),'Invalid training target')
                print(f'{dataset}: fitting existing training fold {number} ({len(train_ids)} participants)',flush=True)
                model.fit(X_train,y_train)
                joblib.dump(model,model_file)
                cached=dict(identity_sha256=identity_sha,model_sha256=sha(model_file),identity=identity)
                sidecar.write_text(json.dumps(cached,indent=2),encoding='utf-8')
                action='new_training_only_clone'
            require(joblib.hash(clone(model).get_params(deep=True))==signature,'Cached model settings differ')
            mapped,fold_intercept=extract_coefficients(model,data.feature_metadata,reference=reference)
            require(mapped.transformed_feature.tolist()==saved.transformed_feature.tolist(),'Fold category schema differs; align explicitly')
            mapped.insert(0,'dataset',dataset); mapped.insert(1,'scope','training_fold_only'); mapped.insert(2,'validation_fold',number)
            fold_tables.append(mapped)
            X_train,_=data.xy(training)
            reconstructions.append(dict(dataset=dataset,scope=f'training_fold_{number}',**verify_reconstruction(model,mapped,fold_intercept,X_train)))
            all_intercepts.append(dict(dataset=dataset,scope='training_fold_only',validation_fold=number,**fold_intercept))
            contrasts += [dict(dataset=dataset,scope='training_fold_only',validation_fold=number,**r) for r in categorical_contrasts(mapped)]
            provenance.append(dict(**identity,cache_action=action,identity_sha256=identity_sha,model_sha256=cached['model_sha256'],cache_file=model_file.relative_to(REPO).as_posix()))
        folds=pd.concat(fold_tables,ignore_index=True)
        folds.to_csv(tables/f'{dataset}_ridge_fold_coefficients.csv',index=False)
        stability=summarize_stability(folds,saved)
        stability.insert(0,'dataset',dataset)
        stability.to_csv(tables/f'{dataset}_ridge_coefficient_stability.csv',index=False)
        if contrasts:
            pd.DataFrame(contrasts).to_csv(tables/f'{dataset}_ridge_categorical_contrasts.csv',index=False)
        numeric=folds.loc[folds.kind.eq('numeric_standardized')]
        rank_pivot=numeric.pivot(index='transformed_feature',columns='validation_fold',values='absolute_rank_common_reference_sd')
        correlations=rank_pivot.corr(method='spearman')
        correlations.to_csv(tables/f'{dataset}_ridge_fold_rank_correlations.csv')
        # Descriptive glucose-history correlation on retained development rows only.
        history=data.feature_metadata.loc[data.feature_metadata.feature_group.eq('glucose_history'),'feature'].tolist()
        data.frame[history].corr().to_csv(tables/f'{dataset}_glucose_history_correlations.csv')
        pairs=[]
        for a,b in itertools.combinations(history,2):
            pair=data.frame[[a,b]].dropna()
            pairs.append(dict(feature_a=a,feature_b=b,pearson_r=pair[a].corr(pair[b]),pairwise_rows=len(pair)))
        pd.DataFrame(pairs).sort_values('pearson_r',key=lambda s:s.abs(),ascending=False).to_csv(tables/f'{dataset}_glucose_history_correlation_pairs.csv',index=False)
        make_plots(dataset,saved,folds,stability,figures)
        report_bits.append(dict(dataset=dataset,alpha=intercept['alpha'],rows=len(data.frame),features=len(data.features),
            transformed=len(saved),top_numeric=stability.loc[stability.kind.eq('numeric_standardized')].assign(abs_saved=lambda d:d.saved_coefficient.abs()).nlargest(8,'abs_saved').to_dict('records'),
            sign_changing=stability.loc[(stability.positive_folds>0)&(stability.negative_folds>0),'original_feature'].tolist(),
            rank_spearman_mean=float(correlations.to_numpy()[np.triu_indices(5,1)].mean()),
            categorical_contrasts=contrasts,highest_correlation_pairs=pd.DataFrame(pairs).assign(abs_r=lambda d:d.pearson_r.abs()).nlargest(4,'abs_r').drop(columns='abs_r').to_dict('records')))
    pd.DataFrame(all_intercepts).to_csv(tables/'ridge_intercepts.csv',index=False)
    pd.DataFrame(reconstructions).to_csv(tables/'ridge_coefficient_reconstruction_checks.csv',index=False)
    require(all(sha(p)==h for p,h in protected.items()),'Protected source changed during analysis')
    evidence=dict(run_date=datetime.now(ZoneInfo('Asia/Colombo')).isoformat(),versions=versions,
        elapsed_seconds=time.perf_counter()-start,protected_files_checked=len(protected),protected_changes=0,
        sign_zero_tolerance=ZERO_TOL,fold_models=provenance,datasets=report_bits,
        final_test_predictions=0,validation_metrics_recomputed=False,
        scope='all_features; coefficients only; no new hyperparameter selection')
    (reports/'ridge_coefficient_analysis_evidence.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print(json.dumps(report_bits,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-root',required=True,type=Path)
    args=parser.parse_args()
    main(args.source_root)
