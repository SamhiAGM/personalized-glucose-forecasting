"""Verify saved SHAP outputs, train/held-out membership, mapping and source hashes."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tree_shap_analysis import REPO,OUT,TREE,REPORT,CONFIG,collapse,summaries
from permutation_importance import sha,seed_for
from data_access import require


def main(source_root):
    source=Path(source_root); evidence=json.loads((REPORT/'tree_shap_evidence.json').read_text())
    require(len(evidence['runs'])==10 and len(evidence['tree_models'])==4,'Incomplete analysis')
    max_error=0.; count=0
    for dataset in ['shanghai','cgmacros']:
        split=pd.read_csv(source/f'reports/modeling/{dataset}_split.csv',dtype={'participant_id':'string'})
        folds=pd.read_csv(source/f'reports/modeling/{dataset}_development_folds.csv',dtype={'participant_id':'string'})
        meta=pd.read_csv(source/f'reports/data_audit/{dataset}_final_feature_list.csv').sort_values('feature_number')
        dev=set(folds.participant_id); locked=set(split.loc[split.dataset_split.eq('test'),'participant_id'])
        all_heldout=set()
        for fold in range(1,6):
            stem=f'{dataset}_fold{fold}'; record=json.loads((OUT/f'{stem}_runtime.json').read_text())
            background=pd.read_csv(OUT/record['background_file'],dtype={'participant_id':'string'})
            explained=pd.read_csv(OUT/record['sample_file'],dtype={'participant_id':'string'})
            require(sha(OUT/record['background_file'])==record['background_selection_sha256'],'Background hash')
            require(sha(OUT/record['sample_file'])==record['explanation_selection_sha256'],'Explanation hash')
            valid=set(folds.loc[folds.validation_fold.eq(fold),'participant_id']); train=dev-valid
            require(set(background.participant_id)==train and set(explained.participant_id)==valid,'Incorrect sample memberships')
            require(train.isdisjoint(valid|locked) and valid.isdisjoint(locked),'Participant leakage')
            require(set(background.development_row_index).isdisjoint(explained.development_row_index),'Background row leakage')
            require(background.development_row_index.is_unique and explained.development_row_index.is_unique,'Duplicate sample row')
            for table,purpose,quota in [(background,'background',8),(explained,'explain',24)]:
                require(table.groupby('participant_id').size().eq(quota).all(),'Unbalanced sampling')
                for pid,group in table.groupby('participant_id'):
                    require(group.selection_seed.eq(seed_for('SHAP',dataset,fold,purpose,str(pid))).all(),'Sample seed mismatch')
            prior=pd.read_csv(REPO/f'results/member4/permutation/{stem}_selected_rows.local.csv')
            require(set(explained.development_row_index).issubset(prior.development_row_index),'Not reused prior validation selection')
            require(sha(REPO/f'models/member4/{dataset}_ridge_fold{fold}.joblib')==record['model_sha256'],'Fold model changed')
            mapping=pd.read_csv(OUT/f'{stem}_feature_mapping.csv')
            arrays=np.load(OUT/f'local/{stem}_row_explanations.npz')
            values=arrays['values']; require(values.shape==(len(explained),len(mapping)),'SHAP dimensions')
            reconstructed=arrays['base_values']+values.sum(axis=1)
            error=float(np.abs(reconstructed-arrays['predictions']).max()); max_error=max(error,max_error)
            np.testing.assert_allclose(reconstructed,arrays['predictions'],atol=CONFIG['atol'],rtol=CONFIG['rtol'])
            originals=collapse(values,mapping,'original_feature',meta.feature.tolist())
            groups=meta.feature_group.drop_duplicates().tolist()
            grouped=collapse(values,mapping,'feature_group',groups)
            for level,v,names in [('transformed',values,mapping.transformed_feature.tolist()),('original',originals,meta.feature.tolist()),('group',grouped,groups)]:
                expected=summaries(v,names,explained.participant_id)
                saved=pd.read_csv(OUT/f'{stem}_{level}_summary.csv')
                require(saved.name.tolist()==names,'Summary feature mapping')
                np.testing.assert_allclose(saved.mean_abs_shap,expected.mean_abs_shap,atol=1e-10,rtol=1e-10)
            if fold==1:
                examples=json.loads((OUT/f'local/{stem}_local_examples.json').read_text())
                ids=sorted(valid); expected_ids=[ids[0],ids[len(ids)//2],ids[-1]]
                require([r['participant_id'] for r in examples]==expected_ids,'Local example selection changed')
                for ex in examples:
                    sub=explained.loc[explained.participant_id.eq(ex['participant_id'])]
                    require(ex['explained_row_position']==int(pd.to_datetime(sub.timestamp).idxmin()),'Example timestamp rule')
                    require((OUT/f'local/{stem}_example{ex["example"]}.png').is_file(),'Missing waterfall')
            all_heldout.update(valid); count+=len(explained)
        require(all_heldout==dev,'Development participants omitted')
        for model in ['random_forest','gradient_boosting']:
            transformed=pd.read_csv(TREE/f'{dataset}_{model}_transformed.csv')
            original=pd.read_csv(TREE/f'{dataset}_{model}_original.csv')
            grouped=transformed.groupby('original_feature').importance.sum()
            np.testing.assert_allclose(original.set_index('original_feature').importance.sort_index(),grouped.sort_index(),atol=1e-12)
            np.testing.assert_allclose(original.importance.sum(),1,atol=1e-10)
    fold_tables=pd.read_csv(OUT/'ridge_shap_fold_summaries.csv')
    summary=pd.read_csv(OUT/'ridge_shap_aggregate.csv')
    for r in summary.itertuples():
        fold=fold_tables.loc[fold_tables.dataset.eq(r.dataset)&fold_tables.level.eq(r.level)&fold_tables.name.eq(r.name)]
        require(len(fold)==5,'Summary lacks five folds')
        np.testing.assert_allclose([r.mean_abs_shap,r.participant_weighted_mean_abs_shap],
            [fold.mean_abs_shap.mean(),np.average(fold.mean_abs_shap,weights=fold.participants)],atol=1e-10)
    prior=pd.read_csv(REPORT/'member4_protected_checksums.csv')
    for r in prior.itertuples(): require(sha(source/r.source_relative_path)==r.sha256,'Protected source changed')
    record=dict(status='PASS',folds=10,tree_models=4,heldout_explanations=count,max_additivity_error=max_error,
        protected_files=len(prior),protected_changes=0,checks=['train-only backgrounds','heldout-only explanation memberships',
        'sample balance and seeds','reused validation rows','model/selection hashes','additivity','signed original/group collapse',
        'summary reconstruction','tree MDI aggregation','local-example preselection rule','source baseline integrity'])
    (REPORT/'tree_shap_verification.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    paths=[p for folder in [OUT,TREE] for p in folder.rglob('*') if p.is_file() and OUT/'local' not in p.parents]
    paths+=list((REPO/'notebooks/member4').glob('*tree_shap*.py'))
    paths+=list((REPO/'results/member4/figures').glob('*shap*.png'))
    paths+=list((REPO/'results/member4/figures').glob('*tree_importance.png'))
    paths+=list(REPORT.glob('tree_shap*.json'))
    paths+=[REPORT/'tree_shap_interpretation.md']
    with (REPORT/'tree_shap_output_manifest.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['relative_path','size_bytes','sha256']); writer.writeheader()
        for path in sorted(set(paths)):
            require(path.is_file(),'Missing planned artifact: '+str(path))
            writer.writerow(dict(relative_path=path.relative_to(REPO).as_posix(),size_bytes=path.stat().st_size,sha256=sha(path)))
    # Row-linked artifact hashes stay private along with the artifacts themselves.
    with (OUT/'local/local_output_manifest.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['relative_path','size_bytes','sha256']); writer.writeheader()
        for path in sorted((OUT/'local').glob('*')):
            if path.is_file() and path.name!='local_output_manifest.csv':
                writer.writerow(dict(relative_path=path.relative_to(OUT/'local').as_posix(),size_bytes=path.stat().st_size,sha256=sha(path)))
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--source-root',required=True,type=Path)
    main(parser.parse_args().source_root)
