"""Verify existing permutation artifacts; no fitting, predictions or raw row reads."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from permutation_importance import OUT,REPORTS,CONFIG,sha,seed_for
from data_access import require


def verify(source_root):
    source=Path(source_root)
    evidence=json.loads((REPORTS/'permutation_analysis_evidence.json').read_text())
    all_results=[]
    for dataset,expected_participants,expected_items in [('shanghai',80,29),('cgmacros',36,44)]:
        folds=pd.read_csv(source/f'reports/modeling/{dataset}_development_folds.csv',dtype={'participant_id':'string'})
        split=pd.read_csv(source/f'reports/modeling/{dataset}_split.csv',dtype={'participant_id':'string'})
        locked=set(split.loc[split.dataset_split.eq('test'),'participant_id'])
        covered=set(); selected_indices=[]
        for fold in range(1,6):
            stem=f'{dataset}_fold{fold}'
            runtime=json.loads((OUT/f'{stem}_runtime.json').read_text())
            result_path=OUT/f'{stem}_repeat_results.csv'
            selection_path=OUT/f'{stem}_selected_rows.local.csv'
            require(sha(result_path)==runtime['result_sha256'],'Repeat result hash mismatch')
            require(sha(selection_path)==runtime['identity']['selection_sha256'],'Selected-row hash mismatch')
            selection=pd.read_csv(selection_path,dtype={'participant_id':'string'})
            require(set(selection.participant_id)==set(folds.loc[folds.validation_fold.eq(fold),'participant_id']),'Wrong held-out participants')
            require(set(selection.participant_id).isdisjoint(locked),'Locked participant selected')
            require(selection.validation_fold.eq(fold).all(),'Wrong fold label')
            require(selection.groupby('participant_id').size().eq(runtime['rows_per_participant']).all(),'Unbalanced sample')
            design=runtime['identity']['sampling_design']
            for _,block in selection.groupby(['participant_id','selected_block_id']):
                require(len(block)==design['block_rows'],'Incomplete selected block')
                delta=pd.to_datetime(block.timestamp).diff().dropna().dt.total_seconds()/60
                require(np.isclose(delta,design['nominal_step_minutes'],atol=1e-6,rtol=0).all(),'Block crossed a sampling gap')
            covered.update(selection.participant_id); selected_indices.extend(selection.development_row_index.tolist())
            results=pd.read_csv(result_path)
            require(len(results)==expected_items*CONFIG['repeats']*2,'Incorrect result count')
            require(results.groupby(['method','level','name']).repeat.nunique().eq(CONFIG['repeats']).all(),'Incorrect repeat count')
            np.testing.assert_allclose(results.importance,results.permuted_mae-results.original_mae,rtol=1e-12,atol=1e-12)
            for (method,repeat),group in results.groupby(['method','repeat']):
                expected_seed=seed_for(dataset,fold,method,int(repeat)-1)
                require(group.permutation_seed.eq(expected_seed).all(),'Incorrect recorded permutation seed')
            constant=results.method.eq('within_participant_block')&results.sample_constant_participants.eq(results.participants)
            require(results.loc[constant,'importance'].eq(0).all() and results.loc[constant,'feature_values_changed_fraction'].eq(0).all(),
                    'Constant-within-participant feature unexpectedly changed')
            all_results.append(results)
        require(len(covered)==expected_participants and covered==set(folds.participant_id),'Incomplete development coverage')
        require(len(selected_indices)==len(set(selected_indices)),'Sampled source row repeated across held-out folds')
    results=pd.concat(all_results,ignore_index=True)
    require(not results.duplicated(['dataset','validation_fold','method','level','name','repeat']).any(),'Duplicate result identity')
    aggregate=pd.read_csv(OUT/'permutation_aggregate_summary.csv')
    fold_summary=pd.read_csv(OUT/'permutation_fold_summaries.csv')
    require(len(results)==7300 and len(aggregate)==146 and len(fold_summary)==730,'Incomplete summary dimensions')
    require(aggregate.folds.eq(5).all(),'Not five folds in summary')
    actual=results.groupby(['dataset','validation_fold','method','level','name']).importance.mean()
    saved=fold_summary.set_index(['dataset','validation_fold','method','level','name']).importance_mean
    np.testing.assert_allclose(saved.sort_index(),actual.sort_index(),atol=1e-12,rtol=1e-12)
    expected=actual.groupby(['dataset','method','level','name']).mean()
    saved_aggregate=aggregate.set_index(['dataset','method','level','name']).importance_mean
    np.testing.assert_allclose(saved_aggregate.sort_index(),expected.sort_index(),atol=1e-12,rtol=1e-12)
    require(int(results.importance.lt(0).sum())==evidence['output_counts']['negative_repeat_results'],'Negative results lost')
    checks=dict(status='PASS',repeat_results=len(results),negative_results=int(results.importance.lt(0).sum()),
        heldout_participants={'shanghai':80,'cgmacros':36},
        checks=['output/sample hashes','exact held-out fold membership','locked ID exclusion','participant-balanced counts',
                'contiguous complete blocks','unique selected rows','MAE degradation signs','seeds and repeat counts',
                'within-participant constants','fold and aggregate mean reconstruction'],
        model_fits=0,predictions=0)
    (REPORTS/'permutation_verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-root',required=True,type=Path)
    verify(parser.parse_args().source_root)
