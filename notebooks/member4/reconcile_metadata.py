"""Check imported definitions against read-only development artifacts; no fitting."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from data_access import load_development,require
from permutation_importance import sha

REPO=Path(__file__).resolve().parents[2]
MAIN='7e75bda108b3abf6f6401bd7620cec9d00eda4ab'


def main(source_root):
    source=Path(source_root).resolve(); reports=REPO/'reports/member4'
    baseline=pd.read_csv(reports/'member4_protected_checksums.csv')
    for r in baseline.itertuples(): require(sha(source/r.source_relative_path)==r.sha256,'Protected baseline changed')
    outputs=[]
    for dataset in ['shanghai','cgmacros']:
        data=load_development(source,dataset); df=data.frame
        imported=pd.read_csv(REPO/f'reports/data_audit/{dataset}_final_feature_list.csv').sort_values('feature_number').reset_index(drop=True)
        pd.testing.assert_frame_equal(imported,data.feature_metadata,check_dtype=False)
        dictionary=pd.read_csv(REPO/f'reports/data_audit/{dataset}_glucose_feature_dictionary.csv')
        history=data.feature_metadata.loc[data.feature_metadata.feature_group.eq('glucose_history'),'feature']
        require(set(dictionary.feature)==set(history),'Glucose dictionary coverage mismatch')
        require(dictionary.loc[~dictionary.feature.eq('glucose_slope_15min'),'unit'].eq('mg/dL').all(),'Glucose units differ')
        require(dictionary.loc[dictionary.feature.eq('glucose_slope_15min'),'unit'].eq('mg/dL/min').all(),'Slope units differ')
        checks={}
        for name,expected in [('glucose_change_15min',df.current_glucose-df.glucose_lag_15min),
                              ('glucose_change_30min',df.current_glucose-df.glucose_lag_30min),
                              ('glucose_slope_15min',df.glucose_change_15min/15)]:
            residual=np.abs(df[name]-expected)
            checks[name]=dict(rows=len(df),max_absolute_residual=float(residual.max()),mismatches=int((~np.isclose(df[name],expected,atol=1e-8,rtol=1e-9)).sum()))
            require(checks[name]['mismatches']==0,'Derived feature mismatch')
        timestamp=pd.to_datetime(df.timestamp); target_time=pd.to_datetime(df.target_timestamp_30min)
        require(((target_time-timestamp)==pd.Timedelta(minutes=30)).all(),'Target horizon mismatch')
        require(not df.duplicated(['participant_id','timestamp']).any(),'Ambiguous future lookup in ML-ready rows')
        lookup=df[['participant_id','timestamp','current_glucose']].rename(columns={'timestamp':'target_timestamp_30min','current_glucose':'future_current_glucose'})
        matched=df[['participant_id','target_timestamp_30min','target_glucose_30min']].merge(lookup,on=['participant_id','target_timestamp_30min'],how='left',validate='many_to_one')
        available=matched.future_current_glucose.notna()
        require(available.any(),'No eligible target matches')
        error=np.abs(matched.loc[available,'target_glucose_30min']-matched.loc[available,'future_current_glucose'])
        mismatches=int((~np.isclose(matched.loc[available,'target_glucose_30min'],matched.loc[available,'future_current_glucose'],atol=1e-8,rtol=1e-9)).sum())
        require(mismatches==0,'Target value mismatch on eligible future ML-ready rows')
        t=timestamp
        hour=t.dt.hour+t.dt.minute/60+t.dt.second/3600
        day=((hour>=6)&(hour<18)).astype(int)
        require(df.day_night_indicator.eq(day).all(),'Day/night definition mismatch')
        # Source target labels are checked only when their future timestamp remains
        # in this same development ML-ready subset. No raw or locked data lookup.
        outputs.append(dict(dataset=dataset,development_participants=len(data.development_ids),development_rows=len(df),
            approved_features=len(data.features),feature_list_semantic_match=True,
            local_feature_list_sha256=sha(source/f'reports/data_audit/{dataset}_final_feature_list.csv'),
            imported_feature_list_sha256=sha(REPO/f'reports/data_audit/{dataset}_final_feature_list.csv'),
            glucose_dictionary_sha256=sha(REPO/f'reports/data_audit/{dataset}_glucose_feature_dictionary.csv'),
            dataset_sha256=sha(source/f'data/processed/{dataset}_ml_ready.csv'),derived_checks=checks,
            exact_horizon_rows=len(df),eligible_target_value_checks=int(available.sum()),
            targets_without_future_ml_ready_row=int((~available).sum()),target_value_mismatches=mismatches,
            max_target_absolute_residual=float(error.max()),day_night_mismatches=0,
            unit_conclusion='mg/dL supported by imported definitions/code and consistent development relationships; no conversion applied'))
    for r in baseline.itertuples(): require(sha(source/r.source_relative_path)==r.sha256,'Protected source changed during reconciliation')
    evidence=dict(imported_main_commit=MAIN,datasets=outputs,protected_files=len(baseline),protected_changes=0,
        model_fits=0,predictions=0,locked_test_measurements_analyzed=0,
        limitation='Source code and schema support lineage; original raw execution/artifact chain is not independently reproduced')
    (reports/'metadata_reconciliation_evidence.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print(json.dumps(evidence,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--source-root',type=Path,required=True)
    main(parser.parse_args().source_root)
