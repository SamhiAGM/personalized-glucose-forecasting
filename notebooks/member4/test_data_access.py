"""Focused leakage tests with synthetic rows and copied assignment metadata."""
import os
import shutil
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from data_access import FORBIDDEN, load_development

SOURCE=Path(os.environ['MEMBER4_SOURCE_ROOT'])

class LeakageChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        for name in ['reports/modeling/shanghai_split.csv','reports/modeling/shanghai_development_folds.csv',
                     'reports/data_audit/shanghai_final_feature_list.csv']:
            dst=self.root/name; dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(SOURCE/name,dst)
        split=pd.read_csv(self.root/'reports/modeling/shanghai_split.csv',dtype={'participant_id':'string'})
        features=pd.read_csv(self.root/'reports/data_audit/shanghai_final_feature_list.csv').feature.tolist()
        frame=pd.DataFrame({feature:[1]*len(split) for feature in features})
        frame['participant_id']=split.participant_id
        frame['timestamp']='2026-01-01 00:00:00'
        frame['target_timestamp_30min']='2026-01-01 00:30:00'
        frame['target_glucose_30min']=1
        self.path=self.root/'data/processed/shanghai_ml_ready.csv'
        self.path.parent.mkdir(parents=True)
        frame.to_csv(self.path,index=False)

    def test_exact_folds_features_and_test_exclusion(self):
        data=load_development(self.root,'shanghai',chunksize=7)
        self.assertEqual(len(data.frame),80)
        self.assertEqual(data.excluded_rows,20)
        X,_=data.xy()
        self.assertEqual(list(X.columns),data.features)
        self.assertFalse(set(X)&FORBIDDEN)
        seen=[]
        for fold in range(1,6):
            train,valid=data.fold(fold)
            self.assertFalse(set(train.participant_id)&set(valid.participant_id))
            seen.extend(valid.participant_id.tolist())
        self.assertEqual(len(seen),len(set(seen)))
        self.assertEqual(set(seen),set(data.development_ids))

    def test_duplicate_fold_assignment_rejected(self):
        p=self.root/'reports/modeling/shanghai_development_folds.csv'
        df=pd.read_csv(p); pd.concat([df,df.iloc[[0]]]).to_csv(p,index=False)
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            load_development(self.root,'shanghai')

    def test_unknown_source_id_rejected(self):
        df=pd.read_csv(self.path,dtype={'participant_id':'string'}); df.loc[0,'participant_id']='unmapped'; df.to_csv(self.path,index=False)
        with self.assertRaisesRegex(ValueError,'Unmapped'):
            load_development(self.root,'shanghai')

    def test_target_in_features_rejected(self):
        p=self.root/'reports/data_audit/shanghai_final_feature_list.csv'
        df=pd.read_csv(p); df.loc[0,'feature']='target_glucose_30min'; df.to_csv(p,index=False)
        with self.assertRaisesRegex(ValueError,'Metadata or target'):
            load_development(self.root,'shanghai')

    def test_locked_frame_rejected_by_xy(self):
        data=load_development(self.root,'shanghai')
        bad=data.frame.iloc[[0]].copy(); bad['participant_id']=next(iter(data.locked_ids))
        with self.assertRaisesRegex(ValueError,'Non-development'):
            data.xy(bad)

if __name__=='__main__':
    unittest.main()
