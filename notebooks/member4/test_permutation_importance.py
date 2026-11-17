"""Focused synthetic tests for grouping, block boundaries and signed MAE loss."""
import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from permutation_importance import sample_development,row_mapping,permuted_inputs,importance_score


class SignalPredictor:
    def __init__(self,sign=1): self.sign=sign
    def predict(self,X): return X.signal.to_numpy()*self.sign


class PermutationChecks(unittest.TestCase):
    def fake_data(self):
        rows=[]
        for pid,n in [('A',20),('B',16)]:
            for i in range(n):
                rows.append(dict(participant_id=pid,timestamp=pd.Timestamp('2026-01-01')+pd.Timedelta(minutes=15*i+(30 if i>=8 else 0))))
        frame=pd.DataFrame(rows)
        assignments=pd.DataFrame({'participant_id':['A','B'],'validation_fold':[1,1]})
        return SimpleNamespace(dataset='shanghai',frame=frame,assignments=assignments)

    def test_sample_balanced_deterministic_and_gap_safe(self):
        data=self.fake_data()
        selection,coverage,design=sample_development(data)
        again,_,_=sample_development(data)
        pd.testing.assert_frame_equal(selection,again)
        self.assertEqual(selection.groupby('participant_id').size().tolist(),[16,16])
        self.assertTrue(selection.development_row_index.is_unique)
        for _,block in selection.groupby(['participant_id','selected_block_id']):
            times=pd.to_datetime(block.timestamp)
            self.assertTrue(times.diff().dropna().eq(pd.Timedelta(minutes=15)).all())

    def test_block_mapping_never_crosses_participants_or_reverses_block_order(self):
        selection,_,_=sample_development(self.fake_data())
        mapping=row_mapping(selection,'within_participant_block',42)
        np.testing.assert_array_equal(selection.participant_id,selection.participant_id.iloc[mapping])
        np.testing.assert_array_equal(selection.within_block_position,selection.within_block_position.iloc[mapping])
        np.testing.assert_array_equal(np.sort(mapping),np.arange(len(selection)))

    def test_group_uses_shared_mapping_and_keeps_source_unchanged(self):
        X=pd.DataFrame({'a':[1.,2.,3.,4.],'b':[2.,4.,6.,8.],'category':['A','B','C','D'],'unselected':[5,6,7,8]})
        before=X.copy(deep=True)
        mapping=np.array([3,1,0,2])
        result=permuted_inputs(X,['a','b','category'],mapping)
        np.testing.assert_array_equal(result.b,2*result.a)
        self.assertEqual(result.category.tolist(),X.category.iloc[mapping].tolist())
        self.assertEqual(result.unselected.tolist(),X.unselected.tolist())
        pd.testing.assert_frame_equal(X,before)

    def test_positive_degradation_and_negative_results_are_retained(self):
        X=pd.DataFrame({'signal':[-2.,-1.,0.,1.,2.]}); y=X.signal.to_numpy()
        mapping=np.arange(4,-1,-1)
        good=importance_score(SignalPredictor(),X,y,['signal'],mapping,0.)
        bad=importance_score(SignalPredictor(-1),X,y,['signal'],mapping,2.4)
        self.assertAlmostEqual(good['importance'],2.4)
        self.assertAlmostEqual(bad['importance'],-2.4)

    def test_participant_constant_feature_not_identifiable_within_participant(self):
        selection,_,_=sample_development(self.fake_data())
        X=pd.DataFrame({'signal':selection.participant_id.map({'A':1.,'B':2.})})
        y=X.signal.to_numpy()
        within=row_mapping(selection,'within_participant_block',42)
        global_rows=row_mapping(selection,'global_row',42)
        self.assertEqual(importance_score(SignalPredictor(),X,y,['signal'],within,0.)['importance'],0.)
        self.assertGreater(importance_score(SignalPredictor(),X,y,['signal'],global_rows,0.)['importance'],0.)


if __name__=='__main__':
    unittest.main()
