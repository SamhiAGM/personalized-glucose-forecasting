"""Synthetic tests: output mapping, grouped attribution algebra and additivity."""
import unittest
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import Ridge
from tree_shap_analysis import feature_mapping,collapse,linear_explanation,select_rows,summaries


class TreeShapChecks(unittest.TestCase):
    def setUp(self):
        X=pd.DataFrame({'numeric_input':np.arange(20.),'sex_code_name':['category_A','category_B']*10})
        pre=ColumnTransformer([('categorical',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),
            ('encoder',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),['sex_code_name']),
            ('numeric',Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())]),['numeric_input'])])
        self.model=Pipeline([('preprocessor',pre),('model',Ridge())]).fit(X,2*X.numeric_input+X.sex_code_name.eq('category_B'))
        self.X=X; self.meta=pd.DataFrame({'feature':X.columns,'feature_group':['glucose','clinical']})

    def test_reordered_branches_and_underscored_categories(self):
        mapping=feature_mapping(self.model,self.meta)
        self.assertEqual(mapping.original_feature.tolist(),['sex_code_name','sex_code_name','numeric_input'])
        self.assertEqual(mapping.category.tolist(),['category_A','category_B',''])
        mdi=np.array([.1,.2,.7]); original=collapse(mdi[None,:],mapping,'original_feature',self.X.columns)
        np.testing.assert_allclose(original,[[.7,.3]])

    def test_shap_additivity_and_exact_linear_formula(self):
        z=self.model.named_steps['preprocessor'].transform(self.X)
        explanation,values,bases=linear_explanation(self.model.named_steps['model'],z[:10],z[10:])
        np.testing.assert_allclose(values.sum(axis=1)+bases,self.model.predict(self.X.iloc[10:]),atol=1e-7)
        mapping=feature_mapping(self.model,self.meta)
        original=collapse(values,mapping,'original_feature',self.X.columns)
        np.testing.assert_allclose(original.sum(axis=1),values.sum(axis=1))

    def test_signed_collapse_before_absolute(self):
        mapping=feature_mapping(self.model,self.meta)
        values=np.array([[3.,-3.,1.]])
        grouped=collapse(values,mapping,'original_feature',self.X.columns)
        np.testing.assert_allclose(grouped,[[1.,0.]])

    def test_balanced_seeded_sampling_and_participant_weighting(self):
        frame=pd.DataFrame({'participant_id':['A']*10+['B']*20,'timestamp':pd.date_range('2026-01-01',periods=30,freq='min')})
        a=select_rows(frame,3,'synthetic',1,'background'); b=select_rows(frame,3,'synthetic',1,'background')
        pd.testing.assert_frame_equal(a,b); self.assertEqual(a.groupby('participant_id').size().tolist(),[3,3])
        result=summaries(np.array([[1.],[3.],[10.]]),['x'],['A','A','B'])
        self.assertEqual(result.mean_abs_shap.iloc[0],6.)


if __name__=='__main__': unittest.main()
