"""Algebra and encoding tests use synthetic participants only."""
import unittest
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ridge_coefficients import extract_coefficients, verify_reconstruction, categorical_contrasts


class CoefficientMappingChecks(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(42)
        self.X=pd.DataFrame({'large_scale':rng.normal(1000,200,80),
            'sex_with_underscore':['category_A','category_B']*40,
            'small_scale':rng.normal(0,.02,80),'constant_value':np.ones(80)})
        y=.2*self.X.large_scale-30*self.X.small_scale+7*self.X.sex_with_underscore.eq('category_B')
        self.X.loc[0,'small_scale']=np.nan
        numeric=Pipeline([('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),('scaler',StandardScaler())])
        category=Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('encoder',OneHotEncoder(handle_unknown='ignore',sparse_output=False))])
        # Categorical first deliberately challenges output-position mapping.
        pre=ColumnTransformer([('categorical',category,['sex_with_underscore']),
            ('numeric',numeric,['small_scale','constant_value','large_scale'])])
        self.model=Pipeline([('preprocessor',pre),('model',Ridge(alpha=.1))]).fit(self.X,y)
        self.meta=pd.DataFrame({'feature':self.X.columns,'feature_group':['synthetic']*4})

    def test_reconstruct_with_reordered_branches_missing_and_constant(self):
        table,intercept=extract_coefficients(self.model,self.meta)
        result=verify_reconstruction(self.model,table,intercept,self.X)
        self.assertLess(result['original_scale_max_absolute_error'],1e-7)
        const=table.loc[table.original_feature.eq('constant_value')].iloc[0]
        self.assertFalse(const.rank_eligible)
        self.assertEqual(const.reference_sd,0)
        self.assertEqual(table.transformed_feature.tolist(),list(self.model.named_steps['preprocessor'].get_feature_names_out()))

    def test_category_contrast_matches_prediction_difference(self):
        table,_=extract_coefficients(self.model,self.meta)
        contrast=categorical_contrasts(table)[0]
        x=self.X.iloc[[1,1]].copy()
        x['sex_with_underscore']=['category_A','category_B']
        delta=float(np.diff(self.model.predict(x))[0])
        self.assertAlmostEqual(contrast['coefficient_contrast'],delta,places=10)
        self.assertEqual(contrast['contrast'],'category_B minus category_A')
        self.assertTrue(table.loc[table.kind.eq('categorical_indicator'),'drop_reference'].eq('none_full_dummy_coding').all())

    def test_common_reference_changes_reporting_only(self):
        table,_=extract_coefficients(self.model,self.meta)
        ref=table.loc[table.kind.eq('numeric_standardized')].set_index('original_feature').copy()
        ref['reference_sd']*=2
        changed,_=extract_coefficients(self.model,self.meta,reference=ref)
        numeric=changed.kind.eq('numeric_standardized')
        np.testing.assert_allclose(changed.loc[numeric,'coefficient_common_reference_sd'],
                                   table.loc[numeric,'coefficient_common_reference_sd']*2)
        np.testing.assert_allclose(changed.coefficient_transformed,table.coefficient_transformed)


if __name__=='__main__':
    unittest.main()
