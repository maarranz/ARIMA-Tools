"""Observed-data identification: gaps must not erase intervention information."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import arima_tools as at
from arima_tools._regressors import identification


class MissingIdentificationTests(unittest.TestCase):
    def example(self):
        n, event = 48, 20
        index = pd.period_range('2000Q1', periods=n, freq='Q')
        innovations = np.random.default_rng(2026).normal(scale=.3, size=n)
        innovations[[0, event, event+1]] = 0
        X = at.outlier_regressors(index, positions=event, types='LS')
        y = pd.Series(np.cumsum(innovations) + 3*X.iloc[:,0], index=index)
        return y, X

    def test_audit_missing_event_remains_identified(self):
        y, X = self.example()
        complete = ARIMA(y, order=(0,1,0), trend='n', exog=X)
        self.assertIs(identification(complete)[0], True)
        y.iloc[20] = np.nan
        model = ARIMA(y, order=(0,1,0), trend='n', exog=X)
        fitted = model.fit(cov_type='oim')
        self.assertTrue(fitted.mle_retvals['converged'])
        self.assertAlmostEqual(fitted.params[X.columns[0]], 3, places=4)
        # Independent observed transition and likelihood curvature, not merely
        # the optimizer flag, establish that the shift carries information.
        self.assertAlmostEqual(y.iloc[21]-y.iloc[19], 3)
        ll = [model.loglike([beta,.09]) for beta in (2.,3.,4.)]
        self.assertAlmostEqual(ll[1]-ll[0], 1/(4*.09), places=7)
        self.assertAlmostEqual(ll[1]-ll[2], 1/(4*.09), places=7)
        self.assertIs(identification(model)[0], True)

    def test_missing_before_event_after_and_multiple_gap(self):
        for missing in ([19], [20], [21], [19,20,21]):
            with self.subTest(missing=missing):
                y, X = self.example(); y.iloc[missing] = np.nan
                original = y.copy(); original_X = X.copy()
                model = ARIMA(y, order=(0,1,0), trend='n', exog=X)
                self.assertIs(identification(model)[0], True)
                before = np.flatnonzero(y.iloc[:20].notna())[-1]
                after = 20 + np.flatnonzero(y.iloc[20:].notna())[0]
                self.assertEqual(X.iloc[after,0]-X.iloc[before,0], 1)
                # Independent random-walk covariance on the actual observed grid.
                times = np.flatnonzero(y.notna())[1:]
                G = (times[:,None] >= np.arange(1,48)[None,:]).astype(float)
                V = .09 * G @ G.T
                x = X.iloc[times,0].to_numpy()
                information = x @ np.linalg.solve(V,x)
                expected = 1 / (.09 * (after-before))
                self.assertAlmostEqual(information, expected, places=7)
                pd.testing.assert_series_equal(y,original)
                pd.testing.assert_frame_equal(X,original_X)

    def test_stationary_gaps_use_observed_regression_rank(self):
        y, X = self.example(); y.iloc[20] = np.nan
        self.assertIs(identification(ARIMA(y,order=(0,0,0),trend='c',exog=X))[0], True)
        constant = at.outlier_regressors(y,positions=0,types='LS')
        self.assertIs(identification(ARIMA(y,order=(0,0,0),trend='n',exog=constant))[0], True)

    def test_true_zero_and_duplicate_columns_despite_gaps(self):
        y, X = self.example(); y.iloc[20] = np.nan
        AO = at.outlier_regressors(y,positions=20,types='AO')
        model = ARIMA(y,order=(0,1,0),trend='n',exog=AO)
        self.assertIs(identification(model)[0], False)
        # AO is present only at an unobserved date: observed likelihood is flat.
        self.assertAlmostEqual(model.loglike([1.,.09]), model.loglike([9.,.09]), places=10)
        duplicate = pd.DataFrame({'a':X.iloc[:,0], 'b':X.iloc[:,0]})
        self.assertIs(identification(ARIMA(y,order=(0,1,0),trend='n',exog=duplicate))[0], False)

    def test_exact_diffuse_integration_confounding_is_unidentified(self):
        y, _ = self.example(); y.iloc[20] = np.nan
        for d in (1,2):
            # Constants and affine paths lie in the I(1)/I(2) integration nullspace.
            X = pd.DataFrame({'level':np.ones(48)},index=y.index)
            if d == 2: X['slope'] = np.arange(48)
            model = ARIMA(y,order=(0,d,0),trend='n',exog=X)
            model.use_exact_diffuse=True; model.initialize_default(); model.loglikelihood_burn=0
            state,status = identification(model)
            self.assertIs(state,False)
            self.assertIn('diffuse integration directions',status)

    def test_finite_approximate_diffuse_confounding_is_inconclusive(self):
        y, _ = self.example(); y.iloc[20] = np.nan
        X = at.outlier_regressors(y,positions=0,types='LS')
        model = ARIMA(y,order=(0,1,0),trend='n',exog=X)
        state,status = identification(model)
        self.assertIsNone(state)
        self.assertIn('Identification not established',status)
        table = at.order_search(y,p_max=0,q_max=0,d_min=1,d_max=1,trend='n',exog=X)
        self.assertIsNone(table.Identified.iloc[0])
        self.assertIn('Identification not established',table['Identification Status'].iloc[0])
        self.assertTrue(any('Identification not established' in w for w in table.Warnings.iloc[0]))
        with self.assertRaises(ValueError): at.select_model(table)
        # Contributions remain computable conditional on supplied coefficients,
        # accompanied by an explicit identification warning.
        fitted = model.filter([3.,.09])
        with self.assertWarnsRegex(UserWarning,'Identification not established'):
            effects = at.intervention_effects(fitted,X)
        np.testing.assert_array_equal(effects.Combined,3.)
        comparison=at.compare_models(fitted,fitted,names=['one','two'])
        self.assertTrue(comparison.Identified.isna().all())

    def test_fit_failure_preserves_separate_identification_result(self):
        y, X = self.example(); y.iloc[20] = np.nan
        first = at.outlier_regressors(y, positions=0, types='LS')
        for exog, expected in ((X, True), (first, None)):
            with self.subTest(expected=expected):
                with patch.object(ARIMA, 'fit', side_effect=RuntimeError('fit failed')):
                    table = at.order_search(y, p_max=0, q_max=0, d_min=1,
                                            d_max=1, trend='n', exog=exog)
                actual = table.Identified.iloc[0]
                if expected is None:
                    self.assertIsNone(actual)
                    self.assertIn('Identification not established', table['Identification Status'].iloc[0])
                    self.assertTrue(table.Warnings.iloc[0])
                else:
                    self.assertTrue(actual)
                    self.assertIn('observed-data', table['Identification Status'].iloc[0])
                self.assertIn('fit failed', table.Status.iloc[0])
                self.assertFalse(table.Converged.iloc[0])

    def test_unsupported_initialization_is_inconclusive_not_unidentified(self):
        y, X = self.example(); y.iloc[20] = np.nan
        model=ARIMA(y,order=(0,1,0),trend='n',exog=X)
        model.initialize_known(np.zeros(model.k_states),np.eye(model.k_states))
        self.assertIsNone(identification(model)[0])
        custom=ARIMA(y,order=(0,1,0),trend='n',exog=X)
        custom.loglikelihood_burn=5
        self.assertIsNone(identification(custom)[0])

    def test_i2_identification_and_independent_observed_covariance(self):
        n,event=48,20
        index=pd.period_range('2000Q1',periods=n,freq='Q')
        e=np.random.default_rng(203).normal(scale=.3,size=n)
        e[[0,1,event-1,event,event+1,event+2]]=0
        X=at.outlier_regressors(index,positions=event,types='LS')
        y=pd.Series(np.cumsum(np.cumsum(e))+3*X.iloc[:,0],index=index)
        y.iloc[event]=np.nan
        # This contrast annihilates an affine path on the original grid.
        self.assertAlmostEqual(y.iloc[event+1]-3*y.iloc[event-1]+2*y.iloc[event-2],3)
        model=ARIMA(y,order=(0,2,0),trend='n',exog=X)
        self.assertIs(identification(model)[0],True)
        times=np.flatnonzero(y.notna())[2:]
        innovations=np.arange(2,n)
        G=np.maximum(times[:,None]-innovations[None,:]+1,0).astype(float)
        V=.09*G@G.T
        x=X.iloc[times,0].to_numpy()
        information=x@np.linalg.solve(V,x)
        beta=x@np.linalg.solve(V,y.iloc[times].to_numpy())/information
        self.assertGreater(information,0)
        self.assertAlmostEqual(beta,3,places=6)
        ll=[model.loglike([b,.09]) for b in (2.,3.,4.)]
        self.assertAlmostEqual(ll[1]-ll[0],information/2,places=5)
        self.assertAlmostEqual(ll[1]-ll[2],information/2,places=5)
        # A quadratic deterministic term survives second differencing.
        quadratic=ARIMA(y,order=(0,2,0),trend=[0,0,1],exog=X)
        self.assertIs(identification(quadratic)[0],True)
        model.use_exact_diffuse=True;model.initialize_default();model.loglikelihood_burn=0
        self.assertIs(identification(model)[0],True)

    def test_complete_data_decisions_remain_unchanged(self):
        y,X=self.example()
        self.assertIs(identification(ARIMA(y,order=(0,1,0),trend='n',exog=X))[0],True)
        first=at.outlier_regressors(y,positions=0,types='LS')
        self.assertIs(identification(ARIMA(y,order=(0,1,0),trend='n',exog=first))[0],False)
        duplicate=pd.DataFrame({'a':X.iloc[:,0],'b':X.iloc[:,0]})
        self.assertIs(identification(ARIMA(y,order=(0,0,0),trend='n',exog=duplicate))[0],False)

    def test_search_effects_and_selection_integrate_gap_correction(self):
        y,X=self.example();y.iloc[20]=np.nan
        table=at.order_search(y,p_max=0,q_max=0,d_min=1,d_max=1,trend='n',exog=X)
        self.assertIs(table.Identified.iloc[0].item(),True)
        self.assertIn('observed-data',table['Identification Status'].iloc[0])
        at.select_model(table)
        fitted=ARIMA(y,order=(0,1,0),trend='n',exog=X).fit()
        effects=at.intervention_effects(fitted,X)
        np.testing.assert_allclose(effects.Combined,fitted.params[X.columns[0]]*X.iloc[:,0])
        self.assertTrue(np.isnan(effects.Adjusted.iloc[20]))
        self.assertEqual(len(effects),48)
        comparison=at.compare_models(fitted,fitted,names=['first','second'])
        self.assertTrue(comparison.Identified.all())
        at.select_model(comparison)
        # Stage 2 rejects altered response values or missing positions.
        changed=y.copy();changed.iloc[10]=np.nan
        other=ARIMA(changed,order=(0,1,0),trend='n',exog=X).fit()
        with self.assertRaisesRegex(ValueError,'same sample'):
            at.select_model(at.compare_models(fitted,other,names=['a','b']))

    def test_recursive_ls_control_rank_transition(self):
        n,event=48,30
        index=pd.period_range('2000Q1',periods=n,freq='Q')
        X=at.outlier_regressors(index,positions=event,types='LS')
        control=X.iloc[:,0].copy().rename('control')
        control.iloc[event+3:]+=.2*np.arange(1,n-event-2)
        Z=pd.concat([control,X],axis=1)
        y=pd.Series(5+2*control+4*X.iloc[:,0]+np.random.default_rng(44).normal(scale=.2,size=n),index=index)
        table=at.recursive_estimation(y,(0,0,0),initial=28,trend='c',exog=Z)
        self.assertTrue(table.loc[table.N<=33,'Estimate'].isna().all())
        self.assertTrue(table.loc[table.N.between(31,33),'Status'].str.contains('dependent').all())
        self.assertEqual(np.linalg.matrix_rank(np.column_stack([np.ones(33),Z.iloc[:33]])),2)
        self.assertEqual(np.linalg.matrix_rank(np.column_stack([np.ones(34),Z.iloc[:34]])),3)
        self.assertEqual(table.loc[table.N==34,'Status'].unique().tolist(),['OK'])
        for _,window in table[table.N>=34].groupby('N'):
            self.assertEqual(set(window.Parameter),{'const','control',X.columns[0],'sigma2'})
        self.assertEqual(table.Endpoint.iloc[-1],index[-1])
        # Preserve the established no-gap policy for recursive estimation.
        y.iloc[20]=np.nan
        with self.assertRaisesRegex(ValueError,'finite'):
            at.recursive_estimation(y,(0,0,0),initial=28,trend='c',exog=Z)


if __name__ == '__main__':
    unittest.main()
