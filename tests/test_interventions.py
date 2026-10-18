"""Analytical intervention checks and integration with the existing ARIMA API."""
from copy import deepcopy
import unittest
import warnings
from unittest.mock import patch
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import arima_tools as at
from arima_tools.interventions import _validated

KEY = 'arima_tools_interventions'


class InterventionTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.period_range('2000Q1', periods=100, freq='Q')
        self.y = pd.Series(np.random.default_rng(91).normal(size=100), index=self.index, name='Output')

    def frame(self, y=None):
        return at.outlier_regressors(self.y if y is None else y,
                                     positions=[20, 40, 60], types=['AO', 'LS', 'TC'], delta=.5)

    def test_analytical_definitions(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            X = at.outlier_regressors(np.zeros(5), positions=[2, 2, 2], types=['AO', 'LS', 'TC'], delta=.5)
        np.testing.assert_array_equal(X.to_numpy().T, [[0,0,1,0,0], [0,0,1,1,1], [0,0,1,.5,.25]])
        self.assertEqual(X.attrs[KEY]['version'], 1)
        self.assertEqual(X.attrs[KEY]['events'][0]['position'], 2)
        self.assertEqual(at.outlier_regressors(np.zeros(5), positions=2, types='TC').iloc[3,0], .7)

    def test_multiple_events_names_and_same_date(self):
        X = at.outlier_regressors(self.y, positions=[2, 3, 3], types=['AO', 'AO', 'LS'])
        self.assertTrue(X.columns.is_unique)
        pd.testing.assert_index_equal(X.index, self.index)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            at.outlier_regressors(self.y, positions=[2,2], types='AO')
        with self.assertRaises(ValueError):
            at.outlier_regressors(self.y, positions=[2,3,4], types=['AO','LS'])

    def test_period_and_datetime_matching(self):
        X = at.outlier_regressors(self.y, dates=['2000Q2', '2001-08-15'], types=['AO','LS'])
        self.assertEqual([e['position'] for e in X.attrs[KEY]['events']], [1,6])
        index = pd.date_range('2020-01-01', periods=10, freq='D', tz='Europe/Madrid')
        X = at.outlier_regressors(index, dates=[index[2]], types='AO')
        pd.testing.assert_index_equal(X.index, index)
        with self.assertRaises(ValueError): at.outlier_regressors(index, dates=['2020-01-03'])
        with self.assertRaises(ValueError): at.outlier_regressors(index, dates=['2020'])
        with self.assertRaises(ValueError): at.outlier_regressors(self.y, dates=['2100Q1'])
        with self.assertRaises(ValueError): at.outlier_regressors(self.y, dates=['2000Q5'])
        with self.assertRaises(ValueError): at.outlier_regressors(self.y, dates=['2000'])
        monthly = pd.period_range('2000-01', periods=12, freq='M')
        with self.assertRaises(ValueError): at.outlier_regressors(monthly, dates=['2000Q1'])

    def test_invalid_positions_types_and_delta(self):
        for position in (-1, 100, 1.5, True, None, np.nan):
            with self.subTest(position=position), self.assertRaises((ValueError, TypeError)):
                at.outlier_regressors(self.y, positions=position)
        for decay in (0, 1, -.1, np.nan, np.inf, True, '.7', None):
            with self.subTest(delta=decay), self.assertRaises(ValueError):
                at.outlier_regressors(self.y, positions=2, types='TC', delta=decay)
        for kind in ('XX', [], ['AO','LS'], True, None):
            with self.assertRaises(ValueError): at.outlier_regressors(self.y, positions=2, types=kind)
        with self.assertRaises(ValueError): at.outlier_regressors(self.y)
        with self.assertRaises(ValueError): at.outlier_regressors(self.y, dates=['2000Q1'], positions=[0])

    def test_indexes_missing_values_and_validation(self):
        missing = self.y.copy(); missing.iloc[3] = np.nan
        pd.testing.assert_frame_equal(self.frame(missing), self.frame())
        for index in (pd.Index([1,1,2]), pd.Index([2,1,3]), pd.Index([1,np.nan,3]),
                      self.index.delete(4), pd.to_datetime(['2020-01-01','2020-01-02','2020-01-04'])):
            with self.assertRaises(ValueError): at.outlier_regressors(index, positions=0)
        for index in (pd.RangeIndex(10,210,2), pd.Index(np.arange(100.)*1.5)):
            pd.testing.assert_index_equal(at.outlier_regressors(index, positions=2).index,index)
        with self.assertRaises(ValueError): at.outlier_regressors(np.zeros((4,2)), positions=0)

    def test_metadata_detects_numeric_column_and_event_changes(self):
        for modification in ('values','columns','position','label','index','delta','missing'):
            X = self.frame()
            if modification == 'values': X.iloc[0,0] = 1
            elif modification == 'columns': X.columns = ['other',*X.columns[1:]]
            elif modification == 'position': X.attrs[KEY]['events'][0]['position'] = 21
            elif modification == 'label': X.attrs[KEY]['events'][0]['label'] = 'wrong'
            elif modification == 'index': X.index = X.index+1
            elif modification == 'delta': X.attrs[KEY]['events'][2]['delta'] = .9
            else: X.attrs.clear()
            with self.subTest(change=modification), self.assertRaises(ValueError): _validated(X)

    def analytical_model(self):
        X = self.frame()
        control = pd.Series(np.linspace(0, 1, len(self.y)), index=self.index, name='control')
        Z = pd.concat([control, X], axis=1)
        values = dict(const=10., control=2., sigma2=1.)
        values.update(dict(zip(X.columns,[3., -4., 5.])))
        y = 10 + 2*control + X.to_numpy() @ np.array([3.,-4.,5.])
        model = ARIMA(y, order=(0,0,0), trend='c', exog=Z)
        fitted = model.filter([values[name] for name in model.param_names])
        return y, X, Z, fitted

    def test_effects_match_named_coefficients_and_exclude_controls(self):
        y, X, Z, result = self.analytical_model()
        table = at.intervention_effects(result, X)
        pd.testing.assert_index_equal(table.index, y.index)
        expected = X.to_numpy() * np.array([3.,-4.,5.])
        np.testing.assert_array_equal(table[list(X)].to_numpy(), expected)
        np.testing.assert_allclose(table.Combined, expected.sum(axis=1))
        np.testing.assert_allclose(table.Adjusted, 10+2*Z.control)
        self.assertFalse(np.allclose(table.Adjusted,result.resid))
        # Reorder the fitted regressors: contributions must still match by name.
        order = list(reversed(Z.columns))
        model = ARIMA(y, order=(0,0,0), trend='c', exog=Z[order])
        params = dict(zip(result.model.param_names,result.params))
        reordered = model.filter([params[name] for name in model.param_names])
        pd.testing.assert_frame_equal(table, at.intervention_effects(reordered, X))

    def test_effects_reject_wrong_sample_values_and_missing_metadata(self):
        _, X, _, result = self.analytical_model()
        wrong = at.outlier_regressors(self.y, positions=[21,40,60], types=['AO','LS','TC'],delta=.5)
        with self.assertRaises(ValueError): at.intervention_effects(result, wrong)
        with self.assertRaises(ValueError): at.intervention_effects(result, X.set_axis(self.index+1))
        with self.assertRaises(ValueError): at.intervention_effects(result, X.rename(columns={X.columns[0]:'control'}))

    def test_future_values_and_chained_extension(self):
        X = self.frame()
        future = at.extend_interventions(X, steps=3)
        self.assertEqual(list(future), list(X))
        np.testing.assert_array_equal(future.iloc[:,0],0)
        np.testing.assert_array_equal(future.iloc[:,1],1)
        np.testing.assert_array_equal(future.iloc[:,2], .5**np.arange(40,43))
        again = at.extend_interventions(future,steps=2)
        np.testing.assert_array_equal(again.iloc[:,2], .5**np.arange(43,45))
        self.assertEqual(again.index[0],self.index[-1]+4)
        _validated(future); _validated(again)

    def test_future_indexes_and_invalid_steps(self):
        for index in (pd.date_range('2020',periods=10,freq='D',tz='Europe/Madrid'),
                      pd.RangeIndex(10,30,2), pd.Index([1.,2.5,4.,5.5])):
            X = at.outlier_regressors(index, positions=2, types='LS')
            future = at.extend_interventions(X,steps=2)
            self.assertGreater(future.index[0],index[-1]); _validated(future)
        X = at.outlier_regressors(pd.Index([1,2,4]), positions=1)
        with self.assertRaises(ValueError): at.extend_interventions(X,steps=2)
        explicit = pd.Index([5,7])
        pd.testing.assert_index_equal(at.extend_interventions(X,steps=2,future_index=explicit).index, explicit)
        for steps in (0,-1,True,2.5):
            with self.assertRaises(ValueError): at.extend_interventions(self.frame(),steps)
        with self.assertRaises(ValueError): at.extend_interventions(self.frame(),2,future_index=self.index[:2])

    def test_plot_values_shared_markers_and_numerical_delegation(self):
        y, X, _, result = self.analytical_model()
        table = at.intervention_effects(result,X)
        fig = at.plot_interventions(y,X,results=result,individual=True)
        np.testing.assert_array_equal(next(t.y for t in fig.data if t.name=='Adjusted'),table.Adjusted)
        np.testing.assert_array_equal(next(t.y for t in fig.data if t.name=='Combined intervention effect'),table.Combined)
        for column in X: np.testing.assert_array_equal(next(t.y for t in fig.data if t.name==column),table[column])
        import arima_tools.interventions as module
        with patch.object(module,'intervention_effects',return_value=table) as numerical:
            at.plot_interventions(y,X,results=result)
            numerical.assert_called_once_with(result,X)
        shared = at.outlier_regressors(y,positions=[20,20],types=['AO','LS'])
        fig = at.plot_interventions(y,shared)
        markers = next(t for t in fig.data if t.name=='Intervention dates')
        self.assertEqual(len(markers.x),1)
        self.assertIn('AO_',markers.text[0]); self.assertIn('LS_',markers.text[0])
        self.assertEqual(markers.marker.symbol[0],'diamond')
        fig.to_json()
        with self.assertRaises(ValueError): at.plot_interventions(y+1,X,results=result)

    def test_order_search_exog_alignment_and_identification(self):
        X = self.frame()
        table = at.order_search(self.y,p_max=1,q_max=0,trend='n',exog=X)
        self.assertTrue(table.Identified.all())
        at.select_model(table)
        with self.assertRaises(ValueError): at.order_search(self.y,exog=X.set_axis(self.index+1))
        invalid = X.copy(); invalid.iloc[0,0] = np.nan
        with self.assertRaises(ValueError): at.order_search(self.y,exog=invalid)
        duplicated = pd.DataFrame({'a':X.iloc[:,1], 'b':X.iloc[:,1]})
        failed = at.order_search(self.y,p_max=0,q_max=0,trend='n',exog=duplicated)
        self.assertFalse(failed.Identified.any())
        self.assertIn('Unidentified',failed.Status.iloc[0])
        with self.assertRaises(ValueError): at.select_model(failed)

    def test_first_observation_ls_and_differencing(self):
        X = at.outlier_regressors(self.y,positions=0,types='LS')
        raw = at.order_search(self.y,p_max=0,q_max=0,exog=X)
        self.assertFalse(raw.Identified.any())
        differenced = at.order_search(self.y,p_max=0,q_max=0,d_min=1,d_max=1,trend='n',exog=X)
        self.assertIn('zero columns after differencing',differenced.Status.iloc[0])

    def test_recursive_unidentified_windows_and_parameter_count(self):
        X = at.outlier_regressors(self.y,positions=70)
        table = at.recursive_estimation(self.y,(0,0,0),initial=68,trend='n',exog=X)
        early = table[table.N <= 70]
        self.assertTrue(early.Estimate.isna().all())
        self.assertTrue(early.Status.str.contains('Unidentified').all())
        self.assertIn(X.columns[0],set(table.loc[table.N>70,'Parameter']))
        with self.assertRaises(ValueError): at.recursive_estimation(self.y,(0,0,0),initial=2,trend='n',exog=X)
        with self.assertRaises(ValueError): at.recursive_estimation(self.y,(0,0,0),exog=X.set_axis(self.index+1))

    def test_comparisons_allow_different_exog_reject_response_changes(self):
        X = at.outlier_regressors(self.y,positions=20)
        Z = at.outlier_regressors(self.y,positions=40,types='LS')
        a = ARIMA(self.y,order=(0,0,0),trend='n',exog=X).fit()
        b = ARIMA(self.y,order=(0,0,0),trend='n',exog=Z).fit()
        table = at.compare_models(a,b,names=['AO','LS'])
        self.assertEqual(table['Sample ID'].nunique(),1)
        at.select_model(table)
        changed = ARIMA(self.y+1,order=(0,0,0),trend='n',exog=Z).fit()
        with self.assertRaises(ValueError): at.select_model(at.compare_models(a,changed,names=['a','changed']))
        duplicate = pd.DataFrame({'a':Z.iloc[:,0], 'b':Z.iloc[:,0]})
        bad = ARIMA(self.y,order=(0,0,0),trend='n',exog=duplicate).fit()
        comparison = at.compare_models(a,bad,names=['good','bad'])
        comparison.loc['bad','BIC']=-1e10
        self.assertFalse(comparison.loc['bad','Identified'])
        self.assertEqual(at.select_model(comparison).name,'good')

    def test_dynamic_forecast_requires_all_future_regressors(self):
        X = self.frame()
        control = pd.Series(np.linspace(0,1,100),index=self.index,name='control')
        Z = pd.concat([control,X],axis=1)
        fitted = ARIMA(self.y,order=(1,0,0),trend='c',exog=Z).fit()
        future = at.extend_interventions(X,steps=4)
        with self.assertRaises(ValueError): at.dynamic_forecast(fitted,steps=4)
        with self.assertRaises(ValueError): at.dynamic_forecast(fitted,steps=4,exog=future)
        full = pd.concat([pd.Series(np.ones(4),index=future.index,name='control'),future],axis=1)
        table = at.dynamic_forecast(fitted,steps=4,exog=full)
        np.testing.assert_allclose(table.Forecast,fitted.forecast(4,exog=full))
        np.testing.assert_array_equal(table.FEV,at.forecast_error_variance(fitted,4))
        for function in (at.plot_forecast,at.plot_fan_chart):
            figure = function(fitted,steps=4,exog=full)
            np.testing.assert_array_equal(figure.data[-1].y[1:],table.Forecast)
        for invalid in (full.iloc[::-1],full[full.columns[::-1]],full.iloc[:2],full*float('nan')):
            with self.assertRaises(ValueError): at.dynamic_forecast(fitted,steps=4,exog=invalid)

    def test_integrated_effects_forecasts_and_no_pre_differencing(self):
        X = self.frame()
        y = self.y.cumsum() + X.to_numpy() @ np.array([2.,-3.,4.])
        result = ARIMA(y,order=(0,1,0),trend='n',exog=X).fit()
        expected = X.mul(pd.Series(dict(zip(result.model.param_names,result.params)))[list(X)],axis=1)
        table = at.intervention_effects(result,X)
        np.testing.assert_array_equal(table[list(X)].to_numpy(),expected.to_numpy())
        np.testing.assert_allclose(table.Adjusted,y-expected.sum(axis=1))
        future = at.extend_interventions(X,steps=3)
        np.testing.assert_allclose(at.dynamic_forecast(result,steps=3,exog=future).Forecast,
                                   result.forecast(3,exog=future))

    def test_missing_response_effects_preserve_gaps(self):
        y, X, Z, result = self.analytical_model()
        y = y.copy(); y.iloc[10] = np.nan
        model = ARIMA(y,order=(0,0,0),trend='c',exog=Z)
        fitted = model.filter(result.params)
        table = at.intervention_effects(fitted,X)
        self.assertTrue(np.isnan(table.Observed.iloc[10]))
        self.assertTrue(np.isnan(table.Adjusted.iloc[10]))
        self.assertTrue(np.isfinite(table.Combined.iloc[10]))
        at.plot_interventions(y,X,results=fitted).to_json()

    def test_last_observation_collinearity_and_nearby_valid_tc(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            X = at.outlier_regressors(self.y,positions=[99,99],types=['AO','TC'])
        self.assertTrue(any('dependent' in str(w.message) for w in caught))
        failed = at.order_search(self.y,p_max=0,q_max=0,trend='n',exog=X)
        self.assertFalse(failed.Identified.any())
        near = at.outlier_regressors(self.y,positions=[20,20],types=['LS','TC'],delta=.99999)
        table = at.order_search(self.y,p_max=0,q_max=0,trend='n',exog=near)
        self.assertTrue(table.Identified.all())

    def test_future_short_explicit_dates_and_array_exog_labels(self):
        index = pd.date_range('2020',periods=80,freq='MS')
        y = pd.Series(self.y.to_numpy()[:80],index=index)
        X = at.outlier_regressors(y,positions=20,types='LS')
        future = at.extend_interventions(X,steps=1,future_index=[index[-1]+pd.offsets.MonthBegin()])
        _validated(future)
        at.extend_interventions(future,steps=1)
        recursive = at.recursive_estimation(y.to_numpy(),(0,0,0),initial=78,trend='n',exog=X)
        self.assertEqual(recursive.Endpoint.iloc[-1],index[-1])
        pd.testing.assert_index_equal(X.index,index)

    def test_exog_numeric_validation_and_forecast_without_regressors(self):
        for exog in (np.ones((3,2)), np.full(100,np.inf),
                     pd.DataFrame(np.ones((100,2)),index=self.index,columns=['a','a'])):
            with self.assertRaises(ValueError): at.order_search(self.y,exog=exog)
        fitted = ARIMA(self.y,order=(0,0,0)).fit()
        with self.assertRaises(ValueError): at.dynamic_forecast(fitted,steps=2,exog=np.ones((2,1)))

    def test_effects_warn_about_actual_nonconvergence(self):
        X = self.frame()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fitted = ARIMA(self.y,order=(1,0,0),trend='n',exog=X).fit(method_kwargs={'maxiter':0})
        self.assertFalse(fitted.mle_retvals['converged'])
        with self.assertWarnsRegex(UserWarning,'did not converge'):
            at.intervention_effects(fitted,X)
        with self.assertRaises(TypeError): at.intervention_effects(object(),X)
        with self.assertRaises(ValueError): at.plot_interventions(self.y,X,individual='yes')


if __name__ == '__main__':
    unittest.main()
