"""Regression checks for R-inspired selection safeguards and diagnostics."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import jarque_bera
import arima_tools as at
from arima_tools.model_selection import _ic_values
from arima_tools.diagnostics import _residual_sample


class SelectionDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.y = pd.Series(np.random.default_rng(42).normal(size=100),
                          index=pd.date_range('2000-01-01', periods=100, freq='MS'))
        cls.a = ARIMA(cls.y, order=(1, 0, 0)).fit()
        cls.b = ARIMA(cls.y, order=(0, 0, 1)).fit()

    def comparison(self, other):
        return at.compare_models(self.a, other, names=['a', 'b'])

    def test_identical_samples_and_convergence(self):
        table = self.comparison(self.b)
        self.assertEqual(table['Sample ID'].nunique(), 1)
        table.loc['a', 'Converged'] = False
        self.assertEqual(at.select_model(table).name, 'b')
        table.loc['b', 'Converged'] = False
        with self.assertRaisesRegex(ValueError, 'No eligible'):
            at.select_model(table)

    def test_values_order_missing_and_indexes(self):
        variants = [self.y + 1, self.y.iloc[::-1],
                    self.y.set_axis(self.y.index + pd.DateOffset(months=1)),
                    self.y.reset_index(drop=True)]
        missing = self.y.copy(); missing.iloc[20] = np.nan
        variants.append(missing)
        for y in variants:
            with self.subTest(index=type(y.index).__name__):
                other = ARIMA(y, order=(0, 0, 1)).fit()
                with self.assertRaisesRegex(ValueError, 'same sample'):
                    at.select_model(self.comparison(other))

    def test_differencing_and_likelihood_burn(self):
        integrated = ARIMA(self.y, order=(0, 1, 0)).fit()
        table = self.comparison(integrated)
        with self.assertRaisesRegex(ValueError, 'choose d'):
            at.select_model(table)
        self.assertEqual(at.select_model(table, d=1).name, 'b')
        model = ARIMA(self.y, order=(0, 0, 1))
        model.loglikelihood_burn = 2
        other = model.fit()
        with self.assertRaisesRegex(ValueError, 'same sample'):
            at.select_model(self.comparison(other))

    def test_labels_and_invalid_selection(self):
        with self.assertRaisesRegex(ValueError, 'names'):
            at.compare_models(self.a, self.a)
        table = self.comparison(self.b)
        with self.assertRaises(ValueError): at.select_model(table, d=True)
        table['BIC'] = np.nan
        with self.assertRaises(ValueError): at.select_model(table)

    def test_ic_scale_and_fixed_parameters(self):
        concentrated = ARIMA(self.y, order=(1, 0, 0), concentrate_scale=True).fit()
        model = ARIMA(self.y, order=(1, 0, 0), trend='n')
        with model.fix_params({'ar.L1': 0.2}):
            fixed = model.fit()
        for result in (self.a, concentrated, fixed):
            row = _ic_values(result)
            self.assertEqual(row['Parameters'], result.df_model)
            self.assertEqual(row['N'], result.nobs_effective)
            self.assertAlmostEqual(row['AICc'], result.aicc)
        self.assertEqual(_ic_values(concentrated)['Parameters'], len(concentrated.params)+1)
        self.assertEqual(_ic_values(fixed)['Parameters'], len(fixed.params)-1)

    def test_search_grouping(self):
        table = at.order_search(self.y, p_max=0, q_max=0, d_max=1)
        self.assertEqual(list(table.d), [0, 1])
        with self.assertRaises(ValueError): at.select_model(table)
        self.assertEqual(at.select_model(table, d=0)['d'], 0)

    def test_residual_gaps_and_burn(self):
        y = self.y.copy(); y.iloc[30] = np.nan
        result = ARIMA(y, order=(1, 0, 0)).fit()
        for function in (at.residual_correlations, at.ljung_box_test):
            with self.assertRaisesRegex(ValueError, 'gaps'): function(result, lags=5)
        integrated = ARIMA(self.y, order=(0, 1, 0)).fit()
        residuals = _residual_sample(integrated)
        self.assertEqual(len(residuals), len(self.y)-integrated.loglikelihood_burn)
        expected = acorr_ljungbox(residuals, lags=[5], model_df=0)
        self.assertAlmostEqual(at.ljung_box_test(integrated, [5]).iloc[0, 0], expected.iloc[0, 0])
        self.assertAlmostEqual(at.jarque_bera_test(integrated).iloc[0], jarque_bera(residuals).statistic)

    def test_lag_validation_and_seasonal_df(self):
        for lag in (True, 2.5, [], 100):
            with self.assertRaises(ValueError): at.ljung_box_test(self.a, lag)
        seasonal = ARIMA(self.y, order=(1, 0, 0), seasonal_order=(1, 0, 0, 4)).fit()
        self.assertEqual(at.ljung_box_test(seasonal, 5).iloc[0]['Degrees of freedom'], 3)

    def test_recursive_missing_and_failure_status(self):
        y = self.y.copy(); y.iloc[20] = np.nan
        with self.assertRaises(ValueError): at.recursive_estimation(y, (1, 0, 0))
        with patch('arima_tools.diagnostics.ARIMA.fit', side_effect=ValueError('deliberate')):
            table = at.recursive_estimation(self.y, (1, 0, 0), initial=98)
        self.assertTrue(table.Status.str.contains('deliberate').all())

    def test_stability_reference_requires_final_convergence(self):
        table = pd.DataFrame({'Endpoint': [1, 2], 'N': [20, 21], 'Parameter': ['ar.L1']*2,
                              'Estimate': [.2, .3], 'Lower CI': [.1, .2],
                              'Upper CI': [.3, .4], 'Converged': [True, False]})
        figure = at.plot_parameter_stability(table)
        self.assertFalse(any('Full sample' in a.text for a in figure.layout.annotations))

    def test_root_summary_and_plot_extent(self):
        result = SimpleNamespace(arroots=np.array([.5]), maroots=np.array([4.]))
        self.assertAlmostEqual(at.root_summary(result)['Minimum AR-MA distance'], 1.75)
        figure = at.plot_inverse_roots(result)
        self.assertGreater(figure.layout.xaxis.range[1], 2)

    def test_same_missing_sample_and_unrecoverable_metadata(self):
        y = self.y.copy(); y.iloc[20] = np.nan
        a = ARIMA(y, order=(1, 0, 0)).fit()
        b = ARIMA(y.copy(), order=(0, 0, 1)).fit()
        table = at.compare_models(a, b)
        self.assertEqual(table['Sample ID'].nunique(), 1)
        at.select_model(table)
        with self.assertRaisesRegex(ValueError, 'identity is unavailable'):
            at.select_model(table.drop(columns='Sample ID'))

    def test_jb_moments_reproduce_statistic(self):
        table = at.jarque_bera_test(self.a)
        n = len(_residual_sample(self.a, consecutive=False))
        expected = n / 6 * (table['Skewness']**2 + (table['Kurtosis']-3)**2/4)
        self.assertAlmostEqual(table['Jarque-Bera statistic'], expected)

    def test_recursive_trend_count_and_success(self):
        table = at.recursive_estimation(self.y, (0, 0, 0), initial=98, trend='ct')
        self.assertEqual(set(table.Parameter), {'const', 'x1', 'sigma2'})
        self.assertTrue(table.Converged.all())
        self.assertTrue((table.Status == 'OK').all())


if __name__ == '__main__':
    unittest.main()
