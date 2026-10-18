"""Regression cases for Stage 2 initialization and convergence blockers."""
import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.initialization import Initialization
from statsmodels.tsa.stattools import pacf

import arima_tools as at
from arima_tools.model_selection import _initialization_info, _converged, _convergence_info


class BlockingCorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.y = np.random.default_rng(123).normal(size=80)

    def fitted(self, order=(1, 0, 0), variance=None, exact=False, **kwargs):
        model = ARIMA(self.y, order=order, trend='n', **kwargs)
        if variance is not None:
            model.initialize_approximate_diffuse(variance=variance)
        if exact:
            model.use_exact_diffuse = True
            model.initialize_default()
            model.loglikelihood_burn = 0
        return model.fit()

    def compare(self, first, second):
        return at.compare_models(first, second, names=['first', 'second'])

    def test_reported_approximate_diffuse_variance_failure(self):
        first = self.fitted(variance=1e6)
        second = self.fitted(variance=1e12)
        table = self.compare(first, second)  # Reporting still permits inspection.
        self.assertEqual(table['Sample ID'].nunique(), 1)
        self.assertGreater(abs(first.aic - second.aic), 10)
        self.assertNotEqual(table.Initialization.iloc[0], table.Initialization.iloc[1])
        with self.assertRaisesRegex(ValueError, 'Initialization'):
            at.select_model(table)

    def test_compatible_approximate_diffuse_and_scheme_mismatch(self):
        first, second = self.fitted(variance=1e6), self.fitted(variance=1e6)
        at.select_model(self.compare(first, second))
        with self.assertRaisesRegex(ValueError, 'Initialization'):
            at.select_model(self.compare(first, self.fitted()))

    def test_stationary_dimensions_and_deterministic_terms(self):
        first = self.fitted()
        second = self.fitted(order=(0, 0, 2))
        self.assertNotEqual(first.model.k_states, second.model.k_states)
        at.select_model(self.compare(first, second))
        constant = ARIMA(self.y, order=(1, 0, 0), trend='c').fit()
        at.select_model(self.compare(first, constant))
        concentrated = self.fitted(concentrate_scale=True)
        at.select_model(self.compare(first, concentrated))

    def test_integrated_default_different_state_dimensions(self):
        first = self.fitted(order=(1, 1, 0))
        second = self.fitted(order=(0, 1, 2))
        self.assertNotEqual(first.model.k_states, second.model.k_states)
        at.select_model(self.compare(first, second))
        concentrated = self.fitted(order=(1, 1, 0), concentrate_scale=True)
        with self.assertRaisesRegex(ValueError, "Initialization"):
            at.select_model(self.compare(first, concentrated))

    def test_exact_diffuse_compatible_different_state_dimensions(self):
        first = self.fitted(order=(1, 1, 0), exact=True)
        second = self.fitted(order=(0, 1, 2), exact=True)
        self.assertGreater(first.nobs_diffuse, 0)
        self.assertNotEqual(first.model.k_states, second.model.k_states)
        table = self.compare(first, second)
        self.assertEqual(table.Initialization.nunique(), 1)
        at.select_model(table)
        concentrated = self.fitted(order=(1, 1, 0), exact=True, concentrate_scale=True)
        at.select_model(self.compare(first, concentrated))
        approximate = self.fitted(order=(1, 1, 0))
        with self.assertRaises(ValueError):
            at.select_model(self.compare(first, approximate))

    def test_unknown_and_known_initialization_are_reported_not_selected(self):
        model = ARIMA(self.y, order=(1, 0, 0), trend='n')
        model.initialize_known(np.array([0.]), np.eye(1))
        known = model.fit()
        table = self.compare(self.fitted(), known)
        self.assertEqual(table['Initialization Status'].iloc[1], 'Unsupported or unavailable')
        with self.assertRaisesRegex(ValueError, 'unsupported or unavailable'):
            at.select_model(table)
        unavailable = SimpleNamespace(model=SimpleNamespace())
        self.assertEqual(_initialization_info(unavailable)['Initialization Status'],
                         'Unsupported or unavailable')
        table = self.compare(self.fitted(), self.fitted(order=(0, 0, 1)))
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            at.select_model(table.drop(columns='Initialization Status'))

    def test_custom_diffuse_blocks_normalize_stationary_dimensions(self):
        # Equivalent explicit blocks must agree with Statsmodels defaults.
        first = self.fitted(order=(1, 1, 0))
        model = ARIMA(self.y, order=(0, 1, 2), trend='n')
        init = Initialization(model.k_states)
        init.set((0, 1), 'approximate_diffuse', approximate_diffuse_variance=1e6)
        init.set((1, model.k_states), 'stationary')
        model.ssm.initialization = init
        at.select_model(self.compare(first, model.fit()))

    def test_real_failed_innovations_optimizer(self):
        failed = ARIMA(self.y, order=(1, 0, 1), trend='n').fit(
            method='innovations_mle',
            method_kwargs={'minimize_kwargs': {'options': {'maxiter': 0}}})
        self.assertFalse(failed.fit_details.minimize_results.success)
        self.assertFalse(_converged(failed))
        table = self.compare(self.fitted(), failed)
        self.assertEqual(table['Convergence Status'].iloc[1], 'Not converged')
        table.loc['second', 'BIC'] = -1e10
        self.assertEqual(at.select_model(table).name, 'first')

    def test_successful_innovations_and_unknown_convergence(self):
        fitted = ARIMA(self.y, order=(1, 0, 0), trend='n').fit(method='innovations_mle')
        self.assertTrue(fitted.fit_details.minimize_results.success)
        self.assertTrue(_converged(fitted))
        self.assertEqual(_convergence_info(SimpleNamespace()), (None, 'Unknown'))
        table = self.compare(self.fitted(), fitted)
        table['Converged'] = table['Converged'].astype(object)
        table.loc['second', 'Converged'] = None
        table.loc['second', 'Convergence Status'] = 'Unknown'
        table.loc['second', 'BIC'] = -1e10
        self.assertEqual(at.select_model(table).name, 'first')

    def test_non_optimizer_estimators_are_distinct_from_success(self):
        for method, order in [('yule_walker', (1, 0, 0)), ('burg', (1, 0, 0)),
                              ('innovations', (0, 0, 1)), ('hannan_rissanen', (1, 0, 0))]:
            with self.subTest(method=method):
                fitted = ARIMA(self.y, order=order, trend='n').fit(method=method)
                self.assertIsNone(_converged(fitted))
                self.assertIn('Not applicable', _convergence_info(fitted)[1])
                at.select_model(self.compare(self.fitted(), fitted))

    def test_pacf_odd_and_even_lag_boundaries(self):
        for size in (9, 10):
            fitted = ARIMA(self.y[:size], order=(0, 0, 0), trend='n').fit()
            limit = size // 2
            table = at.residual_correlations(fitted, lags=limit)
            np.testing.assert_allclose(table.PACF, pacf(fitted.resid, nlags=limit, method='ywm')[1:])
            with self.assertRaises(ValueError):
                at.residual_correlations(fitted, lags=limit+1)

    def test_failed_fits_preserve_prior_warnings(self):
        def fail(*args, **kwargs):
            warnings.warn('warning before failed fit', RuntimeWarning)
            raise ValueError('deliberate failure')
        with patch('statsmodels.tsa.arima.model.ARIMA.fit', side_effect=fail):
            search = at.order_search(self.y, p_max=0, q_max=0)
            recursive = at.recursive_estimation(self.y, (1, 0, 0), initial=78)
        for table in (search, recursive):
            self.assertTrue(table.Warnings.map(lambda w: 'warning before failed fit' in w).all())
            self.assertTrue(table.Status.str.contains('deliberate failure').all())
            self.assertFalse(table.Converged.any())

    def test_gls_outer_convergence_is_not_inferred_from_inner_success(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fixed_iterations = ARIMA(self.y, order=(1, 0, 0)).fit(
                method='innovations_mle', gls_kwargs={'n_iter': 1})
            failed = ARIMA(self.y, order=(1, 0, 0)).fit(
                method='innovations_mle', gls_kwargs={'max_iter': 1})
            converged = ARIMA(self.y, order=(1, 0, 0)).fit(method='innovations_mle')
        self.assertIsNone(fixed_iterations.fit_details.converged)
        self.assertEqual(_convergence_info(fixed_iterations), (None, 'Unknown'))
        self.assertFalse(_converged(failed))
        self.assertTrue(_converged(converged))

    def test_gls_inner_statespace_failure_overrides_outer_success(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fitted = ARIMA(self.y, order=(1, 0, 0)).fit(
                method='statespace', gls=True,
                method_kwargs={'fit_kwargs': {'maxiter': 0, 'disp': False}})
        flags = [inner.statespace_results.mle_retvals['converged']
                 for inner in fitted.fit_details.arma_results if inner is not None]
        self.assertIn(False, flags)
        self.assertFalse(_converged(fitted))

    def test_missing_initialization_signature_and_convergence_evidence(self):
        table = self.compare(self.fitted(), self.fitted(order=(0, 0, 1)))
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            at.select_model(table.drop(columns='Initialization'))
        # A filtering-only result is not evidence of optimizer convergence.
        model = ARIMA(self.y, order=(1, 0, 0), trend='n')
        filtered = model.filter(self.fitted().params)
        self.assertEqual(_convergence_info(filtered), (None, 'Unknown'))
        table = self.compare(self.fitted(), filtered)
        self.assertEqual(at.select_model(table).name, 'first')


if __name__ == '__main__':
    unittest.main()
