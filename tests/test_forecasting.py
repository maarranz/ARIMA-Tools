"""Controlled numerical checks for the flat forecasting API."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess

import arima_tools as at


class ForecastingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(20261005)
        cls.models = {}
        for name, ar, ma, order in [
            ('AR(1)', [1, -0.65], [1], (1, 0, 0)),
            ('ARMA(1,1)', [1, -0.55], [1, 0.35], (1, 0, 1)),
        ]:
            y = ArmaProcess(ar, ma).generate_sample(
                nsample=1200, burnin=300, distrvs=rng.standard_normal
            )
            model = ARIMA(y, order=order, trend='n')
            # Disable the steady-state shortcut for precision comparisons.
            model.ssm.tolerance = 0.0
            cls.models[name] = model.fit()
        cls.walk = np.cumsum(rng.normal(size=1200))
        cls.models['Random walk'] = ARIMA(
            cls.walk, order=(0, 1, 0), trend='n'
        ).fit()

    def test_controlled_variances_and_limits(self):
        for name, result in self.models.items():
            with self.subTest(model=name):
                sigma2 = result.params[result.param_names.index('sigma2')]
                fev = at.forecast_error_variance(result, steps=1000)
                reference = np.asarray(result.get_forecast(1000).var_pred_mean)
                np.testing.assert_allclose(fev, reference, rtol=1e-10, atol=1e-10)
                h = np.arange(1, 1001)
                if name == 'Random walk':
                    analytical = h * sigma2
                    np.testing.assert_allclose(fev, analytical, rtol=1e-13)
                    np.testing.assert_allclose(np.diff(fev), sigma2, rtol=1e-10)
                    limit = None
                else:
                    phi = result.arparams[0]
                    if name == 'AR(1)':
                        analytical = sigma2 * (1 - phi ** (2 * h)) / (1 - phi**2)
                        np.testing.assert_allclose(fev, analytical, rtol=1e-13)
                        limit = sigma2 / (1 - phi**2)
                    else:
                        theta = result.maparams[0]
                        limit = sigma2 * (1 + (phi + theta)**2 / (1 - phi**2))
                    self.assertAlmostEqual(fev[-1], limit, places=12)
                    self.assertGreaterEqual(fev[-1], fev[0])
                print(
                    f'{name}: max |FEV - Statsmodels|={np.max(np.abs(fev-reference)):.3g}; '
                    f'V1={fev[0]:.9f}; V20={fev[19]:.9f}; '
                    f'V1000={fev[-1]:.9f}; stationary limit={limit}',
                    flush=True,
                )

    def test_forecast_table_and_intervals(self):
        for result in self.models.values():
            table = at.dynamic_forecast(result, steps=12, level=0.90)
            self.assertEqual(list(table.columns), ['Forecast', 'FEV', 'SE', 'Lower', 'Upper'])
            pd.testing.assert_index_equal(table.index, pd.RangeIndex(1, 13, name='Horizon'))
            np.testing.assert_allclose(table.Forecast, result.forecast(12))
            np.testing.assert_allclose(table.FEV, at.forecast_error_variance(result, 12))
            np.testing.assert_allclose(table.SE, np.sqrt(table.FEV))
            np.testing.assert_allclose(table.Lower, table.Forecast - norm.ppf(.95) * table.SE)
            np.testing.assert_allclose(table.Upper, table.Forecast + norm.ppf(.95) * table.SE)

    def test_variance_does_not_call_forecast(self):
        result = self.models['ARMA(1,1)']
        expected = at.forecast_error_variance(result, 8)
        with patch.object(result, 'get_forecast', side_effect=AssertionError('Must not call')):
            with patch.object(result, 'forecast', side_effect=AssertionError('Must not call')):
                np.testing.assert_array_equal(at.forecast_error_variance(result, 8), expected)
        with patch.object(result, 'forecast', return_value=np.arange(8)):
            with patch.object(result, 'get_forecast', side_effect=AssertionError('Must not call')):
                np.testing.assert_array_equal(at.dynamic_forecast(result, 8).FEV, expected)

    def test_integration_and_concentrated_scale(self):
        for order, seasonal in [((0, 2, 0), (0, 0, 0, 0)),
                                ((0, 1, 0), (0, 1, 0, 4))]:
            # Integration-specific psi weights: second integration gives j+1;
            # combined ordinary/seasonal integration gives floor(j/4)+1.
            rng = np.random.default_rng(18)
            y = np.cumsum(np.cumsum(rng.normal(size=400)))
            result = ARIMA(y, order=order, seasonal_order=seasonal, trend='n').fit()
            sigma2 = result.params[result.param_names.index('sigma2')]
            j = np.arange(20)
            psi = j + 1 if order[1] == 2 else j // 4 + 1
            np.testing.assert_allclose(at.forecast_error_variance(result), sigma2 * np.cumsum(psi**2))
            np.testing.assert_allclose(at.forecast_error_variance(result),
                                       result.get_forecast(20).var_pred_mean, rtol=1e-7)
        result = ARIMA(self.walk, order=(0, 1, 0), trend='n', concentrate_scale=True).filter([])
        np.testing.assert_allclose(at.forecast_error_variance(result),
                                   np.arange(1, 21) * result.scale, rtol=1e-13)
        np.testing.assert_allclose(at.forecast_error_variance(result),
                                   result.get_forecast(20).var_pred_mean, rtol=1e-10)

    def test_validation_and_numpy_scalars(self):
        result = self.models['AR(1)']
        for steps in [0, -1, 1.5, True, np.bool_(False), '3', None, np.nan]:
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                at.forecast_error_variance(result, steps)
        for level in [0, 1, -0.1, 1.1, np.nan, np.inf, True, None, '0.95']:
            with self.subTest(level=level), self.assertRaises(ValueError):
                at.dynamic_forecast(result, level=level)
        with self.assertRaises(TypeError):
            at.forecast_error_variance(object())
        self.assertEqual(at.forecast_error_variance(result, np.int64(1)).shape, (1,))
        self.assertEqual(at.dynamic_forecast(result, np.int64(1), np.float64(.95)).shape, (1, 5))
        self.assertEqual(len(at.__all__), 26)

    def test_plot_values_and_band_nesting(self):
        for name, result in self.models.items():
            for level in (.95, .8):
                fig = at.plot_forecast(result, steps=40, level=level)
                table = at.dynamic_forecast(result, 40, level)
                self.assertEqual(len(fig.data[0].y), 50)
                np.testing.assert_array_equal(fig.data[0].y, result.model.endog[-50:, 0])
                self.assertTrue(all(t.fill in (None, 'none') for t in fig.data))
                for t in fig.data:
                    if t.meta: np.testing.assert_array_equal(t.y, table[t.meta['bound']])
                np.testing.assert_array_equal(fig.data[-1].y[1:], table.Forecast)
                self.assertEqual(fig.data[-1].x[0], fig.data[0].x[-1])
                self.assertEqual(len(fig.layout.shapes), 1)
            for levels in ((.5, .7, .8, .9, .95, .99), (.95, .5, .8), (.8,)):
                fig = at.plot_fan_chart(result, steps=40, levels=levels)
                bands = {}
                for t in fig.data:
                    if t.meta:
                        bands.setdefault(t.meta['level'], {})[t.meta['bound']] = np.asarray(t.y)[1:]
                        table = at.dynamic_forecast(result, 40, t.meta['level'])
                        np.testing.assert_array_equal(t.y[1:], table[t.meta['bound']])
                        self.assertEqual(t.x[0], fig.data[0].x[-1])
                self.assertEqual(sorted(bands), sorted(levels))
                self.assertEqual(sum(t.fill == 'tonexty' for t in fig.data), len(levels))
                self.assertEqual(sum(t.showlegend is not False for t in fig.data), 3)
                ordered = sorted(bands)
                for narrow, wide in zip(ordered, ordered[1:]):
                    self.assertTrue(np.all(bands[wide]['Lower'] <= bands[narrow]['Lower']))
                    self.assertTrue(np.all(bands[wide]['Upper'] >= bands[narrow]['Upper']))
                widths = bands[max(levels)]['Upper'] - bands[max(levels)]['Lower']
                self.assertTrue(np.all(np.diff(widths) >= -1e-12))
                if name == 'Random walk': self.assertGreater(widths[-1], 1.9 * widths[9])
                else: self.assertAlmostEqual(widths[-1], widths[-5], places=10)
        r = self.models['AR(1)']
        self.assertEqual(at.plot_fan_chart(r, levels=(.5, .8, .95)).to_json(),
                         at.plot_fan_chart(r, levels=(.95, .5, .8)).to_json())
        self.assertEqual(sorted({t.meta['level'] for t in at.plot_fan_chart(r).data if t.meta}),
                         [.5, .7, .8, .9, .95, .99])

    def test_plot_history_and_validation(self):
        r = self.models['AR(1)']
        for plot in (at.plot_forecast, at.plot_fan_chart):
            for history in (None, 5000):
                self.assertEqual(len(plot(r, history=history).data[0].y), 1200)
            self.assertEqual(len(plot(r, history=np.int64(1)).data[0].y), 1)
            one = plot(r, steps=1, history=1)
            self.assertIn('markers', one.data[0].mode)
            self.assertEqual(len(one.data[-1].y), 2)
            for history in (0, -1, 1.5, True, np.bool_(False), '50', np.nan):
                with self.subTest(history=history), self.assertRaises(ValueError): plot(r, history=history)
            for steps in (0, -1, 1.5, True):
                with self.assertRaises(ValueError): plot(r, steps=steps)
        for level in (0, 1, -1, np.nan, np.inf, True, '0.95'):
            with self.assertRaises(ValueError): at.plot_forecast(r, level=level)
        for levels in ((), None, .95, (.5, .5), (.5, 1), (0, .9), (np.nan,),
                       (np.inf,), (True,), ('0.95',), [[.5]]):
            with self.subTest(levels=levels), self.assertRaises(ValueError): at.plot_fan_chart(r, levels=levels)

    def test_plot_indices(self):
        y = np.random.default_rng(72).normal(size=80)
        indices = [pd.date_range('2020-01-01', periods=80, freq='MS'),
                   pd.date_range('2020-01-01', periods=80, freq='D', tz='Europe/Madrid'),
                   pd.period_range('2020-01', periods=80, freq='M'),
                   pd.RangeIndex(10, 170, 2, name='Sample')]
        for index in indices:
            r = ARIMA(pd.Series(y, index=index, name='Output'), order=(1, 0, 0), trend='n').fit()
            expected, historical = r.forecast(5).index, index[-8:]
            if isinstance(index, pd.PeriodIndex):
                expected, historical = expected.to_timestamp(), historical.to_timestamp()
            for plot in (at.plot_forecast, at.plot_fan_chart):
                fig = plot(r, steps=5, history=8)
                self.assertEqual(list(fig.data[0].x), list(historical))
                self.assertEqual(list(fig.data[-1].x[1:]), list(expected))
                self.assertEqual(fig.layout.yaxis.title.text, 'Output')
                fig.to_json()
        irregular = pd.to_datetime(['2020-01-01', '2020-01-03', '2020-01-08', '2020-01-09'])
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = ARIMA(pd.Series(y[:4], index=irregular), order=(0, 0, 0), trend='n').fit()
            # Test positional rendering separately from Statsmodels' unsupported-index errors.
            with patch.object(r, 'forecast', return_value=np.zeros(3)):
                fig = at.plot_forecast(r, steps=3, history=None)
        self.assertEqual(list(fig.data[0].x), [1, 2, 3, 4])
        self.assertEqual(list(fig.data[-1].x), [4, 5, 6, 7])

    def test_plots_delegate_numerical_work(self):
        import arima_tools.forecasting as forecasting
        r = self.models['ARMA(1,1)']
        fake = pd.DataFrame({'Forecast': [10., 20.], 'FEV': [25., 36.], 'SE': [5., 6.],
                             'Lower': [-11., -12.], 'Upper': [31., 32.]})
        with patch.object(forecasting, 'dynamic_forecast', return_value=fake) as numerical:
            with patch.object(r, 'get_forecast', side_effect=AssertionError('No uncertainty access')):
                with patch.object(r, 'forecast', side_effect=AssertionError('No direct calculation')):
                    fig = at.plot_forecast(r, steps=2, level=.8)
                    numerical.assert_called_once_with(r, steps=2, level=.8)
                    np.testing.assert_array_equal(fig.data[-1].y[1:], fake.Forecast)
                    np.testing.assert_array_equal(fig.data[1].y, fake.Lower)
                    numerical.reset_mock()
                    fan = at.plot_fan_chart(r, steps=2, levels=(.5, .9))
                    self.assertEqual(numerical.call_count, 2)
                    for t in fan.data:
                        if t.meta: np.testing.assert_array_equal(t.y[1:], fake[t.meta['bound']])


if __name__ == '__main__':
    unittest.main()
