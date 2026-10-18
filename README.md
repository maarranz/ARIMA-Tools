# ARIMA-Tools

Teaching-oriented Python tools for empirical ARIMA modelling, developed for **Applied Time Series Econometrics (ATSE)** at **Universidad Carlos III de Madrid**. The package complements Statsmodels and StatsForecast with tables, summaries, numerical forecasts, and interactive Plotly figures.

## Public API and capabilities

```python
import arima_tools as at
```

The API is deliberately flat: use `at.function(...)`. The 30 public functions are exposed at package level; users do not need to import individual submodules.

- **Estimation and model properties:** coefficient inference, parameter correlations, joint tests, AR/MA roots, causality, invertibility, and common-root checks.
- **Diagnostics:** parameters and model structure; residual ACF/PACF, Ljung–Box and normality tests; dynamics and recursive parameter stability.
- **Model comparison and selection:** information-criterion tables, nonseasonal order searches, and StatsForecast AutoARIMA selection and summaries.
- **Forecasting:** independent forecast-error variances, recursive forecasts with normal-theory intervals, forecast-line plots, and shaded fan charts. Default fan coverages are `(0.50, 0.70, 0.80, 0.90, 0.95, 0.99)`.

The implementation is organized in `arima_tools/`, with `__init__.py`, `estimation.py`, `diagnostics.py`, `model_selection.py`, `forecasting.py`, and `interventions.py` (with shared internal regressor checks).

## Documentation and examples

[Public API documentation — rendered HTML](docs/index.html) · [Quarto source](docs/ARIMA_Tools_Documentation.qmd)

The API document gives signatures, returns, and short examples. Open the HTML locally in a browser; it is also prepared for eventual GitHub Pages publication. See the root-level [ARIMA_Tools_Development.ipynb](ARIMA_Tools_Development.ipynb) for the fuller demonstration workflow.

This documentation describes the software. **ATSE course material** provides the econometric theory, and the **Labs** provide fuller worked examples and exercises. The public course/Labs link will be added when available.

## Environment

[environment.yml](environment.yml) specifies Python 3.14 and the required scientific Python, Plotly, Statsmodels, StatsForecast, and Jupyter packages. From the repository root:

```bash
mamba env create -f environment.yml
mamba activate atse_arma_tools
jupyter lab ARIMA_Tools_Development.ipynb
```

Create the environment only if needed; otherwise activate the existing one. Run Python or JupyterLab from the repository root to import the local package.

## Development status and related projects

The package is under active development; the API may evolve. Numerical and graphical forecasting are implemented. Static forecasting, forecast evaluation and comparison belong to **Forecasting-Tools**.

- **ARMA-Theory** — theoretical ARMA properties, roots, ACF/PACF, and impulse responses.
- **ADL-Theory** — theoretical autoregressive distributed lag models.
- **Forecasting-Tools** — static forecasting, forecast evaluation and comparison.
- **ADL-Tools** — empirical ADL modelling (planned).

## License

[MIT License](LICENSE).

## Author

**Miguel A. Arranz, PhD**\
Department of Economics\
Universidad Carlos III de Madrid (UC3M)\
Email: [maarranz@eco.uc3m.es](mailto:maarranz@eco.uc3m.es)\
ORCID: [https://orcid.org/0000-0002-5951-2284](https://orcid.org/0000-0002-5951-2284)

## Model-selection and diagnostic conventions

`compare_models()` reports candidates; `select_model(table, criterion="BIC", d=None)`
checks comparability before choosing. Mixed-d tables require an explicit `d`.
Grid-search tables are grouped by d, with criteria ranked only within each group.
Eligible models must share ordered response values, missing-value positions,
original time indexes (including frequency and timezone), effective observations,
and initialization metadata. Tables without sample fingerprints must be rebuilt
before selecting among multiple eligible models. Failed, unknown-convergence
and nonfinite candidates are excluded; duplicate model labels require explicit unique names.

`N` is Statsmodels `nobs_effective`; `Parameters` is `df_model`, including free
parameters, concentrated innovation scale and exact diffuse state terms.
AIC/BIC/HQIC use Statsmodels likelihood conventions; AICc uses the same N and k
and is undefined when N-k-1 <= 0. These conventions can differ numerically from
R's `forecast::Arima`; cross-engine numerical equality is not assumed.

Residual diagnostics omit initialization periods (the maximum of likelihood burn
and diffuse periods) and unobserved responses. ACF/PACF and Ljung–Box reject
interior gaps; Jarque–Bera uses remaining finite residuals and reports the same
moment convention as its statistic. Ljung–Box counts ordinary and seasonal AR/MA
coefficients, without subtracting differencing, deterministic terms or variance.
Recursive estimation rejects missing input rather than closing gaps, records
warnings/failure reasons, and excludes nonconverged estimates from stability
plots. A full-sample reference is drawn only for a valid final-window estimate.

Run regression checks from the repository root with
`python -m unittest discover -s tests -v` in the existing environment.

### Likelihood initialization and optimizer evidence

`compare_models()` remains a reporting function: incompatible candidates can be
inspected together. It records an `Initialization` signature, an
`Initialization Status`, fitted initial moments in `Initialization Detail`, and
`Convergence Status`. `select_model()` enforces these safeguards.

Initialization signatures identify stationary, approximate-diffuse and
exact-diffuse states. Diffuse means, approximate-diffuse variances and variance
units must agree; burn-in, diffuse periods and differencing checks still apply.
Different stationary ARMA state dimensions do not alone prevent comparison.
Known/custom priors without an established equivalence rule, unavailable metadata
and unsupported representations are reported but excluded from selection.
Concentrated-scale filtering expresses approximate-diffuse variance relative to
innovation scale; mixing that prior with an absolute-variance diffuse prior is
conservatively rejected. Stationary and compatible exact-diffuse comparisons can
still mix concentrated and explicitly estimated scale.

Convergence is read from state-space `mle_retvals` and, where present,
`fit_details.minimize_results.success`, including inner GLS optimizers. Explicit
failure overrides success, and unknown required convergence (including fixed-count
GLS iterations) remains unknown. `Converged` is `None` for unknown evidence and
recognized non-optimizer estimators; `Convergence Status` distinguishes them.
Recognized Yule-Walker, Burg, innovations and Hannan-Rissanen result structures
are marked not applicable and remain selectable. That status does not establish
maximum-likelihood estimation or statistical adequacy.

PACF accepts lags through `floor(T / 2)` for T usable residuals. Both grid and
recursive estimation retain warnings emitted before a fit fails, alongside its
failure reason.

## Manual intervention modelling

Interventions describe specified events; no dates or decay rates are detected or
estimated automatically. `at.outlier_regressors(y, dates=None, types="AO",
delta=0.7, *, positions=None)` returns a named pandas DataFrame ready for
Statsmodels `ARIMA(..., exog=X).fit()`.

For a chosen observation T0:

| Type | Regressor | Coefficient interpretation |
|---|---|---|
| AO | One at T0, zero elsewhere | One-observation displacement |
| LS | Zero before T0, one from T0 onward | Sustained level displacement |
| TC | Zero before T0, delta**(t-T0) afterward | Initial displacement, decaying by delta per observation |

The default TC decay is 0.7, following ARIMA-Tools-R; require 0 < delta < 1.
One scalar decay applies to all TC events. Multiple interventions and repeated
types are allowed; duplicate type/date pairs are rejected. Different types on
the same date are allowed, but can be unidentified in short samples.

### Complete Statsmodels example

The following simulated example specifies a permanent shift, a one-quarter
outlier, and a temporary change. In empirical work, choose dates from the
substantive question and inspect the fitted diagnostics.

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import arima_tools as at

index = pd.period_range("2000Q1", periods=104, freq="Q")
X = at.outlier_regressors(
    index, dates=["2008Q4", "2020Q2", "2021Q1"],
    types=["LS", "AO", "TC"], delta=0.7,
)
rng = np.random.default_rng(1234)
y = pd.Series(
    100 + 0.2 * np.arange(len(index))
    + X.to_numpy() @ np.array([5.0, -8.0, 3.0])
    + rng.normal(size=len(index)),
    index=index, name="Output",
)
results = ARIMA(y, order=(1, 0, 0), trend="ct", exog=X).fit(
    cov_type="oim", method_kwargs={"maxiter": 500},
)
print("Converged:", results.mle_retvals["converged"])
print(at.coefficient_diagnostics(results))
effects = at.intervention_effects(results, X)
figure = at.plot_interventions(y, X, results=results, individual=True)
future_X = at.extend_interventions(X, steps=8)
forecast = at.dynamic_forecast(results, steps=8, exog=future_X)
fan = at.plot_fan_chart(results, steps=8, exog=future_X)
```

The example explicitly requests observed-information covariance (`cov_type="oim")
and allows more optimizer iterations. Inspect convergence and standard errors:
sparse, single-observation AO regressors can give unstable default OPG covariance
estimates. An AO coefficient is supported by very limited event information;
normal/Wald inference remains approximate and should not determine event inclusion
automatically. The package does not change Statsmodels fitting defaults.

### Dates, positions and metadata

Pass a Series, pandas Index, or one-dimensional array. Exactly one of `dates` or
`positions` is required. Positions are **zero-based**; arrays receive a RangeIndex.
Period dates resolve to their containing period using the index frequency.
Datetime events require exact, complete dates with matching timezone. Calendar
indexes must be regular, unique and increasing, without missing labels. Numerical
indexes must be unique and increasing. No observation is sorted or dropped;
missing response values retain their positions. TC decay advances one observation
at a time, rather than by calendar-day distance.

Generated columns include names such as `AO_2020Q2`, `LS_2008-09`, and `TC_pos12`.
`X.attrs["arima_tools_interventions"]` is a version-1 dictionary with:

- `origin_length`, `origin_index`: original sample length and index description.
- `row_start`, `index`: current block's offset and index description.
- `events`: type, zero-based position, label, column, and delta (None for AO/LS).

Descriptions contain index kind, dtype, frequency, timezone and ordered labels.
`extend_interventions` preserves the original event specifications while updating
the current block. Keep the original X: pandas transformations may discard attrs.
Effects, plotting and extension validate metadata against the numerical columns;
interventions are never inferred from prefixes alone. Ordinary controls can be
combined with X for fitting; pass the original intervention-only X to the effects
and plotting functions.

### Effects, adjustment and identification

`at.intervention_effects(results, X)` returns an indexed DataFrame: `Observed`,
one named contribution per intervention, `Combined`, and `Adjusted`. Each
contribution is beta_hat*X; Combined sums only those intervention contributions.
Adjusted is Observed minus Combined. Ordinary controls, deterministic terms and
ARIMA errors are not removed. **The intervention-adjusted series is not a
residual.** Plotting delegates to this same numerical helper.

`at.plot_interventions(y, X, *, results=None, individual=False, title="Intervention
Effects")` returns a Plotly Figure. Without results it shows observations and
chosen dates. With results, the upper panel shows observed and adjusted series;
the lower panel shows the combined effect. `individual=True` adds component
curves. Same-date events share one marker and tooltip.

All effects stay on the modeled response scale. If y is log GDP, adjustment
remains in logarithms. Statsmodels estimates a regression with ARIMA errors:
pass the **undifferenced regression columns** even when d > 0. An LS remains a
step in the supplied response; its first difference is a pulse. An AO's first
difference has adjacent positive and negative pulses. TC remains geometric on
the supplied scale. ARIMA integration does not turn these regressors into
innovation impulse responses. There is no automatic back-transformation.

`order_search(..., exog=None)` and `recursive_estimation(..., exog=None)` accept
keyword-only finite aligned regressors. DataFrame indexes must match Series y
exactly; arrays align by position, with indexed exog supplying labels when y is
an array. All external columns and built-in trends enter identification checks,
but the original columns are passed to Statsmodels. Complete response samples
retain the differenced-design check. With missing responses, the check preserves
the original time grid and tests observed regressors modulo the visible
integration directions: constants for I(1), affine paths for I(2). Thus a contrast
across a gap can identify an LS even when its event-date response is missing.
Rank uses column scaling and machine-precision SVD tolerance, not a discretionary
near-collinearity cutoff. This algebraic check does not guarantee precise estimates.

An LS at the first observation can duplicate a constant or disappear under
differencing. Events at the last observation can generate identical columns.
The builder warns about dependent event columns; fitting helpers report failed
or unidentified specifications without removing regressors. Recursive windows
before an event is observed retain missing estimates and an explanatory status.
Parameter counts include external coefficients. Stage 2 convergence, effective
sample, initialization and differencing safeguards remain in force. Comparison
reports `Identified` and `Identification Status`; selection requires identification
to be established (`Identified=True`). Different exogenous specifications remain comparable when their
response sample and likelihood conventions are compatible; X equality is not
required.

### Dynamic forecasting with interventions

`at.extend_interventions(X, steps=20, *, future_index=None)` returns **future rows
only**. AO stays zero, LS stays one, and TC decay continues from the original
event, without restarting at the forecast origin. It continues existing events;
it does not introduce new future events. Regular period, datetime, RangeIndex
and equally spaced numerical indexes can continue automatically; otherwise supply
an increasing future index. Explicit continuations must match known frequencies.

`dynamic_forecast`, `plot_forecast` and `plot_fan_chart` accept keyword-only
`exog=None` for all future external regressors. DataFrame columns must have the
fitted names in the fitted order and match the model's forecast index; unnamed
arrays must have the correct shape and are interpreted in fitted column order.
Supply ordinary controls too, when present; built-in trends need no supplied
columns. Forecast-error variances are unchanged and condition on the parameters
and future regressors. No coefficient, date, decay or model-selection uncertainty
is added. Existing missing-history/initialization limitations of innovation-based
FEV still apply. Static forecasting and forecast evaluation remain the
responsibility of Forecasting-Tools.
