# ARIMA-Tools

Teaching-oriented Python tools for empirical ARIMA modelling, developed for **Applied Time Series Econometrics (ATSE)** at **Universidad Carlos III de Madrid**. The package complements Statsmodels and StatsForecast with tables, summaries, numerical forecasts, and interactive Plotly figures.

## Public API and capabilities

```python
import arima_tools as at
```

The API is deliberately flat: use `at.function(...)`. The 26 public functions are exposed at package level; users do not need to import individual submodules.

- **Estimation and model properties:** coefficient inference, parameter correlations, joint tests, AR/MA roots, causality, invertibility, and common-root checks.
- **Diagnostics:** parameters and model structure; residual ACF/PACF, Ljung–Box and normality tests; dynamics and recursive parameter stability.
- **Model comparison and selection:** information-criterion tables, nonseasonal order searches, and StatsForecast AutoARIMA selection and summaries.
- **Forecasting:** independent forecast-error variances, recursive forecasts with normal-theory intervals, forecast-line plots, and shaded fan charts. Default fan coverages are `(0.50, 0.70, 0.80, 0.90, 0.95, 0.99)`.

The implementation is organized in `arima_tools/`, with `__init__.py`, `estimation.py`, `diagnostics.py`, `model_selection.py`, and `forecasting.py`.

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

The package is under active development; the API may evolve. Numerical and graphical forecasting are implemented. Forecast evaluation and comparison are planned separately in **Forecast-Tools**.

- **ARMA-Theory** — theoretical ARMA properties, roots, ACF/PACF, and impulse responses.
- **ADL-Theory** — theoretical autoregressive distributed lag models.
- **Forecast-Tools** — forecast evaluation and comparison (planned).
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
