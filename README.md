# ARIMA-Tools

Teaching-oriented Python tools for empirical ARIMA modelling, developed for
**Applied Time Series Econometrics (ATSE)** at **Universidad Carlos III de Madrid**.
For students, instructors and researchers, the package complements Statsmodels
and StatsForecast with transparent numerical tables and interactive Plotly figures.

## Capabilities

Use the flat API, `import arima_tools as at`, for 30 public functions covering:

- Coefficient inference, joint Wald tests, parameter correlations and AR/MA roots.
- Residual autocorrelation and normality diagnostics, innovation dynamics and
  recursive parameter stability.
- Information-criterion comparisons, nonseasonal order searches and an AutoARIMA
  benchmark.
- Dynamic forecasts, innovation-based forecast-error variances and normal-theory
  prediction intervals, including forecast plots and fan charts.
- Manual AO (one-observation displacement), LS (level shift) and TC (geometrically
  decaying change) regressors, fitted contributions and future regressor paths.

Intervention dates and decay rates are specified by the user. The package does
not detect events automatically or establish their economic causality.

## GitHub setup

Clone the repository and create its environment if needed:

```bash
git clone https://github.com/maarranz/ARIMA-Tools.git
cd ARIMA-Tools
mamba env create -f environment.yml
mamba activate atse_arma_tools
```

For an existing checkout/environment, change to the repository root and activate
`atse_arma_tools`. [environment.yml](environment.yml) specifies Python 3.14 and the
scientific Python, Statsmodels, StatsForecast, Plotly and Jupyter dependencies.
Run Python or Jupyter from this root so the local `arima_tools` directory is
importable; no package-install or PyPI step is required.

This is the GitHub-based workflow. Fresh environment recreation and a complete
Google Colab setup are reserved for Stage 4C. The current dependency file is not
a version lock; successful examples in an existing environment do not establish
clean-environment compatibility.

## Short example

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import arima_tools as at

rng = np.random.default_rng(12345)
y = pd.Series(rng.normal(size=120).cumsum(), name="Simulated random walk")
result = ARIMA(y, order=(0, 1, 0), trend="n").fit()
print(at.coefficient_diagnostics(result))
forecast = at.dynamic_forecast(result, steps=8)
print(forecast)
figure = at.plot_fan_chart(result, steps=8)
# In Jupyter, display figure or call figure.show().
```

Here d = 1 follows the known simulated process. Forecast intervals condition on
the fitted model and parameters; they omit parameter-estimation uncertainty.
Intervention and ordinary external regressors require supplied future values.

## Documentation, notebook and tests

- [Complete documentation — HTML](docs/index.html): open locally in a browser.
- [Single-file Quarto source](docs/ARIMA_Tools_Documentation.qmd): definitions,
  assumptions, examples, numerical conventions and all public signatures.
- [Development notebook](ARIMA_Tools_Development.ipynb): demonstrations including
  deliberately incompatible comparisons and the safeguards that reject selection.

From the activated environment at the repository root:

```bash
jupyter lab ARIMA_Tools_Development.ipynb
python -m unittest discover -s tests -v
```

The Quarto source also documents the standalone HTML build. ATSE course material
supplies the fuller theory and Labs; public links will be added when available.

## Key econometric conventions

`compare_models()` reports candidates; `select_model()` checks their ordered
response samples, missing positions, indexes, effective observations and likelihood
initialization. Mixed-d tables require an explicit `d`: integration order is
chosen separately from information-criterion selection of ARMA orders. Unknown
optimizer convergence and unresolved regression identification are not treated
as success. See the complete reference for supported initialization conventions.

Residual lag diagnostics exclude initialization periods and reject interior gaps.
Recursive estimation requires complete finite responses. Forecast uncertainty
uses ARIMA innovation weights and can differ from state-space variances when
past innovations or initialization remain uncertain.

Pass intervention regressors undifferenced to Statsmodels. Adjusted observations
remove only fitted intervention contributions and are not residuals. Identification
can be established, rejected or inconclusive; none of these outcomes establishes
economic causality. Full definitions and limitations are in the Quarto reference.

## ATSE software suite

ARIMA-Tools belongs to a collection of complementary econometric teaching tools.
Each project is developed and documented separately.

| Project | Purpose |
|---|---|
| ARMA-Theory | Theoretical properties of ARMA processes |
| ADL-Theory | Stability and dynamic effects of ADL models |
| ARIMA-Tools | Empirical ARIMA modelling, inference, diagnostics, selection, dynamic forecasting and interventions |
| ADL-Tools | Intended scope: empirical ADL modelling, estimation and hypothesis testing; completion and public availability are not asserted here |
| Forecasting-Tools | Static forecasting, forecast evaluation and forecast comparison |

## Development, license and author

The package remains under active development. [MIT License](LICENSE).

**Miguel A. Arranz, PhD**\
Department of Economics, Universidad Carlos III de Madrid (UC3M)\
[maarranz@eco.uc3m.es](mailto:maarranz@eco.uc3m.es) ·
[ORCID: 0000-0002-5951-2284](https://orcid.org/0000-0002-5951-2284)
