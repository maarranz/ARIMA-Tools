# ARIMA-Tools

Teaching-oriented Python tools for empirical ARIMA modelling: estimation and properties of the fitted model, diagnostics, and model comparison and selection.

The project is being developed primarily for use in **Applied Time Series Econometrics (ATSE)**, but the tools are intended to be useful more generally for teaching and applied time-series analysis.

## Objectives

`ARIMA-Tools` complements existing Python time-series libraries rather than replacing them. It builds on `statsmodels` for ARIMA estimation and on `statsforecast` for automatic ARIMA model selection, adding tables, summaries, and interactive plots suited to teaching and applied analysis.

## Current capabilities

### Estimation and properties of the fitted model

Models are fitted using `statsmodels.tsa.arima.model.ARIMA`. ARIMA-Tools then provides coefficient estimates with standard errors, asymptotic inference and confidence intervals, parameter correlation matrices, and joint Wald tests. It also examines AR and MA roots, causality, invertibility, and common or near-common roots, with summaries and inverse-root plots.

### Diagnostics

Diagnostics are understood broadly, with three complementary layers:

1. **Parameters and model structure:** coefficient inference, parameter correlations, root properties, and possible AR–MA cancellation help assess the fitted specification.
2. **Residual behavior:** residual ACF/PACF tables, Ljung–Box tests, Jarque–Bera normality tests, and interactive residual-diagnostic plots help assess unexplained behavior.
3. **Model behavior:** dynamic and cumulative effects of an innovation, recursive expanding-window estimation, and parameter-stability plots help assess dynamics and stability over the sample.

These conceptual areas overlap: parameter and root diagnostics are implemented in `estimation.py`, while residual diagnostics, dynamics, and stability are implemented in `diagnostics.py`. For integrated models, root properties and dynamic effects refer to the stationary ARMA component. A breakpoint marker in a stability plot is a graphical reference, not a formal structural-break test.

### Model comparison and selection

Compare fitted models using likelihood and AIC, AICc, BIC, and HQIC; search a specified grid of nonseasonal ARIMA orders; and select the eligible model minimizing an information criterion. The grid search uses the differencing range chosen by the user. Automatic nonseasonal selection is available through StatsForecast AutoARIMA, with a compact summary of the selected specification and its mean or drift component.

Information criteria should be compared for models fitted to the same dependent variable and sample. Selection does not replace diagnostic assessment.

## Package structure and public API

```text
ARIMA-Tools/
├── arima_tools/
│   ├── __init__.py          # Flat public API
│   ├── estimation.py        # Parameter inference and model/root properties
│   ├── diagnostics.py       # Residuals, dynamics, and parameter stability
│   └── model_selection.py   # Comparison, order search, and AutoARIMA
├── ARIMA_Tools_Development.ipynb
├── environment.yml
├── README.md
└── LICENSE
```

The former single-file `arima_tools.py` has been replaced by the `arima_tools/` package. Its 22 public functions are exposed through `__init__.py`, preserving the existing interface:

```python
import arima_tools as at
```

The public API is deliberately flat: normally call functions as `at.function(...)`. Submodules organize the implementation; users do not need to remember or import from individual submodules.

## Requirements and environment

The current development environment uses Python 3.14 and the following principal packages:

- NumPy
- pandas
- SciPy
- Matplotlib
- Plotly
- statsmodels
- StatsForecast
- JupyterLab

The environment can be recreated from [environment.yml](environment.yml). From the repository root:

```bash
mamba env create -f environment.yml
mamba activate atse_arma_tools
```

If the environment already exists, activate it without recreating it. Run Python or JupyterLab from the repository root so that the local package is available for import.

## Example workflow

This small example simulates an ARMA(1,1) process, fits a model, examines its properties and residuals, and searches candidate orders:

```python
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
import arima_tools as at

np.random.seed(12345)
y = ArmaProcess([1, -0.7], [1, 0.4]).generate_sample(nsample=500)
result = ARIMA(y, order=(1, 0, 1), trend="n").fit()

print(at.coefficient_diagnostics(result))
print(at.root_summary(result))
print(at.ljung_box_test(result, lags=[5, 10, 20]))
at.plot_dynamic_effects(result, horizon=20).show()

search = at.order_search(y, p_max=2, q_max=2, criterion="BIC", trend="n")
print(at.select_model(search, criterion="BIC"))
```

See [ARIMA_Tools_Development.ipynb](ARIMA_Tools_Development.ipynb) at the repository root for the fuller workflow, including near-common and complex roots, residual diagnostics, automatic selection, recursive estimation, and parameter stability. Launch it from the same directory:

```bash
jupyter lab ARIMA_Tools_Development.ipynb
```

## Development status

This project is under active development, and the API may evolve. The current package implements the capabilities described above. Forecasting is planned future work and has not yet been implemented in ARIMA-Tools.

Forecast evaluation and comparison procedures are planned separately in **Forecast-Tools**, for use with forecasts from ARIMA, ADL, and other models.

## Related projects

- **ARMA-Theory** — theoretical analysis of ARMA models, including theoretical ACF/PACF, causality, invertibility, roots, and impulse responses.
- **ADL-Theory** — theoretical analysis of autoregressive distributed lag models.
- **Forecast-Tools** — forecast evaluation and comparison tools (planned).
- **ADL-Tools** — estimation, diagnostics, causality, and forecasting for empirical ADL models (planned).

## License

[MIT License](LICENSE).

## Author

Miguel A. Arranz
