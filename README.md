# ARIMA-Tools

Python tools for the estimation, diagnostics, model selection, and forecasting of ARIMA models.

The project is being developed primarily for use in **Applied Time Series Econometrics (ATSE)**, but the tools are intended to be useful more generally for teaching and applied time-series analysis.

## Objectives

`ARIMA-Tools` complements existing Python time-series libraries rather than replacing them. In particular, it builds on functionality provided by:

- `statsmodels` for ARIMA estimation, diagnostics, and forecasting;
- `statsforecast` for automatic ARIMA model selection.

The project adds teaching-oriented and applied tools that are not directly available in the form required for the course.

Planned functionality includes:

- AR and MA root diagnostics;
- inverse-root calculations and plots;
- causality and invertibility checks;
- detection of common and near-common AR and MA roots;
- residual diagnostics;
- parameter stability and recursive estimation;
- information-criterion-based model comparison;
- integration with automatic ARIMA selection;
- static one-step-ahead forecasting;
- multi-step forecasting;
- forecasting of transformed variables and reconstruction of forecasts on the original scale.

Forecast evaluation and comparison procedures, such as the Morgan–Granger–Newbold and Diebold–Mariano tests, will be developed separately in **Forecast-Tools** so that they can also be used with forecasts from ADL and other models.

## Requirements

The current development environment uses Python 3.14 and the following principal packages:

- NumPy
- pandas
- SciPy
- Matplotlib
- Plotly
- statsmodels
- StatsForecast
- JupyterLab

The environment can be recreated from `environment.yml`.

```bash
mamba env create -f environment.yml
mamba activate atse_arma_tools
```

## Development status

This project is currently under active development.

The API may change while the initial diagnostic and forecasting functions are being implemented and tested.

## Related projects

- **ARMA-Theory** — theoretical analysis of ARMA models, including theoretical ACF/PACF, causality, invertibility, roots, and impulse responses.
- **ADL-Theory** — theoretical analysis of autoregressive distributed lag models.
- **Forecast-Tools** — forecast evaluation and comparison tools (planned).
- **ADL-Tools** — estimation, diagnostics, causality, and forecasting for empirical ADL models (planned).

## License

MIT License.

## Author

Miguel A. Arranz
