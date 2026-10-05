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
