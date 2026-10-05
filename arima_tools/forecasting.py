"""Numerical forecasts and innovation-based ARIMA forecast uncertainty."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import arma2ma


def forecast_error_variance(result, steps=20):
    """
    Calculate forecast-error variances on the original dependent-variable scale.

    Parameters
    ----------
    result : statsmodels ARIMAResults
        Fitted statsmodels.tsa.arima.model.ARIMA result.
    steps : int, default 20
        Positive number of out-of-sample forecast horizons.

    Returns
    -------
    numpy.ndarray
        Variances for horizons 1 through `steps`, in that order.

    Notes
    -----
    Expand the full ARIMA transfer function, including ordinary and seasonal
    integration, to obtain psi_0 through psi_(steps-1). Each variance equals
    the estimated innovation variance times the cumulative sum of squared
    psi weights. No Statsmodels forecast variances are used.

    Parameters are treated as fixed. The formula assumes the past innovations
    are known; finite-sample state uncertainty (including missing observations
    or unresolved initialization) can make Statsmodels state-space forecast
    variances differ. Parameter-estimation uncertainty is not included.
    """

    if (
        isinstance(steps, (bool, np.bool_))
        or not isinstance(steps, (int, np.integer))
        or steps <= 0
    ):
        raise ValueError("steps must be a positive integer.")

    if not isinstance(getattr(result, "model", None), ARIMA):
        raise TypeError("result must be a fitted Statsmodels ARIMA result.")

    if result.model.concentrate_scale:
        sigma2 = float(result.scale)
    else:
        sigma2 = float(
            np.asarray(result.params)[result.param_names.index("sigma2")]
        )

    if not np.isfinite(sigma2) or sigma2 < 0:
        raise ValueError("The estimated innovation variance must be finite and nonnegative.")

    # Reduced polynomials already combine nonseasonal and seasonal AR/MA terms,
    # but exclude differencing. Polynomial coefficients are in ascending lags.
    ar = np.asarray(result.polynomial_reduced_ar, dtype=float)
    ma = np.asarray(result.polynomial_reduced_ma, dtype=float)
    d = result.model.order[1]
    _, D, _, period = result.model.seasonal_order

    for _ in range(d):
        ar = np.convolve(ar, [1.0, -1.0])

    if D:
        seasonal_difference = np.zeros(period + 1)
        seasonal_difference[0] = 1.0
        seasonal_difference[-1] = -1.0
        for _ in range(D):
            ar = np.convolve(ar, seasonal_difference)

    psi = arma2ma(ar, ma, lags=int(steps))
    return sigma2 * np.cumsum(np.square(psi))


def dynamic_forecast(result, steps=20, level=0.95):
    """
    Produce recursive forecasts from the end of the observed sample.

    Parameters
    ----------
    result : statsmodels ARIMAResults
        Fitted Statsmodels ARIMA result. Models requiring future external
        regressors cannot be forecast through this interface; deterministic
        trends handled by Statsmodels are supported.
    steps : int, default 20
        Positive number of forecast horizons.
    level : float, default 0.95
        Normal-theory prediction-interval coverage, strictly between 0 and 1.

    Returns
    -------
    pandas.DataFrame
        Index named 'Horizon', running from 1 through `steps`. Columns are
        Forecast, FEV, SE, Lower, and Upper. SE is sqrt(FEV); bounds use the
        requested normal quantile and the independently calculated FEV.

    Notes
    -----
    Point forecasts use the fitted result's forecast method. Uncertainty uses
    forecast_error_variance(), with its fixed-parameter and known-innovation
    assumptions, rather than Statsmodels prediction intervals.
    """

    if (
        isinstance(level, (bool, np.bool_))
        or not isinstance(level, (int, float, np.integer, np.floating))
        or not np.isfinite(level)
        or not 0 < level < 1
    ):
        raise ValueError("level must lie strictly between 0 and 1.")

    fev = forecast_error_variance(result, steps=steps)
    forecasts = np.asarray(result.forecast(steps=int(steps)), dtype=float)
    se = np.sqrt(fev)
    critical = norm.isf((1.0 - level) / 2.0)

    return pd.DataFrame(
        {
            "Forecast": forecasts,
            "FEV": fev,
            "SE": se,
            "Lower": forecasts - critical * se,
            "Upper": forecasts + critical * se,
        },
        index=pd.RangeIndex(1, int(steps) + 1, name="Horizon"),
    )


def _forecast_plot_data(result, steps, history):
    """Join supported model indices without calculating forecast values."""
    if history is not None and (
        isinstance(history, (bool, np.bool_))
        or not isinstance(history, (int, np.integer))
        or history <= 0
    ):
        raise ValueError("history must be a positive integer or None.")

    observed = np.asarray(result.model.endog, dtype=float).reshape(-1)
    index = result.model._index
    if isinstance(index, pd.DatetimeIndex):
        future = pd.date_range(index[-1], periods=steps + 1, freq=index.freq)[1:]
        axis_title = "Time"
    elif isinstance(index, pd.PeriodIndex):
        future = pd.period_range(index[-1], periods=steps + 1, freq=index.freq)[1:]
        # Plotly cannot serialize Period objects. Both sides use period starts.
        index = index.to_timestamp()
        future = future.to_timestamp()
        axis_title = "Time"
    elif isinstance(index, pd.RangeIndex) and not result.model._index_generated:
        future = index[-1] + index.step * np.arange(1, steps + 1)
        axis_title = index.name or "Observation"
    else:
        # Unsupported labels (including irregular dates) have no reliable
        # forecast continuation. Use one coherent positional axis throughout.
        index = pd.RangeIndex(1, len(observed) + 1)
        future = np.arange(len(observed) + 1, len(observed) + steps + 1)
        axis_title = "Observation"

    start = 0 if history is None else max(0, len(observed) - int(history))
    return index[start:], observed[start:], future, axis_title


def _forecast_figure(result, x, observed, axis_title, title):
    """Apply the package's common Plotly layout and mark the forecast origin."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=observed,
        mode="lines+markers" if len(observed) == 1 else "lines",
        name="Observed", marker=dict(size=5),
        line=dict(color="#444444", width=1.5),
    ))
    # A shape avoids Plotly's date arithmetic in annotated add_vline calls.
    fig.add_shape(
        type="line", x0=x[-1], x1=x[-1], y0=0, y1=1,
        xref="x", yref="paper", layer="below",
        line=dict(color="#888888", width=1, dash="dot"),
    )
    fig.add_annotation(
        x=x[-1], y=1, xref="x", yref="paper", text="Forecast origin",
        showarrow=False, yshift=12, font=dict(size=11, color="#666666"),
    )
    fig.update_xaxes(title_text=axis_title, showline=True, linewidth=1)
    fig.update_yaxes(
        title_text=str(result.model.endog_names), showline=True,
        linewidth=1, zeroline=False,
    )
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        template="plotly_white", width=1000, height=550,
        margin=dict(l=80, r=45, t=110, b=90), hovermode="x",
        legend=dict(orientation="h", x=0, y=-0.16, traceorder="normal"),
    )
    return fig


def plot_forecast(result, steps=20, level=0.95, history=50):
    """
    Plot observed values, recursive forecasts, and prediction-interval lines.

    Parameters
    ----------
    result : statsmodels ARIMAResults
        Fitted Statsmodels ARIMA result accepted by dynamic_forecast().
    steps : int, default 20
        Positive number of forecast horizons.
    level : float, default 0.95
        Central prediction-interval coverage, strictly between 0 and 1.
    history : int or None, default 50
        Positive number of final observed values to display, or None for all
        observations. Values larger than the sample display the whole sample.

    Returns
    -------
    plotly.graph_objects.Figure
        Observed and forecast lines, two unfilled interval boundaries, and a
        dotted forecast-origin marker. The function does not display the figure.

    Notes
    -----
    All numerical forecasts and intervals come from dynamic_forecast().
    Supported model dates and numeric indices are continued into the forecast.
    Period indices are displayed at period starts. Unsupported indices use
    observation numbers for both history and forecasts when Statsmodels can
    produce the numerical forecast. Statsmodels forecasting errors propagate.
    The point-forecast
    line starts at the last observation to make the transition visible.
    """
    # Validate history before doing any numerical work.
    if history is not None and (
        isinstance(history, (bool, np.bool_))
        or not isinstance(history, (int, np.integer)) or history <= 0
    ):
        raise ValueError("history must be a positive integer or None.")
    forecast = dynamic_forecast(result, steps=steps, level=level)
    x, observed, future, axis_title = _forecast_plot_data(result, int(steps), history)
    fig = _forecast_figure(result, x, observed, axis_title, "Dynamic Forecast")
    for bound in ("Lower", "Upper"):
        fig.add_trace(go.Scatter(
            x=future, y=forecast[bound].to_numpy(),
            mode="lines+markers" if len(future) == 1 else "lines",
            marker=dict(size=4),
            name=f"{100 * level:g}% prediction interval",
            legendgroup="interval", showlegend=bound == "Lower",
            line=dict(color="#6484A6", width=1.5, dash="dash"),
            meta=dict(level=float(level), bound=bound),
            hovertemplate=f"{bound}: %{{y:.4g}}<extra>{100 * level:g}% PI</extra>",
        ))
    fig.add_trace(go.Scatter(
        x=[x[-1], *future], y=np.r_[observed[-1], forecast["Forecast"]],
        mode="lines", name="Forecast", line=dict(color="#1F5A96", width=2.5),
    ))
    return fig


def plot_fan_chart(
    result,
    steps=20,
    levels=(0.50, 0.70, 0.80, 0.90, 0.95, 0.99),
    history=50,
):
    """
    Plot recursive forecasts with nested central prediction-interval bands.

    Parameters
    ----------
    result : statsmodels ARIMAResults
        Fitted Statsmodels ARIMA result accepted by dynamic_forecast().
    steps : int, default 20
        Positive number of forecast horizons.
    levels : iterable of float, default (0.50, 0.70, 0.80, 0.90, 0.95, 0.99)
        Nonempty collection of distinct central coverages strictly between
        zero and one. Input order does not affect the chart.
    history : int or None, default 50
        Positive number of final observed values to display, or None for all.

    Returns
    -------
    plotly.graph_objects.Figure
        Nested blue shaded bands, strongest centrally and fading outward,
        with observed values and a point forecast drawn over the fan.

    Notes
    -----
    Each interval is obtained from dynamic_forecast(); the plotting code does
    not calculate uncertainty. Bands are drawn widest first and join the last
    observation at horizon zero. Index handling matches plot_forecast().
    A single grouped legend entry represents all intervals; the title lists
    their coverages. The function does not display the figure.
    """
    try:
        levels = list(levels)
    except TypeError as exc:
        raise ValueError("levels must be a nonempty iterable of distinct coverages.") from exc
    if not levels or any(
        isinstance(level, (bool, np.bool_))
        or not isinstance(level, (int, float, np.integer, np.floating))
        or not np.isfinite(level) or not 0 < level < 1
        for level in levels
    ):
        raise ValueError("Every level must lie strictly between 0 and 1.")
    if len(set(levels)) != len(levels):
        raise ValueError("levels must not contain duplicates.")
    if history is not None and (
        isinstance(history, (bool, np.bool_))
        or not isinstance(history, (int, np.integer)) or history <= 0
    ):
        raise ValueError("history must be a positive integer or None.")

    levels = sorted(levels, reverse=True)
    tables = [dynamic_forecast(result, steps=steps, level=level) for level in levels]
    x, observed, future, axis_title = _forecast_plot_data(result, int(steps), history)
    coverages = ", ".join(f"{100 * level:g}%" for level in reversed(levels))
    fig = _forecast_figure(
        result, x, observed, axis_title,
        f"Forecast Fan Chart<br><sup>Central prediction intervals: {coverages}</sup>",
    )
    band_x = [x[-1], *future]
    for i, (level, table) in enumerate(zip(levels, tables)):
        opacity = 0.08 + 0.12 * i / max(1, len(levels) - 1)
        for bound in ("Lower", "Upper"):
            fig.add_trace(go.Scatter(
                x=band_x, y=np.r_[observed[-1], table[bound]], mode="lines",
                name="Prediction intervals", legendgroup="intervals",
                showlegend=i == 0 and bound == "Upper",
                line=dict(width=0),
                fill="tonexty" if bound == "Upper" else None,
                fillcolor=f"rgba(31,90,150,{opacity:.3f})",
                meta=dict(level=float(level), bound=bound),
                hovertemplate=f"{bound}: %{{y:.4g}}<extra>{100 * level:g}% PI</extra>",
            ))
    # Keep the observed series and forecast legible over the shaded fan.
    fig.add_trace(go.Scatter(
        x=band_x, y=np.r_[observed[-1], tables[0]["Forecast"]],
        mode="lines", name="Forecast", line=dict(color="#1F5A96", width=2.5),
    ))
    return fig
