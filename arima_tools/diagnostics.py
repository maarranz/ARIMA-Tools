"""Residual diagnostics, dynamic effects, and recursive parameter stability."""

import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from plotly.subplots import make_subplots
from scipy.stats import norm, jarque_bera, skew, kurtosis

def ljung_box_test(results, lags):
    """
    Perform Ljung-Box tests for residual autocorrelation.

    Parameters
    ----------
    results : statsmodels ARIMAResults
        Fitted ARIMA model results.
    lags : int or sequence of int
        Lag or lags at which the Ljung-Box test is performed.

    Returns
    -------
    pandas.DataFrame
        Table containing the Ljung-Box statistic, degrees of freedom,
        and p-value for each requested lag.

    Notes
    -----
    The null hypothesis is that the residual autocorrelations up to
    the specified lag are jointly equal to zero.

    The degrees of freedom are adjusted for the number of estimated
    AR and MA parameters:

        df = h - p - q

    where h is the Ljung-Box lag.

    Tests with nonpositive degrees of freedom are not defined and
    therefore cannot be computed.
    """

    if np.isscalar(lags):
        lags = [int(lags)]
    else:
        lags = [int(lag) for lag in lags]

    if any(lag <= 0 for lag in lags):
        raise ValueError("All lags must be positive integers.")

    p = results.model.order[0]
    q = results.model.order[2]
    model_df = p + q

    invalid_lags = [lag for lag in lags if lag <= model_df]

    if invalid_lags:
        raise ValueError(
            f"Ljung-Box lags must be greater than p + q = {model_df}. "
            f"Invalid lag(s): {invalid_lags}"
        )

    lb = acorr_ljungbox(
        results.resid,
        lags=lags,
        model_df=model_df,
        return_df=True,
    )

    table = lb.rename(
        columns={
            "lb_stat": "Ljung-Box statistic",
            "lb_pvalue": "P-value",
        }
    )

    table.index.name = "Lag"
    table["Degrees of freedom"] = table.index - model_df

    return table[
        [
            "Ljung-Box statistic",
            "Degrees of freedom",
            "P-value",
        ]
    ]


def residual_correlations(results, lags=20, confidence=0.95):
    """
    Return residual autocorrelations and partial autocorrelations.

    Parameters
    ----------
    results : statsmodels ARIMAResults
        Fitted ARIMA model results.
    lags : int, default 20
        Maximum lag for the residual ACF and PACF.
    confidence : float, default 0.95
        Confidence level used to construct approximate white-noise
        bounds for the residual correlations.

    Returns
    -------
    pandas.DataFrame
        Table containing residual ACF and PACF values from lag 1
        through the requested maximum lag, together with approximate
        confidence bounds.

    Notes
    -----
    Lag zero is omitted because its ACF and PACF are equal to one
    by construction and are not informative for residual diagnostics.

    The confidence bounds are the approximate white-noise bounds

        +/- z / sqrt(T),

    where z is the appropriate standard Normal quantile and T is the
    number of residual observations used in the calculation.

    These are individual approximate bounds for inspecting the
    correlogram. They are not a joint test of residual autocorrelation.
    The Ljung-Box test should be used for joint testing.
    """

    if not isinstance(lags, (int, np.integer)) or lags <= 0:
        raise ValueError("lags must be a positive integer.")

    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1.")

    residuals = np.asarray(results.resid)
    residuals = residuals[np.isfinite(residuals)]

    nobs = len(residuals)

    if lags >= nobs:
        raise ValueError(
            f"lags must be smaller than the number of residual "
            f"observations ({nobs})."
        )

    acf_values = acf(
        residuals,
        nlags=lags,
        fft=True,
    )

    pacf_values = pacf(
        residuals,
        nlags=lags,
        method="ywm",
    )

    alpha = 1 - confidence
    z = norm.ppf(1 - alpha / 2)
    bound = z / np.sqrt(nobs)

    return pd.DataFrame(
        {
            "Lag": np.arange(1, lags + 1),
            "ACF": acf_values[1:],
            "PACF": pacf_values[1:],
            "Lower Bound": -bound,
            "Upper Bound": bound,
        }
    )


def plot_residual_diagnostics(
    results,
    lags=20,
    confidence=0.95,
    width=1000,
    height=1200,
    title="Residual Diagnostics",
):
    """
    Plot residual diagnostics for a fitted ARIMA model.

    The figure contains four vertically stacked panels:

    1. Residuals over time
    2. Residual autocorrelation function (ACF)
    3. Residual partial autocorrelation function (PACF)
    4. Ljung-Box test p-values by lag

    Parameters
    ----------
    results : statsmodels ARIMAResults
        Fitted ARIMA model results.
    lags : int, default 20
        Maximum lag displayed in the ACF, PACF, and Ljung-Box panels.
    confidence : float, default 0.95
        Confidence level used for the approximate white-noise bounds
        in the ACF and PACF panels.
    width : int, default 1000
        Figure width in pixels.
    height : int, default 1200
        Figure height in pixels.
    title : str, default "Residual Diagnostics"
        Figure title.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive Plotly figure.

    Notes
    -----
    Lag zero is omitted from the ACF and PACF panels because it is
    equal to one by construction and can obscure the structure of
    the remaining correlations.

    The shaded regions in the ACF and PACF panels are approximate
    individual white-noise bounds. They are not joint tests of
    residual autocorrelation.

    The Ljung-Box panel reports p-values only for lags for which the
    adjusted degrees of freedom are positive. A horizontal reference
    line is shown at p = 0.05.
    """

    # ------------------------------------------------------------
    # Numerical diagnostics
    # ------------------------------------------------------------

    correlations = residual_correlations(
        results,
        lags=lags,
        confidence=confidence,
    )

    residuals = pd.Series(results.resid).dropna()

    lag_values = correlations["Lag"].to_numpy()
    acf_values = correlations["ACF"].to_numpy()
    pacf_values = correlations["PACF"].to_numpy()

    lower = correlations["Lower Bound"].iloc[0]
    upper = correlations["Upper Bound"].iloc[0]

    # Ljung-Box tests are defined only when h > p + q.
    p = results.model.order[0]
    q = results.model.order[2]
    first_lb_lag = p + q + 1

    if first_lb_lag <= lags:
        lb_lags = list(range(first_lb_lag, lags + 1))
        lb_table = ljung_box_test(results, lb_lags)

        lb_lag_values = lb_table.index.to_numpy()
        lb_pvalues = lb_table["P-value"].to_numpy()
    else:
        lb_lag_values = np.array([])
        lb_pvalues = np.array([])

    # ------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.06,
        row_heights=[0.30, 0.24, 0.24, 0.22],
        subplot_titles=[
            "Residuals",
            "Autocorrelation Function",
            "Partial Autocorrelation Function",
            "Ljung-Box Test",
        ],
    )

    # ============================================================
    # Panel 1: Residuals
    # ============================================================

    fig.add_trace(
        go.Scatter(
            x=residuals.index,
            y=residuals.to_numpy(),
            mode="lines",
            line=dict(width=1.5),
            hovertemplate=(
                "<b>Observation:</b> %{x}<br>"
                "<b>Residual:</b> %{y:.4f}"
                "<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_hline(
        y=0,
        line_width=1,
        row=1,
        col=1,
    )

    # ============================================================
    # Panel 2: ACF
    # ============================================================

    # Confidence region
    fig.add_hrect(
        y0=lower,
        y1=upper,
        line_width=0,
        opacity=0.12,
        row=2,
        col=1,
    )

    # Stems
    for lag, value in zip(lag_values, acf_values):
        fig.add_shape(
            type="line",
            x0=lag,
            x1=lag,
            y0=0,
            y1=value,
            line=dict(width=2),
            row=2,
            col=1,
        )

    # Markers
    fig.add_trace(
        go.Scatter(
            x=lag_values,
            y=acf_values,
            mode="markers",
            marker=dict(
                size=8,
                symbol="circle",
            ),
            hovertemplate=(
                "<b>Lag:</b> %{x}<br>"
                "<b>ACF:</b> %{y:.4f}"
                "<extra></extra>"
            ),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.add_hline(
        y=0,
        line_width=1,
        row=2,
        col=1,
    )

    # ============================================================
    # Panel 3: PACF
    # ============================================================

    # Confidence region
    fig.add_hrect(
        y0=lower,
        y1=upper,
        line_width=0,
        opacity=0.12,
        row=3,
        col=1,
    )

    # Stems
    for lag, value in zip(lag_values, pacf_values):
        fig.add_shape(
            type="line",
            x0=lag,
            x1=lag,
            y0=0,
            y1=value,
            line=dict(width=2),
            row=3,
            col=1,
        )

    # Markers
    fig.add_trace(
        go.Scatter(
            x=lag_values,
            y=pacf_values,
            mode="markers",
            marker=dict(
                size=8,
                symbol="circle",
            ),
            hovertemplate=(
                "<b>Lag:</b> %{x}<br>"
                "<b>PACF:</b> %{y:.4f}"
                "<extra></extra>"
            ),
            showlegend=False,
        ),
        row=3,
        col=1,
    )

    fig.add_hline(
        y=0,
        line_width=1,
        row=3,
        col=1,
    )

    # ============================================================
    # Panel 4: Ljung-Box p-values
    # ============================================================

    if lb_pvalues.size > 0:
        fig.add_trace(
            go.Scatter(
                x=lb_lag_values,
                y=lb_pvalues,
                mode="lines+markers",
                line=dict(width=2),
                marker=dict(size=7),
                hovertemplate=(
                    "<b>Lag:</b> %{x}<br>"
                    "<b>p-value:</b> %{y:.4f}"
                    "<extra></extra>"
                ),
                showlegend=False,
            ),
            row=4,
            col=1,
        )

    # 5% reference line
    fig.add_hline(
        y=0.05,
        line_dash="dash",
        line_width=1.5,
        annotation_text="0.05",
        annotation_position="top right",
        row=4,
        col=1,
    )

    # ============================================================
    # Axes
    # ============================================================

    fig.update_xaxes(
        title_text="Observation",
        showline=True,
        linewidth=1,
        row=1,
        col=1,
    )

    fig.update_yaxes(
        title_text="Residual",
        showline=True,
        linewidth=1,
        zeroline=False,
        row=1,
        col=1,
    )

    # ACF and PACF axes
    for row in (2, 3):
        fig.update_xaxes(
            title_text="Lag",
            dtick=1,
            range=[0.5, lags + 0.5],
            showline=True,
            linewidth=1,
            row=row,
            col=1,
        )

        fig.update_yaxes(
            title_text="Correlation",
            showline=True,
            linewidth=1,
            zeroline=False,
            row=row,
            col=1,
        )

    # Separate adaptive symmetric scales for ACF and PACF.
    for row, values in (
        (2, acf_values),
        (3, pacf_values),
    ):
        ymax = max(
            np.max(np.abs(values)),
            abs(lower),
            abs(upper),
        )

        ymax = min(
            1.0,
            max(0.25, 1.15 * ymax),
        )

        fig.update_yaxes(
            range=[-ymax, ymax],
            row=row,
            col=1,
        )

    # Ljung-Box axes
    fig.update_xaxes(
        title_text="Lag",
        dtick=1,
        range=[0.5, lags + 0.5],
        showline=True,
        linewidth=1,
        row=4,
        col=1,
    )

    fig.update_yaxes(
        title_text="p-value",
        range=[0, 1],
        showline=True,
        linewidth=1,
        zeroline=False,
        row=4,
        col=1,
    )

    # ============================================================
    # Overall layout
    # ============================================================

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
        ),
        template="plotly_white",
        width=width,
        height=height,
        margin=dict(
            l=80,
            r=45,
            t=100,
            b=65,
        ),
        hovermode="closest",
        showlegend=False,
    )

    return fig


def jarque_bera_test(results):
    """
    Perform the Jarque-Bera test of residual normality.

    Parameters
    ----------
    results : statsmodels ARIMAResults
        Fitted ARIMA model results.

    Returns
    -------
    pandas.Series
        Jarque-Bera statistic, degrees of freedom, p-value,
        residual skewness, and residual kurtosis.

    Notes
    -----
    The null hypothesis is that the residuals are normally distributed.

    The Jarque-Bera test is based on the sample skewness and kurtosis
    of the residuals. Under the null hypothesis, the test statistic
    has an asymptotic chi-squared distribution with two degrees of
    freedom.

    Skewness and kurtosis are reported alongside the test statistic
    to help identify the source of departures from normality.
    Kurtosis is reported in the conventional form for which the
    Normal distribution has kurtosis equal to 3.
    """

    residuals = np.asarray(results.resid)
    residuals = residuals[np.isfinite(residuals)]

    if residuals.size < 3:
        raise ValueError(
            "At least three finite residual observations are required."
        )

    jb = jarque_bera(residuals)

    residual_skewness = skew(
        residuals,
        bias=False,
    )

    residual_kurtosis = kurtosis(
        residuals,
        fisher=False,
        bias=False,
    )

    return pd.Series(
        {
            "Jarque-Bera statistic": float(jb.statistic),
            "Degrees of freedom": 2,
            "P-value": float(jb.pvalue),
            "Skewness": float(residual_skewness),
            "Kurtosis": float(residual_kurtosis),
        },
        name="Jarque-Bera test",
    )


def dynamic_effects(results, horizon=20):
    """
    Compute dynamic effects of a one-unit innovation for a fitted
    ARMA/ARIMA model.

    Parameters
    ----------
    results : statsmodels ARIMAResults
        Fitted ARIMA model results.
    horizon : int, default 20
        Maximum response horizon. The table contains effects from
        horizon 0 through `horizon`.

    Returns
    -------
    pandas.DataFrame
        Table containing 'Horizon', 'Dynamic Effect', and 'Cumulative Effect'.
        'Dynamic Effect' reports the impulse response of the fitted model's
        original endogenous/dependent variable to a one-unit innovation.
        'Cumulative Effect' is the cumulative sum of those dynamic effects.

    Notes
    -----
    The impulse responses are obtained from the fitted Statsmodels result.
    For integrated ARIMA models, they incorporate the integration structure
    and need not decay to zero.

    For integrated models, 'Cumulative Effect' sums responses already
    expressed on the original dependent-variable scale. It is not a
    transformation used to reconstruct levels from differenced responses.
    """

    if not isinstance(horizon, (int, np.integer)) or horizon < 0:
        raise ValueError("horizon must be a non-negative integer.")

    # statsmodels impulse responses:
    # steps=0 returns psi_0 only,
    # steps=horizon returns psi_0,...,psi_horizon.
    effects = np.asarray(
        results.impulse_responses(steps=horizon)
    )

    cumulative = np.cumsum(effects)

    return pd.DataFrame(
        {
            "Horizon": np.arange(horizon + 1),
            "Dynamic Effect": effects,
            "Cumulative Effect": cumulative,
        }
    )


def plot_dynamic_effects(
    results,
    horizon=20,
    width=900,
    height=700,
    title="Dynamic Effects of an Innovation",
):
    """
    Plot dynamic and cumulative effects of a one-unit innovation.

    Parameters
    ----------
    results : statsmodels ARIMAResults
        Fitted ARIMA model results.
    horizon : int, default 20
        Maximum response horizon.
    width : int, default 900
        Figure width in pixels.
    height : int, default 700
        Figure height in pixels.
    title : str, default "Dynamic Effects of an Innovation"
        Figure title.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive Plotly figure.

    Notes
    -----
    The upper panel shows the individual psi weights. The lower
    panel shows their cumulative sum.

    For integrated ARIMA models, these effects refer to the stationary
    ARMA representation and are not automatically effects on the
    level of the original variable.
    """

    table = dynamic_effects(
        results,
        horizon=horizon,
    )

    h = table["Horizon"].to_numpy()
    effects = table["Dynamic Effect"].to_numpy()
    cumulative = table["Cumulative Effect"].to_numpy()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        row_heights=[0.50, 0.50],
        subplot_titles=[
            "Dynamic Effects",
            "Cumulative Effects",
        ],
    )

    # ------------------------------------------------------------
    # Dynamic effects
    # ------------------------------------------------------------

    for lag, value in zip(h, effects):
        fig.add_shape(
            type="line",
            x0=lag,
            x1=lag,
            y0=0,
            y1=value,
            line=dict(width=2),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=h,
            y=effects,
            mode="markers",
            marker=dict(
                size=8,
                symbol="circle",
            ),
            hovertemplate=(
                "<b>Horizon:</b> %{x}<br>"
                "<b>Dynamic effect:</b> %{y:.5f}"
                "<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_hline(
        y=0,
        line_width=1,
        row=1,
        col=1,
    )

    # ------------------------------------------------------------
    # Cumulative effects
    # ------------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=h,
            y=cumulative,
            mode="lines+markers",
            line=dict(width=2),
            marker=dict(size=6),
            hovertemplate=(
                "<b>Horizon:</b> %{x}<br>"
                "<b>Cumulative effect:</b> %{y:.5f}"
                "<extra></extra>"
            ),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.add_hline(
        y=0,
        line_width=1,
        row=2,
        col=1,
    )

    # ------------------------------------------------------------
    # Axes
    # ------------------------------------------------------------

    fig.update_xaxes(
        dtick=1,
        range=[-0.5, horizon + 0.5],
        showline=True,
        linewidth=1,
        row=1,
        col=1,
    )

    fig.update_xaxes(
        title_text="Horizon",
        dtick=1,
        range=[-0.5, horizon + 0.5],
        showline=True,
        linewidth=1,
        row=2,
        col=1,
    )

    fig.update_yaxes(
        title_text="Effect",
        showline=True,
        linewidth=1,
        zeroline=False,
        row=1,
        col=1,
    )

    fig.update_yaxes(
        title_text="Cumulative effect",
        showline=True,
        linewidth=1,
        zeroline=False,
        row=2,
        col=1,
    )

    # ------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
        ),
        template="plotly_white",
        width=width,
        height=height,
        margin=dict(
            l=80,
            r=45,
            t=95,
            b=65,
        ),
        hovermode="closest",
        showlegend=False,
    )

    return fig


def recursive_estimation(
    y,
    order,
    initial=None,
    min_df=20,
    confidence=0.95,
    trend=None,
):
    """
    Perform recursive expanding-window estimation of an ARIMA model.

    Parameters
    ----------
    y : array-like
        Time series to be modeled. A pandas Series is recommended
        when dates or other meaningful index values are available.
    order : tuple of int
        ARIMA order (p, d, q).
    initial : None, float, or int, default None
        Initial estimation window.

        If None, the initial window is chosen automatically as the
        larger of 10 percent of the sample and the window required
        to provide at least `min_df` residual degrees of freedom.

        If a float between 0 and 1, it is interpreted as a fraction
        of the full sample.

        If an integer, it is interpreted as the number of observations
        in the initial estimation window.
    min_df : int, default 20
        Minimum residual degrees of freedom used by the automatic
        initial-window rule.
    confidence : float, default 0.95
        Confidence level for the parameter confidence intervals.
    trend : str or None, default None
        Trend specification passed to statsmodels ARIMA.

    Returns
    -------
    pandas.DataFrame
        Long-format table containing the endpoint of each recursive
        sample, parameter estimates, standard errors, confidence
        intervals, and convergence status.

    Notes
    -----
    Estimation uses an expanding window. Starting from the initial
    sample, one observation is added at each step until the full
    sample is reached.

    Routine statsmodels estimation warnings are suppressed locally.
    Actual convergence status is retained in the output.

    The automatic initial window uses a 10 percent starting fraction
    subject to a degrees-of-freedom safeguard.
    """

    # ------------------------------------------------------------
    # Validate ARIMA order
    # ------------------------------------------------------------

    if (
        not isinstance(order, (tuple, list))
        or len(order) != 3
    ):
        raise ValueError(
            "order must be a tuple (p, d, q)."
        )

    p, d, q = order

    for value, name in zip(order, ["p", "d", "q"]):
        if not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(
                f"{name} must be a non-negative integer."
            )

    # ------------------------------------------------------------
    # Prepare data
    # ------------------------------------------------------------

    if isinstance(y, pd.Series):
        y_series = y.dropna().copy()
    else:
        y_array = np.asarray(y, dtype=float)

        if y_array.ndim != 1:
            raise ValueError("y must be one-dimensional.")

        y_series = pd.Series(y_array).dropna()

    if not np.all(np.isfinite(y_series.to_numpy(dtype=float))):
        raise ValueError(
            "y must contain only finite observations."
        )

    n = len(y_series)

    # ------------------------------------------------------------
    # Validate other arguments
    # ------------------------------------------------------------

    if not isinstance(min_df, (int, np.integer)) or min_df < 1:
        raise ValueError(
            "min_df must be a positive integer."
        )

    if not 0 < confidence < 1:
        raise ValueError(
            "confidence must lie strictly between 0 and 1."
        )

    # ------------------------------------------------------------
    # Approximate number of estimated parameters
    #
    # AR coefficients       : p
    # MA coefficients       : q
    # innovation variance   : 1
    # deterministic term    : depends on trend
    # ------------------------------------------------------------

    k = p + q + 1

    if trend is not None and trend != "n":
        k += 1

    # statsmodels default trend:
    # d = 0 -> constant
    # d > 0 -> no constant
    if trend is None and d == 0:
        k += 1

    # Differencing reduces the effective information available
    # in the initial sample.
    df_required_window = k + min_df + d

    # ------------------------------------------------------------
    # Determine initial window
    # ------------------------------------------------------------

    if initial is None:

        initial_n = max(
            int(np.floor(0.10 * n)),
            df_required_window,
        )

    elif isinstance(initial, (float, np.floating)):

        if not 0 < initial < 1:
            raise ValueError(
                "When initial is a float, it must lie "
                "strictly between 0 and 1."
            )

        initial_n = int(np.floor(initial * n))

    elif isinstance(initial, (int, np.integer)):

        if initial < 1:
            raise ValueError(
                "initial must be a positive integer."
            )

        initial_n = int(initial)

    else:
        raise TypeError(
            "initial must be None, a fraction between 0 and 1, "
            "or a positive integer."
        )

    # ------------------------------------------------------------
    # Feasibility checks
    # ------------------------------------------------------------

    if initial_n >= n:
        raise ValueError(
            "The initial estimation window is too large for the "
            "available sample. No recursive estimation period remains."
        )

    # Explicit choices are allowed, but they must at least leave
    # positive approximate residual degrees of freedom.
    approximate_df = initial_n - d - k

    if approximate_df <= 0:
        raise ValueError(
            "The initial estimation window is too small for the "
            f"ARIMA{tuple(order)} specification. "
            f"Approximate residual degrees of freedom: "
            f"{approximate_df}."
        )

    # ------------------------------------------------------------
    # Critical value
    # ------------------------------------------------------------

    alpha = 1.0 - confidence
    critical = norm.ppf(1.0 - alpha / 2.0)

    # ------------------------------------------------------------
    # Recursive estimation
    # ------------------------------------------------------------

    rows = []

    for end in range(initial_n, n + 1):

        sample = y_series.iloc[:end]
        endpoint = sample.index[-1]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                model = ARIMA(
                    sample,
                    order=tuple(order),
                    trend=trend,
                )

                result = model.fit()

            converged = bool(
                result.mle_retvals.get("converged", True)
            )

            params = pd.Series(
                np.asarray(result.params),
                index=result.param_names,
            )

            bse = pd.Series(
                np.asarray(result.bse),
                index=result.param_names,
            )

            for parameter in result.param_names:

                estimate = float(params[parameter])
                std_error = float(bse[parameter])

                rows.append(
                    {
                        "Endpoint": endpoint,
                        "N": end,
                        "Parameter": parameter,
                        "Estimate": estimate,
                        "Std. Error": std_error,
                        "Lower CI": (
                            estimate - critical * std_error
                        ),
                        "Upper CI": (
                            estimate + critical * std_error
                        ),
                        "Converged": converged,
                    }
                )

        except Exception:
            # Keep the recursive endpoint visible even when
            # estimation fails.
            rows.append(
                {
                    "Endpoint": endpoint,
                    "N": end,
                    "Parameter": np.nan,
                    "Estimate": np.nan,
                    "Std. Error": np.nan,
                    "Lower CI": np.nan,
                    "Upper CI": np.nan,
                    "Converged": False,
                }
            )

    return pd.DataFrame(rows)


def plot_parameter_stability(
    recursive,
    parameters=None,
    reference=True,
    breakpoint=None,
    width=950,
    height_per_panel=300,
    title="Recursive Parameter Stability",
):
    """
    Plot recursive parameter estimates and confidence intervals.

    Parameters
    ----------
    recursive : pandas.DataFrame
        Long-format table produced by recursive_estimation().

    parameters : str, sequence of str, or None, default None
        Parameter or parameters to plot.

        If a string is supplied, one parameter is plotted.

        If a sequence is supplied, one vertically stacked subplot
        is created for each selected parameter.

        If None, all estimated parameters except 'sigma2' are plotted.

    reference : bool, default True
        If True, show the full-sample estimate as a horizontal
        reference line in each subplot.

    breakpoint : optional, default None
        Known or hypothesized structural-break point.

        If supplied, a vertical dashed line is added at this endpoint
        in every subplot. The value should be compatible with the
        Endpoint index used in the recursive-estimation table.

        If None, no breakpoint line is shown.

    width : int, default 950
        Figure width in pixels.

    height_per_panel : int, default 300
        Approximate height of each parameter subplot.

    title : str, default "Recursive Parameter Stability"
        Figure title.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive Plotly figure containing one subplot for each
        selected parameter.

    Notes
    -----
    The recursive estimate is shown together with its confidence band.

    The final recursive estimate corresponds to the full-sample
    estimate and is optionally displayed as a horizontal reference
    line.

    The innovation variance ('sigma2') is excluded by default but
    can be requested explicitly.

    A breakpoint is purely a graphical reference. It does not affect
    estimation or constitute a formal structural-break test.
    """

    # ------------------------------------------------------------
    # Validate input table
    # ------------------------------------------------------------

    required_columns = {
        "Endpoint",
        "N",
        "Parameter",
        "Estimate",
        "Lower CI",
        "Upper CI",
        "Converged",
    }

    missing = required_columns.difference(recursive.columns)

    if missing:
        raise ValueError(
            "recursive is missing required columns: "
            + ", ".join(sorted(missing))
        )

    # ------------------------------------------------------------
    # Available parameters
    # ------------------------------------------------------------

    available = [
        p
        for p in recursive["Parameter"].dropna().unique()
    ]

    if not available:
        raise ValueError(
            "No parameter estimates are available for plotting."
        )

    # ------------------------------------------------------------
    # Select parameters
    # ------------------------------------------------------------

    if parameters is None:

        selected = [
            p for p in available
            if p != "sigma2"
        ]

    elif isinstance(parameters, str):

        selected = [parameters]

    else:

        selected = list(parameters)

    if not selected:
        raise ValueError(
            "No parameters have been selected for plotting."
        )

    unknown = [
        p for p in selected
        if p not in available
    ]

    if unknown:
        raise ValueError(
            "Unknown parameter(s): "
            + ", ".join(unknown)
            + ". Available parameters are: "
            + ", ".join(available)
        )

    # Avoid accidental duplicate panels
    selected = list(dict.fromkeys(selected))

    n_parameters = len(selected)

    # ------------------------------------------------------------
    # Create figure
    # ------------------------------------------------------------

    fig = make_subplots(
        rows=n_parameters,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=min(
            0.08,
            0.20 / n_parameters,
        ),
        subplot_titles=selected,
    )

    # ------------------------------------------------------------
    # Add each parameter
    # ------------------------------------------------------------

    for row, parameter in enumerate(selected, start=1):

        data = (
            recursive[
                recursive["Parameter"] == parameter
            ]
            .sort_values("N")
            .copy()
        )

        x = data["Endpoint"]
        estimate = data["Estimate"]
        lower = data["Lower CI"]
        upper = data["Upper CI"]

        # --------------------------------------------------------
        # Confidence band
        # --------------------------------------------------------

        fig.add_trace(
            go.Scatter(
                x=x,
                y=lower,
                mode="lines",
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=upper,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(120,120,120,0.20)",
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=1,
        )

        # --------------------------------------------------------
        # Recursive estimate
        # --------------------------------------------------------

        fig.add_trace(
            go.Scatter(
                x=x,
                y=estimate,
                mode="lines",
                line=dict(width=2),
                name="Recursive estimate",
                hovertemplate=(
                    "<b>Endpoint:</b> %{x}<br>"
                    "<b>Estimate:</b> %{y:.5f}"
                    "<extra></extra>"
                ),
                showlegend=(row == 1),
            ),
            row=row,
            col=1,
        )

        # --------------------------------------------------------
        # Full-sample estimate
        # --------------------------------------------------------

        if reference:

            valid = data[
                data["Estimate"].notna()
            ]

            if not valid.empty:

                full_sample = float(
                    valid.iloc[-1]["Estimate"]
                )

                fig.add_hline(
                    y=full_sample,
                    line_dash="dash",
                    line_width=1.5,
                    annotation_text=(
                        f"Full sample: {full_sample:.4f}"
                    ),
                    annotation_position="top right",
                    row=row,
                    col=1,
                )

        # --------------------------------------------------------
        # Structural-break reference
        # --------------------------------------------------------

        if breakpoint is not None:

            fig.add_vline(
                x=breakpoint,
                line_dash="dash",
                line_width=1.5,
                row=row,
                col=1,
            )

        # --------------------------------------------------------
        # Axis formatting
        # --------------------------------------------------------

        fig.update_yaxes(
            title_text="Estimate",
            showline=True,
            linewidth=1,
            zeroline=False,
            row=row,
            col=1,
        )

        fig.update_xaxes(
            showline=True,
            linewidth=1,
            row=row,
            col=1,
        )

    # ------------------------------------------------------------
    # X-axis title only on final subplot
    # ------------------------------------------------------------

    fig.update_xaxes(
        title_text="Recursive sample endpoint",
        row=n_parameters,
        col=1,
    )

    # ------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------

    height = max(
        450,
        height_per_panel * n_parameters,
    )

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
        ),
        template="plotly_white",
        width=width,
        height=height,
        margin=dict(
            l=80,
            r=55,
            t=100,
            b=70,
        ),
        hovermode="x",
        showlegend=True,
    )

    return fig

