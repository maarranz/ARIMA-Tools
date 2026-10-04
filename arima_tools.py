"""
ARIMA-Tools
===========

Teaching-oriented tools for empirical ARIMA modelling.

This module complements statsmodels and statsforecast with utilities for
diagnostics, model selection, parameter stability, and forecasting.

Developed for Applied Time Series Econometrics (ATSE).

Author
------
Miguel A. Arranz
"""

import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from plotly.subplots import make_subplots
from scipy.stats import chi2, norm, jarque_bera, skew, kurtosis
from statsforecast.models import AutoARIMA
from statsforecast.arima import arima_string

__version__ = "0.2.0"

__all__ = [
    "root_diagnostics",
    "is_causal",
    "is_invertible",
    "common_roots",
    "plot_inverse_roots",
    "root_summary",
    "coefficient_diagnostics",
    "parameter_correlation",
    "joint_significance",
    "ljung_box_test",
    "residual_correlation",
    "plot_residual_diagnostics",
    "jarque_bera_test",
    "dynamic_effects",
    "plot_dynamic_effects",
    "compare_models",
    "select_model",
    "order_search",
    "auto_arima",
    "recursive_estimation",
    "plot_parameter_stability",
    "auto_arima_summary",
]

def root_diagnostics(results):
    """
    Return AR and MA root diagnostics for a fitted statsmodels ARIMA model.

    Parameters
    ----------
    results : statsmodels ARIMAResults
        Fitted ARIMA model results.

    Returns
    -------
    pandas.DataFrame
        Table containing the AR and MA roots, their moduli,
        inverse roots, and inverse-root moduli.

    Notes
    -----
    For the stationary ARMA component of an ARIMA model:

    - AR roots outside the unit circle imply causality/stationarity.
    - MA roots outside the unit circle imply invertibility.
    - Equivalently, the corresponding inverse roots lie inside
      the unit circle.
    """

    rows = []

    for root_type, roots in [
        ("AR", results.arroots),
        ("MA", results.maroots),
    ]:
        for i, root in enumerate(roots, start=1):
            inverse_root = 1 / root

            rows.append(
                {
                    "Type": root_type,
                    "Root": i,
                    "Value": root,
                    "Modulus": np.abs(root),
                    "Inverse Root": inverse_root,
                    "Inverse Modulus": np.abs(inverse_root),
                }
            )

    return pd.DataFrame(rows)

def is_causal(results, tol=1e-10):
    """
    Check whether the AR component of a fitted ARIMA model is causal.

    Parameters
    ----------
    results : statsmodels ARIMAResults
        Fitted ARIMA model results.
    tol : float, default 1e-10
        Numerical tolerance used when comparing root moduli with one.

    Returns
    -------
    bool
        True if all AR roots lie outside the unit circle.

    Notes
    -----
    For an ARIMA(p,d,q) model, this condition refers to the AR
    polynomial of the stationary ARMA component, not to the
    stationarity of the original series when d > 0.

    An AR component is causal when all roots of its AR polynomial
    have modulus greater than one.
    """
    roots = np.asarray(results.arroots)

    if roots.size == 0:
        return True

    return bool(np.all(np.abs(roots) > 1.0 + tol))


def is_invertible(results, tol=1e-10):
    """
    Check whether the MA component of a fitted ARIMA model is invertible.

    Parameters
    ----------
    results : statsmodels ARIMAResults
        Fitted ARIMA model results.
    tol : float, default 1e-10
        Numerical tolerance used when comparing root moduli with one.

    Returns
    -------
    bool
        True if all MA roots lie outside the unit circle.

    Notes
    -----
    An MA component is invertible when all roots of its MA polynomial
    have modulus greater than one.
    """
    roots = np.asarray(results.maroots)

    if roots.size == 0:
        return True

    return bool(np.all(np.abs(roots) > 1.0 + tol))

def common_roots(results, tol=0.05, exact_tol=1e-10):
    """
    Identify common or near-common AR and MA roots.

    Parameters
    ----------
    results : statsmodels ARIMAResults
        Fitted ARIMA model results.
    tol : float, default 0.05
        Maximum distance between an AR inverse root and an MA
        inverse root for them to be classified as near-common.
    exact_tol : float, default 1e-10
        Numerical tolerance used to classify roots as exactly common.

    Returns
    -------
    pandas.DataFrame
        Table containing AR-MA root pairs whose inverse roots are
        within the specified tolerance.

    Notes
    -----
    Common or near-common roots may indicate cancellation between
    the AR and MA polynomials and possible overparameterization.

    Distances are calculated in the complex plane using inverse roots.

    A pair is classified as "Common" when its distance is less than
    or equal to `exact_tol`, and as "Near-common" when its distance
    is greater than `exact_tol` but less than or equal to `tol`.
    """

    ar_roots = np.asarray(results.arroots)
    ma_roots = np.asarray(results.maroots)

    columns = [
        "AR Root",
        "MA Root",
        "AR Inverse Root",
        "MA Inverse Root",
        "Distance",
        "Classification",
    ]

    if ar_roots.size == 0 or ma_roots.size == 0:
        return pd.DataFrame(columns=columns)

    rows = []

    for i, ar_root in enumerate(ar_roots, start=1):
        ar_inverse = 1 / ar_root

        for j, ma_root in enumerate(ma_roots, start=1):
            ma_inverse = 1 / ma_root

            distance = np.abs(ar_inverse - ma_inverse)

            if distance <= tol:
                classification = (
                    "Common"
                    if distance <= exact_tol
                    else "Near-common"
                )

                rows.append(
                    {
                        "AR Root": i,
                        "MA Root": j,
                        "AR Inverse Root": ar_inverse,
                        "MA Inverse Root": ma_inverse,
                        "Distance": distance,
                        "Classification": classification,
                    }
                )

    return pd.DataFrame(rows, columns=columns)

def plot_inverse_roots(
    results,
    tol=0.05,
    exact_tol=1e-10,
    title="Inverse Roots",
):
    """
    Plot the inverse AR and MA roots of a fitted ARIMA model.

    Parameters
    ----------
    results : statsmodels ARIMAResults
        Fitted ARIMA model results.
    tol : float, default 0.05
        Maximum distance between an AR inverse root and an MA
        inverse root for them to be classified as near-common.
    exact_tol : float, default 1e-10
        Numerical tolerance used to classify roots as exactly common.
    title : str, default "Inverse Roots"
        Title of the figure.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive Plotly figure.

    Notes
    -----
    The unit circle is shown as a reference.

    - AR inverse roots are displayed as squares.
    - MA inverse roots are displayed as circles.
    - Common and near-common AR-MA root pairs are marked with crosses.

    For near-common roots, the cross is placed at the midpoint of
    the corresponding AR and MA inverse roots. The original root
    markers remain visible.

    For a causal AR component, all AR inverse roots must lie inside
    the unit circle.

    For an invertible MA component, all MA inverse roots must lie
    inside the unit circle.
    """

    ar_roots = np.asarray(results.arroots)
    ma_roots = np.asarray(results.maroots)

    ar_inverse = 1 / ar_roots if ar_roots.size else np.array([])
    ma_inverse = 1 / ma_roots if ma_roots.size else np.array([])

    fig = go.Figure()

    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 500)

    fig.add_trace(
        go.Scatter(
            x=np.cos(theta),
            y=np.sin(theta),
            mode="lines",
            name="Unit circle",
            hoverinfo="skip",
        )
    )

    # AR inverse roots
    if ar_inverse.size:
        fig.add_trace(
            go.Scatter(
                x=ar_inverse.real,
                y=ar_inverse.imag,
                mode="markers",
                name="AR inverse roots",
                marker=dict(
                    symbol="square",
                    size=11,
                ),
                customdata=np.abs(ar_inverse),
                hovertemplate=(
                    "AR inverse root<br>"
                    "Real: %{x:.4f}<br>"
                    "Imaginary: %{y:.4f}<br>"
                    "Modulus: %{customdata:.4f}"
                    "<extra></extra>"
                ),
            )
        )

    # MA inverse roots
    if ma_inverse.size:
        fig.add_trace(
            go.Scatter(
                x=ma_inverse.real,
                y=ma_inverse.imag,
                mode="markers",
                name="MA inverse roots",
                marker=dict(
                    symbol="circle",
                    size=11,
                ),
                customdata=np.abs(ma_inverse),
                hovertemplate=(
                    "MA inverse root<br>"
                    "Real: %{x:.4f}<br>"
                    "Imaginary: %{y:.4f}<br>"
                    "Modulus: %{customdata:.4f}"
                    "<extra></extra>"
                ),
            )
        )

    # Common and near-common roots
    pairs = common_roots(
        results,
        tol=tol,
        exact_tol=exact_tol,
    )

    if not pairs.empty:
        midpoints = (
            pairs["AR Inverse Root"].to_numpy()
            + pairs["MA Inverse Root"].to_numpy()
        ) / 2

        customdata = np.column_stack(
            [
                pairs["Distance"].to_numpy(),
                pairs["Classification"].to_numpy(),
            ]
        )

        fig.add_trace(
            go.Scatter(
                x=midpoints.real,
                y=midpoints.imag,
                mode="markers",
                name="Common / near-common",
                marker=dict(
                    symbol="x",
                    size=14,
                    line=dict(width=2),
                ),
                customdata=customdata,
                hovertemplate=(
                    "%{customdata[1]} roots<br>"
                    "Real: %{x:.4f}<br>"
                    "Imaginary: %{y:.4f}<br>"
                    "Distance: %{customdata[0]:.4f}"
                    "<extra></extra>"
                ),
            )
        )

    # Real and imaginary axes
    fig.add_hline(y=0, line_width=1)
    fig.add_vline(x=0, line_width=1)

    fig.update_layout(
        title=title,
        xaxis_title="Real",
        yaxis_title="Imaginary",
        width=650,
        height=650,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
    )

    fig.update_xaxes(
        range=[-1.1, 1.1],
        scaleanchor="y",
        scaleratio=1,
        zeroline=False,
    )

    fig.update_yaxes(
        range=[-1.1, 1.1],
        zeroline=False,
    )

    return fig

def root_summary(results, tol=0.05, exact_tol=1e-10):
    """
    Summarize the root properties of a fitted ARIMA model.

    Parameters
    ----------
    results : statsmodels ARIMAResults
        Fitted ARIMA model results.
    tol : float, default 0.05
        Maximum distance between an AR inverse root and an MA
        inverse root for them to be classified as near-common.
    exact_tol : float, default 1e-10
        Numerical tolerance used to classify roots as exactly common.

    Returns
    -------
    pandas.Series
        Summary of causality, invertibility, and common-root
        diagnostics.

    Notes
    -----
    Causality refers to the AR polynomial of the stationary ARMA
    component of the fitted ARIMA model.

    Common and near-common roots may indicate cancellation between
    the AR and MA polynomials and possible overparameterization.
    """

    causal = is_causal(results)
    invertible = is_invertible(results)

    pairs = common_roots(
        results,
        tol=tol,
        exact_tol=exact_tol,
    )

    n_common = 0
    n_near_common = 0
    min_distance = np.nan

    if not pairs.empty:
        n_common = int(
            (pairs["Classification"] == "Common").sum()
        )

        n_near_common = int(
            (pairs["Classification"] == "Near-common").sum()
        )

        min_distance = pairs["Distance"].min()

    return pd.Series(
        {
            "Causal AR component": causal,
            "Invertible MA component": invertible,
            "Common root pairs": n_common,
            "Near-common root pairs": n_near_common,
            "Minimum AR-MA distance": min_distance,
        },
        name="Root diagnostics",
    )

def coefficient_diagnostics(results, confidence=0.95):
    """
    Return coefficient diagnostics for a fitted ARIMA model.

    Parameters
    ----------
    results : statsmodels ARIMAResults
        Fitted ARIMA model results.
    confidence : float, default 0.95
        Confidence level used to construct the confidence intervals.
        Must lie strictly between 0 and 1.

    Returns
    -------
    pandas.DataFrame
        Table containing coefficient estimates, standard errors,
        z-statistics, asymptotic p-values, and confidence intervals.

    Notes
    -----
    The reported z-statistics and p-values use the asymptotic normal
    approximation employed by statsmodels for ARIMA maximum-likelihood
    estimates.

    No automatic significance classification is provided. In dynamic
    models, coefficient relevance should not be assessed mechanically
    using a fixed p-value threshold.
    """

    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1.")

    alpha = 1 - confidence
    conf_int = np.asarray(results.conf_int(alpha=alpha))

    ci_percent = 100 * confidence

    return pd.DataFrame(
        {
            "Coefficient": results.param_names,
            "Estimate": np.asarray(results.params),
            "Std. Error": np.asarray(results.bse),
            "z-statistic": np.asarray(results.tvalues),
            "P-value": np.asarray(results.pvalues),
            f"{ci_percent:g}% CI Lower": conf_int[:, 0],
            f"{ci_percent:g}% CI Upper": conf_int[:, 1],
        }
    )

def parameter_correlation(results, include_variance=False):
    """
    Return the estimated correlation matrix of model parameters.

    Parameters
    ----------
    results : statsmodels ARIMAResults
        Fitted ARIMA model results.
    include_variance : bool, default False
        If True, include the innovation variance parameter (sigma2)
        in the correlation matrix.

    Returns
    -------
    pandas.DataFrame
        Estimated parameter correlation matrix.

    Notes
    -----
    Large correlations between parameter estimates may indicate that
    individual coefficients are imprecisely identified or that the
    model is overparameterized.

    Parameter correlation should be interpreted together with other
    model diagnostics and should not be used mechanically as a model
    selection rule.
    """

    cov = np.asarray(results.cov_params())
    names = np.asarray(results.param_names)

    # Optionally remove sigma2
    if not include_variance:
        keep = names != "sigma2"
        cov = cov[np.ix_(keep, keep)]
        names = names[keep]

    std = np.sqrt(np.diag(cov))

    corr = cov / np.outer(std, std)

    return pd.DataFrame(
        corr,
        index=names,
        columns=names,
    )

def joint_significance(results, parameters):
    """
    Test whether a group of model parameters is jointly equal to zero.

    Parameters
    ----------
    results : statsmodels ARIMAResults
        Fitted ARIMA model results.
    parameters : sequence of str
        Names of the parameters to be tested jointly.

    Returns
    -------
    pandas.Series
        Wald chi-squared statistic, degrees of freedom, and
        asymptotic p-value.

    Notes
    -----
    The null hypothesis is that all selected parameters are jointly
    equal to zero.

    The test is a Wald test based on the estimated covariance matrix
    of the fitted ARIMA model. Under the null hypothesis, the Wald
    statistic has an asymptotic chi-squared distribution with degrees
    of freedom equal to the number of restrictions.
    """

    parameters = list(parameters)
    names = list(results.param_names)

    if len(parameters) == 0:
        raise ValueError("At least one parameter must be specified.")

    unknown = [p for p in parameters if p not in names]

    if unknown:
        raise ValueError(
            "Unknown parameter(s): " + ", ".join(unknown)
        )

    if len(set(parameters)) != len(parameters):
        raise ValueError("Parameter names must not be repeated.")

    indices = [names.index(p) for p in parameters]

    beta = np.asarray(results.params)[indices]
    cov = np.asarray(results.cov_params())[np.ix_(indices, indices)]

    statistic = float(beta @ np.linalg.solve(cov, beta))
    df = len(parameters)


    p_value = float(chi2.sf(statistic, df))

    return pd.Series(
        {
            "Wald statistic": statistic,
            "Degrees of freedom": df,
            "P-value": p_value,
        },
        name="Joint significance test",
    )

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
        Table containing the dynamic effect (psi weight) and the
        cumulative effect at each horizon.

    Notes
    -----
    The dynamic effects are the coefficients of the MA(infinity)
    representation

        y_t = psi(L) * epsilon_t,

    where psi_0 = 1.

    For an integrated ARIMA model, the reported effects refer to the
    stationary ARMA representation associated with the fitted model.
    They should not automatically be interpreted as effects on the
    level of the original integrated variable.

    The cumulative effect is the running sum of the psi weights.
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

def compare_models(*results, names=None):
    """
    Compare fitted ARIMA models using likelihood and information criteria.

    Parameters
    ----------
    *results : statsmodels ARIMAResults
        Two or more fitted ARIMA model results.
    names : sequence of str, optional
        Labels for the models. If omitted, labels are constructed
        from the ARIMA orders.

    Returns
    -------
    pandas.DataFrame
        Model comparison table containing the ARIMA order, number of
        observations, number of estimated parameters, log likelihood,
        AIC, AICc, BIC, and HQIC.

    Notes
    -----
    AICc is calculated as

        AICc = AIC + 2*k*(k + 1) / (n - k - 1),

    where k is the number of estimated parameters and n is the number
    of observations used in estimation.

    Information criteria should only be compared across models
    estimated on the same dependent variable and sample.
    """

    if len(results) < 2:
        raise ValueError("At least two fitted models must be supplied.")

    if names is not None:
        names = list(names)

        if len(names) != len(results):
            raise ValueError(
                "The number of names must equal the number of models."
            )

        if len(set(names)) != len(names):
            raise ValueError("Model names must be unique.")

    rows = []

    for i, result in enumerate(results):

        order = tuple(result.model.order)
        nobs = int(result.nobs)
        k = len(result.params)

        if names is None:
            label = f"ARIMA{order}"
        else:
            label = names[i]

        denominator = nobs - k - 1

        if denominator > 0:
            aicc = (
                result.aic
                + (2 * k * (k + 1)) / denominator
            )
        else:
            aicc = np.nan

        rows.append(
            {
                "Model": label,
                "Order": order,
                "N": nobs,
                "Parameters": k,
                "Log Likelihood": float(result.llf),
                "AIC": float(result.aic),
                "AICc": float(aicc),
                "BIC": float(result.bic),
                "HQIC": float(result.hqic),
            }
        )

    table = pd.DataFrame(rows)

    return table.set_index("Model")

def order_search(
    y,
    p_max=4,
    q_max=4,
    d_min=0,
    d_max=0,
    criterion="BIC",
    trend=None,
):
    """
    Search over a grid of nonseasonal ARIMA(p,d,q) models using
    information criteria.

    Parameters
    ----------
    y : array-like
        Time series to be modeled.

    p_max : int, default 4
        Maximum AR order considered. The search includes
        p = 0, ..., p_max.

    q_max : int, default 4
        Maximum MA order considered. The search includes
        q = 0, ..., q_max.

    d_min : int, default 0
        Minimum order of integration considered.

    d_max : int, default 0
        Maximum order of integration considered. The search includes
        d = d_min, ..., d_max.

    criterion : {"AIC", "AICc", "BIC", "HQIC"}, default "BIC"
        Information criterion used to sort the results.

    trend : str or None, default None
        Trend specification passed to statsmodels ARIMA.

    Returns
    -------
    pandas.DataFrame
        Table containing all candidate models, their information
        criteria, and convergence status.

    Notes
    -----
    The function performs a transparent grid search over the specified
    nonseasonal ARIMA(p,d,q) model space.

    All four information criteria are reported regardless of the
    criterion used for sorting.

    Estimation warnings generated during the grid search are
    suppressed locally so that repeated starting-value and
    convergence warnings do not flood the notebook.

    Actual convergence status is retained in the output table.

    AICc is calculated as

        AICc = AIC + 2*k*(k + 1) / (n - k - 1),

    where k is the number of estimated parameters and n is the
    number of observations used in estimation.

    This function does not perform automatic differencing tests.
    The range of integration orders is explicitly chosen by the user.
    """

    # ------------------------------------------------------------
    # Validate integer inputs
    # ------------------------------------------------------------

    for value, name in [
        (p_max, "p_max"),
        (q_max, "q_max"),
        (d_min, "d_min"),
        (d_max, "d_max"),
    ]:
        if not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(
                f"{name} must be a non-negative integer."
            )

    if d_min > d_max:
        raise ValueError(
            "d_min must be less than or equal to d_max."
        )

    # ------------------------------------------------------------
    # Validate information criterion
    # ------------------------------------------------------------

    criteria = {
        "aic": "AIC",
        "aicc": "AICc",
        "bic": "BIC",
        "hqic": "HQIC",
    }

    key = str(criterion).lower()

    if key not in criteria:
        raise ValueError(
            "criterion must be one of: AIC, AICc, BIC, HQIC."
        )

    criterion_column = criteria[key]

    # ------------------------------------------------------------
    # Estimate candidate models
    # ------------------------------------------------------------

    rows = []

    for d in range(d_min, d_max + 1):

        for p in range(p_max + 1):

            for q in range(q_max + 1):

                order = (p, d, q)

                try:
                    # Suppress repetitive statsmodels warnings only
                    # during estimation of this candidate model.
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")

                        model = ARIMA(
                            y,
                            order=order,
                            trend=trend,
                        )

                        result = model.fit()

                    # ------------------------------------------------
                    # Number of observations and parameters
                    # ------------------------------------------------

                    nobs = int(result.nobs)
                    k = len(result.params)

                    # ------------------------------------------------
                    # AICc
                    # ------------------------------------------------

                    denominator = nobs - k - 1

                    if denominator > 0:
                        aicc = (
                            result.aic
                            + (2 * k * (k + 1))
                            / denominator
                        )
                    else:
                        aicc = np.nan

                    # ------------------------------------------------
                    # Convergence information
                    # ------------------------------------------------

                    converged = bool(
                        result.mle_retvals.get(
                            "converged",
                            True,
                        )
                    )

                    rows.append(
                        {
                            "Order": order,
                            "p": p,
                            "d": d,
                            "q": q,
                            "N": nobs,
                            "Parameters": k,
                            "Log Likelihood": float(
                                result.llf
                            ),
                            "AIC": float(
                                result.aic
                            ),
                            "AICc": float(
                                aicc
                            ),
                            "BIC": float(
                                result.bic
                            ),
                            "HQIC": float(
                                result.hqic
                            ),
                            "Converged": converged,
                            "Status": (
                                "OK"
                                if converged
                                else "Not converged"
                            ),
                        }
                    )

                except Exception as exc:

                    rows.append(
                        {
                            "Order": order,
                            "p": p,
                            "d": d,
                            "q": q,
                            "N": np.nan,
                            "Parameters": np.nan,
                            "Log Likelihood": np.nan,
                            "AIC": np.nan,
                            "AICc": np.nan,
                            "BIC": np.nan,
                            "HQIC": np.nan,
                            "Converged": False,
                            "Status": (
                                f"Failed: "
                                f"{type(exc).__name__}: {exc}"
                            ),
                        }
                    )

    # ------------------------------------------------------------
    # Construct results table
    # ------------------------------------------------------------

    table = pd.DataFrame(rows)

    # ------------------------------------------------------------
    # Sort results
    #
    # Converged models first, ordered by the selected IC.
    # Non-converged and failed models remain visible at the bottom.
    # ------------------------------------------------------------

    table = table.sort_values(
        by=["Converged", criterion_column],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)

    return table




def select_model(table, criterion="BIC"):
    """
    Select the model minimizing a chosen information criterion.

    Parameters
    ----------
    table : pandas.DataFrame
        Model comparison or order-search table.
    criterion : {"AIC", "AICc", "BIC", "HQIC"}, default "BIC"
        Information criterion used for model selection.

    Returns
    -------
    pandas.Series
        Row corresponding to the eligible model with the smallest
        value of the selected information criterion.

    Notes
    -----
    If the table contains a 'Converged' column, non-converged models
    are excluded from selection.

    This function identifies the minimum information criterion. It
    does not imply that the selected model is adequate in terms of
    residual diagnostics, parameter stability, or other model
    properties.
    """

    criteria = {
        "aic": "AIC",
        "aicc": "AICc",
        "bic": "BIC",
        "hqic": "HQIC",
    }

    key = str(criterion).lower()

    if key not in criteria:
        raise ValueError(
            "criterion must be one of: AIC, AICc, BIC, HQIC."
        )

    column = criteria[key]

    if column not in table.columns:
        raise ValueError(
            f"The table does not contain a '{column}' column."
        )

    eligible = table.copy()

    # Exclude non-converged models when convergence information
    # is available.
    if "Converged" in eligible.columns:
        eligible = eligible[eligible["Converged"]]

    # Keep only finite values of the selected criterion.
    eligible = eligible[np.isfinite(eligible[column])]

    if eligible.empty:
        raise ValueError(
            f"No eligible models with finite {column} values "
            "are available."
        )

    model_index = eligible[column].idxmin()

    return table.loc[model_index]


def auto_arima(
    y,
    criterion="BIC",
    d=None,
    max_p=5,
    max_q=5,
    stepwise=True,
    allow_mean=True,
    allow_drift=True,
):
    """
    Select a nonseasonal ARIMA model using StatsForecast AutoARIMA.

    Parameters
    ----------
    y : array-like
        Time series to be modeled.
    criterion : {"AIC", "AICc", "BIC"}, default "BIC"
        Information criterion used by AutoARIMA.
    d : int or None, default None
        Order of nonseasonal differencing. If None, AutoARIMA
        selects the differencing order using its unit-root procedure.
    max_p : int, default 5
        Maximum AR order considered.
    max_q : int, default 5
        Maximum MA order considered.
    stepwise : bool, default True
        If True, use the stepwise AutoARIMA search. If False,
        perform a more exhaustive search.
    allow_mean : bool, default True
        Allow a non-zero mean when appropriate.
    allow_drift : bool, default True
        Allow a drift term when appropriate.

    Returns
    -------
    statsforecast.models.AutoARIMA
        Fitted StatsForecast AutoARIMA model.

    Notes
    -----
    This function provides access to the StatsForecast implementation
    of the Hyndman-Khandakar AutoARIMA procedure.

    The function is deliberately restricted to nonseasonal models.
    It is intended as an automatic model-selection benchmark rather
    than a replacement for specification analysis and diagnostics.
    """

    criteria = {
        "aic": "aic",
        "aicc": "aicc",
        "bic": "bic",
    }

    key = str(criterion).lower()

    if key not in criteria:
        raise ValueError(
            "criterion must be one of: AIC, AICc, BIC."
        )

    if d is not None:
        if not isinstance(d, (int, np.integer)) or d < 0:
            raise ValueError(
                "d must be a non-negative integer or None."
            )

    for value, name in [
        (max_p, "max_p"),
        (max_q, "max_q"),
    ]:
        if not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(
                f"{name} must be a non-negative integer."
            )

    y_array = np.asarray(y, dtype=float)

    if y_array.ndim != 1:
        raise ValueError("y must be one-dimensional.")

    if not np.all(np.isfinite(y_array)):
        raise ValueError("y must contain only finite observations.")

    model = AutoARIMA(
        d=d,
        max_p=max_p,
        max_q=max_q,
        seasonal=False,
        season_length=1,
        ic=criteria[key],
        stepwise=stepwise,
        allowmean=allow_mean,
        allowdrift=allow_drift,
    )

    model.fit(y_array)

    return model

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

def auto_arima_summary(model):
    """
    Summarize a fitted StatsForecast AutoARIMA model.

    Parameters
    ----------
    model : statsforecast.models.AutoARIMA
        Fitted AutoARIMA object returned by auto_arima().

    Returns
    -------
    pandas.Series
        Compact summary of the selected ARIMA specification,
        deterministic component, selection criterion, information
        criteria, likelihood, sample size, and innovation variance.

    Notes
    -----
    The deterministic component is described explicitly to distinguish
    between a mean in a stationary model and drift in an integrated
    model.

    For d = 0, a mean refers to the unconditional mean of the
    stationary ARMA process.

    For d = 1, drift corresponds to a constant in the differenced
    representation and therefore to linear drift in the level of
    the series.
    """

    # ------------------------------------------------------------
    # Check that the AutoARIMA object has been fitted
    # ------------------------------------------------------------

    if not hasattr(model, "model_"):
        raise ValueError(
            "The AutoARIMA model has not been fitted."
        )

    fitted = model.model_

    # ------------------------------------------------------------
    # Obtain StatsForecast model description
    # ------------------------------------------------------------

    description = arima_string(fitted)

    # Examples:
    #
    # ARIMA(2,0,0) with zero mean
    # ARIMA(2,0,0) with non-zero mean
    # ARIMA(1,1,1) with drift
    # ------------------------------------------------------------

    description_lower = description.lower()

    # ------------------------------------------------------------
    # Selected ARIMA specification
    # ------------------------------------------------------------

    if " with " in description:
        selected_model = description.split(" with ")[0]
    else:
        selected_model = description

    # ------------------------------------------------------------
    # Deterministic component
    # ------------------------------------------------------------

    if "with drift" in description_lower:

        deterministic = "Drift"

        interpretation = (
            "Constant in differenced model; "
            "linear drift in level"
        )

    elif "with non-zero mean" in description_lower:

        deterministic = "Mean"

        interpretation = (
            "Non-zero unconditional mean"
        )

    elif "with zero mean" in description_lower:

        deterministic = "None"

        interpretation = (
            "Zero unconditional mean"
        )

    else:

        deterministic = "None"

        interpretation = (
            "No deterministic component reported"
        )

    # ------------------------------------------------------------
    # Selection criterion
    # ------------------------------------------------------------

    criterion = str(model.ic).lower()

    criterion_values = {
        "aic": fitted.get("aic"),
        "aicc": fitted.get("aicc"),
        "bic": fitted.get("bic"),
    }

    criterion_value = criterion_values.get(
        criterion,
        np.nan,
    )

    # ------------------------------------------------------------
    # Helper for safely converting optional values
    # ------------------------------------------------------------

    def _safe_float(value):
        if value is None:
            return np.nan
        return float(value)

    # ------------------------------------------------------------
    # Construct summary
    # ------------------------------------------------------------

    summary = pd.Series(
        {
            "Selected model": selected_model,
            "Deterministic component": deterministic,
            "Interpretation": interpretation,
            "Selection criterion": criterion.upper(),
            "Criterion value": _safe_float(
                criterion_value
            ),
            "AIC": _safe_float(
                fitted.get("aic")
            ),
            "AICc": _safe_float(
                fitted.get("aicc")
            ),
            "BIC": _safe_float(
                fitted.get("bic")
            ),
            "Log Likelihood": _safe_float(
                fitted.get("loglik")
            ),
            "Observations": (
                int(fitted["nobs"])
                if fitted.get("nobs") is not None
                else np.nan
            ),
            "Innovation variance": _safe_float(
                fitted.get("sigma2")
            ),
        },
        name="AutoARIMA Summary",
    )

    return summary


