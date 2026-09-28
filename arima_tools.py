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

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, pacf
from plotly.subplots import make_subplots
from scipy.stats import chi2, norm, jarque_bera, skew, kurtosis

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
