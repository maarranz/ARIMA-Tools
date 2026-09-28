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


__version__ = "0.1.0"

__all__ = [
    "root_diagnostics",
    "is_causal",
    "is_invertible",
    "common_roots",
    "plot_inverse_roots",
    "root_summary",
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



