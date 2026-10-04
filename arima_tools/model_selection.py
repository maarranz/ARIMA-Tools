"""Model comparison, order searches, and automatic ARIMA selection."""

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsforecast.models import AutoARIMA
from statsforecast.arima import arima_string

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

