"""Model comparison, order searches, and automatic ARIMA selection."""

import warnings
import hashlib
import pickle
from collections.abc import Mapping
from statsmodels.tsa.arima.specification import SARIMAXSpecification
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsforecast.models import AutoARIMA
from statsforecast.arima import arima_string

def _initialization_info(result):
    """Describe fitted initialization without comparing stationary state sizes.

    Only the standard ARIMA state ordering is recognized. Known/custom state
    priors have no automatic cross-specification equivalence rule and are reported
    but excluded from selection. Initial moments come from the filter snapshot,
    not a parameter-dependent stationary covariance or a guessed default.
    """
    unsupported = {"Initialization": None,
                   "Initialization Status": "Unsupported or unavailable"}
    try:
        model = result.model
        if not isinstance(model, ARIMA) or model.hamilton_representation or model.simple_differencing:
            return unsupported
        fitted = result.filter_results
        init = fitted.initialization
        size = model.k_states
        integration = model._k_states_diff
        regimes = [None] * size
        configured_variances = [None] * size

        def visit(block, states):
            kind = block.initialization_type
            if kind is None:
                for positions, child in block.blocks.items():
                    visit(child, tuple(states[int(i)] for i in positions))
            else:
                for state in states:
                    if regimes[state] is not None:
                        raise ValueError("Overlapping initialization blocks")
                    regimes[state] = kind
                    if kind == "approximate_diffuse":
                        configured_variances[state] = float(block.approximate_diffuse_variance)

        visit(init, tuple(range(size)))
        mean = np.asarray(fitted.initial_state, dtype=float)
        covariance = np.asarray(fitted.initial_state_cov, dtype=float)
        diffuse = (np.zeros_like(covariance) if fitted.initial_diffuse_state_cov is None
                   else np.asarray(fitted.initial_diffuse_state_cov, dtype=float))
        if not (np.isfinite(mean).all() and np.isfinite(covariance).all()
                and np.isfinite(diffuse).all()):
            return unsupported
        detail = {"Regimes": tuple(regimes), "Initial Mean": tuple(mean),
                  "Initial Covariance": tuple(map(tuple, covariance)),
                  "Initial Diffuse Covariance": tuple(map(tuple, diffuse)),
                  "Scale Concentrated": bool(fitted.filter_concentrated)}
        if any(kind not in {"stationary", "diffuse", "approximate_diffuse"}
               for kind in regimes):
            return {**unsupported, "Initialization Detail": detail}
        profile = []
        for i, kind in enumerate(regimes):
            if kind == "stationary":
                profile.append((kind,))
                continue
            # Supported diffuse blocks are independent with diagonal priors.
            # Retain the fitted mean and variance even when burn-in is positive.
            others = [j for j in range(size) if j != i]
            if (np.any(covariance[i, others] != 0)
                    or np.any(diffuse[i, others] != 0)):
                return unsupported
            variance = float(covariance[i, i])
            # Concentrated filtering stores moments in fitted scale units;
            # compare the initializer's variance convention before that scaling.
            if fitted.filter_concentrated:
                if not np.isfinite(fitted.scale) or fitted.scale <= 0:
                    return unsupported
                variance /= float(fitted.scale)
            if kind == "approximate_diffuse":
                configured = configured_variances[i]
                if (variance <= 0 or diffuse[i, i] != 0
                        or not np.isclose(variance, configured, rtol=1e-12, atol=0)):
                    return unsupported
                # Use the exact configured value after checking the snapshot;
                # scale multiplication/division can introduce rounding noise.
                variance = configured
            if kind == "diffuse" and (diffuse[i, i] != 1 or variance != 0):
                return unsupported
            entry = (kind, float(mean[i]), variance)
            if kind == "approximate_diffuse":
                # Relative and absolute diffuse variances are different priors,
                # even if both have the same numeric initializer setting.
                entry += ("relative to scale" if fitted.filter_concentrated else "absolute",)
            profile.append(entry)
        integrated = tuple(profile[:integration])
        arma = tuple(profile[integration:])
        # Stationary ARMA states may have different dimensions across p and q.
        # Diffuse ARMA states retain dimension because their likelihood
        # normalization changes with the number of diffuse directions.
        if arma and all(state == ("stationary",) for state in arma):
            arma = (("stationary",),)
        return {"Initialization": (integrated, arma),
                "Initialization Status": "Supported",
                "Initialization Detail": detail}
    except (AttributeError, TypeError, ValueError, IndexError, KeyError):
        return unsupported


def _sample_info(result):
    """Fingerprint ordered response, missing positions and original time labels."""
    response = result.model.data.orig_endog
    if response is None:
        raise ValueError("The estimation sample cannot be recovered.")
    values = np.asarray(response, dtype=float).reshape(-1).copy()
    values[np.isnan(values)] = np.nan
    index = result.model.data.row_labels
    # Include index class, frequency and timezone, not response/column names.
    labels = None if index is None else (
        type(index).__name__, str(index.dtype), getattr(index, "freqstr", None),
        str(getattr(index, "tz", None)), tuple(index.tolist()))
    identity = (values.shape, values.tobytes(), labels,
                tuple(getattr(result.model.data, "missing_row_idx", [])))
    sample_id = hashlib.sha256(pickle.dumps(identity, protocol=4)).hexdigest()
    burn = int(result.loglikelihood_burn)
    return {"Sample ID": sample_id, "Start": index[0] if index is not None else 0,
            "End": index[-1] if index is not None else len(values) - 1,
            "Burn": burn, "Diffuse": int(result.nobs_diffuse),
            "Seasonal Differencing": (result.model.seasonal_order[1],
                                       result.model.seasonal_order[3])}


def _ic_values(result):
    # Statsmodels counts free parameters, concentrated scale and diffuse states.
    # Use its likelihood convention consistently for all four criteria.
    n, k = int(result.nobs_effective), int(result.df_model)
    denominator = n - k - 1
    return {"N": n, "Parameters": k, "Log Likelihood": float(result.llf),
            "AIC": float(result.aic),
            "AICc": float(result.aic + 2*k*(k+1)/denominator) if denominator > 0 else np.nan,
            "BIC": float(result.bic), "HQIC": float(result.hqic)}


def _convergence_info(result):
    """Return explicit convergence evidence, N/A, or unknown conservatively."""
    evidence = []

    def flag(value):
        return bool(value) if isinstance(value, (bool, np.bool_)) else None

    retvals = getattr(result, "mle_retvals", None)
    if isinstance(retvals, Mapping):
        evidence.append(flag(retvals.get("converged")))
    details = getattr(result, "fit_details", None)

    def inspect_details(details):
        if not isinstance(details, Mapping):
            return
        if "minimize_results" in details:
            evidence.append(flag(getattr(details["minimize_results"], "success", None)))
        if "converged" in details:
            evidence.append(flag(details["converged"]))
        if "statespace_results" in details:
            inner_retvals = getattr(details["statespace_results"], "mle_retvals", None)
            evidence.append(flag(inner_retvals.get("converged"))
                            if isinstance(inner_retvals, Mapping) else None)
        # GLS may contain inner innovations-MLE or state-space optimizers.
        for inner in details.get("arma_results", []):
            inspect_details(inner)
        if "mle_retvals" in details:
            inner = details["mle_retvals"]
            evidence.append(flag(inner.get("converged")) if isinstance(inner, Mapping) else None)

    inspect_details(details)
    if evidence:
        if False in evidence:
            return False, "Not converged"
        if None in evidence:
            return None, "Unknown"
        return True, "Converged"
    # These are the documented ancillary-result shapes for Yule-Walker,
    # Burg, innovations and Hannan-Rissanen: no optimizer is involved.
    if (isinstance(details, Mapping)
            and isinstance(details.get("spec"), SARIMAXSpecification)
            and set(details) in ({"spec"}, {"spec", "initial_ar_order", "resid"})):
        return None, "Not applicable (non-optimizer estimator)"
    return None, "Unknown"


def _converged(result):
    return _convergence_info(result)[0]



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

    Comparison reports models without choosing among them. Sample identity,
    differencing, convergence and likelihood metadata let select_model enforce
    comparability. N is Statsmodels nobs_effective and Parameters is df_model
    (free parameters plus concentrated scale and exact diffuse state terms).
    Initialization signatures preserve standard ARIMA integration conventions
    while allowing different stationary state dimensions. Known/custom priors
    without an equivalence rule are reported as unsupported. Convergence Status
    distinguishes optimizer success/failure, non-optimizer estimators and unknown
    evidence. Numerical differences from R likelihood initialization are expected.
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
        label = f"ARIMA{order}" if names is None else names[i]
        converged, convergence_status = _convergence_info(result)
        rows.append({"Model": label, "Order": order, "d": order[1],
                     **_ic_values(result), **_sample_info(result),
                     **_initialization_info(result),
                     "Converged": converged,
                     "Convergence Status": convergence_status})

    if len({row["Model"] for row in rows}) != len(rows):
        raise ValueError("Model names must be unique; supply names for repeated orders.")

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
    criterion used for sorting within each differencing order.

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
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
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

                caught = []
                try:
                    # Suppress repetitive statsmodels warnings only
                    # during estimation of this candidate model.
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")

                        model = ARIMA(
                            y,
                            order=order,
                            trend=trend,
                        )

                        result = model.fit()

                    # ------------------------------------------------
                    # Number of observations and parameters
                    # ------------------------------------------------

                    ic = _ic_values(result)
                    converged, convergence_status = _convergence_info(result)

                    rows.append(
                        {
                            "Order": order,
                            "p": p,
                            "d": d,
                            "q": q,
                            **ic,
                            **_sample_info(result),
                            **_initialization_info(result),
                            "Convergence Status": convergence_status,
                            "Warnings": tuple(str(w.message) for w in caught),
                            "Converged": converged,
                            "Status": (
                                "OK"
                                if converged
                                else convergence_status
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
                            "Warnings": tuple(str(w.message) for w in caught),
                            "Converged": False,
                            "Convergence Status": "Fit failed",
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
    # Group by d; converged models first within each group.
    # Non-converged and failed models remain visible at the bottom.
    # ------------------------------------------------------------

    table = table.sort_values(
        by=["d", "Converged", criterion_column],
        ascending=[True, False, True],
        na_position="last",
    ).reset_index(drop=True)

    return table


def select_model(table, criterion="BIC", d=None):
    """
    Select the model minimizing a chosen information criterion.

    Parameters
    ----------
    table : pandas.DataFrame
        Model comparison or order-search table.
    criterion : {"AIC", "AICc", "BIC", "HQIC"}, default "BIC"
        Information criterion used for model selection.
    d : int or None, default None
        Differencing order to select within; required for mixed-d tables.

    Returns
    -------
    pandas.Series
        Row corresponding to the eligible model with the smallest
        value of the selected information criterion.

    Notes
    -----
    Failed, unknown-convergence and non-finite candidates are excluded.
    Recognized non-optimizer estimators are marked not applicable rather than
    converged; they remain eligible. Unsupported initialization is excluded.
    Mixed-d tables
    require d. Eligible models must have identical response values, ordering,
    missing positions and time labels, effective N and initialization metadata.
    Legacy tables without sample identity cannot establish comparability for
    multiple eligible models; rebuild them with compare_models/order_search.

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

    if not table.index.is_unique:
        raise ValueError("Model table index must be unique.")
    eligible = table.copy()
    orders = eligible["d"] if "d" in eligible else (
        eligible["Order"].map(lambda order: order[1]) if "Order" in eligible else None)
    if d is not None:
        if isinstance(d, (bool, np.bool_)) or not isinstance(d, (int, np.integer)) or d < 0:
            raise ValueError("d must be a non-negative integer.")
        if orders is None:
            raise ValueError("The table has no differencing-order information.")
        eligible = eligible.loc[orders == d]
    elif orders is not None and orders.dropna().nunique() > 1:
        raise ValueError("Models have different differencing orders; choose d explicitly.")
    if "Converged" in eligible:
        confirmed = eligible["Converged"].eq(True).fillna(False)
        if "Convergence Status" in eligible:
            confirmed |= eligible["Convergence Status"].eq(
                "Not applicable (non-optimizer estimator)")
        eligible = eligible.loc[confirmed]
    eligible = eligible.loc[np.isfinite(eligible[column])]
    if eligible.empty:
        raise ValueError(f"No eligible models with finite {column} values are available.")
    if len(eligible) > 1 and "Sample ID" not in eligible:
        raise ValueError("Sample identity is unavailable; rebuild the table with compare_models/order_search.")
    if ("Initialization" not in eligible or "Initialization Status" not in eligible
            or not eligible["Initialization Status"].eq("Supported").all()):
        raise ValueError("Likelihood initialization is unsupported or unavailable; comparability cannot be established.")
    for field in ("Initialization", "Sample ID", "N", "Burn", "Diffuse", "Seasonal Differencing"):
        if field in eligible and (eligible[field].isna().any() or eligible[field].nunique() != 1):
            raise ValueError(f"Models do not have the same sample/likelihood basis ({field}).")
    return eligible.loc[eligible[column].idxmin()]



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
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
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

