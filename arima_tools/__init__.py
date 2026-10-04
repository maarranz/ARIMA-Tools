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
    "residual_correlations",
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

from .estimation import (
    root_diagnostics,
    is_causal,
    is_invertible,
    common_roots,
    plot_inverse_roots,
    root_summary,
    coefficient_diagnostics,
    parameter_correlation,
    joint_significance,
)

from .diagnostics import (
    ljung_box_test,
    residual_correlations,
    plot_residual_diagnostics,
    jarque_bera_test,
    dynamic_effects,
    plot_dynamic_effects,
    recursive_estimation,
    plot_parameter_stability,
)

from .model_selection import (
    compare_models,
    order_search,
    select_model,
    auto_arima,
    auto_arima_summary,
)

