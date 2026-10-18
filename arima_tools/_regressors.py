"""Internal alignment and identification checks; never change observations."""
import numpy as np
import pandas as pd


def validate_exog(y, exog):
    if exog is None:
        return None
    if isinstance(exog, pd.Series):
        exog = exog.to_frame()
    if isinstance(exog, pd.DataFrame):
        if isinstance(y, pd.Series) and not exog.index.equals(y.index):
            raise ValueError("exog index must equal the response index in the same order.")
        if not exog.index.is_unique or not exog.index.is_monotonic_increasing or exog.index.hasnans:
            raise ValueError("exog index must be unique, increasing and without missing labels.")
        if not exog.columns.is_unique or not all(isinstance(c, str) for c in exog.columns):
            raise ValueError("exog columns must have unique string names.")
        output = exog.copy()
    else:
        values = np.asarray(exog)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2:
            raise ValueError("exog must be a one- or two-dimensional numeric matrix.")
        index = y.index if isinstance(y, pd.Series) else None
        if len(values) != len(y):
            raise ValueError("exog must have one row per response observation.")
        output = pd.DataFrame(values, index=index,
                              columns=[f"x{i+1}" for i in range(values.shape[1])])
    if len(output) != len(y) or output.shape[1] == 0:
        raise ValueError("exog must have one row per observation and at least one column.")
    try:
        values = output.to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError("exog must contain finite real numeric values.") from exc
    if np.iscomplexobj(output.to_numpy()) or not np.isfinite(values).all():
        raise ValueError("exog must contain finite real numeric values, without missing values.")
    return output.astype(float)


def _column_rank(matrix):
    """Numerical rank with column scaling and the usual machine-precision SVD."""
    if not matrix.shape[0] or not matrix.shape[1]:
        return 0
    scales = np.max(np.abs(matrix), axis=0)
    active = scales > 0
    return int(np.linalg.matrix_rank(matrix[:, active] / scales[active])) if active.any() else 0


def _integration_basis(model, n):
    """Full-grid nullspace of (1-L)^d (1-L^s)^D, without deleting time steps.

    Seed the homogeneous difference equation with independent initial values.
    Its solutions span constants for I(1), affine time paths for I(2), and the
    corresponding seasonal integration directions. Orthonormalizing the basis
    changes neither its span nor the rank diagnostic.
    """
    polynomial = np.array([1.])
    for _ in range(model.order[1]):
        polynomial = np.convolve(polynomial, [1., -1.])
    _, seasonal_d, _, period = model.seasonal_order
    for _ in range(seasonal_d):
        difference = np.zeros(period + 1)
        difference[[0, -1]] = [1., -1.]
        polynomial = np.convolve(polynomial, difference)
    degree = len(polynomial) - 1
    if degree == 0:
        return np.empty((n, 0)), degree
    if degree >= n:
        return np.eye(n), degree
    basis = np.zeros((n, degree))
    basis[:degree] = np.eye(degree)
    with np.errstate(over='raise', invalid='raise'):
        for t in range(degree, n):
            basis[t] = -polynomial[1:] @ basis[t-degree:t][::-1]
    if _column_rank(basis) != degree:
        raise ValueError("Integration basis is numerically unresolved.")
    return np.linalg.qr(basis, mode='reduced')[0], degree


def _gap_initialization(model, degree):
    """Recognize integration-only diffuse states and proper stationary ARMA states.

    Statsmodels exposes initial state blocks in its state-space representation.
    Extra diffuse ARMA directions, known/custom priors and custom likelihood burn
    need a model-specific observed likelihood diagnostic, not this nullspace test.
    """
    if (model.simple_differencing or model.hamilton_representation
            or not model.enforce_stationarity or model._k_states_diff != degree):
        return None
    regimes = [None] * model.k_states

    def visit(block, states):
        if block.initialization_type is None:
            for positions, child in block.blocks.items():
                visit(child, tuple(states[int(i)] for i in positions))
        else:
            for state in states:
                if regimes[state] is not None:
                    raise ValueError("Overlapping initialization blocks.")
                regimes[state] = block.initialization_type

    visit(model.ssm.initialization, tuple(range(model.k_states)))
    if any(kind != 'stationary' for kind in regimes[degree:]):
        return None
    if degree == 0:
        return 'stationary' if model.loglikelihood_burn == 0 else None
    kinds = set(regimes[:degree])
    if kinds == {'diffuse'} and model.loglikelihood_burn == 0:
        return 'diffuse'
    if kinds == {'approximate_diffuse'} and model.loglikelihood_burn == degree:
        return 'approximate_diffuse'
    return None


def _observed_identification(model, design, observed, names):
    """Rank of observed regressors modulo visible integration directions.

    Return None when this diagnostic cannot establish identification. Finite
    approximate-diffuse priors may supply information absent from diffuse-limit
    contrasts, so contrast deficiency alone is not called exact non-identification
    under approximate initialization. Raw observed-column dependencies are exact.
    """
    visible = np.isfinite(observed)
    regressors = design[visible]
    if not len(regressors):
        return False, "Unidentified specification: no observed responses."
    if not np.isfinite(regressors).all():
        return None, "Identification not established: nonfinite observed regressors."
    if _column_rank(regressors) < len(names):
        return False, "Unidentified specification: dependent or zero observed regression columns: " + ", ".join(names)
    try:
        basis, degree = _integration_basis(model, len(observed))
        convention = _gap_initialization(model, degree)
        if convention is None:
            return None, "Identification not established: unsupported initialization/likelihood convention with response gaps."
        nuisance = basis[visible]
        increment = _column_rank(np.column_stack([nuisance, regressors])) - _column_rank(nuisance)
    except (AttributeError, IndexError, TypeError, ValueError, FloatingPointError, np.linalg.LinAlgError):
        return None, "Identification not established: observed-data rank calculation is inconclusive."
    if increment == len(names):
        return True, "Identified (observed-data integration contrasts)"
    if convention == 'diffuse':
        return False, "Unidentified specification: observed regressors confounded with diffuse integration directions."
    return None, "Identification not established: deficient diffuse-limit contrasts with finite approximate-diffuse initialization."


def identification(model, fixed_params=()):
    """Check regression rank, including trends; return True, False or None.

    Column normalization and the standard machine-precision SVD rank tolerance
    avoid a user-chosen near-collinearity cutoff. This is an algebraic check,
    not a claim of precise estimates or general statistical identification.
    Complete-data checks retain the existing differenced-design convention.
    With response gaps, observed-data contrasts preserve the original grid;
    None explicitly means that this diagnostic did not establish identification.
    """
    if model.exog is None:
        return True, "Identified"
    names = list(model.exog_names)
    if len(set(model.param_names)) != len(model.param_names):
        return False, "Unidentified specification: repeated parameter names."
    keep = [i for i, name in enumerate(names) if name not in fixed_params]
    if not keep:
        return True, "Identified"
    design = np.asarray(model.exog, dtype=float)[:, keep]
    names = [names[i] for i in keep]
    observed = np.asarray(model.endog, dtype=float).reshape(-1)
    if not np.isfinite(observed).all():
        return _observed_identification(model, design, observed, names)
    for lag, count in [(1, model.order[1]),
                       (model.seasonal_order[3], model.seasonal_order[1])]:
        for _ in range(count):
            design = design[lag:] - design[:-lag]
            observed = observed[lag:] - observed[:-lag]
    design = design[np.isfinite(observed)]
    if not len(design):
        return False, "Unidentified specification: no usable regression observations."
    scales = np.max(np.abs(design), axis=0)
    zero = [name for name, scale in zip(names, scales) if scale == 0]
    if zero:
        return False, "Unidentified specification: zero columns after differencing: " + ", ".join(zero)
    normalized = design / scales
    if np.linalg.matrix_rank(normalized) < len(names):
        return False, "Unidentified specification: dependent regression/trend columns: " + ", ".join(names)
    return True, "Identified"


def future_exog(result, exog, steps):
    """Validate all future external columns, excluding Statsmodels trends."""
    count = result.model.k_exog
    if not count:
        if exog is not None:
            raise ValueError("The fitted model has no external regressors.")
        return None
    if exog is None:
        raise ValueError("Supply future exog for all fitted external regressors.")
    names = list(result.model.exog_names[result.model.k_trend:])
    if isinstance(exog, pd.Series):
        exog = exog.to_frame()
    if isinstance(exog, pd.DataFrame) and list(exog.columns) != names:
        raise ValueError("Future exog column names and order must match: " + ", ".join(names))
    output = validate_exog(np.empty(steps), exog)
    if output.shape[1] != count:
        raise ValueError("Future exog must contain all fitted external regressors.")
    if isinstance(exog, pd.DataFrame):
        _, _, _, expected = result.model._get_prediction_index(result.nobs, result.nobs + steps - 1)
        if expected is not None and not exog.index.equals(expected):
            raise ValueError("Future exog index must continue the fitted model index.")
    return output
