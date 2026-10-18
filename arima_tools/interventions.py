"""Manual AO, LS and TC regressors and effects on the modeled response scale.

No events or decay rates are estimated automatically. Metadata version 1 stores
origin_length, origin_index, row_start, index, and events (type, position, label,
column, delta). Positions always count observations from zero at the origin.
"""
from copy import deepcopy
import re
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
from ._regressors import identification

_KEY = "arima_tools_interventions"


def _count(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _index(y):
    if isinstance(y, pd.Series):
        index = y.index.copy()
    elif isinstance(y, pd.Index):
        index = y.copy()
    else:
        values = np.asarray(y)
        if values.ndim != 1:
            raise ValueError("y must be one-dimensional.")
        index = pd.RangeIndex(len(values))
    if isinstance(index, pd.MultiIndex):
        raise ValueError("Use a one-dimensional time or numerical index.")
    if not len(index) or not index.is_unique or not index.is_monotonic_increasing or index.hasnans:
        raise ValueError("Index must be nonempty, unique, increasing and without missing labels.")
    if isinstance(index, pd.PeriodIndex):
        if not index.equals(pd.period_range(index[0], periods=len(index), freq=index.freq)):
            raise ValueError("PeriodIndex must be regular without gaps.")
    elif isinstance(index, pd.DatetimeIndex):
        freq = index.freq or (pd.infer_freq(index) if len(index) >= 3 else None)
        if freq is None or not index.equals(pd.date_range(index[0], periods=len(index), freq=freq)):
            raise ValueError("DatetimeIndex must be regular with a known or inferable frequency.")
    elif not pd.api.types.is_numeric_dtype(index.dtype):
        raise ValueError("Use a time index or a numerical observation index.")
    return index


def _description(index):
    return {"kind": type(index).__name__, "dtype": str(index.dtype),
            "frequency": getattr(index, "freqstr", None),
            "timezone": str(getattr(index, "tz", None)), "labels": [str(v) for v in index]}


def _list(value):
    if pd.api.types.is_scalar(value):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _decay(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value) or not 0 < value < 1:
        raise ValueError("delta must be a finite number strictly between zero and one.")
    return float(value)


def _values(events, start, length):
    time = np.arange(start, start + length)
    columns = []
    for event in events:
        age = time - event['position']
        if event['type'] == 'AO':
            column = (age == 0).astype(float)
        elif event['type'] == 'LS':
            column = (age >= 0).astype(float)
        else:
            column = np.zeros(length)
            active = age >= 0
            column[active] = event['delta'] ** age[active]
        columns.append(column)
    return np.column_stack(columns)


def outlier_regressors(y, dates=None, types="AO", delta=0.7, *, positions=None):
    """Build manual AO pulses, LS steps and TC geometric decays as named exog.

    Give exactly one of dates or zero-based positions. Period dates resolve to
    their containing period; datetime labels match exactly. TC uses one shared
    delta in (0, 1), default .7. Missing response values never remove rows.
    Regressors are on the supplied response scale; do not pre-difference them.
    """
    index = _index(y)
    decay = _decay(delta)
    if (dates is None) == (positions is None):
        raise ValueError("Give exactly one of dates or positions.")
    given = _list(positions if positions is not None else dates)
    if not given:
        raise ValueError("Supply at least one intervention.")
    resolved = []
    for value in given:
        if positions is not None:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 0 <= value < len(index):
                raise ValueError("positions must be zero-based integers inside the sample.")
            pos = int(value)
        else:
            try:
                if isinstance(index, pd.PeriodIndex):
                    if isinstance(value, str):
                        if re.fullmatch(r"\d{4}[Qq]\d+", value) and not index.freqstr.startswith('Q'):
                            raise ValueError("Quarterly event label requires a quarterly index.")
                        if re.fullmatch(r"\d{4}", value) and not index.freqstr.startswith(('Y', 'A')):
                            raise ValueError("A bare year is ambiguous; supply the complete period.")
                    if isinstance(value, pd.Period) and value.freq != index.freq:
                        raise ValueError("Event period frequency differs from the index.")
                    label = pd.Period(value, freq=index.freq)
                elif isinstance(index, pd.DatetimeIndex):
                    if isinstance(value, str) and len(value.strip()) < 10:
                        raise ValueError("Datetime events require a complete date, not a partial label.")
                    if pd.api.types.is_number(value):
                        raise ValueError("Use a complete calendar date or positions, not a numeric timestamp.")
                    label = pd.Timestamp(value)
                    if (label.tz is None) != (index.tz is None) or (label.tz is not None and str(label.tz) != str(index.tz)):
                        raise ValueError("Event timezone must match the index.")
                else:
                    raise ValueError("Use positions for numerical observation indexes.")
                pos = index.get_loc(label)
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"Intervention date {value!r} does not match an observation: {exc}") from exc
        resolved.append(pos)
    kinds = [types] * len(resolved) if isinstance(types, str) else _list(types)
    if len(kinds) != len(resolved) or any(not isinstance(t, str) or t.upper() not in {'AO', 'LS', 'TC'} for t in kinds):
        raise ValueError("types must be AO, LS or TC, scalar or one per intervention.")
    events = []
    seen = set()
    for pos, kind in zip(resolved, kinds):
        kind = kind.upper()
        if (kind, pos) in seen:
            raise ValueError("Duplicate intervention type and observation.")
        seen.add((kind, pos))
        label = str(index[pos])
        suffix = label if isinstance(index, (pd.PeriodIndex, pd.DatetimeIndex)) else f"pos{pos}"
        events.append(dict(type=kind, position=int(pos), label=label,
                           column=f"{kind}_{suffix}", delta=decay if kind == 'TC' else None))
    columns = [e['column'] for e in events]
    if len(set(columns)) != len(columns):
        raise ValueError("Intervention column names must be unique.")
    X = pd.DataFrame(_values(events, 0, len(index)), index=index, columns=columns)
    X.attrs[_KEY] = dict(version=1, origin_length=len(index), origin_index=_description(index),
                         row_start=0, index=_description(index), events=events)
    if np.linalg.matrix_rank(X.to_numpy()) < X.shape[1]:
        warnings.warn("Intervention columns are dependent in this sample; individual effects are unidentified.", UserWarning, stacklevel=2)
    return X


def _validated(X):
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be the intervention DataFrame with versioned metadata.")
    meta = X.attrs.get(_KEY)
    try:
        if not isinstance(meta, dict) or type(meta['version']) is not int or meta['version'] != 1:
            raise ValueError("Unsupported or missing intervention metadata.")
        index = _index(X.index)
        if meta['index'] != _description(index) or not X.columns.is_unique:
            raise ValueError("Intervention index/columns do not match metadata.")
        start, origin = meta['row_start'], meta['origin_length']
        if isinstance(start, bool) or not isinstance(start, int) or start < 0 or isinstance(origin, bool) or not isinstance(origin, int) or origin <= 0:
            raise ValueError("Invalid metadata observation positions.")
        if len(meta['origin_index']['labels']) != origin:
            raise ValueError("Origin metadata length is inconsistent.")
        if start == 0 and (len(X) != origin or meta['origin_index'] != _description(index)):
            raise ValueError("Original intervention sample does not match metadata.")
        if 0 < start < origin:
            raise ValueError("Future metadata cannot begin inside the original sample.")
        events = meta['events']
        if not events or list(X.columns) != [e['column'] for e in events]:
            raise ValueError("Intervention columns do not match metadata.")
        seen = set()
        for event in events:
            kind, pos = event['type'], event['position']
            if kind not in {'AO', 'LS', 'TC'} or isinstance(pos, bool) or not isinstance(pos, int) or not 0 <= pos < origin or (kind, pos) in seen:
                raise ValueError("Invalid or duplicate intervention metadata.")
            seen.add((kind, pos))
            if event['label'] != meta['origin_index']['labels'][pos]:
                raise ValueError("Event label does not match origin metadata.")
            calendar = meta['origin_index']['kind'] in {'PeriodIndex', 'DatetimeIndex'}
            suffix = event['label'] if calendar else f"pos{pos}"
            if event['column'] != f"{kind}_{suffix}":
                raise ValueError("Intervention column name does not match its event.")
            if kind == 'TC':
                _decay(event['delta'])
            elif event['delta'] is not None:
                raise ValueError("Only TC interventions have a decay parameter.")
        if np.iscomplexobj(X.to_numpy()):
            raise ValueError("Intervention regressors must be real numeric values.")
        if not np.array_equal(X.to_numpy(dtype=float), _values(events, start, len(X))):
            raise ValueError("Intervention values do not match metadata definitions.")
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("Invalid intervention metadata.") from exc
    return deepcopy(meta)


def _observed(y, index):
    if isinstance(y, pd.Series) and not y.index.equals(index):
        raise ValueError("Observed series and interventions must have the same index.")
    values = np.asarray(y, dtype=float)
    if values.ndim != 1 or len(values) != len(index) or np.isinf(values).any():
        raise ValueError("Observed series must match X and contain no infinite values.")
    return values


def intervention_effects(results, X):
    """Return Observed, named beta*X contributions, Combined and Adjusted.

    Adjustment removes intervention regression contributions, not ARIMA errors
    or ordinary controls/trends. It stays on the modeled scale (including logs).
    X must be the original intervention frame, not an inferred column prefix.
    """
    if not isinstance(getattr(results, "model", None), ARIMA):
        raise TypeError("results must be fitted Statsmodels ARIMA results.")
    meta = _validated(X)
    if meta['row_start'] != 0 or len(X) != meta['origin_length']:
        raise ValueError("Effects require the complete original intervention sample.")
    model = results.model
    from .model_selection import _convergence_info
    if _convergence_info(results)[0] is False:
        warnings.warn("The fitted model did not converge; effects use its reported coefficients.",
                      UserWarning, stacklevel=2)
    index = model.data.row_labels
    if index is None:
        index = pd.RangeIndex(int(results.nobs))
    if not X.index.equals(index):
        raise ValueError("Intervention index must match the fitted sample.")
    ok, status = identification(model, getattr(results, 'fixed_params', ()))
    if ok is False:
        raise ValueError(status)
    if ok is None:
        warnings.warn(status + " Effects use the supplied fitted coefficients.", UserWarning, stacklevel=2)
    names = list(model.exog_names or [])
    params = dict(zip(model.param_names, np.asarray(results.params)))
    for column in X:
        if column not in names or column not in params or not np.isfinite(params[column]):
            raise ValueError(f"No uniquely named finite fitted coefficient for {column}.")
        if not np.array_equal(np.asarray(model.exog)[:, names.index(column)], X[column].to_numpy()):
            raise ValueError("Supplied interventions differ from fitted exog values.")
    observed = np.asarray(model.endog, dtype=float).reshape(-1)
    if np.isinf(observed).any():
        raise ValueError("The modeled response contains infinite observations.")
    contributions = X.mul(pd.Series({c: params[c] for c in X}), axis='columns')
    combined = contributions.sum(axis=1)
    table = pd.concat([pd.Series(observed, index=index, name='Observed'), contributions,
                       combined.rename('Combined'), (pd.Series(observed, index=index)-combined).rename('Adjusted')], axis=1)
    table.attrs[_KEY] = meta
    return table


def plot_interventions(y, X, *, results=None, individual=False, title="Intervention Effects"):
    """Plot chosen event dates; with results, plot adjusted data and fitted effects."""
    if not isinstance(individual, (bool, np.bool_)):
        raise ValueError("individual must be Boolean.")
    meta = _validated(X)
    observed = _observed(y, X.index)
    if meta['row_start'] != 0 or len(X) != meta['origin_length']:
        raise ValueError("Plotting requires the original intervention sample.")
    table = None if results is None else intervention_effects(results, X)
    if table is not None and not np.array_equal(observed, table.Observed.to_numpy(), equal_nan=True):
        raise ValueError("y differs from the fitted response sample.")
    axis = X.index.to_timestamp() if isinstance(X.index, pd.PeriodIndex) else X.index
    fig = make_subplots(rows=1 if table is None else 2, cols=1, shared_xaxes=True)
    fig.add_trace(go.Scatter(x=axis, y=observed, name='Observed', line=dict(color='#444444')), row=1, col=1)
    if table is not None:
        fig.add_trace(go.Scatter(x=axis, y=table.Adjusted, name='Adjusted', line=dict(color='#1F5A96')), row=1, col=1)
        fig.add_trace(go.Scatter(x=axis, y=table.Combined, name='Combined intervention effect', line=dict(color='#1F5A96')), row=2, col=1)
        if individual:
            for column in X:
                fig.add_trace(go.Scatter(x=axis, y=table[column], name=column), row=2, col=1)
        fig.update_yaxes(title_text='Intervention effect', row=2, col=1)
    positions = sorted({e['position'] for e in meta['events']})
    labels = ['<br>'.join(e['column'] for e in meta['events'] if e['position'] == pos) for pos in positions]
    symbols = ['diamond' if sum(e['position'] == pos for e in meta['events']) > 1 else
               {'AO': 'circle', 'LS': 'square', 'TC': 'triangle-up'}[next(e['type'] for e in meta['events'] if e['position'] == pos)] for pos in positions]
    fig.add_trace(go.Scatter(x=axis[positions], y=observed[positions], mode='markers',
                             marker=dict(color='#B2182B', size=10, symbol=symbols), text=labels,
                             hovertemplate='%{text}<extra>Manual intervention</extra>', name='Intervention dates'), row=1, col=1)
    fig.update_layout(title=dict(text=title, x=.5), template='plotly_white', width=1000,
                      height=450 if table is None else 700, hovermode='x',
                      legend=dict(orientation='h', y=-.15))
    fig.update_yaxes(title_text='Modeled response', row=1, col=1)
    fig.update_xaxes(title_text='Time' if isinstance(X.index, (pd.PeriodIndex, pd.DatetimeIndex)) else 'Observation', row=1 if table is None else 2, col=1)
    return fig


def _next_index(index, steps):
    if isinstance(index, pd.PeriodIndex):
        return pd.period_range(index[-1]+1, periods=steps, freq=index.freq, name=index.name)
    if isinstance(index, pd.DatetimeIndex):
        freq = index.freq or (pd.infer_freq(index) if len(index) >= 3 else None)
        if freq is not None:
            return pd.date_range(index[-1], periods=steps+1, freq=freq, name=index.name)[1:]
    if isinstance(index, pd.RangeIndex):
        return pd.RangeIndex(index[-1]+index.step, index[-1]+index.step*(steps+1), index.step, name=index.name)
    if len(index) >= 2 and pd.api.types.is_numeric_dtype(index.dtype):
        differences = np.diff(index.to_numpy())
        if np.all(differences == differences[0]):
            return pd.Index(index[-1] + differences[0]*np.arange(1, steps+1), name=index.name)
    return None


def extend_interventions(X, steps=20, *, future_index=None):
    """Return future AO=0, LS=1 and TC decay continued from the original event."""
    steps = _count(steps, 'steps')
    meta = _validated(X)
    expected = _next_index(X.index, steps)
    if future_index is None:
        if expected is None:
            raise ValueError("Supply future_index; index continuation cannot be inferred.")
        index = expected
    else:
        index = pd.Index(future_index)
        if expected is not None and index.equals(expected):
            # Recover proven frequency for short explicit date continuations.
            index = expected.rename(index.name)
        else:
            index = _index(index)
        if len(index) != steps or (expected is not None and not index.equals(expected)):
            raise ValueError("future_index must contain the next steps observations.")
        if index[0] <= X.index[-1]:
            raise ValueError("future_index must follow the historical index.")
    start = meta['row_start'] + len(X)
    future = pd.DataFrame(_values(meta['events'], start, steps), index=index, columns=X.columns)
    meta.update(row_start=start, index=_description(index))
    future.attrs[_KEY] = meta
    return future
