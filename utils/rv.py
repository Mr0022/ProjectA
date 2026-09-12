"""
Realized-volatility helpers shared by the HAR baselines and the deep models.

Two conventions are fixed here so every model in the repo agrees:

1. **Series representation.** RV is carried internally as ln(RV), one value per
   trading day. Source files may store either levels (a column named ``RV``) or
   logs (``ln_RV``); :func:`to_log_rv` normalises both to logs.

2. **Forecast target.** The h-day target is the log of the SUM of realized
   variance over the forecast window::

       Y_t^(h) = ln( sum_{k=1..h} RV_{t+k} )

   For h = 1 this is just ln(RV_{t+1}). For h > 1 it differs from the log of the
   window MEAN by the constant ln(h), and from the mean of the logs (the
   convention this repo used previously) by a Jensen gap that grows with h.
"""

import numpy as np
import pandas as pd

# Column names understood as already being in logs; anything else holding RV is
# treated as levels and log-transformed on load.
LOG_PREFIXES = ("ln_", "log_", "ln", "log")
LOG_RV_CANDIDATES = ["ln_RV", "log_RV", "lnRV", "ln_rv"]
LEVEL_RV_CANDIDATES = ["RV", "rv", "realized_volatility", "realized_variance"]


def is_log_column(name: str) -> bool:
    """True when a column name declares that it already holds ln(RV)."""
    lowered = str(name).lower()
    return lowered.startswith("ln_") or lowered.startswith("log_") or lowered in ("lnrv", "logrv")


def pick_rv_column(columns, preferred=None) -> str:
    """
    Resolve which column carries RV.

    `preferred` (e.g. the --target argument) wins when present. Otherwise a log
    column is preferred over a level column, matching the historical default of
    'ln_RV'.
    """
    cols = list(columns)
    if preferred is not None and preferred in cols:
        return preferred
    for c in LOG_RV_CANDIDATES + LEVEL_RV_CANDIDATES:
        if c in cols:
            return c
    raise ValueError(
        f"No realized-volatility column found. Looked for {preferred!r}, then "
        f"{LOG_RV_CANDIDATES + LEVEL_RV_CANDIDATES}; file has {cols}."
    )


def to_log_rv(values, column_name, verbose=True):
    """
    Return `values` as ln(RV), plus a boolean mask of the rows worth keeping.

    A column whose name declares logs is passed through untouched. A level
    column is log-transformed; non-positive entries (market holidays that leak
    into the calendar as RV = 0) cannot be logged and are reported in the mask
    so the caller can drop them rather than propagate -inf.

    Returns
    -------
    ln_rv : np.ndarray (float64)  ln(RV), NaN where the input was unusable
    keep  : np.ndarray (bool)     False for rows the caller should drop
    """
    v = np.asarray(values, dtype=float)

    if is_log_column(column_name):
        keep = np.isfinite(v)
        return v, keep

    keep = np.isfinite(v) & (v > 0)
    ln_rv = np.full_like(v, np.nan)
    ln_rv[keep] = np.log(v[keep])
    dropped = int((~keep).sum())
    if verbose:
        msg = f"[data] '{column_name}' is in levels -> using ln({column_name})"
        if dropped:
            msg += f"; dropped {dropped} non-positive/NaN row(s)"
        print(msg)
    return ln_rv, keep


def prepare_rv_frame(df, target, date_col="date", verbose=True):
    """
    Normalise a raw RV DataFrame for downstream use.

    The target column is converted to ln(RV) in place (keeping its original
    name), rows that cannot be logged are dropped, and the frame is sorted by
    date with a fresh RangeIndex.

    Returns the cleaned DataFrame.
    """
    df = df.dropna(subset=[date_col, target]).copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    ln_rv, keep = to_log_rv(df[target].values, target, verbose=verbose)
    df[target] = ln_rv
    df = df[keep].reset_index(drop=True)
    return df


def forward_log_sum(ln_rv, h: int):
    """
    Build the h-day forecast target Y_t^(h) = ln( sum_{k=1..h} RV_{t+k} ).

    `ln_rv` is a pandas Series of ln(RV). The window [t+1 .. t+h] lies strictly
    in the future of row t, so the newest information the target uses is from
    t+1 and row t's own regressors (which may include RV_t) never leak.

    h = 1 reduces to ln(RV_{t+1}). Entries whose window runs past the end of the
    sample are NaN.
    """
    if h < 1:
        raise ValueError(f"horizon h must be >= 1, got {h}")
    rv = np.exp(ln_rv)
    return np.log(rv.rolling(h).sum()).shift(-h)


def log_sum_np(ln_rv_window, axis=-1):
    """
    ln( sum RV ) for a numpy array of ln(RV) values along `axis`.

    Numerically this is a log-sum-exp; it is written that way so the helper is
    safe for any RV scale, not just the O(0.1) daily values in this repo.
    """
    a = np.asarray(ln_rv_window, dtype=float)
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)
