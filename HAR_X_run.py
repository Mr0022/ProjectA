"""
==============================================================================
HAR-X Model: HAR + Scheduled Macro News Announcements, LASSO-selected
Multi-Horizon Realized Volatility Forecasting

Based on: Plihal, T. Scheduled Macroeconomic News Announcements and Forex
          Volatility Forecasting. (N-HAR / N-HAR-CJ specification.)
          Corsi, F. (2009) for the HAR core.
          Tibshirani, R. (1996) for the LASSO.

--------------------------------------------------------------------------
WHY THIS MODEL EXISTS
--------------------------------------------------------------------------
The event-conditioned deep models in this repo (ModernTCN with --use_events,
i.e. FiLM-TCN) combine THREE things at once: the news calendar, a nonlinear
architecture, and a deep feature extractor. Comparing them only against
HAR-RV / HAR-Q -- neither of which sees the calendar -- leaves any gain
unattributable.

HAR-X closes that gap. It is the LINEAR model that sees exactly the same
information as FiLM-TCN, so the 2x2 becomes identifiable:

                  no news              with news
    linear        HAR / HAR-DOW        HAR-X          <- this file
    nonlinear     ModernTCN            FiLM-TCN

If HAR-X captures most of the FiLM-TCN gain, the calendar is doing the work.
If it does not, the architecture is.

--------------------------------------------------------------------------
THE THREE MODELS (all estimated on IDENTICAL rows, per horizon)
--------------------------------------------------------------------------
    HAR      Y = b0 + b_d*RV_d + b_w*RV_w + b_m*RV_m
             Reproduces HAR_RV_run.py on this frame. Sanity anchor.

    HAR-DOW  HAR + day-of-week interactions (MON/TUE/THU/FRI x RV_d).
             This is Plihal's ACTUAL benchmark, not plain HAR. Including it
             is what stops a "news effect" from being a repackaged
             "NFP lands on Friday" effect. Wednesday is the omitted
             category (the paper's choice; it does not affect results).

    HAR-X    HAR-DOW + the scheduled news block, estimated by LASSO with
             blocked 10-fold cross-validation. The paper's N-HAR.

Reporting the middle model is the point: it splits the raw HAR -> HAR-X gain
into "calendar seasonality" and "genuine news content".

--------------------------------------------------------------------------
SPLIT, TARGET AND METRICS: INHERITED, NOT REIMPLEMENTED
--------------------------------------------------------------------------
load_base_features, build_horizon_target, split_by_year, compute_metrics and
HORIZONS are IMPORTED from HAR_RV_run rather than copied. That is deliberate:
it makes "same train/test split as HAR-RV" a structural guarantee instead of
a claim that can silently drift when one file is edited.

    Train  : year <= 2023   (validation 2022-2023 folded in, as in HAR-RV)
    Test   : year >= 2024   (identical window to the deep models)
    Target : Y_t^(h) = ln( (1/h) * sum_{k=0}^{h-1} RV_{t+k} )
    Metrics: MSE, MAE, QLIKE (Patton, 2011) on the ln(mean RV) scale

Deviation from the paper, stated plainly: Plihal re-estimates on a rolling
1000-day window every day. This script uses the repo's fixed split so the
comparison against the deep models is like for like. Rolling would favour
HAR-X; the fixed split is the conservative choice.

--------------------------------------------------------------------------
HOW THE NEWS ENTERS (this is the part that is easy to get wrong)
--------------------------------------------------------------------------
Scheduled news is usable out-of-sample precisely because the RELEASE DATE is
known in advance even though the OUTCOME is not. So the regressor for a
forecast anchored at t is the calendar over the window being forecast,
[t .. t+h-1] -- the future, not the past. This is the paper's core insight
(it is why they use dummies rather than standardized surprises) and it is the
same quantity FiLM-TCN conditions on:

    ModernTCN.forward:  cond = self.event_embed(event_y).mean(dim=1)

event_y is the known schedule over the pred_len horizon, and it is MEAN-pooled.
This script therefore aggregates each event column by its MEAN over
[t .. t+h-1] as well, so the linear and deep models see the same statistic:

    h=1  -> the 0/1 indicator on day t
    h=5  -> fraction of the 5-day window carrying that release
    h=22 -> fraction of the 22-day window carrying that release

No value is ever read from outside [t .. t+h-1], and nothing about an
announcement's OUTCOME is used -- only that it is scheduled.

--------------------------------------------------------------------------
DIMENSION CONTROL
--------------------------------------------------------------------------
data/events_daily_features.csv carries 286 binary evt_* indicators + 12
n_events* counts. Three screens run BEFORE the LASSO, all fitted on TRAIN ONLY:

  1. Minimum firings (--min_events): drop indicators too rare to estimate.
  2. Persistence (--min_recent over --recent_years): drop columns that have
     gone dead by the end of the training window. Plihal's S3.2 rule, and it
     bites hard here because this calendar has PERSON-NAMED columns: officials
     such as Quarles, Clarida, Fischer and Mersch left office before the test
     period starts, so their columns are structurally zero out of sample.
     Without this screen LASSO selects them as train-period drift markers and
     h=22 comes out WORSE than plain HAR. Only the tail of TRAIN is inspected;
     screening on test-period activity would be look-ahead.
  3. Correlation merge (--corr_threshold, default 0.90): Plihal's rule --
     releases that always land on the same day are statistically identical,
     so keep one representative per group. Applied to the WINDOWED regressors
     (what actually enters the model), so it re-runs per horizon; at h=22 the
     windowed means are much smoother and collapse harder, which is expected.

The representative kept is the densest column in its group.

Note on the feature space: events_daily_features.csv fixes its column set
from years <= 2021 (build_event_features.py --train-end-year), i.e. the DEEP
models' train boundary, not HAR's <= 2023. Keeping that file as-is means
HAR-X and FiLM-TCN share an identical feature space, which matters more here
than reclaiming two extra years of column selection.

--------------------------------------------------------------------------
INFERENCE
--------------------------------------------------------------------------
LASSO coefficients are biased (Castle et al., 2011), so the printed
coefficient table is a POST-LASSO OLS refit on the selected support with
Newey-West HAC standard errors (L = 2*(h-1), the repo convention). The
headline forecasts remain the LASSO's own.

Model comparison uses the Diebold-Mariano test with the Harvey-Leybourne-
Newbold small-sample correction, on both MSE and QLIKE loss differentials.
The repo had no forecast-significance test at all before this file.

Usage:
    python HAR_X_run.py
    python HAR_X_run.py --event_path ./data/events_daily_features.csv
    python HAR_X_run.py --event_set counts      # 12 count columns only
    python HAR_X_run.py --corr_threshold 0.95 --min_events 40
    python HAR_X_run.py --recent_years 3 --min_recent 10   # stricter persistence
    python HAR_X_run.py --placebo 2            # + scrambled-calendar falsification
==============================================================================
"""

# -- Standard Library ----------------------------------------------------------
import argparse
import os
import warnings
warnings.filterwarnings("ignore")

# -- Third-party ---------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.api as sm
from   statsmodels.regression.linear_model import OLS
from   scipy import stats
from   sklearn.linear_model import LassoCV

# -- Inherited from the HAR-RV baseline ----------------------------------------
# Imported, not copied: this is what makes "same split as HAR-RV" structural.
from HAR_RV_run import (
    load_base_features,
    build_horizon_target,
    split_by_year,
    compute_metrics,
    qlike,
    HORIZONS,
    TRAIN_END_YEAR,
    TEST_START_YEAR,
    SEP,
    THIN,
)
import HAR_RV_run as _harrv


# ==============================================================================
# 0.  CONFIGURATION
# ==============================================================================

DATA_FILE  = "./data/EURUSD-RV.csv"
EVENT_FILE = "./data/events_daily_features.csv"
OUTPUT_DIR = "HAR-X results"

N_BLOCKS        = 10      # blocked CV folds (Bergmeir & Benitez, 2012)
CORR_THRESHOLD  = 0.90    # Plihal's >90% merge rule
MIN_EVENTS      = 30      # minimum training-day firings for an evt_ column
RECENT_YEARS    = 2       # tail of TRAIN used to test whether a column is still live
MIN_RECENT      = 5       # minimum firings in that tail (Plihal's persistence rule)
RANDOM_STATE    = 2021

# Day-of-week interactions. Wednesday omitted to avoid the dummy trap.
DOW_KEEP = {0: "MON", 1: "TUE", 3: "THU", 4: "FRI"}

MODELS = ["HAR", "HAR-DOW", "HAR-X"]
C_MODEL = {"HAR": "#8c8c9a", "HAR-DOW": "#2166ac", "HAR-X": "#d73027"}
C_ACTUAL = "#1a1a2e"


def out_path(filename: str) -> str:
    return os.path.join(OUTPUT_DIR, filename)


def print_section(title: str):
    print(f"\n{SEP}\n  {title}\n{SEP}")


# ==============================================================================
# 1.  EVENT CALENDAR LOADING
# ==============================================================================

def load_event_calendar(path: str, event_set: str = "all") -> pd.DataFrame:
    """
    Load the wide daily event matrix produced by build_event_features.py.

    Expected schema (data_provider/data_loader.py:130-137):
        'date' column + numeric per-day features, split by name into
        'n_events*' counts and binary 'evt_*' indicators.

    event_set selects which block enters the model:
        all      -> counts + indicators   (default)
        events   -> indicators only
        counts   -> counts only (a deliberately small, robust baseline)
    """
    ev = pd.read_csv(path)
    date_col = "date" if "date" in ev.columns else ev.columns[0]
    ev[date_col] = pd.to_datetime(ev[date_col], format="mixed")
    ev = ev.dropna(subset=[date_col]).sort_values(date_col)
    ev = ev.drop_duplicates(subset=[date_col], keep="last")
    ev = ev.set_index(date_col)
    ev.index.name = "date"

    counts = [c for c in ev.columns if c.startswith("n_events")]
    dummies = [c for c in ev.columns if c.startswith("evt_")]

    if event_set == "counts":
        cols = counts
    elif event_set == "events":
        cols = dummies
    else:
        cols = counts + dummies

    if not cols:
        raise ValueError(
            f"No event columns found in {path}. Expected 'n_events*' and/or "
            f"'evt_*' columns; file has {list(ev.columns)[:10]} ..."
        )

    ev = ev[cols].apply(pd.to_numeric, errors="coerce")
    print(f"  Event file   : {path}")
    print(f"  Coverage     : {ev.index[0].date()} -> {ev.index[-1].date()}  "
          f"({len(ev):,} rows)")
    print(f"  Columns      : {len(counts)} count + {len(dummies)} indicator "
          f"-> using {len(cols)} ('{event_set}')")
    return ev


def align_events_to_trading_days(ev: pd.DataFrame, trading_index: pd.DatetimeIndex):
    """
    Reindex the calendar onto the RV trading days.

    Days inside the calendar's coverage but absent from it are genuine
    no-event days and are zero-filled. Days OUTSIDE coverage carry no
    information at all -- they are returned as NaN so the caller drops those
    anchors rather than silently labelling a whole span 'no news scheduled',
    which would be a fabricated regressor and would bias the test set.
    """
    lo, hi = ev.index.min(), ev.index.max()
    aligned = ev.reindex(trading_index)

    inside = (trading_index >= lo) & (trading_index <= hi)
    filled_inside = int(aligned[inside].isna().all(axis=1).sum())
    aligned.loc[inside] = aligned.loc[inside].fillna(0.0)
    aligned.loc[~inside] = np.nan

    uncovered = int((~inside).sum())
    if filled_inside:
        print(f"  Zero-filled  : {filled_inside} trading day(s) inside coverage "
              f"with no scheduled release")
    if uncovered:
        print(f"  WARNING      : {uncovered} of {len(trading_index)} trading days fall "
              f"OUTSIDE event coverage ({lo.date()} .. {hi.date()}).")
        print(f"                 Those anchors are DROPPED, not zero-filled. All models "
              f"are then\n                 evaluated on the reduced window so the "
              f"comparison stays valid.")
    else:
        print(f"  Coverage OK  : all {len(trading_index):,} trading days covered")
    return aligned


def window_mean_events(ev_aligned: pd.DataFrame, h: int) -> pd.DataFrame:
    """
    Aggregate each event column to its MEAN over the forecast window [t .. t+h-1].

    rolling(h).mean() at row i covers [i-h+1 .. i]; shifting by -(h-1) moves
    that value onto row i-h+1, so row t ends up holding the mean over
    [t .. t+h-1] -- exactly the span the target Y_t^(h) covers, and exactly
    what FiLM-TCN mean-pools over (event_y).

    Any window overlapping an uncovered day propagates NaN, so incomplete
    windows drop out instead of being silently truncated.
    """
    if h == 1:
        out = ev_aligned.copy()
    else:
        out = ev_aligned.rolling(h, min_periods=h).mean().shift(-(h - 1))
    return out.add_prefix("ev_")


# ==============================================================================
# 2.  DESIGN MATRIX
# ==============================================================================

def add_dow_interactions(df: pd.DataFrame) -> tuple:
    """
    Day-of-week dummies interacted with lagged daily RV (Plihal, Eq. 2).

    Interacted rather than additive because the paper controls for the LEVEL
    of volatility, not just its calendar position. Wednesday is omitted.

    At h=5 and h=22 the forecast window spans every weekday, so these terms
    carry little information; they are kept anyway and LASSO is left to shrink
    them, which is cleaner than special-casing the specification per horizon.
    """
    cols = []
    for wd, name in DOW_KEEP.items():
        c = f"{name}_x_RVd"
        df[c] = (df.index.dayofweek == wd).astype(float) * df["RV_d"]
        cols.append(c)
    return df, cols


def screen_event_columns(train_df: pd.DataFrame, ev_cols: list,
                         min_events: float, corr_threshold: float,
                         recent_years: int = RECENT_YEARS,
                         min_recent: float = MIN_RECENT):
    """
    Three TRAIN-ONLY screens, in order. Fitting them on train alone is what
    keeps the test period from influencing the feature space.

      1. Drop near-constant columns and indicators whose total training-window
         mass is below `min_events` (too rare to estimate).
      2. Drop columns that have gone DEAD: less than `min_recent` mass over the
         final `recent_years` of the training window.
      3. Merge columns correlated above `corr_threshold` (Plihal's >90% rule):
         releases that always co-occur are statistically the same regressor.
         The densest column in each group is kept as its representative.

    Screen 2 is Plihal's persistence rule (S3.2: keep only news "released for
    the majority of our observed period ... the last no sooner than 2016") and
    it matters more here than in the paper, because this calendar carries
    PERSON-NAMED columns: evt_USD_Fed_s_Quarles_speech and friends are not
    recurring releases, they are officials who left office. Without this screen
    LASSO happily selects speakers who stopped appearing before the test period
    even begins -- they fit train-period drift and are then structurally zero
    out of sample, which is how h=22 ends up WORSE than plain HAR.

    Crucially this looks only at the tail of TRAIN, never at test. Screening on
    test-period activity would be look-ahead and would invalidate the whole
    exercise; a speaker who fell silent by the end of train is identifiable
    without ever touching the test years.

    Returns (kept_cols, merge_map, n_dropped_rare, n_dropped_dead).
    """
    X = train_df[ev_cols]

    mass = X.sum()
    var = X.var()
    alive = [c for c in ev_cols if var[c] > 1e-12 and mass[c] >= min_events]
    n_rare = len(ev_cols) - len(alive)
    if not alive:
        return [], {}, n_rare, 0

    # -- Screen 2: still active at the END of the training window -------------
    cutoff = train_df.index.max() - pd.DateOffset(years=recent_years)
    recent = train_df.loc[train_df.index > cutoff, alive]
    if len(recent):
        recent_mass = recent.sum()
        still_live = [c for c in alive if recent_mass[c] >= min_recent]
        n_dead = len(alive) - len(still_live)
        alive = still_live
    else:
        n_dead = 0
    if not alive:
        return [], {}, n_rare, n_dead

    # Densest first, so the representative of each group is the best-populated.
    alive = sorted(alive, key=lambda c: -mass[c])
    corr = X[alive].corr().abs()

    kept, merge_map, absorbed = [], {}, set()
    for c in alive:
        if c in absorbed:
            continue
        kept.append(c)
        absorbed.add(c)
        for other in alive:
            if other in absorbed:
                continue
            if corr.loc[c, other] >= corr_threshold:
                absorbed.add(other)
                merge_map[other] = c
    return kept, merge_map, n_rare, n_dead


def blocked_folds(n: int, k: int = N_BLOCKS, embargo: int = 0):
    """
    Contiguous (not random) CV folds, with an embargo around each fold.

    Random k-fold leaks across autocorrelated neighbours and returns
    optimistic errors on time series; Plihal follows Bergmeir & Benitez (2012)
    and uses sequential blocks instead. So does this.

    Blocking alone is not enough once h > 1. The target Y_t^(h) spans
    [t .. t+h-1], so two rows within h-1 of each other share up to h-1 days of
    the SAME realized variance. A training row sitting just outside a
    validation block therefore carries part of that block's answer. The CV
    error comes back too low, LASSO reads that as "the penalty can be smaller",
    and the fit is left under-regularized -- worst exactly at h=22, where each
    target window overlaps the next by 21 days.

    Purging `embargo` rows from the training side of every fold boundary
    (Lopez de Prado, 2018, ch.7) removes that shared span. At h=1 the embargo
    is 0 and this reduces to plain blocked CV.
    """
    idx = np.arange(n)
    blocks = np.array_split(idx, k)
    folds = []
    for i, val in enumerate(blocks):
        if len(val) == 0:
            continue
        lo, hi = val[0] - embargo, val[-1] + embargo
        train_idx = idx[(idx < lo) | (idx > hi)]
        if len(train_idx) == 0:
            continue
        folds.append((train_idx, val))
    return folds


def build_design(df_base: pd.DataFrame, ev_aligned: pd.DataFrame, h: int,
                 verbose: bool = False):
    """
    Assemble the full design for horizon h and split it.

    Factored out so the placebo check runs through EXACTLY this path. If the
    real and scrambled calendars were assembled by two copies of this logic,
    a divergence between them would be indistinguishable from a real effect,
    which is the one thing a falsification test must not allow.

    The single dropna over the whole design is what guarantees HAR, HAR-DOW
    and HAR-X are estimated and scored on identical rows.
    """
    df_h = build_horizon_target(df_base, h)

    # News over the forecast window [t .. t+h-1]
    ev_h = window_mean_events(ev_aligned, h)
    df_h = df_h.join(ev_h, how="left")

    df_h, dow_cols = add_dow_interactions(df_h)

    ev_cols_all = list(ev_h.columns)
    need = ["Y_h", "RV_d", "RV_w", "RV_m"] + dow_cols + ev_cols_all
    before = len(df_h)
    df_h = df_h.dropna(subset=need)
    if verbose and before - len(df_h):
        print(f"\n  Dropped {before - len(df_h)} anchor(s) whose "
              f"[t .. t+{h-1}] window is not fully event-covered.")

    train, test = split_by_year(df_h)
    return train, test, dow_cols, ev_cols_all


# ==============================================================================
# 3.  ESTIMATION
# ==============================================================================

def fit_ols(train: pd.DataFrame, cols: list, h: int):
    """OLS with Newey-West HAC, L = 2*(h-1) -- the repo/HAR-RV convention."""
    nw_lag = HORIZONS[h]["nw_lag"]
    X = sm.add_constant(train[cols], has_constant="add")
    return OLS(train["Y_h"], X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": max(nw_lag, 1), "use_correction": True},
    )


def fit_lasso(train: pd.DataFrame, test: pd.DataFrame, cols: list, h: int):
    """
    LASSO over the full HAR + DOW + news design, blocked-CV tuned.

    Features are standardized on TRAIN statistics so the L1 penalty is
    comparable across blocks that live on wildly different scales (log-RV
    around -1, indicators in {0,1}, counts in 0..20). y is left in its own
    units. Everything is penalized, including the HAR core -- that is plain
    LASSO, as in the paper; the core terms are strong enough to survive.
    """
    Xtr_raw = train[cols].to_numpy(float)
    Xte_raw = test[cols].to_numpy(float)
    ytr = train["Y_h"].to_numpy(float)

    mu = Xtr_raw.mean(axis=0)
    sd = Xtr_raw.std(axis=0, ddof=0)
    sd[sd < 1e-12] = 1.0
    Xtr = (Xtr_raw - mu) / sd
    Xte = (Xte_raw - mu) / sd

    # The alpha grid is left at the default 100-point path. That default is
    # spelled `n_alphas` in sklearn < 1.7 and `alphas` from 1.7 on, so passing
    # neither keeps this runnable on either side of that rename.
    model = LassoCV(
        cv=blocked_folds(len(train), N_BLOCKS, embargo=h - 1),
        max_iter=200_000,
        tol=1e-4,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ).fit(Xtr, ytr)

    coef = pd.Series(model.coef_ / sd, index=cols)          # back to raw units
    intercept = float(model.intercept_ - (mu / sd * model.coef_).sum())
    selected = [c for c in cols if abs(coef[c]) > 1e-10]

    return {
        "model": model,
        "alpha": float(model.alpha_),
        "coef": coef,
        "intercept": intercept,
        "selected": selected,
        "y_hat_train": pd.Series(model.predict(Xtr), index=train.index),
        "y_hat_test": pd.Series(model.predict(Xte), index=test.index),
    }


def post_lasso_ols(train: pd.DataFrame, selected: list, h: int):
    """
    Debiased refit on the selected support, with HAC SEs.

    LASSO shrinks toward zero, so its coefficients are biased and their
    magnitudes are not directly interpretable (Castle et al., 2011). Refitting
    OLS on the chosen support gives an unbiased coefficient and a t-statistic
    worth printing. Forecasts still come from the LASSO itself.
    """
    if not selected:
        return None
    try:
        return fit_ols(train, selected, h)
    except Exception as exc:       # near-singular support
        print(f"  (post-LASSO OLS refit skipped: {exc})")
        return None


# ==============================================================================
# 4.  FORECAST COMPARISON
# ==============================================================================

def loss_series(actual: np.ndarray, pred: np.ndarray, kind: str) -> np.ndarray:
    """Per-observation loss, so differentials can be tested."""
    if kind == "MSE":
        return (actual - pred) ** 2
    if kind == "MAE":
        return np.abs(actual - pred)
    if kind == "QLIKE":
        ratio = np.exp(actual) / np.exp(pred)
        return ratio - np.log(ratio) - 1.0
    raise ValueError(kind)


def diebold_mariano(loss_bench: np.ndarray, loss_model: np.ndarray, h: int):
    """
    DM test (1995) with the Harvey-Leybourne-Newbold (1997) small-sample
    correction, referred to t_{n-1}.

    d_t = loss_bench - loss_model, so a POSITIVE statistic means the model
    beats the benchmark. The long-run variance uses a Bartlett kernel at the
    repo's HAC bandwidth, which is what makes overlapping h>1 forecasts
    testable at all.
    """
    d = np.asarray(loss_bench, float) - np.asarray(loss_model, float)
    n = len(d)
    if n < 10 or np.allclose(d, 0):
        return np.nan, np.nan

    dbar = d.mean()
    dm_lag = max(HORIZONS[h]["nw_lag"], h - 1)
    dev = d - dbar
    lrv = float(np.mean(dev ** 2))
    for L in range(1, min(dm_lag, n - 1) + 1):
        w = 1.0 - L / (dm_lag + 1.0)
        lrv += 2.0 * w * float(np.mean(dev[L:] * dev[:-L]))
    if lrv <= 0:
        return np.nan, np.nan

    dm = dbar / np.sqrt(lrv / n)
    k = dm_lag + 1
    adj = np.sqrt(max((n + 1 - 2 * k + k * (k - 1) / n) / n, 1e-12))
    dm_hln = dm * adj
    p = 2.0 * (1.0 - stats.t.cdf(abs(dm_hln), df=n - 1))
    return float(dm_hln), float(p)


def pct_improvement(bench: float, model: float) -> float:
    """Percent reduction in loss vs benchmark. Negative = better, as in the paper."""
    return float((model - bench) / abs(bench) * 100.0) if bench else np.nan


def run_placebo(df_base, ev, n_seeds, corr_threshold, min_events,
                recent_years, min_recent):
    """
    Falsification check: re-run HAR-X on a calendar whose ROWS have been
    permuted, destroying the date alignment while leaving every column's
    marginal distribution, sparsity and correlation structure intact.

    A real news effect must vanish here. If a scrambled calendar buys the same
    improvement, the "news effect" was the LASSO fitting 200-odd extra columns
    to noise, and the honest conclusion is that the headline number means
    nothing. This is the first thing a referee will ask for, so it ships with
    the model rather than living in a scratch file.
    """
    print_section("PLACEBO CHECK -- scrambled calendar (date alignment destroyed)")
    print("  A genuine news effect should collapse to ~0% and lose significance.\n")
    print(f"  {'Horizon':<9} {'Calendar':<22} {'dMSE vs HAR-DOW':>17} {'DM p':>9} "
          f"{'news sel.':>10}")
    print(THIN)

    rows = []
    for h in HORIZONS:
        variants = [("REAL", ev)]
        for s in range(n_seeds):
            rng = np.random.default_rng(1000 + s)
            ev_s = ev.copy()
            ev_s.iloc[:, :] = ev.values[rng.permutation(len(ev))]
            variants.append((f"SCRAMBLED seed {1000 + s}", ev_s))

        for tag, ev_v in variants:
            ev_al = align_events_to_trading_days_quiet(ev_v, df_base.index)
            train, test, dow_cols, ev_cols = build_design(df_base, ev_al, h)
            kept, _, _, _ = screen_event_columns(
                train, ev_cols, min_events=min_events, corr_threshold=corr_threshold,
                recent_years=recent_years, min_recent=min_recent)
            har = ["RV_d", "RV_w", "RV_m"]

            res = fit_ols(train, har + dow_cols, h)
            p_dow = res.predict(sm.add_constant(test[har + dow_cols],
                                                has_constant="add"))
            lf = fit_lasso(train, test, har + dow_cols + kept, h)

            a = test["Y_h"].values
            pd_ = np.asarray(p_dow, float)
            px = np.asarray(lf["y_hat_test"], float)
            d = pct_improvement(compute_metrics(a, pd_)["MSE"],
                                compute_metrics(a, px)["MSE"])
            _, p = diebold_mariano(loss_series(a, pd_, "MSE"),
                                   loss_series(a, px, "MSE"), h)
            n_news = len([c for c in lf["selected"] if c.startswith("ev_")])
            print(f"  {('h=' + str(h)):<9} {tag:<22} {d:>16.3f}% {p:>9.4f} {n_news:>10}")
            rows.append({"horizon": h, "calendar": tag, "dMSE_pct_vs_HAR_DOW": d,
                         "DM_p": p, "news_regressors_selected": n_news})
        print(THIN)

    fn = out_path("har_x_placebo.csv")
    pd.DataFrame(rows).to_csv(fn, index=False)
    print(f"  -> Saved: {fn}")


def align_events_to_trading_days_quiet(ev, idx):
    """align_events_to_trading_days without the per-call coverage report."""
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        return align_events_to_trading_days(ev, idx)


# ==============================================================================
# 5.  PRINT TABLES
# ==============================================================================

def print_split_info(train: pd.DataFrame, test: pd.DataFrame):
    print(f"\n  -- Split (inherited from HAR_RV_run, val folded into train) ----")
    print(f"  {'Set':<8} {'Rows':>6}  {'Start':>12}  {'End':>12}")
    print(f"  {THIN[:50]}")
    print(f"  {'Train':<8} {len(train):>6}  {str(train.index[0].date()):>12}  "
          f"{str(train.index[-1].date()):>12}")
    print(f"  {'Test':<8} {len(test):>6}  {str(test.index[0].date()):>12}  "
          f"{str(test.index[-1].date()):>12}")
    print(f"  {THIN[:50]}")


def print_screen_summary(n_raw, n_rare, n_dead, merge_map, kept, h, corr_threshold):
    print(f"\n  -- News block screening (train-only, h={h}) ----")
    print(f"  {'Raw event columns':<46} {n_raw:>6}")
    print(f"  {'Dropped: rare / near-constant':<46} {n_rare:>6}")
    print(f"  {f'Dropped: dead by end of train':<46} {n_dead:>6}")
    print(f"  {f'Dropped: corr-merged (>{corr_threshold})':<46} {len(merge_map):>6}")
    print(f"  {'Retained for LASSO':<46} {len(kept):>6}")


def print_lasso_table(fit, post, h, top_n=20):
    hlabel = HORIZONS[h]["label"]
    nw_lag = HORIZONS[h]["nw_lag"]
    sel = fit["selected"]
    print_section(f"HAR-X  [{hlabel}]  LASSO alpha={fit['alpha']:.6f}  "
                  f"({len(sel)} of {len(fit['coef'])} regressors selected)")

    if not sel:
        print("  LASSO shrank every regressor to zero (intercept-only model).")
        return

    if post is None:
        print(f"  {'Variable':<44} {'LASSO coef':>12}")
        print(THIN)
        for c in sorted(sel, key=lambda x: -abs(fit["coef"][x]))[:top_n]:
            print(f"  {c:<44} {fit['coef'][c]:>12.5f}")
        print(THIN)
        return

    print(f"  Coefficients below are the POST-LASSO OLS refit on the selected")
    print(f"  support (debiased), Newey-West HAC L={nw_lag}. Forecasts use the LASSO.\n")
    params, bse, tv, pv = post.params, post.bse, post.tvalues, post.pvalues
    order = [c for c in params.index if c != "const"]
    order = sorted(order, key=lambda x: -abs(tv[x]))

    print(f"  {'Variable':<44} {'Coeff':>10} {'Std.Err':>10} {'t-stat':>9} {'p':>8}")
    print(THIN)
    if "const" in params.index:
        print(f"  {'Intercept':<44} {params['const']:>10.4f} {bse['const']:>10.4f} "
              f"{tv['const']:>9.3f} {pv['const']:>8.4f}")
    shown = 0
    for c in order:
        if shown >= top_n:
            break
        sig = "***" if pv[c] < 0.01 else ("**" if pv[c] < 0.05 else ("*" if pv[c] < 0.10 else ""))
        print(f"  {c:<44} {params[c]:>10.4f} {bse[c]:>10.4f} "
              f"{tv[c]:>9.3f} {pv[c]:>8.4f}{sig}")
        shown += 1
    print(THIN)
    if len(order) > top_n:
        print(f"  ... {len(order) - top_n} further selected regressor(s) omitted; "
              f"full table in the params CSV.")
    print("  Significance: *** p<0.01  ** p<0.05  * p<0.10")
    print(f"\n  R2 (post-LASSO, train) : {post.rsquared:.4f}    "
          f"Adj. R2 : {post.rsquared_adj:.4f}")
    print(f"  Observations           : {int(post.nobs):,}")


def print_horizon_comparison(metrics_h: dict, dm_h: dict, h: int, n_test: int):
    hlabel = HORIZONS[h]["label"]
    print_section(f"OUT-OF-SAMPLE COMPARISON  [{hlabel}]  "
                  f"Test {TEST_START_YEAR}+  (n={n_test})")
    print(f"  {'Model':<10} {'MSE':>11} {'vs HAR':>9} {'MAE':>11} "
          f"{'QLIKE':>11} {'vs HAR':>9}")
    print(THIN)
    base = metrics_h["HAR"]
    for m in MODELS:
        v = metrics_h[m]
        dmse = "--" if m == "HAR" else f"{pct_improvement(base['MSE'], v['MSE']):+7.3f}%"
        dql = "--" if m == "HAR" else f"{pct_improvement(base['QLIKE'], v['QLIKE']):+7.3f}%"
        print(f"  {m:<10} {v['MSE']:>11.6f} {dmse:>9} {v['MAE']:>11.6f} "
              f"{v['QLIKE']:>11.6f} {dql:>9}")
    print(THIN)
    print("  Negative % = lower loss than plain HAR (the paper's sign convention).")

    print(f"\n  -- Diebold-Mariano (HLN-corrected), positive = model beats benchmark --")
    print(f"  {'Comparison':<28} {'Loss':>7} {'DM':>9} {'p-value':>9}   {'Verdict'}")
    print(THIN)
    for (bench, model), per_loss in dm_h.items():
        for lname, (dm, p) in per_loss.items():
            if np.isnan(dm):
                verdict = "n/a"
            elif p < 0.05:
                verdict = f"{model} better" if dm > 0 else f"{bench} better"
            elif p < 0.10:
                verdict = f"{model} better (10%)" if dm > 0 else f"{bench} better (10%)"
            else:
                verdict = "no difference"
            label = f"{model} vs {bench}"
            print(f"  {label:<28} {lname:>7} {dm:>9.3f} {p:>9.4f}   {verdict}")
    print(THIN)


def print_final_summary(all_metrics: dict, all_dm: dict, test_counts: dict):
    print_section(f"FINAL SUMMARY -- OUT-OF-SAMPLE TEST ({TEST_START_YEAR}+)")
    print(f"  {'Horizon':<12} {'Model':<10} {'MSE':>11} {'MAE':>11} {'QLIKE':>11} "
          f"{'dMSE% vs HAR-DOW':>18} {'DM p':>8}")
    print(THIN)
    for h in HORIZONS:
        dowm = all_metrics[h]["HAR-DOW"]
        for m in MODELS:
            v = all_metrics[h][m]
            if m == "HAR-X":
                d = f"{pct_improvement(dowm['MSE'], v['MSE']):+17.3f}%"
                p = all_dm[h].get(("HAR-DOW", "HAR-X"), {}).get("MSE", (np.nan, np.nan))[1]
                ps = f"{p:>8.4f}" if not np.isnan(p) else f"{'--':>8}"
            else:
                d, ps = f"{'--':>18}", f"{'--':>8}"
            hl = f"h={h}" if m == "HAR" else ""
            print(f"  {hl:<12} {m:<10} {v['MSE']:>11.6f} {v['MAE']:>11.6f} "
                  f"{v['QLIKE']:>11.6f} {d} {ps}")
        print(f"  {THIN[:len(THIN)]}")


# ==============================================================================
# 6.  FIGURES
# ==============================================================================

def figure_forecasts(results: dict):
    horizons = list(HORIZONS.keys())
    fig, axes = plt.subplots(len(horizons), 1, figsize=(11, 3.8 * len(horizons)))
    fig.subplots_adjust(hspace=0.45)
    for ax, h in zip(np.atleast_1d(axes), horizons):
        r = results[h]
        te = r["test"]
        ax.plot(te.index, te["Y_h"], color=C_ACTUAL, lw=1.1, label="Actual", zorder=3)
        for m in ["HAR-DOW", "HAR-X"]:
            ax.plot(te.index, r["pred_test"][m], color=C_MODEL[m], lw=1.0,
                    alpha=0.85, label=m)
        ax.set_title(f"{HORIZONS[h]['label']}  --  out-of-sample forecasts "
                     f"(test {TEST_START_YEAR}+)", loc="left")
        ax.set_ylabel(r"$\ln(\overline{RV})$")
        ax.legend(ncol=3, loc="upper left")
    for ext in ("png", "pdf"):
        fig.savefig(out_path(f"har_x_fig1_forecasts.{ext}"))
    plt.close(fig)
    print(f"  -> Saved: {out_path('har_x_fig1_forecasts.png')} (+ .pdf)")


def figure_loss_comparison(all_metrics: dict):
    metrics = ["MSE", "MAE", "QLIKE"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    horizons = list(HORIZONS.keys())
    x = np.arange(len(horizons))
    w = 0.26
    for ax, met in zip(axes, metrics):
        for i, m in enumerate(MODELS):
            vals = [all_metrics[h][m][met] for h in horizons]
            ax.bar(x + (i - 1) * w, vals, w, label=m, color=C_MODEL[m],
                   edgecolor="white", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([f"h={h}" for h in horizons])
        ax.set_title(met, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Out-of-sample loss")
    axes[-1].legend(loc="upper right")
    fig.suptitle(f"Out-of-sample loss by model and horizon (test {TEST_START_YEAR}+)",
                 x=0.09, ha="left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_path(f"har_x_fig2_loss_comparison.{ext}"))
    plt.close(fig)
    print(f"  -> Saved: {out_path('har_x_fig2_loss_comparison.png')} (+ .pdf)")


def figure_selected_events(results: dict, top_n: int = 15):
    """
    The analogue of Plihal's Tables 8-10: WHICH scheduled releases the model
    actually leans on. With a fixed split there is one LASSO fit per horizon,
    so this ranks selected news regressors by |coefficient| rather than by
    selection frequency across rolling re-estimations.
    """
    horizons = list(HORIZONS.keys())
    fig, axes = plt.subplots(1, len(horizons), figsize=(15, 5.5))
    for ax, h in zip(np.atleast_1d(axes), horizons):
        coef = results[h]["lasso"]["coef"]
        ev = coef[[c for c in coef.index if c.startswith("ev_")]]
        ev = ev[ev.abs() > 1e-10].sort_values(key=abs, ascending=False).head(top_n)
        if len(ev) == 0:
            ax.text(0.5, 0.5, "no news regressor\nselected", ha="center",
                    va="center", transform=ax.transAxes, color="0.4")
            ax.set_title(f"{HORIZONS[h]['label']}", loc="left")
            ax.axis("off")
            continue
        ev = ev.iloc[::-1]
        labels = [c.replace("ev_", "").replace("evt_", "").replace("_", " ")[:38]
                  for c in ev.index]
        ax.barh(np.arange(len(ev)), ev.values,
                color=[C_MODEL["HAR-X"] if v > 0 else "#2166ac" for v in ev.values],
                edgecolor="white", linewidth=0.5)
        ax.set_yticks(np.arange(len(ev)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.axvline(0, color="0.3", lw=0.8)
        ax.set_title(f"{HORIZONS[h]['label']}", loc="left")
        ax.set_xlabel("LASSO coefficient")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Most influential scheduled-news regressors selected by HAR-X",
                 x=0.02, ha="left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_path(f"har_x_fig3_selected_news.{ext}"))
    plt.close(fig)
    print(f"  -> Saved: {out_path('har_x_fig3_selected_news.png')} (+ .pdf)")


def figure_improvement(all_metrics: dict, all_dm: dict):
    horizons = list(HORIZONS.keys())
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(horizons))
    w = 0.36
    for i, (bench, model) in enumerate([("HAR", "HAR-X"), ("HAR-DOW", "HAR-X")]):
        vals, ps = [], []
        for h in horizons:
            vals.append(pct_improvement(all_metrics[h][bench]["MSE"],
                                        all_metrics[h][model]["MSE"]))
            ps.append(all_dm[h].get((bench, model), {}).get("MSE", (np.nan, np.nan))[1])
        bars = ax.bar(x + (i - 0.5) * w, vals, w,
                      label=f"HAR-X vs {bench}",
                      color=["#d73027", "#2166ac"][i], edgecolor="white", linewidth=0.6)
        for b, v, p in zip(bars, vals, ps):
            star = "" if (p is None or np.isnan(p)) else (
                "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else "")
            ax.text(b.get_x() + b.get_width() / 2,
                    v + (0.15 if v >= 0 else -0.45), f"{v:+.2f}%{star}",
                    ha="center", fontsize=8)
    ax.axhline(0, color="0.3", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"h={h}" for h in horizons])
    ax.set_ylabel("MSE change (%)")
    ax.set_title("HAR-X out-of-sample MSE vs benchmarks  "
                 "(negative = better; * DM significance)", loc="left")
    ax.legend()
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_path(f"har_x_fig4_improvement.{ext}"))
    plt.close(fig)
    print(f"  -> Saved: {out_path('har_x_fig4_improvement.png')} (+ .pdf)")


# ==============================================================================
# 7.  EXPORT
# ==============================================================================

def export_all(results: dict, all_metrics: dict, all_dm: dict):
    # -- 7a. Per-horizon forecasts -------------------------------------------
    for h, r in results.items():
        rows = []
        for split, df, preds in (("train", r["train"], r["pred_train"]),
                                 ("test", r["test"], r["pred_test"])):
            out = pd.DataFrame({"Y_h": df["Y_h"]}, index=df.index)
            for m in MODELS:
                out[f"pred_{m}"] = preds[m]
            out["split"] = split
            rows.append(out)
        full = pd.concat(rows).sort_index()
        fn = out_path(f"har_x_h{h:02d}_forecasts.csv")
        full.to_csv(fn)
        print(f"  -> Saved: {fn}")

    # -- 7b. Consolidated metrics --------------------------------------------
    rows = []
    for h in results:
        for split in ("train", "test"):
            for m in MODELS:
                mt = all_metrics[h][m] if split == "test" else results[h]["metrics_train"][m]
                rows.append({"horizon": h, "split": split, "model": m,
                             "MSE": mt["MSE"], "MAE": mt["MAE"], "QLIKE": mt["QLIKE"]})
    fn = out_path("har_x_all_metrics.csv")
    pd.DataFrame(rows).to_csv(fn, index=False)
    print(f"  -> Saved: {fn}")

    # -- 7c. Diebold-Mariano table -------------------------------------------
    rows = []
    for h, per in all_dm.items():
        for (bench, model), losses in per.items():
            for lname, (dm, p) in losses.items():
                rows.append({"horizon": h, "benchmark": bench, "model": model,
                             "loss": lname, "DM_stat": dm, "p_value": p})
    fn = out_path("har_x_dm_tests.csv")
    pd.DataFrame(rows).to_csv(fn, index=False)
    print(f"  -> Saved: {fn}")

    # -- 7d. Selected regressors + post-LASSO inference -----------------------
    for h, r in results.items():
        fit, post = r["lasso"], r["post"]
        tbl = pd.DataFrame({"lasso_coef": fit["coef"]})
        tbl["selected"] = tbl["lasso_coef"].abs() > 1e-10
        tbl["block"] = ["news" if c.startswith("ev_") else
                        ("dow" if c.endswith("_x_RVd") else "har") for c in tbl.index]
        if post is not None:
            ci = post.conf_int()
            tbl["postlasso_coef"] = post.params
            tbl["postlasso_se"] = post.bse
            tbl["postlasso_t"] = post.tvalues
            tbl["postlasso_p"] = post.pvalues
            tbl["postlasso_ci_lo"] = ci[0]
            tbl["postlasso_ci_hi"] = ci[1]
        tbl = tbl.sort_values(["selected", "lasso_coef"],
                              key=lambda s: s.abs() if s.name == "lasso_coef" else s,
                              ascending=[False, False])
        fn = out_path(f"har_x_h{h:02d}_params.csv")
        tbl.to_csv(fn)
        print(f"  -> Saved: {fn}")

    # -- 7e. Correlation-merge map (which releases were treated as one) -------
    rows = []
    for h, r in results.items():
        for dropped, rep in r["merge_map"].items():
            rows.append({"horizon": h, "merged_column": dropped, "representative": rep})
    if rows:
        fn = out_path("har_x_merged_events.csv")
        pd.DataFrame(rows).to_csv(fn, index=False)
        print(f"  -> Saved: {fn}")


# ==============================================================================
# 8.  MAIN PIPELINE
# ==============================================================================

def main(data_file=DATA_FILE, event_file=EVENT_FILE, event_set="all",
         corr_threshold=CORR_THRESHOLD, min_events=MIN_EVENTS,
         recent_years=RECENT_YEARS, min_recent=MIN_RECENT, placebo=0):
    print(f"\n{SEP}")
    print("  HAR-X  --  HAR + Scheduled Macro News (LASSO)")
    print("  Plihal: Scheduled Macroeconomic News Announcements and Forex Volatility")
    print(f"  EUR/USD  |  Horizons: h = 1, 5, 22")
    print(f"  Split: Train <={TRAIN_END_YEAR}  |  Test >={TEST_START_YEAR}  "
          f"(inherited from HAR_RV_run)")
    print(SEP)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n  Output directory: '{OUTPUT_DIR}/'")

    # -- 8.1  Load RV and the calendar ----------------------------------------
    print(f"\n[1] Loading data ...")
    df_base = load_base_features(data_file)
    print(f"  RV series    : {df_base.index[0].date()} -> {df_base.index[-1].date()}  "
          f"({len(df_base):,} rows)")
    ev = load_event_calendar(event_file, event_set)
    ev_aligned = align_events_to_trading_days(ev, df_base.index)

    results, all_metrics, all_dm, test_counts = {}, {}, {}, {}

    # -- 8.2  Per-horizon -----------------------------------------------------
    for h in HORIZONS:
        print(f"\n{'-' * 72}")
        print(f"  HORIZON  h = {h}  [{HORIZONS[h]['label']}]")
        print(f"{'-' * 72}")

        train, test, dow_cols, ev_cols_all = build_design(df_base, ev_aligned, h,
                                                          verbose=True)
        print_split_info(train, test)
        test_counts[h] = len(test)

        # Screen the news block on TRAIN only
        kept, merge_map, n_rare, n_dead = screen_event_columns(
            train, ev_cols_all, min_events=min_events, corr_threshold=corr_threshold,
            recent_years=recent_years, min_recent=min_recent)
        print_screen_summary(len(ev_cols_all), n_rare, n_dead, merge_map, kept, h,
                             corr_threshold)

        har_cols = ["RV_d", "RV_w", "RV_m"]
        cols_by_model = {
            "HAR":     har_cols,
            "HAR-DOW": har_cols + dow_cols,
            "HAR-X":   har_cols + dow_cols + kept,
        }

        # -- Fit ---------------------------------------------------------------
        pred_train, pred_test, metrics_test, metrics_train = {}, {}, {}, {}
        ols_fits = {}
        for m in ("HAR", "HAR-DOW"):
            res = fit_ols(train, cols_by_model[m], h)
            ols_fits[m] = res
            Xtr = sm.add_constant(train[cols_by_model[m]], has_constant="add")
            Xte = sm.add_constant(test[cols_by_model[m]], has_constant="add")
            pred_train[m] = res.predict(Xtr)
            pred_test[m] = res.predict(Xte)

        lfit = fit_lasso(train, test, cols_by_model["HAR-X"], h)
        pred_train["HAR-X"] = lfit["y_hat_train"]
        pred_test["HAR-X"] = lfit["y_hat_test"]
        post = post_lasso_ols(train, lfit["selected"], h)
        print_lasso_table(lfit, post, h)

        for m in MODELS:
            metrics_train[m] = compute_metrics(train["Y_h"].values,
                                               np.asarray(pred_train[m], float))
            metrics_test[m] = compute_metrics(test["Y_h"].values,
                                              np.asarray(pred_test[m], float))

        # -- Diebold-Mariano ---------------------------------------------------
        actual = test["Y_h"].values
        dm_h = {}
        for bench, model in [("HAR", "HAR-DOW"), ("HAR", "HAR-X"), ("HAR-DOW", "HAR-X")]:
            per_loss = {}
            for lname in ("MSE", "QLIKE"):
                lb = loss_series(actual, np.asarray(pred_test[bench], float), lname)
                lm = loss_series(actual, np.asarray(pred_test[model], float), lname)
                per_loss[lname] = diebold_mariano(lb, lm, h)
            dm_h[(bench, model)] = per_loss

        print_horizon_comparison(metrics_test, dm_h, h, len(test))

        results[h] = {
            "train": train, "test": test,
            "pred_train": pred_train, "pred_test": pred_test,
            "metrics_train": metrics_train,
            "lasso": lfit, "post": post, "ols": ols_fits,
            "merge_map": merge_map, "kept": kept,
        }
        all_metrics[h] = metrics_test
        all_dm[h] = dm_h

    # -- 8.3  Summary, figures, export ----------------------------------------
    print_final_summary(all_metrics, all_dm, test_counts)

    print(f"\n[2] Generating figures ...")
    figure_forecasts(results)
    figure_loss_comparison(all_metrics)
    figure_selected_events(results)
    figure_improvement(all_metrics, all_dm)

    print(f"\n[3] Exporting results ...")
    export_all(results, all_metrics, all_dm)

    if placebo > 0:
        run_placebo(df_base, ev, placebo, corr_threshold, min_events,
                    recent_years, min_recent)

    print(f"""
{SEP}
  HOW TO READ THIS
{THIN}
  HAR      -> HAR-DOW  isolates day-of-week seasonality.
  HAR-DOW  -> HAR-X    isolates the SCHEDULED NEWS contribution. This is the
                       number that belongs next to FiLM-TCN: both models see
                       the same calendar over the same window, so whatever
                       FiLM-TCN gains beyond HAR-X is attributable to the
                       architecture rather than to the news.

  The DM columns say whether a gap is distinguishable from zero at all. A
  headline % improvement with p > 0.10 is not evidence of anything.

  Deviation from Plihal, restated: he rolls a 1000-day window and re-estimates
  daily; this uses the repo's fixed split so the deep-model comparison is like
  for like. Rolling would favour HAR-X, so this is the conservative choice.

  References:
    Corsi, F. (2009). A simple approximate long-memory model of realized
      volatility. Journal of Financial Econometrics, 7(2), 174-196.
    Tibshirani, R. (1996). Regression shrinkage and selection via the lasso.
      JRSS-B, 58(1), 267-288.
    Bergmeir, C., & Benitez, J. M. (2012). On the use of cross-validation for
      time series predictor evaluation. Information Sciences, 191, 192-213.
    Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy.
      Journal of Business & Economic Statistics, 13(3), 253-263.
    Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of
      prediction mean squared errors. Int. J. of Forecasting, 13(2), 281-291.
    Patton, A. J. (2011). Volatility forecast comparison using imperfect
      volatility proxies. Journal of Econometrics, 160(1), 246-256.
{SEP}
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HAR-X: HAR + scheduled macro news announcements, LASSO-selected. "
                    "Same train/test split as HAR_RV_run.py.")
    parser.add_argument("--data_path", type=str, default=DATA_FILE,
                        help="RV CSV (date index + 'RV' levels or 'ln_RV').")
    parser.add_argument("--event_path", type=str, default=EVENT_FILE,
                        help="Wide daily event matrix: 'date' + n_events*/evt_* columns.")
    parser.add_argument("--event_set", type=str, default="all",
                        choices=["all", "events", "counts"],
                        help="Which event block enters the model.")
    parser.add_argument("--corr_threshold", type=float, default=CORR_THRESHOLD,
                        help="Merge event columns correlated above this (Plihal: 0.90).")
    parser.add_argument("--min_events", type=float, default=MIN_EVENTS,
                        help="Minimum training-window mass for an event column.")
    parser.add_argument("--recent_years", type=int, default=RECENT_YEARS,
                        help="Tail of TRAIN used to check a column is still live.")
    parser.add_argument("--min_recent", type=float, default=MIN_RECENT,
                        help="Minimum firings in that tail (0 disables the screen).")
    parser.add_argument("--placebo", type=int, default=0, metavar="N",
                        help="Also re-run on N row-scrambled calendars as a "
                             "falsification check (2 is plenty).")
    args = parser.parse_args()
    RECENT_YEARS = args.recent_years
    MIN_RECENT = args.min_recent
    main(args.data_path, args.event_path, args.event_set,
         args.corr_threshold, args.min_events,
         args.recent_years, args.min_recent, args.placebo)
