"""
==============================================================================
HAR-Q Model: Multi-Horizon Realized Volatility Forecasting
Based on: Bollerslev, T., Patton, A. J., & Quaedvlieg, R. (2016). Exploiting
          the errors: A simple approach for improved volatility forecasting.
          Journal of Econometrics, 192(1), 1-18.
          (Builds on Corsi, F. (2009), J. Financial Econometrics, 7(2), 174-196.)

Horizons : h = 1  (daily),  h = 5  (weekly),  h = 22  (monthly)

HAR-Q augments HAR-RV by letting the DAILY-lag coefficient vary with the
measurement error in RV, proxied by the realized quarticity (RQ):

    Y_t^(h) = β₀ + (β_d + β_1Q·√RQ_{t-1})·RV_{t-1}
                 + β_w·RV_w + β_m·RV_m + ε_t

Intuition (BPQ 2016): when yesterday's RV is a noisy estimate of true
integrated variance (large RQ), its signal is less reliable, so the model
attenuates its coefficient toward the smoother weekly/monthly components.
This is the headline "HAR-Q" specification (daily lag only), NOT "HAR-Q-F"
(which would also adjust the weekly and monthly lags).

Target construction:
    For horizon h, the dependent variable is the log of the AVERAGE realized
    variance over the forecast window -- the same target as HAR-RV and the
    deep baselines (--aggregate_mean), and ProjectC's convention:

        Y_t^(h) = ln( (1/h) * Sum_{k=1}^{h}  RV_{t+k} )

    It differs from the log of the window SUM by the constant ln(h), which
    the OLS intercept absorbs.

Split logic mirrors Dataset_Custom (data_provider/data_loader.py):
    Dataset_Custom : train <=2021, val 2022-2023, test >=2024
    HAR-Q (OLS)    : train <=2023,                test >=2024

    The validation window (2022-2023) is folded into the training sample
    because HAR-Q is OLS with no hyperparameters to tune.

------------------------------------------------------------------------------
DATA SOURCE -- DIFFERENT FROM THE OTHER MODELS (read before publishing)
------------------------------------------------------------------------------
HAR-RV, LSTM and ModernTCN default to data/EURUSD-RV.csv. HAR-Q does NOT: it
needs realized quarticity, and that file carries only RV. The sole RQ source in
this repo is data/realized_volatility_with_rqq.csv, whose RV series is a
DIFFERENT series from EURUSD-RV.csv (they disagree on most overlapping dates
and their trading calendars differ), so HAR-Q stays on that file.

    => HAR-Q numbers are NOT directly comparable to the other models.

To put HAR-Q on the same footing, add an 'RQ' column (built from the same
intraday returns as the EURUSD RV) to data/EURUSD-RV.csv and run:

    python HAR_Q_run.py --data_path ./data/EURUSD-RV.csv

------------------------------------------------------------------------------
IMPORTANT — DATA REQUIREMENT
------------------------------------------------------------------------------
HAR-Q requires realized quarticity in addition to realized volatility. This
script looks for one of the following columns in the CSV:
    level : 'RQ', 'rq', 'realized_quarticity', 'quarticity'
    log   : 'ln_RQ', 'log_RQ', 'lnRQ'   (exponentiated internally)
If none is found, the script raises a clear error. RQ cannot be recovered
from RV alone — it must come from the same intraday returns used to build RV,
as RQ_t = (N/3)·Σ r_{t,i}^4.

------------------------------------------------------------------------------
NOTE ON LOG SCALE (read before publishing)
------------------------------------------------------------------------------
The BPQ (2016) HAR-Q is derived in RV *levels*, where the measurement-error
variance of RV is proportional to RQ (so √RQ is the natural noise scale).
This pipeline works in ln(RV) to stay identical to the deep-learning baseline
(metrics on ln(RV) scale). Two Q-transforms are provided:

    Q_TRANSFORM = "sqrt"   →  Q = √RQ_{t-1}                (canonical BPQ form)
    Q_TRANSFORM = "ratio"  →  Q = √RQ_{t-1} / RV_{t-1}     (delta-method form,
                                                            more consistent
                                                            with a log target)

Default is "sqrt" (literal HAR-Q). The "ratio" form is arguably more coherent
with a log-RV target: by the delta method, Var(ln RV − ln IV) ≈ 2·RQ/RV², so
√RQ/RV is the natural log-space noise scale. Pick one and report it.

The √RQ term is demeaned by its TRAINING-sample mean (BPQ convention) purely
for interpretability of β_d — this is an exact reparametrization and does not
change fitted values or out-of-sample forecasts.

HAC bandwidth (Patton & Sheppard, 2009; Bollerslev et al., 2016):
    L = 2*(h-1), applied via Newey-West (1987) Bartlett kernel.
    h=1  → L=0  (standard OLS SEs; no overlap)
    h=5  → L=8
    h=22 → L=42

Metrics   : MSE, MAE, QLIKE (Patton, 2011) — computed on the log scale of the
            target, i.e. on ln(mean RV)

Usage:
    python HAR_Q_run.py
    python HAR_Q_run.py --data_path ./data/realized_volatility_with_rqq.csv
    python HAR_Q_run.py --q_transform ratio
==============================================================================
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import argparse
import os
import warnings
warnings.filterwarnings("ignore")

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates  as mdates
from   matplotlib.ticker import AutoMinorLocator

import statsmodels.api as sm
from   statsmodels.regression.linear_model import OLS
from   statsmodels.stats.stattools         import durbin_watson
from   statsmodels.stats.diagnostic        import acorr_ljungbox
from   scipy                               import stats

from utils.rv import pick_rv_column, to_log_rv, forward_log_mean


# ══════════════════════════════════════════════════════════════════════════════
# 0.  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

DATA_FILE  = "./data/realized_volatility_with_rqq.csv"

# ── Output directory ──────────────────────────────────────────────────────────
# All figures and CSVs are written here. Created automatically if missing.
OUTPUT_DIR = "HAR-Q results"

def out_path(filename: str) -> str:
    """Return the full path inside OUTPUT_DIR for a given output filename."""
    return os.path.join(OUTPUT_DIR, filename)

# ── Year-based split boundaries (mirrors Dataset_Custom in data_loader.py) ───
TRAIN_END_YEAR  = 2023
TEST_START_YEAR = 2024

# Forecast horizons and their Newey-West bandwidths: L = 2*(h-1)
HORIZONS = {
    1  : {"label": "Daily   (h=1)",  "nw_lag":  0},
    5  : {"label": "Weekly  (h=5)",  "nw_lag":  8},
    22 : {"label": "Monthly (h=22)", "nw_lag": 42},
}

LAG_W    = 5    # weekly component window
LAG_M    = 22   # monthly component window

# First/last year actually present in the loaded file. Set by load_base_features
# so printouts and figure captions follow the data instead of hardcoded years.
SAMPLE_START_YEAR = None
SAMPLE_END_YEAR   = None
DPI_SAVE = 300

# ── HAR-Q realized-quarticity settings ───────────────────────────────────────
Q_TRANSFORM = "sqrt"   # "sqrt"  → √RQ_{t-1}   (canonical BPQ 2016)
                       # "ratio" → √RQ_{t-1}/RV_{t-1}  (delta-method, log-scale)
RQ_LEVEL_CANDIDATES = ["RQ", "rq", "realized_quarticity", "quarticity"]
RQ_LOG_CANDIDATES   = ["ln_RQ", "log_RQ", "lnRQ", "ln_rq"]

# ── Publication palette ───────────────────────────────────────────────────────
C_ACTUAL = "#1a1a2e"
C_H      = {1: "#2166ac", 5: "#fc8d59", 22: "#d73027"}
C_TRAIN  = "#2166ac"
C_TEST   = "#d73027"

plt.rcParams.update({
    "font.family":         "serif",
    "font.serif":          ["Times New Roman", "DejaVu Serif"],
    "font.size":           11,
    "axes.titlesize":      11,
    "axes.labelsize":      11,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.grid":           True,
    "grid.alpha":          0.30,
    "grid.linestyle":      ":",
    "grid.linewidth":      0.6,
    "lines.linewidth":     1.0,
    "figure.dpi":          150,
    "savefig.dpi":         DPI_SAVE,
    "savefig.bbox":        "tight",
    "savefig.facecolor":   "white",
    "legend.framealpha":   0.90,
    "legend.edgecolor":    "0.75",
    "legend.fontsize":     9,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
})

SEP  = "=" * 72
THIN = "-" * 72


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING & BASE REGRESSOR CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def _extract_rq_level(raw: pd.DataFrame) -> pd.Series:
    """
    Locate a realized-quarticity column and return it in LEVELS.

    Accepts either a level column (RQ) or a log column (ln_RQ, exponentiated).
    Raises a clear, actionable error if neither is present — RQ cannot be
    derived from RV alone.
    """
    for c in RQ_LEVEL_CANDIDATES:
        if c in raw.columns:
            return raw[c].astype(float)
    for c in RQ_LOG_CANDIDATES:
        if c in raw.columns:
            return np.exp(raw[c].astype(float))
    raise ValueError(
        "HAR-Q requires realized quarticity (RQ), which was not found in "
        f"'{DATA_FILE}'.\n"
        f"    Expected a level column, one of: {RQ_LEVEL_CANDIDATES}\n"
        f"    or a log column,       one of: {RQ_LOG_CANDIDATES}\n"
        "    RQ must be built from the same intraday returns as RV:\n"
        "        RQ_t = (N/3) * sum_i r_{t,i}^4\n"
        "    It cannot be recovered from RV alone. Add the column and re-run."
    )


def load_base_features(filepath: str) -> pd.DataFrame:
    """
    Load ln(RV) + RQ and build the HAR regressors plus the HAR-Q Q-term.

    Regressors identical across horizons — only the target Y^(h) changes.
    All use information strictly before time t (shift-1) → no look-ahead.

        RV_d  : ln(RV_{t-1})
        RV_w  : mean( ln(RV_{t-1}), …, ln(RV_{t-5})  )
        RV_m  : mean( ln(RV_{t-1}), …, ln(RV_{t-22}) )
        Q_raw : √RQ_{t-1}              (Q_TRANSFORM="sqrt")
              : √RQ_{t-1} / RV_{t-1}   (Q_TRANSFORM="ratio")

    The HAR-Q interaction regressor RV_d × (Q_raw − mean) is formed later,
    after the train/test split, so the demeaning uses training data only.
    """
    raw = pd.read_csv(filepath, index_col=0, parse_dates=True)
    raw.index.name = "date"
    raw = raw.sort_index()

    col = pick_rv_column(raw.columns)
    s   = raw[col].astype(float).dropna()

    ln_rv, keep = to_log_rv(s.values, col)
    s = pd.Series(ln_rv, index=s.index)[keep]

    global SAMPLE_START_YEAR, SAMPLE_END_YEAR
    SAMPLE_START_YEAR = int(s.index.year.min())
    SAMPLE_END_YEAR   = int(s.index.year.max())

    df = pd.DataFrame({"ln_RV": s})
    df["RV_d"] = df["ln_RV"].shift(1)
    df["RV_w"] = df["ln_RV"].shift(1).rolling(LAG_W).mean()
    df["RV_m"] = df["ln_RV"].shift(1).rolling(LAG_M).mean()

    # ── Realized quarticity → Q-term (lagged to match RV_d timing) ───────────
    rq_level = _extract_rq_level(raw).reindex(s.index)
    if (rq_level < 0).any():
        raise ValueError("Realized quarticity contains negative values; check the data.")
    rq_lag = rq_level.shift(1)                       # RQ_{t-1}

    if Q_TRANSFORM == "sqrt":
        df["Q_raw"] = np.sqrt(rq_lag)
    elif Q_TRANSFORM == "ratio":
        rv_lag_level = np.exp(df["ln_RV"].shift(1))  # RV_{t-1} in levels
        df["Q_raw"]  = np.sqrt(rq_lag) / rv_lag_level
    else:
        raise ValueError(f"Unknown Q_TRANSFORM='{Q_TRANSFORM}' (use 'sqrt' or 'ratio').")

    return df   # NaN rows dropped per-horizon after target is attached


# ══════════════════════════════════════════════════════════════════════════════
# 2.  HORIZON-SPECIFIC TARGET CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_horizon_target(df_base: pd.DataFrame, h: int) -> pd.DataFrame:
    """
    Construct the log of the h-day forward AVERAGE RV as the dependent variable.

        Y_t^(h) = ln( (1/h) * Σ_{k=0}^{h-1}  RV_{t+k} )

    Alignment note: this script lags its regressors by one day (RV_d =
    ln(RV_{t-1}), see load_base_features), so the forecast window that sits
    one step ahead of the information set is [t .. t+h-1] -- hence the
    shift(-(h-1)) below. HAR_RV_run.py and ProjectC use the same convention,
    which is what makes `year >= TEST_START_YEAR` select the same test
    forecasts, and the same number of them, as Dataset_Custom.

        h=1  -> Y_t = ln(RV_t)
        h=5  -> Y_t = ln( (RV_t + ... + RV_{t+4})  / 5  )
        h=22 -> Y_t = ln( (RV_t + ... + RV_{t+21}) / 22 )
    """
    df = df_base.copy()
    if h == 1:
        df["Y_h"] = df["ln_RV"]
    else:
        # forward_log_mean gives ln(mean RV[t+1 .. t+h]); shift back by one to
        # land on this script's [t .. t+h-1] window.
        df["Y_h"] = forward_log_mean(df["ln_RV"], h).shift(1)
    df = df.dropna(subset=["Y_h", "RV_d", "RV_w", "RV_m", "Q_raw"])
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3.  YEAR-BASED TRAIN / TEST SPLIT   (mirrors Dataset_Custom)
# ══════════════════════════════════════════════════════════════════════════════

def split_by_year(df: pd.DataFrame,
                  train_end_year: int  = TRAIN_END_YEAR,
                  test_start_year: int = TEST_START_YEAR):
    """Chronological year-based split identical to Dataset_Custom."""
    train = df[df.index.year <= train_end_year].copy()
    test  = df[df.index.year >= test_start_year].copy()

    if len(train) == 0:
        raise ValueError(
            f"Train set is empty. Check that your data contains years <= {train_end_year}."
        )
    if len(test) == 0:
        raise ValueError(
            f"Test set is empty. Check that your data contains years >= {test_start_year}."
        )
    return train, test


# ══════════════════════════════════════════════════════════════════════════════
# 4.  HAR-Q DESIGN MATRIX & OLS ESTIMATION  (per horizon)
# ══════════════════════════════════════════════════════════════════════════════

def build_design(df: pd.DataFrame, q_bar: float) -> pd.DataFrame:
    """
    Build the HAR-Q design matrix (constant included).

    Columns (order matters for readable tables):
        const  RV_d  RV_dQ  RV_w  RV_m
    where RV_dQ = (Q_raw − q_bar) · RV_d  is the measurement-error interaction.

    q_bar is the TRAINING-sample mean of Q_raw, passed in so that train, test
    and prediction all use identical construction (no look-ahead on the demean).
    """
    X = pd.DataFrame(index=df.index)
    X["RV_d"]  = df["RV_d"]
    X["RV_dQ"] = (df["Q_raw"] - q_bar) * df["RV_d"]
    X["RV_w"]  = df["RV_w"]
    X["RV_m"]  = df["RV_m"]
    return sm.add_constant(X)


def fit_har_q(train: pd.DataFrame, h: int, q_bar: float):
    """
    Estimate HAR-Q by OLS for forecast horizon h.

        Y_t^(h) = β₀ + β_d·RV_d + β_1Q·(√RQ−mean)·RV_d + β_w·RV_w + β_m·RV_m + ε

    HAC bandwidth L = 2*(h-1), Newey-West (1987) Bartlett kernel.
    """
    nw_lag = HORIZONS[h]["nw_lag"]
    Y = train["Y_h"]
    X = build_design(train, q_bar)
    result = OLS(Y, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": max(nw_lag, 1), "use_correction": True},
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 5.  LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean((actual - predicted) ** 2))

def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))

def qlike(actual_ln: np.ndarray, predicted_ln: np.ndarray) -> float:
    """QLIKE loss (Patton, 2011). Smaller = better."""
    rv_act  = np.exp(actual_ln)
    rv_pred = np.exp(predicted_ln)
    valid   = (rv_act > 0) & (rv_pred > 0)
    ratio   = rv_act[valid] / rv_pred[valid]
    return float(np.mean(ratio - np.log(ratio) - 1))

def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    return {"MSE": mse(actual, predicted),
            "MAE": mae(actual, predicted),
            "QLIKE": qlike(actual, predicted)}


# ══════════════════════════════════════════════════════════════════════════════
# 6.  DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

def residual_diagnostics(residuals: pd.Series, lags: int = 20) -> dict:
    dw            = durbin_watson(residuals)
    lb            = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    jb_stat, jb_p = stats.jarque_bera(residuals)
    return {
        "DW"      : dw,
        "LB_stat" : float(lb["lb_stat"].iloc[0]),
        "LB_p"    : float(lb["lb_pvalue"].iloc[0]),
        "JB_stat" : jb_stat,
        "JB_p"    : jb_p,
        "Skew"    : float(stats.skew(residuals)),
        "Kurt"    : float(stats.kurtosis(residuals)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 7.  PRINT TABLES
# ══════════════════════════════════════════════════════════════════════════════

def print_section(title: str):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def print_split_info(train: pd.DataFrame, test: pd.DataFrame, h: int):
    print(f"\n  ── Split (mirrors Dataset_Custom, val folded into train) ────")
    print(f"  {'Set':<8} {'Rows':>6}  {'Start':>12}  {'End':>12}  {'Years'}")
    print(f"  {THIN[:60]}")
    print(f"  {'Train':<8} {len(train):>6}  "
          f"{str(train.index[0].date()):>12}  "
          f"{str(train.index[-1].date()):>12}  "
          f"{SAMPLE_START_YEAR} – {TRAIN_END_YEAR}")
    print(f"  {'Test':<8} {len(test):>6}  "
          f"{str(test.index[0].date()):>12}  "
          f"{str(test.index[-1].date()):>12}  "
          f"{TEST_START_YEAR} – {SAMPLE_END_YEAR}")
    print(f"  {THIN[:60]}")
    print(f"  Note: validation window (2022-2023) folded into train — OLS")
    print(f"        has no hyperparameters. Test window matches DL model.\n")

def print_estimation_table(result, h: int, q_bar: float):
    nw_lag   = HORIZONS[h]["nw_lag"]
    hlabel   = HORIZONS[h]["label"]
    params   = result.params
    bse      = result.bse
    tvals    = result.tvalues
    pvals    = result.pvalues
    ci       = result.conf_int(alpha=0.05)
    name_map = {"const": "Intercept (β₀)",
                "RV_d" : "Daily RV   (β_d)",
                "RV_dQ": "Daily×Q    (β_1Q)",
                "RV_w" : "Weekly RV  (β_w)",
                "RV_m" : "Monthly RV (β_m)"}

    q_desc = "√RQ_{t-1}" if Q_TRANSFORM == "sqrt" else "√RQ_{t-1}/RV_{t-1}"
    hdr_nw = f"L={nw_lag} — BPQ (2016) / Patton & Sheppard (2009)"
    print_section(f"HAR-Q  [{hlabel}]  Newey-West HAC  {hdr_nw}")
    print(f"  Q-term: β_1Q multiplies (Q − mean)·RV_d,  Q = {q_desc}"
          f"  [demean mean = {q_bar:.6g}]")
    hdr = f"{'Variable':<22} {'Coeff':>10} {'Std.Err':>10} {'t-stat':>10} {'p-value':>10}  {'95% CI'}"
    print(hdr)
    print(THIN)
    for var in params.index:
        sig   = "***" if pvals[var]<0.01 else ("**" if pvals[var]<0.05 else
                ("*"  if pvals[var]<0.10 else ""))
        label = name_map.get(var, var)
        print(f"{label:<22} {params[var]:>10.4f} {bse[var]:>10.4f} "
              f"{tvals[var]:>10.3f} {pvals[var]:>10.4f}  "
              f"[{ci.loc[var,0]:7.4f}, {ci.loc[var,1]:7.4f}]{sig}")
    print(THIN)
    print("Significance: *** p<0.01  ** p<0.05  * p<0.10")
    print(f"\n  R²           : {result.rsquared:.4f}")
    print(f"  Adj. R²      : {result.rsquared_adj:.4f}")
    print(f"  F-statistic  : {result.fvalue:.3f}   [p = {result.f_pvalue:.4f}]")
    print(f"  Observations : {int(result.nobs):,}")
    print(f"  AIC          : {result.aic:.3f}   BIC : {result.bic:.3f}")

def print_metrics_by_horizon(all_metrics: dict):
    print_section(f"OUT-OF-SAMPLE FORECAST EVALUATION — ALL HORIZONS  "
                  f"(Test Set  {TEST_START_YEAR}–{SAMPLE_END_YEAR})")
    print(f"  {'Metric':<10}", end="")
    for h in HORIZONS:
        print(f"  {HORIZONS[h]['label']:>20}", end="")
    print()
    print(THIN)
    for metric in ["MSE", "MAE", "QLIKE"]:
        print(f"  {metric:<10}", end="")
        for h in HORIZONS:
            v = all_metrics[h]["test"][metric]
            print(f"  {v:>20.6f}", end="")
        print()
    print(THIN)
    print("  Note: MSE/MAE on the ln(mean RV) scale. QLIKE compares exp() of the\n        target, i.e. average variance over the window; the ln(h) offset\n        cancels in its ratio, so it is comparable across conventions.")

def print_diagnostics(diag: dict, h: int):
    hlabel = HORIZONS[h]["label"]
    print_section(f"RESIDUAL DIAGNOSTICS  [h={h}, {hlabel}, Training Sample {SAMPLE_START_YEAR}–{TRAIN_END_YEAR}]")
    print(f"  {'Statistic':<30} {'Value':>12}")
    print(THIN)
    for k, v in diag.items():
        print(f"  {k:<30} {v:>12.4f}")
    print(THIN)


# ══════════════════════════════════════════════════════════════════════════════
# 8.  FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def figure_multi_horizon_forecast(results_dict):
    horizons = list(HORIZONS.keys())
    fig, axes = plt.subplots(len(horizons), 1,
                             figsize=(11, 3.8 * len(horizons)),
                             sharex=False)
    fig.subplots_adjust(hspace=0.52)
    panel_letters = ["(A)", "(B)", "(C)"]

    for idx, h in enumerate(horizons):
        ax     = axes[idx]
        t_idx  = results_dict[h]["y_hat_test"].index
        actual = results_dict[h]["df_test"]["Y_h"]
        pred   = results_dict[h]["y_hat_test"]

        ax.plot(t_idx, actual.values, color=C_ACTUAL,  lw=0.9,
                alpha=0.85, label="Actual $Y^{(h)}$")
        ax.plot(t_idx, pred.values,   color=C_H[h],    lw=1.0,
                ls="--", alpha=0.90, label=f"HAR-Q  h={h}")
        ax.fill_between(t_idx, actual.values, pred.values,
                        color="#aaaaaa", alpha=0.18, label="Error")
        ax.set_ylabel(f"$\\bar{{\\ln(RV)}}^{{({h})}}$", fontsize=10)
        ax.set_title(
            f"{panel_letters[idx]}  {HORIZONS[h]['label']} — Out-of-Sample Forecast  "
            f"({TEST_START_YEAR}–{SAMPLE_END_YEAR})",
            loc="left", pad=4)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.legend(loc="upper right", fontsize=8)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    fig.suptitle(
        f"HAR-Q Multi-Horizon Forecasts: h = 1, 5, 22  (Bollerslev et al., 2016)\n"
        f"Train: {SAMPLE_START_YEAR}–{TRAIN_END_YEAR}  |  Test: {TEST_START_YEAR}–{SAMPLE_END_YEAR}",
        fontsize=12, fontweight="bold", y=1.01)
    fig.savefig(out_path("har_q_fig1_multihoriz_forecast.pdf"), bbox_inches="tight")
    fig.savefig(out_path("har_q_fig1_multihoriz_forecast.png"), dpi=DPI_SAVE, bbox_inches="tight")
    print(f"  → Saved: {out_path('har_q_fig1_multihoriz_forecast.pdf')} / .png")
    plt.close(fig)


def figure_loss_comparison(all_metrics: dict):
    metrics_list  = ["MSE", "MAE", "QLIKE"]
    horizons      = list(HORIZONS.keys())
    labels        = [f"h={h}" for h in horizons]
    colors        = [C_H[h] for h in horizons]
    panel_letters = ["(A)", "(B)", "(C)"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.subplots_adjust(wspace=0.4)
    for idx, metric in enumerate(metrics_list):
        ax   = axes[idx]
        vals = [all_metrics[h]["test"][metric] for h in horizons]
        bars = ax.bar(labels, vals, color=colors, alpha=0.75,
                      edgecolor="white", width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.015,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(f"{panel_letters[idx]}  {metric}", loc="left", pad=3)
        ax.set_ylabel(metric)
        ax.set_xlabel("Forecast Horizon")
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    fig.suptitle(
        f"HAR-Q: Out-of-Sample Loss by Horizon — Test Set  {TEST_START_YEAR}–{SAMPLE_END_YEAR}",
        fontsize=12, fontweight="bold")
    fig.savefig(out_path("har_q_fig2_loss_comparison.pdf"), bbox_inches="tight")
    fig.savefig(out_path("har_q_fig2_loss_comparison.png"), dpi=DPI_SAVE, bbox_inches="tight")
    print(f"  → Saved: {out_path('har_q_fig2_loss_comparison.pdf')} / .png")
    plt.close(fig)


def figure_coefficients_across_horizons(results_dict: dict):
    horizons      = list(HORIZONS.keys())
    coef_names    = {"const": "β₀  (Intercept)",
                     "RV_d" : "β_d (Daily)",
                     "RV_dQ": "β_1Q (Daily×Q)",
                     "RV_w" : "β_w (Weekly)",
                     "RV_m" : "β_m (Monthly)"}
    coef_keys     = list(coef_names.keys())
    x_pos         = np.arange(len(horizons))
    x_labels      = [f"h={h}" for h in horizons]
    panel_letters = ["(A)", "(B)", "(C)", "(D)", "(E)"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.subplots_adjust(hspace=0.45, wspace=0.35)
    axes = axes.ravel()
    for i, key in enumerate(coef_keys):
        ax     = axes[i]
        coeffs = [results_dict[h]["result"].params[key] for h in horizons]
        errs   = [1.96 * results_dict[h]["result"].bse[key] for h in horizons]
        ax.errorbar(x_pos, coeffs, yerr=errs,
                    fmt="o-", color=C_TRAIN, lw=1.2, ms=6,
                    capsize=4, capthick=1, elinewidth=0.9,
                    label="Estimate ± 1.96·SE")
        ax.axhline(0, color="gray", lw=0.7, ls="--")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.set_title(f"{panel_letters[i]}  {coef_names[key]}", loc="left", pad=3)
        ax.set_ylabel("Coefficient")
        ax.set_xlabel("Forecast Horizon")
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.legend(fontsize=8)
    for j in range(len(coef_keys), len(axes)):
        axes[j].axis("off")

    fig.suptitle("HAR-Q: Coefficient Paths Across Forecast Horizons  (h = 1, 5, 22)",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.savefig(out_path("har_q_fig3_coeff_paths.pdf"), bbox_inches="tight")
    fig.savefig(out_path("har_q_fig3_coeff_paths.png"), dpi=DPI_SAVE, bbox_inches="tight")
    print(f"  → Saved: {out_path('har_q_fig3_coeff_paths.pdf')} / .png")
    plt.close(fig)


def figure_residual_diagnostics_multihoriz(results_dict: dict):
    horizons      = list(HORIZONS.keys())
    max_lag       = 44
    ci_mult       = 1.96
    panel_letters = ["(A)", "(B)", "(C)"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.subplots_adjust(wspace=0.38)
    for idx, h in enumerate(horizons):
        ax    = axes[idx]
        resid = results_dict[h]["result"].resid
        n     = len(resid)
        acf_sq = [pd.Series(resid.values**2).autocorr(lag=k)
                  for k in range(1, max_lag + 1)]
        ci_b  = ci_mult / np.sqrt(n)
        nw_l  = HORIZONS[h]["nw_lag"]

        ax.bar(range(1, max_lag + 1), acf_sq,
               color=C_H[h], alpha=0.70, width=0.7)
        ax.axhline( ci_b, color="black", ls="--", lw=0.8, label="95% CI")
        ax.axhline(-ci_b, color="black", ls="--", lw=0.8)
        ax.axhline(0,     color="black", lw=0.5)
        if nw_l > 0:
            ax.axvline(nw_l, color=C_TEST, lw=1.0, ls=":",
                       label=f"NW lag L={nw_l}")
        ax.set_title(f"{panel_letters[idx]}  h={h}  [{HORIZONS[h]['label']}]",
                     loc="left", pad=3)
        ax.set_xlabel("Lag (days)")
        ax.set_ylabel("ACF (squared residuals)")
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.legend(fontsize=8)

    fig.suptitle(
        f"HAR-Q: ACF of Squared Residuals — Training Sample {SAMPLE_START_YEAR}–{TRAIN_END_YEAR} "
        f"(Vertical line = NW bandwidth)",
        fontsize=11, fontweight="bold")
    fig.savefig(out_path("har_q_fig4_acf_residuals.pdf"), bbox_inches="tight")
    fig.savefig(out_path("har_q_fig4_acf_residuals.png"), dpi=DPI_SAVE, bbox_inches="tight")
    print(f"  → Saved: {out_path('har_q_fig4_acf_residuals.pdf')} / .png")
    plt.close(fig)


def figure_scatter_grid(results_dict: dict):
    horizons      = list(HORIZONS.keys())
    panel_letters = ["(A)", "(B)", "(C)"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.subplots_adjust(wspace=0.38)
    for idx, h in enumerate(horizons):
        ax   = axes[idx]
        act  = results_dict[h]["df_test"]["Y_h"].values
        pred = results_dict[h]["y_hat_test"].values
        lo   = min(act.min(), pred.min())
        hi   = max(act.max(), pred.max())

        ax.scatter(pred, act, s=10, color=C_H[h],
                   alpha=0.40, edgecolors="none", label="Test obs.")
        ax.plot([lo, hi], [lo, hi], color="black", lw=1.0, ls="--",
                label="45° line")
        slope, intercept, r, *_ = stats.linregress(pred, act)
        xs = np.array([lo, hi])
        ax.plot(xs, intercept + slope * xs, color=C_TEST, lw=1.0,
                label=f"OLS fit  R²={r**2:.3f}")
        ax.set_xlabel(f"Predicted  $\\bar{{\\ln(RV)}}^{{({h})}}$", fontsize=9)
        ax.set_ylabel(f"Actual     $\\bar{{\\ln(RV)}}^{{({h})}}$", fontsize=9)
        ax.set_title(f"{panel_letters[idx]}  h={h}  [{HORIZONS[h]['label']}]",
                     loc="left", pad=3)
        ax.legend(fontsize=8)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    fig.suptitle(
        f"HAR-Q: Actual vs. Predicted — Test Set  {TEST_START_YEAR}–{SAMPLE_END_YEAR}  (all horizons)",
        fontsize=12, fontweight="bold")
    fig.savefig(out_path("har_q_fig5_scatter_grid.pdf"), bbox_inches="tight")
    fig.savefig(out_path("har_q_fig5_scatter_grid.png"), dpi=DPI_SAVE, bbox_inches="tight")
    print(f"  → Saved: {out_path('har_q_fig5_scatter_grid.pdf')} / .png")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 9.  EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_all(results_dict: dict, all_metrics: dict):
    # ── 9a. Fitted values (one CSV per horizon) ──────────────────────────────
    for h, res in results_dict.items():
        out = res["df_train"][["Y_h"]].copy()
        out["fitted"]   = res["y_hat_train"].values
        out["residual"] = out["Y_h"] - out["fitted"]
        out["split"]    = "train"
        test_out = res["df_test"][["Y_h"]].copy()
        test_out["fitted"]   = res["y_hat_test"].values
        test_out["residual"] = test_out["Y_h"] - test_out["fitted"]
        test_out["split"]    = "test"
        full = pd.concat([out, test_out]).sort_index()
        fname = out_path(f"har_q_h{h:02d}_fitted.csv")
        full.to_csv(fname)
        print(f"  → Saved: {fname}")

    # ── 9b. Consolidated metrics table ───────────────────────────────────────
    rows = []
    for h in results_dict:
        for split in ["train", "test"]:
            m = all_metrics[h][split]
            rows.append({"horizon": h, "split": split,
                         "MSE": m["MSE"], "MAE": m["MAE"], "QLIKE": m["QLIKE"]})
    metrics_path = out_path("har_q_all_metrics.csv")
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    print(f"  → Saved: {metrics_path}")

    # ── 9c. Parameter tables (one CSV per horizon) ───────────────────────────
    for h, res in results_dict.items():
        result = res["result"]
        ci     = result.conf_int()
        pt = pd.DataFrame({
            "Coefficient": result.params,
            "Std_Error"  : result.bse,
            "t_stat"     : result.tvalues,
            "p_value"    : result.pvalues,
            "CI_lower"   : ci[0],
            "CI_upper"   : ci[1],
        })
        fname = out_path(f"har_q_h{h:02d}_params.csv")
        pt.to_csv(fname)
        print(f"  → Saved: {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# 10.  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main(data_file: str = DATA_FILE):
    print(f"\n{SEP}")
    print("  HAR-Q MULTI-HORIZON MODEL  —  Bollerslev, Patton & Quaedvlieg (2016)")
    print("  EUR/USD Realized Volatility  |  Horizons: h = 1, 5, 22")
    print(f"  Split: Train <={TRAIN_END_YEAR}  |  Test >={TEST_START_YEAR}")
    print(f"  Q-term: {Q_TRANSFORM}   (daily-lag measurement-error adjustment)")
    print(f"  (Split mirrors Dataset_Custom in data_loader.py for DL comparison)")
    print(SEP)

    # ── 10.0  Prepare output directory ───────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n  Output directory: '{OUTPUT_DIR}/' (figures + CSVs)")

    # ── 10.1  Load base features ─────────────────────────────────────────────
    print(f"\n[1] Loading data from '{data_file}' …")
    df_base = load_base_features(data_file)
    print(f"  Full series  : {df_base.index[0].date()} → {df_base.index[-1].date()}")
    print(f"  Total rows   : {len(df_base):,}  (before per-horizon NaN drop)")

    # ── 10.2  Per-horizon loop ────────────────────────────────────────────────
    results_dict = {}
    all_metrics  = {}

    for h in HORIZONS:
        hlabel = HORIZONS[h]["label"]
        print(f"\n{'─'*72}")
        print(f"  HORIZON  h = {h}  [{hlabel}]")
        print(f"{'─'*72}")

        # Build horizon-specific dataset then apply year-based split
        df_h         = build_horizon_target(df_base, h)
        train, test  = split_by_year(df_h)
        print_split_info(train, test, h)

        # HAR-Q demean uses TRAINING mean of the Q-term only (no look-ahead)
        q_bar = float(train["Q_raw"].mean())

        # Estimate
        result = fit_har_q(train, h, q_bar)
        print_estimation_table(result, h, q_bar)

        # Predict (identical design construction, shared q_bar)
        X_tr        = build_design(train, q_bar)
        X_te        = build_design(test,  q_bar)
        y_hat_train = result.predict(X_tr)
        y_hat_test  = result.predict(X_te)

        # Metrics
        m_train = compute_metrics(train["Y_h"].values, y_hat_train.values)
        m_test  = compute_metrics(test["Y_h"].values,  y_hat_test.values)

        # Diagnostics (on training residuals)
        diag = residual_diagnostics(result.resid)
        print_diagnostics(diag, h)

        # Store
        results_dict[h] = {
            "result"      : result,
            "df_train"    : train,
            "df_test"     : test,
            "y_hat_train" : y_hat_train,
            "y_hat_test"  : y_hat_test,
            "q_bar"       : q_bar,
        }
        all_metrics[h] = {"train": m_train, "test": m_test}

    # ── 10.3  Cross-horizon comparison table ──────────────────────────────────
    print_metrics_by_horizon(all_metrics)

    # ── 10.4  Figures ─────────────────────────────────────────────────────────
    print(f"\n[2] Generating publication figures …")
    figure_multi_horizon_forecast(results_dict)
    figure_loss_comparison(all_metrics)
    figure_coefficients_across_horizons(results_dict)
    figure_residual_diagnostics_multihoriz(results_dict)
    figure_scatter_grid(results_dict)

    # ── 10.5  Export ──────────────────────────────────────────────────────────
    print(f"\n[3] Exporting results …")
    export_all(results_dict, all_metrics)

    # ── 10.6  Summary ─────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  FINAL SUMMARY — OUT-OF-SAMPLE TEST SET  ({TEST_START_YEAR}–{SAMPLE_END_YEAR})")
    print(THIN)
    hdr = f"  {'Horizon':<14} {'NW Lag':>8} {'MSE':>12} {'MAE':>12} {'QLIKE':>12}"
    print(hdr)
    print(THIN)
    for h in HORIZONS:
        m   = all_metrics[h]["test"]
        nwl = HORIZONS[h]["nw_lag"]
        print(f"  {HORIZONS[h]['label']:<14} {nwl:>8}  "
              f"{m['MSE']:>12.6f} {m['MAE']:>12.6f} {m['QLIKE']:>12.6f}")
    print(THIN)
    print(f"""
  Model:
    HAR-Q (Bollerslev, Patton & Quaedvlieg, 2016) — daily lag only.
    Daily coefficient is state-dependent on realized quarticity:
        β_d(t) = β_d + β_1Q · (Q_{{t-1}} − mean),   Q = {Q_TRANSFORM}
    A significant negative β_1Q means the daily lag is attenuated when
    yesterday's RV is a noisy volatility estimate (large RQ).

  Split:
    Train  {SAMPLE_START_YEAR} – {TRAIN_END_YEAR}  (year <= {TRAIN_END_YEAR}, val folded in)
    Test   {TEST_START_YEAR} – {SAMPLE_END_YEAR}   (year >= {TEST_START_YEAR})
    Logic mirrors Dataset_Custom in data_loader.py — test window is
    IDENTICAL to the deep learning baseline for a fair comparison.

  References:
    Bollerslev, T., Patton, A. J., & Quaedvlieg, R. (2016). Exploiting the
      errors: A simple approach for improved volatility forecasting.
      Journal of Econometrics, 192(1), 1-18.
    Corsi, F. (2009). A simple approximate long-memory model of realized
      volatility. Journal of Financial Econometrics, 7(2), 174-196.
    Patton, A. J. (2011). Volatility forecast comparison using imperfect
      volatility proxies. Journal of Econometrics, 160(1), 246-256.
    Patton, A. J., & Sheppard, K. (2009). Evaluating volatility and
      correlation forecasts. Handbook of Financial Time Series. Springer.
""")
    print(SEP)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HAR-Q multi-horizon realized-volatility model (Bollerslev, Patton & Quaedvlieg, 2016).")
    parser.add_argument("--data_path", type=str, default=DATA_FILE,
                        help="Path to the CSV (date index + an 'RV' level or 'ln_RV' column, "
                             "plus a realized-quarticity column).")
    parser.add_argument("--q_transform", type=str, default=Q_TRANSFORM,
                        choices=["sqrt", "ratio"],
                        help="HAR-Q Q-term: 'sqrt'=√RQ_{t-1} (canonical BPQ), "
                             "'ratio'=√RQ_{t-1}/RV_{t-1} (delta-method, log-scale).")
    args = parser.parse_args()
    Q_TRANSFORM = args.q_transform   # override module default from the CLI
    main(args.data_path)
