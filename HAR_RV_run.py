"""
==============================================================================
HAR-RV Model: Multi-Horizon Realized Volatility Forecasting
Based on: Corsi, F. (2009). A simple approximate long-memory model of
          realized volatility. Journal of Financial Econometrics, 7(2), 174-196.

Horizons : h = 1  (daily),  h = 5  (weekly),  h = 22  (monthly)

Split logic mirrors Dataset_Custom (data_provider/data_loader.py) exactly:
    Dataset_Custom : train 2010-2021, val 2022-2023, test 2024-2025
    HAR-RV (OLS)   : train 2010-2023, test 2024-2025

    The validation window (2022-2023) is folded into the training sample
    because OLS has no hyperparameters to tune. The test window (2024-2025)
    is IDENTICAL to the deep learning baseline, enabling a fair
    out-of-sample comparison.

Target construction (Corsi, 2009; Bollerslev et al., 2016):
    For horizon h, the dependent variable is the h-day forward average:

        Y_t^(h) = (1/h) * Sum_{k=1}^{h}  ln(RV_{t+k})

    The regressors (RV_d, RV_w, RV_m) use information available AT time t
    (i.e. they include today's RV_t), so predicting the average over
    [t+1 .. t+h] is a genuine 1-step-ahead forecast. This matches the
    information set used by the HAR-residual deep-learning hybrids
    (Dataset_HAR_Residual), whose look-back window also ends at t. The
    regressors are IDENTICAL across horizons -- only the target changes.

HAC bandwidth (Patton & Sheppard, 2009; Bollerslev et al., 2016):
    L = 2*(h-1), applied via Newey-West (1987) Bartlett kernel.
    h=1  -> L=0  (standard OLS SEs; no overlap)
    h=5  -> L=8
    h=22 -> L=42

Metrics   : MSE, MAE, QLIKE (Patton, 2011) -- computed on ln(RV) scale

Usage:
    python HAR_RV_run.py
    python HAR_RV_run.py --data_path ./data/realized_volatility.csv
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
import matplotlib.dates  as mdates
from   matplotlib.ticker import AutoMinorLocator

import statsmodels.api as sm
from   statsmodels.regression.linear_model import OLS
from   statsmodels.stats.stattools         import durbin_watson
from   statsmodels.stats.diagnostic        import acorr_ljungbox
from   scipy                               import stats


# ==============================================================================
# 0.  CONFIGURATION
# ==============================================================================

DATA_FILE  = "./data/realized_volatility.csv"

# -- Output directory ----------------------------------------------------------
# All figures and CSVs are written here. Created automatically if missing.
OUTPUT_DIR = "HAR-RV results"

def out_path(filename: str) -> str:
    """Return the full path inside OUTPUT_DIR for a given output filename."""
    return os.path.join(OUTPUT_DIR, filename)

# -- Year-based split boundaries (mirrors Dataset_Custom in data_loader.py) ----
#
#   Dataset_Custom does:
#       train_end = (df['date'].dt.year <= 2021).sum()
#       val_end   = (df['date'].dt.year <= 2023).sum()
#       test      = remainder (year >= 2024)
#
#   HAR-RV folds val into train (OLS has no hyperparameters to tune):
#       TRAIN_END_YEAR  = 2023   -> train rows where year <= 2023
#       TEST_START_YEAR = 2024   -> test  rows where year >= 2024
#
#   The test window 2024-2025 is identical to the DL model test set.
# ------------------------------------------------------------------------------
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
DPI_SAVE = 300

# -- Publication palette -------------------------------------------------------
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


# ==============================================================================
# 1.  DATA LOADING & BASE REGRESSOR CONSTRUCTION
# ==============================================================================

def load_base_features(filepath: str) -> pd.DataFrame:
    """
    Load ln(RV) series and build the three HAR regressors.

    The CSV must have a date column (parsed as index) and a numeric column
    named 'ln_RV' (or the first numeric column will be used).

    Regressors are identical across all forecast horizons -- only the
    target variable Y^(h) changes.  All three use information available
    AT time t (they include today's RV_t); the target then spans the
    future window [t+1 .. t+h], so there is no look-ahead.

        RV_d : ln(RV_t)
        RV_w : mean( ln(RV_t), ..., ln(RV_{t-4})  )
        RV_m : mean( ln(RV_t), ..., ln(RV_{t-21}) )
    """
    raw = pd.read_csv(filepath, index_col=0, parse_dates=True)
    raw.index.name = "date"
    col = "ln_RV" if "ln_RV" in raw.columns else raw.select_dtypes("number").columns[0]
    s   = raw[col].sort_index().dropna()

    df = pd.DataFrame({"ln_RV": s})
    df["RV_d"] = df["ln_RV"]
    df["RV_w"] = df["ln_RV"].rolling(LAG_W).mean()
    df["RV_m"] = df["ln_RV"].rolling(LAG_M).mean()
    return df   # NaN rows dropped per-horizon after target is attached


# ==============================================================================
# 2.  HORIZON-SPECIFIC TARGET CONSTRUCTION
# ==============================================================================

def build_horizon_target(df_base: pd.DataFrame, h: int) -> pd.DataFrame:
    """
    Construct the h-day forward average log-RV as the dependent variable.

        Y_t^(h) = (1/h) * Sum_{k=1}^{h}  ln(RV_{t+k})

    The regressors in load_base_features include today's value (RV_d =
    ln(RV_t), etc.), so the newest information available at row t is from t.
    The target therefore spans [t+1 .. t+h], making every horizon a genuine
    1-step-ahead forecast:

        h=1  -> Y_t = ln(RV_{t+1})
        h=5  -> Y_t = mean( ln(RV_{t+1}), ..., ln(RV_{t+5})  )
        h=22 -> Y_t = mean( ln(RV_{t+1}), ..., ln(RV_{t+22}) )

    rolling(h).mean() gives the trailing average over [t-h+1 .. t]; the
    final shift(-h) slides that window forward to [t+1 .. t+h].

    Parameters
    ----------
    df_base : DataFrame with columns ['ln_RV', 'RV_d', 'RV_w', 'RV_m']
    h       : forecast horizon in trading days

    Returns
    -------
    DataFrame with 'Y_h' added; rows with any NaN dropped.
    """
    df = df_base.copy()
    df["Y_h"] = df["ln_RV"].rolling(h).mean().shift(-h)
    df = df.dropna(subset=["Y_h", "RV_d", "RV_w", "RV_m"])
    return df


# ==============================================================================
# 3.  YEAR-BASED TRAIN / TEST SPLIT
#     Mirrors Dataset_Custom in data_loader.py.
#
#     Dataset_Custom logic (condensed):
#         train_end = (df['date'].dt.year <= 2021).sum()
#         val_end   = (df['date'].dt.year <= 2023).sum()
#         border1s  = [0,          train_end - seq_len,  val_end - seq_len]
#         border2s  = [train_end,  val_end,              len(df)]
#
#     For HAR-RV (no seq_len offset needed -- regressors are scalar lags,
#     and val is folded into train):
#         train : rows where index.year <= TRAIN_END_YEAR  (2023)
#         test  : rows where index.year >= TEST_START_YEAR (2024)
#
#     The boundary is computed on the full date-indexed DataFrame BEFORE
#     any NaN-dropping so the year cut-points are stable across horizons.
# ==============================================================================

def split_by_year(df: pd.DataFrame,
                  train_end_year: int  = TRAIN_END_YEAR,
                  test_start_year: int = TEST_START_YEAR):
    """
    Chronological year-based split matching Dataset_Custom (val folded into train).

    Parameters
    ----------
    df               : DatetimeIndex DataFrame (post dropna)
    train_end_year   : last year (inclusive) in training set  -> 2023
    test_start_year  : first year (inclusive) in test set     -> 2024

    Returns
    -------
    train, test : DataFrames
    """
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


# ==============================================================================
# 4.  HAR-RV OLS ESTIMATION  (per horizon)
# ==============================================================================

def fit_har_rv(train: pd.DataFrame, h: int):
    """
    Estimate HAR-RV by OLS for forecast horizon h.

        Y_t^(h) = b0 + b_d*RV_d + b_w*RV_w + b_m*RV_m + e_t

    HAC bandwidth L = 2*(h-1), Newey-West (1987) Bartlett kernel.
    h=1 -> L=0 recovers standard OLS SEs (no overlapping observations).
    """
    nw_lag = HORIZONS[h]["nw_lag"]
    Y = train["Y_h"]
    X = sm.add_constant(train[["RV_d", "RV_w", "RV_m"]])
    result = OLS(Y, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": max(nw_lag, 1), "use_correction": True},
    )
    return result


# ==============================================================================
# 5.  LOSS FUNCTIONS
# ==============================================================================

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


# ==============================================================================
# 6.  DIAGNOSTICS
# ==============================================================================

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


# ==============================================================================
# 7.  PRINT TABLES
# ==============================================================================

def print_section(title: str):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def print_split_info(train: pd.DataFrame, test: pd.DataFrame, h: int):
    """Print split summary matching Dataset_Custom style reporting."""
    print(f"\n  -- Split (mirrors Dataset_Custom, val folded into train) ----")
    print(f"  {'Set':<8} {'Rows':>6}  {'Start':>12}  {'End':>12}  {'Years'}")
    print(f"  {THIN[:60]}")
    print(f"  {'Train':<8} {len(train):>6}  "
          f"{str(train.index[0].date()):>12}  "
          f"{str(train.index[-1].date()):>12}  "
          f"2010 - {TRAIN_END_YEAR}")
    print(f"  {'Test':<8} {len(test):>6}  "
          f"{str(test.index[0].date()):>12}  "
          f"{str(test.index[-1].date()):>12}  "
          f"{TEST_START_YEAR} - 2025")
    print(f"  {THIN[:60]}")
    print(f"  Note: validation window (2022-2023) folded into train -- OLS")
    print(f"        has no hyperparameters. Test window matches DL model.\n")

def print_estimation_table(result, h: int):
    nw_lag   = HORIZONS[h]["nw_lag"]
    hlabel   = HORIZONS[h]["label"]
    params   = result.params
    bse      = result.bse
    tvals    = result.tvalues
    pvals    = result.pvalues
    ci       = result.conf_int(alpha=0.05)
    name_map = {"const": "Intercept (b0)",
                "RV_d" : "Daily RV   (b_d)",
                "RV_w" : "Weekly RV  (b_w)",
                "RV_m" : "Monthly RV (b_m)"}

    hdr_nw = f"L={nw_lag} -- Corsi (2009) / Patton & Sheppard (2009)"
    print_section(f"HAR-RV  [{hlabel}]  Newey-West HAC  {hdr_nw}")
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
    print(f"\n  R2           : {result.rsquared:.4f}")
    print(f"  Adj. R2      : {result.rsquared_adj:.4f}")
    print(f"  F-statistic  : {result.fvalue:.3f}   [p = {result.f_pvalue:.4f}]")
    print(f"  Observations : {int(result.nobs):,}")
    print(f"  AIC          : {result.aic:.3f}   BIC : {result.bic:.3f}")

def print_metrics_by_horizon(all_metrics: dict):
    print_section("OUT-OF-SAMPLE FORECAST EVALUATION -- ALL HORIZONS  (Test Set  2024-2025)")
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
    print("  Note: All metrics on ln(RV) scale.  QLIKE: Patton (2011), smaller = better.")

def print_diagnostics(diag: dict, h: int):
    hlabel = HORIZONS[h]["label"]
    print_section(f"RESIDUAL DIAGNOSTICS  [h={h}, {hlabel}, Training Sample 2010-{TRAIN_END_YEAR}]")
    print(f"  {'Statistic':<30} {'Value':>12}")
    print(THIN)
    for k, v in diag.items():
        print(f"  {k:<30} {v:>12.4f}")
    print(THIN)


# ==============================================================================
# 8.  FIGURES
# ==============================================================================

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
                ls="--", alpha=0.90, label=f"HAR-RV  h={h}")
        ax.fill_between(t_idx, actual.values, pred.values,
                        color="#aaaaaa", alpha=0.18, label="Error")
        ax.set_ylabel(f"$\\bar{{\\ln(RV)}}^{{({h})}}$", fontsize=10)
        ax.set_title(
            f"{panel_letters[idx]}  {HORIZONS[h]['label']} -- Out-of-Sample Forecast  "
            f"({TEST_START_YEAR}-2025)",
            loc="left", pad=4)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.legend(loc="upper right", fontsize=8)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    fig.suptitle(
        f"HAR-RV Multi-Horizon Forecasts: h = 1, 5, 22  (Corsi, 2009)\n"
        f"Train: 2010-{TRAIN_END_YEAR}  |  Test: {TEST_START_YEAR}-2025",
        fontsize=12, fontweight="bold", y=1.01)
    fig.savefig(out_path("har_rv_fig1_multihoriz_forecast.pdf"), bbox_inches="tight")
    fig.savefig(out_path("har_rv_fig1_multihoriz_forecast.png"), dpi=DPI_SAVE, bbox_inches="tight")
    print(f"  -> Saved: {out_path('har_rv_fig1_multihoriz_forecast.pdf')} / .png")
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
        f"HAR-RV: Out-of-Sample Loss by Horizon -- Test Set  {TEST_START_YEAR}-2025",
        fontsize=12, fontweight="bold")
    fig.savefig(out_path("har_rv_fig2_loss_comparison.pdf"), bbox_inches="tight")
    fig.savefig(out_path("har_rv_fig2_loss_comparison.png"), dpi=DPI_SAVE, bbox_inches="tight")
    print(f"  -> Saved: {out_path('har_rv_fig2_loss_comparison.pdf')} / .png")
    plt.close(fig)


def figure_coefficients_across_horizons(results_dict: dict):
    horizons      = list(HORIZONS.keys())
    coef_names    = {"const": "b0  (Intercept)",
                     "RV_d" : "b_d (Daily)",
                     "RV_w" : "b_w (Weekly)",
                     "RV_m" : "b_m (Monthly)"}
    coef_keys     = list(coef_names.keys())
    x_pos         = np.arange(len(horizons))
    x_labels      = [f"h={h}" for h in horizons]
    panel_letters = ["(A)", "(B)", "(C)", "(D)"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.subplots_adjust(hspace=0.45, wspace=0.35)
    for i, key in enumerate(coef_keys):
        ax     = axes[i // 2, i % 2]
        coeffs = [results_dict[h]["result"].params[key] for h in horizons]
        errs   = [1.96 * results_dict[h]["result"].bse[key] for h in horizons]
        ax.errorbar(x_pos, coeffs, yerr=errs,
                    fmt="o-", color=C_TRAIN, lw=1.2, ms=6,
                    capsize=4, capthick=1, elinewidth=0.9,
                    label="Estimate +/- 1.96*SE")
        ax.axhline(0, color="gray", lw=0.7, ls="--")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.set_title(f"{panel_letters[i]}  {coef_names[key]}", loc="left", pad=3)
        ax.set_ylabel("Coefficient")
        ax.set_xlabel("Forecast Horizon")
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.legend(fontsize=8)

    fig.suptitle("HAR-RV: Coefficient Paths Across Forecast Horizons  (h = 1, 5, 22)",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.savefig(out_path("har_rv_fig3_coeff_paths.pdf"), bbox_inches="tight")
    fig.savefig(out_path("har_rv_fig3_coeff_paths.png"), dpi=DPI_SAVE, bbox_inches="tight")
    print(f"  -> Saved: {out_path('har_rv_fig3_coeff_paths.pdf')} / .png")
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
        f"HAR-RV: ACF of Squared Residuals -- Training Sample 2010-{TRAIN_END_YEAR} "
        f"(Vertical line = NW bandwidth)",
        fontsize=11, fontweight="bold")
    fig.savefig(out_path("har_rv_fig4_acf_residuals.pdf"), bbox_inches="tight")
    fig.savefig(out_path("har_rv_fig4_acf_residuals.png"), dpi=DPI_SAVE, bbox_inches="tight")
    print(f"  -> Saved: {out_path('har_rv_fig4_acf_residuals.pdf')} / .png")
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
                label="45 deg line")
        slope, intercept, r, *_ = stats.linregress(pred, act)
        xs = np.array([lo, hi])
        ax.plot(xs, intercept + slope * xs, color=C_TEST, lw=1.0,
                label=f"OLS fit  R2={r**2:.3f}")
        ax.set_xlabel(f"Predicted  $\\bar{{\\ln(RV)}}^{{({h})}}$", fontsize=9)
        ax.set_ylabel(f"Actual     $\\bar{{\\ln(RV)}}^{{({h})}}$", fontsize=9)
        ax.set_title(f"{panel_letters[idx]}  h={h}  [{HORIZONS[h]['label']}]",
                     loc="left", pad=3)
        ax.legend(fontsize=8)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    fig.suptitle(
        f"HAR-RV: Actual vs. Predicted -- Test Set  {TEST_START_YEAR}-2025  (all horizons)",
        fontsize=12, fontweight="bold")
    fig.savefig(out_path("har_rv_fig5_scatter_grid.pdf"), bbox_inches="tight")
    fig.savefig(out_path("har_rv_fig5_scatter_grid.png"), dpi=DPI_SAVE, bbox_inches="tight")
    print(f"  -> Saved: {out_path('har_rv_fig5_scatter_grid.pdf')} / .png")
    plt.close(fig)


# ==============================================================================
# 9.  EXPORT
# ==============================================================================

def export_all(results_dict: dict, all_metrics: dict):
    # -- 9a. Fitted values (one CSV per horizon) ------------------------------
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
        fname = out_path(f"har_rv_h{h:02d}_fitted.csv")
        full.to_csv(fname)
        print(f"  -> Saved: {fname}")

    # -- 9b. Consolidated metrics table ---------------------------------------
    rows = []
    for h in results_dict:
        for split in ["train", "test"]:
            m = all_metrics[h][split]
            rows.append({"horizon": h, "split": split,
                         "MSE": m["MSE"], "MAE": m["MAE"], "QLIKE": m["QLIKE"]})
    metrics_path = out_path("har_rv_all_metrics.csv")
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    print(f"  -> Saved: {metrics_path}")

    # -- 9c. Parameter tables (one CSV per horizon) ---------------------------
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
        fname = out_path(f"har_rv_h{h:02d}_params.csv")
        pt.to_csv(fname)
        print(f"  -> Saved: {fname}")


# ==============================================================================
# 10.  MAIN PIPELINE
# ==============================================================================

def main(data_file: str = DATA_FILE):
    print(f"\n{SEP}")
    print("  HAR-RV MULTI-HORIZON MODEL  --  Corsi (2009)")
    print("  EUR/USD Realized Volatility  |  Horizons: h = 1, 5, 22")
    print(f"  Split: Train 2010-{TRAIN_END_YEAR}  |  Test {TEST_START_YEAR}-2025")
    print(f"  (Split mirrors Dataset_Custom in data_loader.py for DL comparison)")
    print(SEP)

    # -- 10.0  Prepare output directory ---------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n  Output directory: '{OUTPUT_DIR}/' (figures + CSVs)")

    # -- 10.1  Load base features ---------------------------------------------
    print(f"\n[1] Loading data from '{data_file}' ...")
    df_base = load_base_features(data_file)
    print(f"  Full series  : {df_base.index[0].date()} -> {df_base.index[-1].date()}")
    print(f"  Total rows   : {len(df_base):,}  (before per-horizon NaN drop)")

    # -- 10.2  Per-horizon loop ------------------------------------------------
    results_dict = {}
    all_metrics  = {}

    for h in HORIZONS:
        hlabel = HORIZONS[h]["label"]
        print(f"\n{'-'*72}")
        print(f"  HORIZON  h = {h}  [{hlabel}]")
        print(f"{'-'*72}")

        # Build horizon-specific dataset then apply year-based split
        df_h         = build_horizon_target(df_base, h)
        train, test  = split_by_year(df_h)
        print_split_info(train, test, h)

        # Estimate
        result      = fit_har_rv(train, h)
        print_estimation_table(result, h)

        # Predict
        X_tr        = sm.add_constant(train[["RV_d", "RV_w", "RV_m"]])
        X_te        = sm.add_constant(test[["RV_d",  "RV_w", "RV_m"]])
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
        }
        all_metrics[h] = {"train": m_train, "test": m_test}

    # -- 10.3  Cross-horizon comparison table ----------------------------------
    print_metrics_by_horizon(all_metrics)

    # -- 10.4  Figures ---------------------------------------------------------
    print(f"\n[2] Generating publication figures ...")
    figure_multi_horizon_forecast(results_dict)
    figure_loss_comparison(all_metrics)
    figure_coefficients_across_horizons(results_dict)
    figure_residual_diagnostics_multihoriz(results_dict)
    figure_scatter_grid(results_dict)

    # -- 10.5  Export ----------------------------------------------------------
    print(f"\n[3] Exporting results ...")
    export_all(results_dict, all_metrics)

    # -- 10.6  Summary ---------------------------------------------------------
    print(f"\n{SEP}")
    print(f"  FINAL SUMMARY -- OUT-OF-SAMPLE TEST SET  ({TEST_START_YEAR}-2025)")
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
  Split:
    Train  2010 - {TRAIN_END_YEAR}  (year <= {TRAIN_END_YEAR}, val folded in)
    Test   {TEST_START_YEAR} - 2025   (year >= {TEST_START_YEAR})
    Logic mirrors Dataset_Custom in data_loader.py -- test window is
    IDENTICAL to the deep learning baseline for a fair comparison.

  References:
    Corsi, F. (2009). A simple approximate long-memory model of realized
      volatility. Journal of Financial Econometrics, 7(2), 174-196.
    Patton, A. J. (2011). Volatility forecast comparison using imperfect
      volatility proxies. Journal of Econometrics, 160(1), 246-256.
    Patton, A. J., & Sheppard, K. (2009). Evaluating volatility and
      correlation forecasts. Handbook of Financial Time Series. Springer.
    Bollerslev, T., Patton, A. J., & Quaedvlieg, R. (2016). Exploiting
      the errors. Journal of Econometrics, 192(1), 1-18.
""")
    print(SEP)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HAR-RV multi-horizon realized-volatility baseline (Corsi, 2009).")
    parser.add_argument("--data_path", type=str, default=DATA_FILE,
                        help="Path to the ln(RV) CSV (date index + 'ln_RV' column).")
    args = parser.parse_args()
    main(args.data_path)
