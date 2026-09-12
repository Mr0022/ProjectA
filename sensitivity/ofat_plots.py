#!/usr/bin/env python3
"""
Academic figures for the EventTCN OFAT hyperparameter-sensitivity sweep.

Reads sensitivity/ofat_results.csv (produced by ofat_sensitivity.py) and writes,
to sensitivity/figures/ :

  1. ofat_response_mse.(pdf|png)   -- response curve per hyperparameter, MSE
  2. ofat_response_qlike.(pdf|png) -- same, QLIKE (finance loss, Patton 2011)
  3. ofat_tornado_mse.(pdf|png)    -- signed swing vs anchor, ranked
  4. ofat_sensitivity_bar.(pdf|png)-- local sensitivity (metric range / anchor)
  5. ofat_event_dim.(pdf|png)      -- EventTCN embedding-width spotlight
  6. ofat_summary.csv              -- per-point mean/std + ranking table

Design: one measure per axis (never a twin axis); colorblind-safe hues; a mean
line with a +/-1 std band and translucent per-seed dots; the tuned anchor marked
on every panel.  OFAT curves are LOCAL sensitivity -- valid around the optimum.

    python sensitivity/ofat_plots.py
    python sensitivity/ofat_plots.py --results sensitivity/ofat_results.csv --metric mse
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ofat_sensitivity import ANCHOR, GRIDS, ORDER, value_key  # noqa: E402

# --- validated colorblind-safe palette (dataviz reference instance) ----------
BLUE, ORANGE, RED, GREEN = "#2a78d6", "#eb6834", "#e34948", "#008300"
INK, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
METRIC_HUE = {"mse": BLUE, "mae": BLUE, "qlike": ORANGE, "rse": BLUE}

PRETTY = {
    "seq_len": "look-back length  (seq_len)",
    "patch_size": "patch size",
    "patch_stride": "patch stride",
    "dim": "model width  (dims)",
    "ffn_ratio": "ConvFFN ratio",
    "large_size": "large kernel size",
    "small_size": "small kernel size",
    "num_blocks": "blocks per stage",
    "dropout": "dropout",
    "head_dropout": "head dropout",
    "learning_rate": "learning rate",
    "batch_size": "batch size",
    "revin": "RevIN",
    "event_dim": "event embedding dim  (event_dim)",
}
LOG_X = {"learning_rate"}  # numeric log axis; everything else is ordinal


def fmt_val(param, v):
    if param == "learning_rate":
        return f"{v:g}"
    if param in ("dropout", "head_dropout"):
        return f"{v:.2f}"
    if param == "revin":
        return "off" if v == 0 else "on"
    return f"{int(v)}" if float(v).is_integer() else f"{v:g}"


plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK, "text.color": INK,
    "figure.dpi": 120, "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load(results_csv):
    # value is a categorical key (repr of the swept value) -- keep it as text so
    # it matches value_key(); pandas would otherwise coerce it to float.
    df = pd.read_csv(results_csv, dtype={"value": str})
    df["value"] = df["value"].fillna("")
    for c in ["mse", "mae", "rse", "qlike"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    anchor = df[df["param"] == "anchor"]
    if anchor.empty:
        raise SystemExit("No anchor rows in results -- run the anchor point first.")
    return df, anchor


def series_for(df, anchor, param, metric):
    """Return (values, per_seed_lists, means, stds) sorted by value for `param`."""
    grid = sorted(set(GRIDS[param]) | {ANCHOR[param]})
    vals, seeds, means, stds = [], [], [], []
    for v in grid:
        if v == ANCHOR[param]:
            sub = anchor
        else:
            sub = df[(df["param"] == param) & (df["value"] == value_key(v))]
        s = sub[metric].dropna().values
        if len(s) == 0:
            continue
        vals.append(v)
        seeds.append(s)
        means.append(float(np.mean(s)))
        stds.append(float(np.std(s, ddof=1)) if len(s) > 1 else 0.0)
    return vals, seeds, np.array(means), np.array(stds)


# ---------------------------------------------------------------------------
# Figure 1/2 -- response-curve small multiples
# ---------------------------------------------------------------------------
def fig_response(df, anchor, metric, params, out):
    hue = METRIC_HUE[metric]
    n = len(params)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.7 * nrows),
                             squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)

    for i, p in enumerate(params):
        ax = axes.flat[i]
        ax.set_visible(True)
        vals, seeds, means, stds = series_for(df, anchor, p, metric)
        if len(vals) < 2:
            ax.set_title(PRETTY.get(p, p) + "  (insufficient data)", color=MUTED)
            continue

        if p in LOG_X:
            x = np.array(vals, dtype=float)
            ax.set_xscale("log")
            xpos = x
        else:
            xpos = np.arange(len(vals))          # ordinal spacing
            ax.set_xticks(xpos)
            ax.set_xticklabels([fmt_val(p, v) for v in vals], rotation=0)

        # +/-1 std band + mean line
        ax.fill_between(xpos, means - stds, means + stds, color=hue, alpha=0.16,
                        linewidth=0)
        ax.plot(xpos, means, "-", color=hue, lw=1.8, zorder=3)
        ax.plot(xpos, means, "o", color=hue, ms=4.5, zorder=4)

        # translucent per-seed dots
        for xp, s in zip(xpos, seeds):
            jit = (np.random.default_rng(0).random(len(s)) - 0.5) * 0.06
            ax.plot(np.full(len(s), xp) + (0 if p in LOG_X else jit), s,
                    ".", color=hue, alpha=0.30, ms=4, zorder=2)

        # anchor marker
        av = ANCHOR[p]
        axpos = av if p in LOG_X else (vals.index(av) if av in vals else None)
        if axpos is not None:
            ai = vals.index(av)
            ax.axvline(axpos, color=AXIS, ls="--", lw=0.9, zorder=1)
            ax.plot([axpos], [means[ai]], "D", color=RED, ms=7,
                    mec="white", mew=0.8, zorder=6)

        ax.set_title(PRETTY.get(p, p))
        ax.set_ylabel(metric.upper())
        ax.grid(axis="x", visible=False)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle(
        f"EventTCN (h=1) one-factor-at-a-time sensitivity  --  test {metric.upper()}"
        f"\nline = 5-seed mean, band = ±1 std, dots = seeds, "
        f"red ◆ = tuned anchor",
        fontsize=11, y=1.005)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200)
    plt.close(fig)
    print(f"wrote {out}.pdf / .png")


# ---------------------------------------------------------------------------
# Figure 3 -- tornado (signed swing vs anchor)
# ---------------------------------------------------------------------------
def fig_tornado(df, anchor, metric, params, out):
    rows = []
    a_mean = float(anchor[metric].mean())
    for p in params:
        vals, seeds, means, stds = series_for(df, anchor, p, metric)
        if len(vals) < 2:
            continue
        improvement = max(0.0, a_mean - means.min())   # best achievable gain
        degradation = max(0.0, means.max() - a_mean)   # worst worsening
        rows.append((p, improvement, degradation, improvement + degradation))
    rows.sort(key=lambda r: r[3])                      # ascending -> biggest on top
    if not rows:
        return

    ys = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(rows) + 1.5))
    for y, (p, imp, deg, _) in zip(ys, rows):
        ax.barh(y, -imp, color=GREEN, alpha=0.85, height=0.62)
        ax.barh(y,  deg, color=RED,   alpha=0.85, height=0.62)
    ax.axvline(0, color=INK, lw=1.0)
    ax.set_yticks(ys)
    ax.set_yticklabels([PRETTY.get(p, p) for p, *_ in rows])
    ax.set_xlabel(f"Δ test {metric.upper()} vs tuned anchor")
    ax.set_title(f"Local sensitivity of EventTCN (h=1): swing in {metric.upper()} "
                 f"around the anchor")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GREEN, label="better than anchor"),
                       Patch(color=RED, label="worse than anchor")],
              frameon=False, loc="lower right")
    ax.grid(axis="y", visible=False)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200)
    plt.close(fig)
    print(f"wrote {out}.pdf / .png")


# ---------------------------------------------------------------------------
# Figure 4 -- relative-sensitivity bar
# ---------------------------------------------------------------------------
def fig_sensitivity_bar(df, anchor, metric, params, out):
    rows = []
    a_mean = float(anchor[metric].mean())
    for p in params:
        vals, seeds, means, stds = series_for(df, anchor, p, metric)
        if len(vals) < 2:
            continue
        rng = (means.max() - means.min()) / a_mean * 100.0
        rows.append((p, rng))
    rows.sort(key=lambda r: r[1])
    if not rows:
        return
    ys = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(rows) + 1.2))
    ax.barh(ys, [r[1] for r in rows], color=BLUE, alpha=0.9, height=0.62)
    ax.set_yticks(ys)
    ax.set_yticklabels([PRETTY.get(p, p) for p, _ in rows])
    ax.set_xlabel(f"{metric.upper()} range across grid, as % of anchor {metric.upper()}")
    ax.set_title(f"How much each hyperparameter moves EventTCN (h=1) {metric.upper()}")
    ax.grid(axis="y", visible=False)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200)
    plt.close(fig)
    print(f"wrote {out}.pdf / .png")


# ---------------------------------------------------------------------------
# Figure 5 -- event_dim spotlight (two single-axis panels)
# ---------------------------------------------------------------------------
def fig_event_dim(df, anchor, out):
    if "event_dim" not in ORDER:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, metric, hue in zip(axes, ("mse", "qlike"), (BLUE, ORANGE)):
        vals, seeds, means, stds = series_for(df, anchor, "event_dim", metric)
        if len(vals) < 2:
            ax.set_title("event_dim (insufficient data)", color=MUTED)
            continue
        xpos = np.arange(len(vals))
        ax.errorbar(xpos, means, yerr=stds, fmt="-o", color=hue, lw=1.8, ms=6,
                    capsize=4, zorder=3)
        for xp, s in zip(xpos, seeds):
            ax.plot(np.full(len(s), xp), s, ".", color=hue, alpha=0.30, ms=4)
        ai = vals.index(ANCHOR["event_dim"]) if ANCHOR["event_dim"] in vals else None
        if ai is not None:
            ax.plot([xpos[ai]], [means[ai]], "D", color=RED, ms=8, mec="white",
                    mew=0.8, zorder=5)
        ax.set_xticks(xpos)
        ax.set_xticklabels([str(int(v)) for v in vals])
        ax.set_xlabel("event_dim")
        ax.set_ylabel(f"test {metric.upper()}")
        ax.grid(axis="x", visible=False)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("EventTCN news-embedding width (event_dim) sensitivity, h=1"
                 "   (red ◆ = tuned anchor)", fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200)
    plt.close(fig)
    print(f"wrote {out}.pdf / .png")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def write_summary(df, anchor, params, out_csv):
    recs = []
    for p in params:
        for metric in ("mse", "qlike"):
            vals, seeds, means, stds = series_for(df, anchor, p, metric)
            for v, s in zip(vals, seeds):
                recs.append({
                    "param": p, "value": fmt_val(p, v), "metric": metric,
                    "n": len(s), "mean": float(np.mean(s)),
                    "std": float(np.std(s, ddof=1)) if len(s) > 1 else 0.0,
                    "is_anchor": (v == ANCHOR[p]),
                })
    pd.DataFrame(recs).to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="sensitivity/ofat_results.csv")
    ap.add_argument("--outdir", default="sensitivity/figures")
    ap.add_argument("--metric", default="mse", choices=["mse", "mae", "rse", "qlike"],
                    help="metric for the tornado / sensitivity-bar figures")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df, anchor = load(args.results)

    # only plot params that actually have swept rows (+ the anchor)
    present = [p for p in ORDER
               if not df[(df["param"] == p)].empty]
    if not present:
        raise SystemExit("Only the anchor is present -- sweep at least one param.")

    fig_response(df, anchor, "mse", present, os.path.join(args.outdir, "ofat_response_mse"))
    fig_response(df, anchor, "qlike", present, os.path.join(args.outdir, "ofat_response_qlike"))
    fig_tornado(df, anchor, args.metric, present, os.path.join(args.outdir, f"ofat_tornado_{args.metric}"))
    fig_sensitivity_bar(df, anchor, args.metric, present, os.path.join(args.outdir, "ofat_sensitivity_bar"))
    fig_event_dim(df, anchor, os.path.join(args.outdir, "ofat_event_dim"))
    write_summary(df, anchor, present, os.path.join(args.outdir, "ofat_summary.csv"))
    print("\nAll figures in", args.outdir)


if __name__ == "__main__":
    main()
