#!/usr/bin/env python3
"""
Academic figures for the ModernTCN OFAT hyperparameter-sensitivity sweep.

Reads the per-horizon CSV written by ofat_sensitivity.py and writes, to
sensitivity/figures/h<H>/ :

  1. ofat_response_mse.(pdf|png)   -- response curve per hyperparameter, MSE
  2. ofat_response_qlike.(pdf|png) -- same, QLIKE (finance loss, Patton 2011)
  3. ofat_tornado_mse.(pdf|png)    -- signed swing vs anchor, ranked
  4. ofat_sensitivity_bar.(pdf|png)-- local sensitivity (metric range / anchor)
  5. ofat_event_dim.(pdf|png)      -- EventTCN embedding-width spotlight (events runs only)
  6. ofat_summary.csv              -- per-point mean/std + ranking table

and, with --compare, one cross-horizon figure covering every horizon swept.

Design: one measure per axis (never a twin axis); colorblind-safe hues; a mean
line with a +/-1 std band and translucent per-seed dots; the tuned anchor marked
on every panel.  OFAT curves are LOCAL sensitivity -- valid around the optimum.

    python sensitivity/ofat_plots.py --pred_len 1
    python sensitivity/ofat_plots.py --pred_len 22 --metric qlike
    python sensitivity/ofat_plots.py --compare 1 5 22      # cross-horizon figure
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
import ofat_sensitivity as ofat  # noqa: E402
from ofat_sensitivity import (ANCHORS, GRIDS, ORDER, ORDER_EVENTS,  # noqa: E402
                              anchor_for, load_anchor, results_path, value_key)

# --- validated colorblind-safe palette (dataviz reference instance) ----------
# Categorical slots 1-3, in fixed order. Validated all-pairs on both surfaces:
#   light  worst CVD dE 9.2, worst normal-vision dE 24.0
#   dark   worst CVD dE 9.4, worst normal-vision dE 20.9
# Used for the three horizons in the cross-horizon figure.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
HORIZON_HUE = {1: BLUE, 5: ORANGE, 22: AQUA}

# Diverging pair for "better / worse than anchor" (polarity, not identity).
BETTER, WORSE = "#1baf7a", "#e34948"

INK, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"

# One series per panel, so the metric hue never sits adjacent to another series.
# The anchor is an ANNOTATION, not a series: it is marked by shape + a direct
# label in ink rather than by a second hue, which is what the old red diamond
# did -- red #e34948 against orange #eb6834 fails the normal-vision floor
# (dE 7.1, below 15), so on the QLIKE panels it was unreadable.
METRIC_HUE = {"mse": BLUE, "mae": BLUE, "qlike": ORANGE, "rse": BLUE}
ANCHOR_INK = INK

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


def model_name(use_events):
    return "EventTCN" if use_events else "ModernTCN"


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
    grid = sorted(set(GRIDS[param]) | {ofat.ANCHOR[param]})
    vals, seeds, means, stds = [], [], [], []
    for v in grid:
        if v == ofat.ANCHOR[param]:
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
def fig_response(df, anchor, metric, params, out, title_tag=""):
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
        av = ofat.ANCHOR[p]
        axpos = av if p in LOG_X else (vals.index(av) if av in vals else None)
        if axpos is not None:
            ai = vals.index(av)
            ax.axvline(axpos, color=AXIS, ls="--", lw=0.9, zorder=1)
            ax.plot([axpos], [means[ai]], "D", color=ANCHOR_INK, ms=7,
                    mec="white", mew=1.2, zorder=6)
            ax.annotate("tuned", (axpos, means[ai]), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=7, color=MUTED)

        ax.set_title(PRETTY.get(p, p))
        ax.set_ylabel(metric.upper())
        ax.grid(axis="x", visible=False)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle(
        f"{title_tag} one-factor-at-a-time sensitivity  --  test {metric.upper()}"
        f"\nline = seed mean, band = ±1 std, dots = individual seeds, "
        f"◆ = tuned anchor",
        fontsize=11, y=1.005)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200)
    plt.close(fig)
    print(f"wrote {out}.pdf / .png")


# ---------------------------------------------------------------------------
# Figure 3 -- tornado (signed swing vs anchor)
# ---------------------------------------------------------------------------
def fig_tornado(df, anchor, metric, params, out, title_tag=""):
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
        ax.barh(y, -imp, color=BETTER, alpha=0.9, height=0.62)
        ax.barh(y,  deg, color=WORSE,  alpha=0.9, height=0.62)
    ax.axvline(0, color=INK, lw=1.0)
    ax.set_yticks(ys)
    ax.set_yticklabels([PRETTY.get(p, p) for p, *_ in rows])
    ax.set_xlabel(f"Δ test {metric.upper()} vs tuned anchor")
    ax.set_title(f"Local sensitivity of {title_tag}: swing in {metric.upper()} "
                 f"around the tuned anchor")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BETTER, label="better than anchor"),
                       Patch(color=WORSE, label="worse than anchor")],
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
def fig_sensitivity_bar(df, anchor, metric, params, out, title_tag=""):
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
    ax.set_title(f"How much each hyperparameter moves {title_tag} {metric.upper()}")
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
def fig_event_dim(df, anchor, out, title_tag=""):
    if df[df["param"] == "event_dim"].empty:
        return   # plain-ModernTCN sweep: no event embedding to show
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
        a_ed = ofat.ANCHOR["event_dim"]
        ai = vals.index(a_ed) if a_ed in vals else None
        if ai is not None:
            ax.plot([xpos[ai]], [means[ai]], "D", color=ANCHOR_INK, ms=8,
                    mec="white", mew=1.2, zorder=5)
            ax.annotate("tuned", (xpos[ai], means[ai]), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=7, color=MUTED)
        ax.set_xticks(xpos)
        ax.set_xticklabels([str(int(v)) for v in vals])
        ax.set_xlabel("event_dim")
        ax.set_ylabel(f"test {metric.upper()}")
        ax.grid(axis="x", visible=False)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle(f"{title_tag} news-embedding width (event_dim) sensitivity"
                 "   (◆ = tuned anchor)", fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200)
    plt.close(fig)
    print(f"wrote {out}.pdf / .png")



# ---------------------------------------------------------------------------
# Cross-horizon figure -- which knobs matter, side by side across h
# ---------------------------------------------------------------------------
def fig_cross_horizon(per_h, metric, out, anchors=None):
    """per_h: {pred_len: (df, anchor, params)}. One grouped bar per parameter,
    one colour per horizon (categorical slots 1-3, validated all-pairs)."""
    horizons = sorted(per_h)
    anchors = anchors or {h: ANCHORS[h] for h in horizons}
    shared = [p for p in ORDER_EVENTS
              if all(p in per_h[h][2] for h in horizons)]
    if not shared or len(horizons) < 2:
        return

    # sensitivity = swing across the grid, as % of that horizon's anchor loss,
    # so the three horizons are on one comparable scale despite different losses
    data = {}
    for h in horizons:
        df, anchor, _ = per_h[h]
        anchor_for(h, anchors[h])
        a = float(anchor[metric].mean())
        data[h] = []
        for prm in shared:
            _, _, means, _ = series_for(df, anchor, prm, metric)
            data[h].append((means.max() - means.min()) / a * 100.0 if len(means) > 1 else np.nan)

    order = np.argsort([np.nanmean([data[h][i] for h in horizons])
                        for i in range(len(shared))])
    shared = [shared[i] for i in order]
    for h in horizons:
        data[h] = [data[h][i] for i in order]

    ys = np.arange(len(shared))
    bar_h = 0.8 / len(horizons)
    fig, ax = plt.subplots(figsize=(8.2, 0.62 * len(shared) + 1.8))
    for k, h in enumerate(horizons):
        off = (k - (len(horizons) - 1) / 2) * bar_h
        ax.barh(ys + off, data[h], height=bar_h * 0.88,      # 2px surface gap
                color=HORIZON_HUE.get(h, BLUE), label=f"h = {h}")
    ax.set_yticks(ys)
    ax.set_yticklabels([PRETTY.get(p, p) for p in shared])
    ax.set_xlabel(f"{metric.upper()} swing across the grid, as % of that horizon's anchor")
    ax.set_title("Which hyperparameters matter, by horizon\n"
                 "local OFAT sensitivity around each horizon's own tuned anchor",
                 fontsize=11)
    ax.legend(frameon=False, loc="lower right", title=None)
    ax.grid(axis="y", visible=False)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
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
                    "is_anchor": (v == ofat.ANCHOR[p]),
                })
    pd.DataFrame(recs).to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")


# ---------------------------------------------------------------------------
def _load_one(pred_len, use_events, results=None, tag=None):
    """Load one horizon's sweep and pin the matching anchor. -> (df, anchor, params)

    The anchor comes from the sidecar written by the runner, so a study centred
    somewhere other than the tuned default is plotted around ITS anchor.
    """
    path = results or results_path(pred_len, use_events, tag)
    if not os.path.exists(path):
        raise SystemExit(f"no results for h={pred_len}: {path} not found "
                         f"(run ofat_sensitivity.py --pred_len {pred_len} first)")
    anchor_for(pred_len, load_anchor(path, pred_len))
    df, anchor = load(path)
    names = ORDER_EVENTS if use_events else ORDER
    params = [p for p in names if not df[df["param"] == p].empty]
    if not params:
        raise SystemExit(f"h={pred_len}: only the anchor is present -- sweep at least one param.")
    return df, anchor, params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_len", type=int, default=None, choices=sorted(ANCHORS),
                    help="horizon to plot (omit when using --compare only)")
    ap.add_argument("--compare", type=int, nargs="+", default=None,
                    choices=sorted(ANCHORS),
                    help="also write one cross-horizon figure over these horizons")
    ap.add_argument("--use_events", action="store_true",
                    help="plot the EventTCN sweep instead of plain ModernTCN")
    ap.add_argument("--results", default=None,
                    help="override the results CSV (single-horizon mode only)")
    ap.add_argument("--tag", default=None,
                    help="study tag used by the sweep (e.g. --tag custom)")
    ap.add_argument("--outdir", default="sensitivity/figures")
    ap.add_argument("--metric", default="mse", choices=["mse", "mae", "rse", "qlike"],
                    help="metric for the tornado / sensitivity-bar / compare figures")
    args = ap.parse_args()

    if args.pred_len is None and not args.compare:
        ap.error("give --pred_len, --compare, or both")

    tag_model = model_name(args.use_events)

    if args.pred_len is not None:
        df, anchor, params = _load_one(args.pred_len, args.use_events,
                                       args.results, args.tag)
        outdir = os.path.join(args.outdir,
                              f"h{args.pred_len}" + (f"_{args.tag}" if args.tag else ""))
        os.makedirs(outdir, exist_ok=True)
        tag = f"{tag_model} (h={args.pred_len})"
        j = lambda n: os.path.join(outdir, n)
        fig_response(df, anchor, "mse", params, j("ofat_response_mse"), tag)
        fig_response(df, anchor, "qlike", params, j("ofat_response_qlike"), tag)
        fig_tornado(df, anchor, args.metric, params, j(f"ofat_tornado_{args.metric}"), tag)
        fig_sensitivity_bar(df, anchor, args.metric, params, j("ofat_sensitivity_bar"), tag)
        fig_event_dim(df, anchor, j("ofat_event_dim"), tag)
        write_summary(df, anchor, params, j("ofat_summary.csv"))
        print("\nFigures for h=%s in %s" % (args.pred_len, outdir))

    if args.compare:
        per_h = {h: _load_one(h, args.use_events, tag=args.tag) for h in args.compare}
        os.makedirs(args.outdir, exist_ok=True)
        anchors = {h: load_anchor(results_path(h, args.use_events, args.tag), h)
                   for h in args.compare}
        fig_cross_horizon(per_h, args.metric, os.path.join(
            args.outdir, f"ofat_cross_horizon_{args.metric}"), anchors)
        print("\nCross-horizon figure in", args.outdir)


if __name__ == "__main__":
    main()
