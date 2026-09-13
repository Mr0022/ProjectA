#!/usr/bin/env python3
"""
OFAT (one-factor-at-a-time) hyperparameter-sensitivity harness for ModernTCN.

Runs at a chosen horizon (h = 1, 5 or 22), each anchored at that horizon's own
tuned config. Results are written to a per-horizon CSV so the three studies stay
independent and can be plotted separately.

Every hyperparameter is anchored at the tuned best config (ANCHOR below) and
ONE is swept at a time over the Optuna tuning-search grid.  Each swept point is
trained with ``run.py --itr 5`` so it carries a 5-seed mean +/- std.  Per-seed
test metrics (MSE / MAE / RSE / QLIKE) are parsed straight from run.py's stdout
and appended to a tidy long-format CSV that ``ofat_plots.py`` turns into the
academic figures.

Design notes
------------
* Local sensitivity: the curve is only valid *around* the optimum -- that is the
  point of OFAT, and it is stated as such in the figures.
* The anchor config is trained exactly once and reused as the centre point of
  every panel (no need to retrain the same config 14 times).
* Resumable: any (param, value) already present in the CSV is skipped, so an
  interrupted sweep just restarts where it left off.
* patch_stride is clamped to ``min(patch_stride, patch_size)`` exactly as
  ``tune.py`` does, so the patch_size sweep never produces a stride > patch_size.

Target: Y_t^(h) = ln((1/h) * sum_{k=1..h} RV_{t+k}) on data/EURUSD-RV.csv
(--aggregate_mean), the same target and test split every other model uses.

Run from the repository root:

    python sensitivity/ofat_sensitivity.py --pred_len 1           # full sweep, h=1
    python sensitivity/ofat_sensitivity.py --pred_len 22 --params dropout learning_rate
    python sensitivity/ofat_sensitivity.py --pred_len 5 --quick   # 2 seeds x 15 epochs
    python sensitivity/ofat_sensitivity.py --pred_len 1 --dry_run # print the plan only

Pass --use_events to study EventTCN (ModernTCN + the news-event calendar)
instead; that adds event_dim to the sweep.
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Anchors = the tuned best hyperparameters per horizon, from
# tuningresults/ModernTCN{1,5,22}/best_params.json.
# (dim -> dims/dw_dims; num_blocks/large_size/small_size are repeated x4.)
#
# NOTE: these were tuned on the PREVIOUS dataset and target. The sweep is still
# a valid local sensitivity study around them, but they are not guaranteed to be
# the optimum for the current data/target -- re-run tune.py for that.
# ---------------------------------------------------------------------------
ANCHORS = {
    1: {
        "seq_len": 70, "patch_size": 16, "patch_stride": 8, "ffn_ratio": 2,
        "num_blocks": 2, "large_size": 27, "small_size": 5, "dim": 32,
        "dropout": 0.33157505058759384, "head_dropout": 0.13413677333143775,
        "revin": 1, "learning_rate": 0.0063484758647924695, "batch_size": 256,
        "event_dim": 8,
    },
    5: {
        "seq_len": 22, "patch_size": 16, "patch_stride": 8, "ffn_ratio": 1,
        "num_blocks": 1, "large_size": 13, "small_size": 3, "dim": 128,
        "dropout": 0.4744918542935045, "head_dropout": 0.16435589854180058,
        "revin": 1, "learning_rate": 9.048320833685613e-05, "batch_size": 256,
        "event_dim": 8,
    },
    22: {
        "seq_len": 22, "patch_size": 16, "patch_stride": 2, "ffn_ratio": 3,
        "num_blocks": 1, "large_size": 51, "small_size": 7, "dim": 256,
        "dropout": 0.3222005482423507, "head_dropout": 0.18550313066719137,
        "revin": 1, "learning_rate": 0.0001385051157761346, "batch_size": 256,
        "event_dim": 8,
    },
}
HORIZONS = sorted(ANCHORS)

# Set by main()/anchor_for() so the plotting module can import a concrete anchor.
ANCHOR = ANCHORS[1]


def anchor_for(pred_len, override=None):
    """Return (and pin as the module-level ANCHOR) the anchor for a horizon.

    `override` is a full anchor dict (e.g. from a results sidecar) that replaces
    the tuned default -- used when a study is anchored somewhere else.
    """
    global ANCHOR
    if override is not None:
        ANCHOR = dict(override)
        return ANCHOR
    if pred_len not in ANCHORS:
        raise SystemExit(f"no tuned anchor for h={pred_len}; have {HORIZONS}")
    ANCHOR = ANCHORS[pred_len]
    return ANCHOR


def results_path(pred_len, use_events=False, tag=None):
    """Per-horizon results CSV, so separate studies never mix.

    `tag` names a study that uses a non-default anchor, e.g. tag="custom" ->
    ofat_moderntcn_h5_custom.csv, keeping it apart from the tuned-anchor run.
    """
    kind = "eventtcn" if use_events else "moderntcn"
    suffix = f"_{tag}" if tag else ""
    return os.path.join("sensitivity", f"ofat_{kind}_h{pred_len}{suffix}.csv")


def anchor_sidecar(csv_path):
    """Path of the JSON recording which anchor produced a results CSV."""
    return os.path.splitext(csv_path)[0] + ".anchor.json"


def parse_anchor_overrides(pairs, base):
    """Apply `key=value` overrides to a copy of `base`, typed like the base."""
    cfg = dict(base)
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"--anchor expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        if k not in base:
            raise SystemExit(f"unknown anchor key {k!r}; valid: {sorted(base)}")
        cfg[k] = float(v) if isinstance(base[k], float) else type(base[k])(v)
    return cfg


def save_anchor(csv_path, cfg, pred_len):
    with open(anchor_sidecar(csv_path), "w") as f:
        json.dump({"pred_len": pred_len, "anchor": cfg}, f, indent=2)


def load_anchor(csv_path, pred_len):
    """Anchor that produced this CSV: the sidecar if present, else the tuned one."""
    side = anchor_sidecar(csv_path)
    if os.path.exists(side):
        with open(side) as f:
            return json.load(f)["anchor"]
    return ANCHORS[pred_len]


# ---------------------------------------------------------------------------
# OFAT grids -- the exact value sets searched by tune.py (continuous knobs get
# an evenly spaced grid across the searched range).  The anchor value is always
# added in and reused, so every curve passes through the tuned point.
# ---------------------------------------------------------------------------
GRIDS = {
    "seq_len":       [22, 35, 70, 180],
    "patch_size":    [4, 8, 16, 32],
    "patch_stride":  [2, 4, 8],
    "dim":           [32, 64, 128, 256],
    "ffn_ratio":     [1, 2, 3, 4],
    "large_size":    [13, 27, 31, 51],
    "small_size":    [3, 5, 7],
    "num_blocks":    [1, 2, 3],
    "dropout":       [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "head_dropout":  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "learning_rate": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    "batch_size":    [64, 128, 256, 512],
    "revin":         [0, 1],
    "event_dim":     [4, 8, 16],
}

# Canonical order used for CLI defaults and plotting. event_dim only applies to
# EventTCN, so it is not part of the plain-ModernTCN sweep.
ORDER = [k for k in GRIDS if k != "event_dim"]
ORDER_EVENTS = list(GRIDS.keys())

METRICS = ["mse", "mae", "rse", "qlike"]

# run.py: run ii uses seed (random_seed + ii); default random_seed = 2021.
BASE_SEED = 2021

# one metric line is printed per seed by Exp_Main.test():
#   "mse:0.28, mae:0.39, rse:0.51, qlike:0.19"
_METRIC_RE = re.compile(
    r"mse:\s*([-\d.eE+]+),\s*mae:\s*([-\d.eE+]+),\s*"
    r"rse:\s*([-\d.eE+]+),\s*qlike:\s*([-\d.eE+]+)"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def value_key(v):
    """Stable string key for a swept value (kept identical runner<->plotter)."""
    if isinstance(v, float):
        return repr(v)
    return str(v)


def build_cmd(param, value, itr, epochs, model_id, pred_len, use_events=False,
              anchor=None):
    """Assemble the run.py command for one OFAT point (all others = anchor)."""
    cfg = dict(anchor if anchor is not None else anchor_for(pred_len))
    if param is not None:
        cfg[param] = value
    # mirror tune.py: stride never exceeds patch size
    cfg["patch_stride"] = min(cfg["patch_stride"], cfg["patch_size"])
    dim = cfg["dim"]

    def rep4(x):
        return [str(x)] * 4

    cmd = [
        sys.executable, "run.py",
        "--is_training", "1",
        "--model_id", model_id,
        "--model", "ModernTCN",
        "--data", "custom",
        "--root_path", "./data/",
        "--data_path", "EURUSD-RV.csv",
        "--features", "S", "--target", "RV",
        "--enc_in", "1", "--dec_in", "1", "--c_out", "1",
        "--aggregate_mean",
        "--seq_len", str(cfg["seq_len"]),
        "--pred_len", str(pred_len),
        "--patch_size", str(cfg["patch_size"]),
        "--patch_stride", str(cfg["patch_stride"]),
        "--ffn_ratio", str(cfg["ffn_ratio"]),
        "--num_blocks", *rep4(cfg["num_blocks"]),
        "--large_size", *rep4(cfg["large_size"]),
        "--small_size", *rep4(cfg["small_size"]),
        "--dims", *rep4(dim),
        "--dw_dims", *rep4(dim),
        "--dropout", repr(cfg["dropout"]),
        "--head_dropout", repr(cfg["head_dropout"]),
        "--revin", str(cfg["revin"]),
        "--use_multi_scale", "False",
        "--lradj", "TST", "--pct_start", "0.3",
        "--learning_rate", repr(cfg["learning_rate"]),
        "--batch_size", str(cfg["batch_size"]),
        "--train_epochs", str(epochs),
        "--patience", "8",
        "--num_workers", "0",
        "--itr", str(itr),
        "--des", f"ofat{pred_len}",
    ]
    if use_events:
        cmd += ["--use_events", "--event_dim", str(cfg["event_dim"]),
                "--event_fusion", "channel"]
    return cmd


def parse_metrics(text, itr):
    """Extract one dict per seed from run.py stdout (order = seed order)."""
    rows = []
    for i, m in enumerate(_METRIC_RE.finditer(text)):
        rows.append({
            "seed":  BASE_SEED + i,
            "mse":   float(m.group(1)),
            "mae":   float(m.group(2)),
            "rse":   float(m.group(3)),
            "qlike": float(m.group(4)),
        })
    return rows


def load_done(csv_path):
    """Return set of (param, value_key) already recorded."""
    done = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            done.add((r["param"], r["value"]))
    return done


def append_rows(csv_path, rows):
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["param", "value", "seed", "mse", "mae", "rse", "qlike",
                        "itr", "epochs"])
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------------
# Run one OFAT point
# ---------------------------------------------------------------------------
def run_point(param, value, itr, epochs, timeout, dry_run, pred_len,
              use_events=False, anchor=None):
    tag = "anchor" if param is None else f"{param}_{value_key(value)}"
    model_id = f"OFAT_h{pred_len}_{tag}".replace(".", "p").replace("-", "m")
    cmd = build_cmd(param, value, itr, epochs, model_id, pred_len, use_events, anchor)

    print(f"\n{'='*70}\n[OFAT] {tag}\n{' '.join(cmd)}\n{'='*70}", flush=True)
    if dry_run:
        return []

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    dt = time.time() - t0
    if proc.returncode != 0:
        print(f"[WARN] {tag}: run.py exited {proc.returncode} after {dt:.0f}s; "
              f"last stderr:\n{proc.stderr[-800:]}", flush=True)
        return []

    metrics = parse_metrics(proc.stdout, itr)
    if len(metrics) < itr:
        print(f"[WARN] {tag}: parsed {len(metrics)}/{itr} seed metrics "
              f"(config may be invalid); skipping.", flush=True)
        # still surface a bit of stdout for debugging
        print(proc.stdout[-800:], flush=True)
        return []

    pv = "" if param is None else value_key(value)
    rows = [[param or "anchor", pv, m["seed"], m["mse"], m["mae"], m["rse"],
             m["qlike"], itr, epochs] for m in metrics]
    mmse = sum(m["mse"] for m in metrics) / len(metrics)
    print(f"[OK] {tag}: {len(metrics)} seeds, mean MSE={mmse:.5f}  ({dt:.0f}s)",
          flush=True)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="OFAT sensitivity sweep for ModernTCN")
    ap.add_argument("--pred_len", type=int, default=1, choices=HORIZONS,
                    help="forecast horizon; each has its own tuned anchor and CSV")
    ap.add_argument("--use_events", action="store_true",
                    help="study EventTCN (adds the event calendar and event_dim)")
    ap.add_argument("--anchor", nargs="+", metavar="KEY=VALUE", default=None,
                    help="override anchor entries, e.g. --anchor seq_len=35 dim=64. "
                         "The resolved anchor is saved beside the results CSV.")
    ap.add_argument("--tag", default=None,
                    help="name this study, keeping it in its own CSV "
                         "(e.g. --tag custom -> ofat_moderntcn_h5_custom.csv)")
    ap.add_argument("--params", nargs="+", default=None, choices=ORDER_EVENTS,
                    help="which hyperparameters to sweep (default: all applicable)")
    ap.add_argument("--itr", type=int, default=5, help="seeds per point")
    ap.add_argument("--train_epochs", type=int, default=40)
    ap.add_argument("--out", default=None,
                    help="results CSV (default: sensitivity/ofat_<kind>_h<H>.csv)")
    ap.add_argument("--timeout", type=int, default=3 * 3600,
                    help="per-config subprocess timeout (s)")
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: 2 seeds, 15 epochs")
    ap.add_argument("--dry_run", action="store_true",
                    help="print the plan and the commands, train nothing")
    args = ap.parse_args()

    if args.quick:
        args.itr, args.train_epochs = 2, 15

    if args.out is None:
        args.out = results_path(args.pred_len, args.use_events, args.tag)
    # An existing study keeps the anchor it was started with, so a resumed sweep
    # can never silently mix points from two different centres.
    base = load_anchor(args.out, args.pred_len)
    anchor = anchor_for(args.pred_len,
                        parse_anchor_overrides(args.anchor, base))
    if args.params is None:
        args.params = ORDER_EVENTS if args.use_events else ORDER
    elif not args.use_events and "event_dim" in args.params:
        raise SystemExit("event_dim only applies with --use_events")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = load_done(args.out)
    if not args.dry_run:
        save_anchor(args.out, anchor, args.pred_len)

    # Build the full plan (anchor first, then each swept point).
    plan = []
    if ("anchor", "") not in done:
        plan.append((None, None))
    for p in args.params:
        grid = sorted(set(GRIDS[p]) | {anchor[p]})
        for v in grid:
            if v == anchor[p]:
                continue                       # centre point == anchor, reused
            if (p, value_key(v)) in done:
                continue
            plan.append((p, v))

    kind = "EventTCN" if args.use_events else "ModernTCN"
    print(f"OFAT plan [{kind}, h={args.pred_len}]: {len(plan)} config(s) to train "
          f"(itr={args.itr}, epochs={args.train_epochs}); "
          f"already done: {len(done)} row-group(s). -> {args.out}")
    diffs = {k: v for k, v in anchor.items() if ANCHORS[args.pred_len].get(k) != v}
    if diffs:
        print("   anchor differs from the tuned default: "
              + ", ".join(f"{k}={v}" for k, v in sorted(diffs.items())))
    for p, v in plan:
        print(f"   - {'anchor' if p is None else f'{p} = {v}'}")

    if args.dry_run:
        for p, v in plan:
            run_point(p, v, args.itr, args.train_epochs, args.timeout, True,
                      args.pred_len, args.use_events, anchor)
        print("\n[dry-run] no training performed.")
        return

    for i, (p, v) in enumerate(plan, 1):
        print(f"\n########## [{i}/{len(plan)}] ##########")
        rows = run_point(p, v, args.itr, args.train_epochs, args.timeout, False,
                         args.pred_len, args.use_events, anchor)
        if rows:
            append_rows(args.out, rows)

    print(f"\nDone. Results in {args.out}")
    tagarg = f" --tag {args.tag}" if args.tag else ""
    print(f"Next:  python sensitivity/ofat_plots.py --pred_len {args.pred_len}{tagarg}")


if __name__ == "__main__":
    main()
