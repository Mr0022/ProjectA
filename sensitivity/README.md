# EventTCN — Local (OFAT) hyperparameter-sensitivity study

One-factor-at-a-time (OFAT) sensitivity analysis for **EventTCN** (ModernTCN +
news-event conditioning) at horizon **h = 1**. Every hyperparameter is pinned at
the tuned best config and one is varied at a time over the Optuna search grid;
each point is trained with `--itr 5` so it carries a 5-seed mean ± std. The curve
is therefore a **local** sensitivity around the tuned optimum — that is stated on
every figure.

## Anchor (tuned best config, h = 1)

```
seq_len 70 · patch_size 16 · patch_stride 8 · ffn_ratio 2 · num_blocks 2
large_size 27 · small_size 5 · dims 32 · dropout 0.3316 · head_dropout 0.1341
revin 1 · learning_rate 0.006348 · batch_size 256 · event_dim 8
event_fusion channel · use_events
```

## Hyperparameters swept (the tuning search space)

`seq_len, patch_size, patch_stride, dim, ffn_ratio, large_size, small_size,
num_blocks, dropout, head_dropout, learning_rate, batch_size, revin, event_dim`

## Run it (from `ModernTCN-Long-term-forecasting/`)

```bash
# 0. sanity check — prints the plan and commands, trains nothing
python sensitivity/ofat_sensitivity.py --dry_run

# 1. smoke test on the cheap knobs (2 seeds, 15 epochs)
python sensitivity/ofat_sensitivity.py --params event_dim dropout --quick

# 2. full sweep (5 seeds, 40 epochs) — resumable, anchor trained once & reused
python sensitivity/ofat_sensitivity.py

# 3. figures + summary table
python sensitivity/ofat_plots.py
```

`ofat_sensitivity.py` appends per-seed test metrics to `ofat_results.csv`
(long format). It is **resumable**: re-running skips any `(param, value)` already
recorded, so you can sweep a subset with `--params ...` and add more later.

## Outputs (`sensitivity/figures/`)

| file | what |
|---|---|
| `ofat_response_mse.{pdf,png}`   | response curve per hyperparameter — test MSE (mean line, ±1 std band, per-seed dots, anchor ◆) |
| `ofat_response_qlike.{pdf,png}` | same, QLIKE (Patton 2011 volatility loss) |
| `ofat_tornado_mse.{pdf,png}`    | signed swing vs anchor, ranked (which knobs matter locally) |
| `ofat_sensitivity_bar.{pdf,png}`| metric range as % of the anchor, ranked |
| `ofat_event_dim.{pdf,png}`      | EventTCN embedding-width spotlight (MSE + QLIKE) |
| `ofat_summary.csv`              | per-point mean/std for the paper tables |

## Notes / caveats

* **`patch_stride` is clamped** to `min(patch_stride, patch_size)` (as in
  `tune.py`), so the `patch_size` sweep never yields a stride larger than the
  patch — the coupling is intentional and matches how the search actually trained.
* OFAT curves are **local**: they measure sensitivity around the tuned optimum,
  not a global variance decomposition. For the global view use the Optuna
  `param_importances` / `parallel_coordinate` artifacts in `tuningresults/`.
* For other horizons, change `ANCHOR`/`--pred_len` (h = 5, h = 22 anchors are in
  `tuningresults/EVENTTCN5` and `EVENTTCN22`).
* An **events on/off** ablation (plain ModernTCN vs EventTCN) is a separate run
  (`event_dim` starts at 4, not "off"); add it by running the anchor without
  `--use_events` if you want that bar.
```
