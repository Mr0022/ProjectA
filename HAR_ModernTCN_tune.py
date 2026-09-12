"""
Bayesian hyperparameter tuning for the HAR-ModernTCN hybrid using Optuna
(TPE sampler + Median pruner).

HAR-ModernTCN = HAR-RV linear model (Corsi, 2009) + a ModernTCN residual learner.
Only the ModernTCN residual learner has hyperparameters; the HAR-RV part is fit by
OLS on the train window inside the dataset. This script tunes the ModernTCN
residual learner against the validation loss on the (standardised) HAR residual.

Run ONE study per horizon h (= --pred_len), since the best ModernTCN configuration
is expected to differ across h = 1, 5, 22. The look-back window is fixed at
seq_len=22 and is not part of the search space.

Usage:
    python HAR_ModernTCN_tune.py \
        --data har_residual \
        --root_path ./data/ \
        --data_path realized_volatility.csv \
        --target ln_RV --features S \
        --pred_len 5 \
        --n_trials 50 --train_epochs 30 --patience 7 --num_workers 0

Results are saved to 'HAR-ModernTCN results/<study_name>/' as a CSV and a
best_params.json. The Optuna study is persisted to
'HAR-ModernTCN results/<study_name>/study.db' (an SQLite RDB), so interrupted
runs resume automatically by re-running the same command. Override the location
with --output_dir.

Optuna Dashboard
----------------
The SQLite storage is fully compatible with optuna-dashboard:

    pip install optuna-dashboard
    optuna-dashboard sqlite:///"HAR-ModernTCN results/<study_name>/study.db"

The exact command is printed at the end of every run.
"""

import argparse
import copy
import json
import os
import random
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

warnings.filterwarnings('ignore')

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp.exp_HAR_ModernTCN import Exp_HAR_ModernTCN
from utils.tools import EarlyStopping, adjust_learning_rate


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Experiment subclass with per-epoch pruning support
# ---------------------------------------------------------------------------

class TunableHARModernTCN(Exp_HAR_ModernTCN):
    """Exp_HAR_ModernTCN with Optuna trial reporting and pruning hooks."""

    def train_with_trial(self, setting: str, trial: optuna.Trial) -> float:
        train_data, train_loader = self._get_data(flag='train')
        _,          vali_loader  = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        train_steps  = len(train_loader)
        early_stop   = EarlyStopping(patience=self.args.patience, verbose=False)
        model_optim  = self._select_optimizer()
        criterion    = self._select_criterion()

        scheduler = lr_scheduler.OneCycleLR(
            optimizer       = model_optim,
            steps_per_epoch = train_steps,
            pct_start       = self.args.pct_start,
            epochs          = self.args.train_epochs,
            max_lr          = self.args.learning_rate,
        )

        best_vali_loss = float('inf')
        best_epoch     = -1

        for epoch in range(self.args.train_epochs):
            self.model.train()

            for seq_x, resid, har_pred, y_true in train_loader:
                model_optim.zero_grad()

                seq_x = seq_x.float().to(self.device)
                resid = resid.float().to(self.device)

                outputs = self.model(seq_x)            # (B, 1, 1)
                outputs = outputs[:, -1:, :]

                loss = criterion(outputs, resid)
                loss.backward()
                model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1,
                                         self.args, printout=False)
                    scheduler.step()

            vali_loss = self.vali(vali_loader, criterion)

            if vali_loss < best_vali_loss:
                best_vali_loss = vali_loss
                best_epoch     = epoch
                torch.save(self.model.state_dict(), os.path.join(path, 'checkpoint.pth'))

            # Optuna: report intermediate value and check for pruning
            trial.report(float(vali_loss), epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            early_stop(vali_loss, self.model, path)
            if early_stop.early_stop:
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1,
                                     self.args, printout=False)

        trial.set_user_attr('best_epoch', best_epoch)
        return best_vali_loss


# ---------------------------------------------------------------------------
# Search space definition
# ---------------------------------------------------------------------------

def sample_hyperparameters(trial: optuna.Trial, base_cfg: argparse.Namespace) -> argparse.Namespace:
    cfg = copy.copy(base_cfg)

    # --- Patch stem ---
    patch_size = trial.suggest_categorical('patch_size', [4, 8, 16])
    # Optuna requires a categorical's value set to be identical across trials,
    # so we always sample from the full set and clamp the stride afterwards
    # (the stem conv requires patch_stride <= patch_size).
    patch_stride_raw = trial.suggest_categorical('patch_stride', [2, 4, 8])
    patch_stride = min(patch_stride_raw, patch_size)

    # --- Backbone width & depth (kept small for short volatility windows) ---
    dim        = trial.suggest_categorical('dim', [16, 32, 64])
    ffn_ratio  = trial.suggest_int('ffn_ratio', 1, 4)
    # large_size must be odd and >= small_size (=5); the ReparamLargeKernelConv
    # asserts small_kernel <= kernel_size.
    large_size = trial.suggest_categorical('large_size', [7, 13, 21])
    num_blocks = trial.suggest_int('num_blocks', 1, 3)

    # --- Regularisation ---
    dropout      = trial.suggest_float('dropout',      0.0, 0.5)
    head_dropout = trial.suggest_float('head_dropout', 0.0, 0.5)

    # --- Optimisation ---
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size    = trial.suggest_categorical('batch_size', base_cfg._batch_choices)
    pct_start     = trial.suggest_float('pct_start', 0.1, 0.5)

    # --- Look-back window ---
    # seq_len is fixed at 22 (see build_base_config); not part of the search space.

    # Write sampled values into config (4-stage backbone -> length-4 lists).
    cfg.patch_size   = patch_size
    cfg.patch_stride = patch_stride
    cfg.dims         = [dim] * 4
    cfg.dw_dims      = [dim] * 4
    cfg.ffn_ratio    = ffn_ratio
    cfg.large_size   = [large_size] * 4
    cfg.small_size   = [5] * 4
    cfg.num_blocks   = [num_blocks] * 4
    cfg.dropout      = dropout
    cfg.head_dropout = head_dropout
    cfg.learning_rate = learning_rate
    cfg.batch_size   = batch_size
    cfg.pct_start    = pct_start

    # RevIN is intentionally NOT searched: the HAR-residual dataset standardises
    # internally, so RevIN is redundant. use_multi_scale stays False (the head is
    # sized for the downsampled length; True produces a runtime size mismatch).
    cfg.revin           = 0
    cfg.use_multi_scale = False

    return cfg


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def objective(trial: optuna.Trial, base_cfg: argparse.Namespace) -> float:
    args = sample_hyperparameters(trial, base_cfg)
    set_seed(args.random_seed)

    setting = f'harmtcn_optuna_trial_{trial.number}'
    trial.set_user_attr('setting', setting)

    try:
        exp            = TunableHARModernTCN(args)
        best_vali_loss = exp.train_with_trial(setting, trial)
    except optuna.TrialPruned:
        raise
    except Exception as exc:
        print(f'[Trial {trial.number}] failed with error: {exc}')
        raise optuna.TrialPruned()
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return best_vali_loss



# ---------------------------------------------------------------------------
# Base config builder (fixed settings that do not change across trials)
# ---------------------------------------------------------------------------

def build_base_config(tune_args: argparse.Namespace) -> argparse.Namespace:
    cfg = argparse.Namespace()

    # Identifiers
    cfg.model    = 'ModernTCN'   # exp builds ModernTCN.Model
    cfg.model_id = 'optuna'
    cfg.des      = 'optuna'

    # Dataset
    cfg.data      = tune_args.data
    cfg.root_path = tune_args.root_path
    cfg.data_path = tune_args.data_path
    cfg.features  = tune_args.features
    cfg.target    = tune_args.target
    cfg.freq      = tune_args.freq   # 'h' to satisfy ModernTCN's (unused) time embed
    cfg.embed     = tune_args.embed

    # Sequence (pred_len IS the horizon h; the head always emits 1 value)
    cfg.seq_len   = tune_args.seq_len
    cfg.label_len = 0
    cfg.pred_len  = tune_args.pred_len
    cfg.enc_in    = 1
    cfg.dec_in    = 1
    cfg.c_out     = 1

    # Training budget (reduced for speed during search)
    cfg.train_epochs = tune_args.train_epochs
    cfg.patience     = tune_args.patience
    cfg.num_workers  = tune_args.num_workers
    cfg.checkpoints  = tune_args.checkpoints

    # Optimiser fixed settings
    cfg.lradj   = 'TST'   # OneCycleLR — best for search
    cfg.use_amp = False
    cfg.loss    = 'mse'
    cfg.itr     = 1

    # Fixed ModernTCN structural params
    cfg.stem_ratio              = 6
    cfg.downsample_ratio        = 2
    cfg.small_kernel_merged     = False
    cfg.call_structural_reparam = False
    cfg.test_flop               = False

    # RevIN / decomposition (revin fixed to 0 in sample_hyperparameters)
    cfg.affine        = 0
    cfg.subtract_last = 0
    cfg.decomposition = 0
    cfg.kernel_size   = 25
    cfg.individual    = 0

    # Misc fields required by run/exp/data_provider plumbing
    cfg.do_predict       = False
    cfg.aggregate_mean   = False   # HAR target is already the h-day mean
    cfg.output_attention = False
    cfg.embed_type       = 0

    # GPU
    cfg.use_gpu       = torch.cuda.is_available()
    cfg.gpu           = 0
    cfg.use_multi_gpu = False
    cfg.devices       = '0'
    cfg.device_ids    = [0]

    cfg.random_seed   = tune_args.random_seed

    # Search-space choice sets (attached for sample_hyperparameters)
    cfg._batch_choices   = tune_args.batch_choices

    return cfg


# ---------------------------------------------------------------------------
# CLI argument parser for the tuning script itself
# ---------------------------------------------------------------------------

def _int_list(s: str):
    return [int(x) for x in s.split(',') if x.strip()]


def parse_tune_args():
    p = argparse.ArgumentParser(description='Optuna HPO for the HAR-ModernTCN hybrid')

    # Dataset
    p.add_argument('--data',      type=str, default='har_residual', help='Dataset type')
    p.add_argument('--root_path', type=str, default='./data/',      help='Root path to data directory')
    p.add_argument('--data_path', type=str, default='realized_volatility.csv', help='CSV filename')

    # Task
    p.add_argument('--features',  type=str, default='S',     help='S: univariate (ln RV)')
    p.add_argument('--target',    type=str, default='ln_RV', help='Value column')
    p.add_argument('--freq',      type=str, default='h',     help="kept 'h' for ModernTCN's unused time embed")
    p.add_argument('--embed',     type=str, default='timeF', help='Time embedding type')
    p.add_argument('--seq_len',   type=int, default=22,      help='Look-back window (fixed, not searched)')
    p.add_argument('--pred_len',  type=int, default=1,       help='Forecast horizon h (1/5/22)')

    # Search-space choice sets
    p.add_argument('--batch_choices',   type=_int_list, default=[32, 64, 128, 256],
                   help='Comma-separated batch-size candidates (kept <=256: the daily-RV '
                        'val split is small and the loader drops the last partial batch)')

    # Tuning budget
    p.add_argument('--n_trials',     type=int, default=50,  help='Number of Optuna trials')
    p.add_argument('--train_epochs', type=int, default=30,  help='Max epochs per trial (keep small)')
    p.add_argument('--patience',     type=int, default=7,   help='Early-stop patience per trial')
    p.add_argument('--num_workers',  type=int, default=0,   help='DataLoader workers (0 is fastest for this in-memory dataset)')
    p.add_argument('--checkpoints',  type=str, default='./optuna_checkpoints/', help='Checkpoint dir')

    # Study settings
    p.add_argument('--study_name',   type=str, default=None,  help='Optuna study name (auto-generated if omitted)')
    p.add_argument('--output_dir',   type=str, default='./HAR-ModernTCN results/', help='Results output directory')
    p.add_argument('--random_seed',  type=int, default=2021,  help='Random seed')
    p.add_argument('--n_startup_trials', type=int, default=10,
                   help='Random trials before TPE kicks in (exploration phase)')
    p.add_argument('--n_warmup_steps',   type=int, default=5,
                   help='Epochs before pruner starts evaluating a trial')
    p.add_argument('--reset', action='store_true',
                   help='Delete existing study DB and start fresh (use after changing search space)')

    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    tune_args = parse_tune_args()
    set_seed(tune_args.random_seed)

    base_cfg = build_base_config(tune_args)

    study_name = tune_args.study_name or (
        f"harmtcn_{tune_args.data}_h{tune_args.pred_len}"
    )
    out_dir = os.path.join(tune_args.output_dir, study_name)
    os.makedirs(out_dir, exist_ok=True)
    db_path  = os.path.join(out_dir, 'study.db')
    # SQLite URIs require forward slashes even on Windows, where os.path.join
    # would otherwise produce backslashes that the storage URL cannot resolve.
    storage  = 'sqlite:///' + db_path.replace(os.sep, '/')

    if tune_args.reset and os.path.exists(db_path):
        os.remove(db_path)
        print(f'Removed existing study DB: {db_path}')

    print(f'\n{"="*60}')
    print(f'  Optuna HPO — HAR-ModernTCN')
    print(f'  Dataset     : {tune_args.data}  |  horizon h={tune_args.pred_len}')
    print(f'  seq_len     : {base_cfg.seq_len} (fixed)')
    print(f'  Trials      : {tune_args.n_trials}')
    print(f'  Epochs/trial: {tune_args.train_epochs}  patience={tune_args.patience}')
    print(f'  Results     : {out_dir}')
    print(f'{"="*60}\n')

    sampler = TPESampler(
        n_startup_trials=tune_args.n_startup_trials,
        seed=tune_args.random_seed,
    )
    pruner = MedianPruner(
        n_startup_trials=tune_args.n_startup_trials,
        n_warmup_steps=tune_args.n_warmup_steps,
        interval_steps=1,
    )

    study = optuna.create_study(
        study_name     = study_name,
        direction      = 'minimize',
        sampler        = sampler,
        pruner         = pruner,
        storage        = storage,
        load_if_exists = True,   # resume if interrupted
    )

    # Metadata surfaced in optuna-dashboard.
    study.set_user_attr('model', 'HAR_ModernTCN')
    study.set_user_attr('dataset', tune_args.data)
    study.set_user_attr('horizon', tune_args.pred_len)
    study.set_user_attr('metric', 'val_mse_residual')

    study.optimize(
        lambda trial: objective(trial, base_cfg),
        n_trials          = tune_args.n_trials,
        show_progress_bar = True,
    )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    best = study.best_trial
    print(f'\n{"="*60}')
    print(f'  Best trial : #{best.number}')
    print(f'  Vali MSE   : {best.value:.6f}  (standardised residual)')
    print(f'  Best params:')
    for k, v in best.params.items():
        print(f'    {k}: {v}')
    print(f'{"="*60}\n')

    best_params_path = os.path.join(out_dir, 'best_params.json')
    with open(best_params_path, 'w') as f:
        json.dump({'best_value': best.value, 'params': best.params}, f, indent=2)
    print(f'Best params saved to: {best_params_path}')

    df = study.trials_dataframe()
    csv_path = os.path.join(out_dir, 'all_trials.csv')
    df.to_csv(csv_path, index=False)
    print(f'All trials saved to : {csv_path}')

    try:
        import plotly  # noqa: F401
        optuna.visualization.plot_optimization_history(study).write_html(
            os.path.join(out_dir, 'optimization_history.html'))
        optuna.visualization.plot_param_importances(study).write_html(
            os.path.join(out_dir, 'param_importances.html'))
        optuna.visualization.plot_parallel_coordinate(study).write_html(
            os.path.join(out_dir, 'parallel_coordinate.html'))
        print(f'Visualisations saved to: {out_dir}')
    except ImportError:
        print('Install plotly to generate HTML visualisations: pip install plotly')

    print(f'\nInspect this study live with optuna-dashboard:')
    print(f'    optuna-dashboard {storage}\n')


if __name__ == '__main__':
    main()
