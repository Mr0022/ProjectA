"""
Bayesian hyperparameter tuning for ModernTCN using Optuna (TPE sampler + Median pruner).

Usage:
    python tune.py \
        --data ETTh1 \
        --root_path ./all_six_datasets/ETT-small \
        --data_path ETTh1.csv \
        --enc_in 7 \
        --seq_len 336 \
        --pred_len 96 \
        --n_trials 50 \
        --train_epochs 30 \
        --patience 7

Results are saved to optuna_results/<study_name>/ as a CSV and a best_params.json file.
The Optuna study is persisted to optuna_results/<study_name>/study.db so interrupted
runs can be resumed automatically by re-running the same command.
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

from exp.exp_ModernTCN import Exp_Main
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

class TunableExp(Exp_Main):
    """Exp_Main with Optuna trial reporting and pruning hooks."""

    def train_with_trial(self, setting: str, trial: optuna.Trial) -> float:
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader   = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        train_steps   = len(train_loader)
        early_stop    = EarlyStopping(patience=self.args.patience, verbose=False)
        model_optim   = self._select_optimizer()
        criterion     = self._select_criterion()

        scheduler = lr_scheduler.OneCycleLR(
            optimizer      = model_optim,
            steps_per_epoch= train_steps,
            pct_start      = self.args.pct_start,
            epochs         = self.args.train_epochs,
            max_lr         = self.args.learning_rate,
        )

        best_vali_loss = float('inf')
        f_dim = -1 if self.args.features == 'MS' else 0

        for epoch in range(self.args.train_epochs):
            self.model.train()

            for batch in train_loader:
                # _unpack_batch handles both 4-tensor (no events) and 6-tensor
                # (events) batches, moving event_x/event_y to the device
                batch_x, batch_y, batch_x_mark, batch_y_mark, event_x, event_y = self._unpack_batch(batch)
                model_optim.zero_grad()

                batch_x      = batch_x.float().to(self.device)
                batch_y      = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs  = self.model(batch_x, batch_x_mark, event_x=event_x, event_y=event_y)
                outputs  = outputs[:, -self.args.pred_len:, f_dim:]
                targets  = self._get_target(batch_y, f_dim)
                loss     = criterion(outputs, targets)

                loss.backward()
                model_optim.step()
                scheduler.step()

            vali_loss = self.vali(vali_data, vali_loader, criterion)

            if vali_loss < best_vali_loss:
                best_vali_loss = vali_loss
                torch.save(self.model.state_dict(), os.path.join(path, 'checkpoint.pth'))

            # Optuna: report intermediate value and check for pruning
            trial.report(float(vali_loss), epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            early_stop(vali_loss, self.model, path)
            if early_stop.early_stop:
                break

        return best_vali_loss


# ---------------------------------------------------------------------------
# Search space definition
# ---------------------------------------------------------------------------

def sample_hyperparameters(trial: optuna.Trial, base_cfg: argparse.Namespace) -> argparse.Namespace:
    cfg = copy.copy(base_cfg)

    # --- Look-back window (input length) ---
    seq_len = trial.suggest_categorical('seq_len', [22, 35, 70, 180])

    # --- Patch stem ---
    patch_size = trial.suggest_categorical('patch_size', [4, 8, 16, 32])
    # Always suggest from the full set; clamp so stride never exceeds patch_size.
    # Optuna requires a parameter's choice set to be identical across all trials
    # (CategoricalDistribution does not support dynamic value space), so we
    # cannot filter the list conditionally — we clamp after sampling instead.
    patch_stride_raw = trial.suggest_categorical('patch_stride', [2, 4, 8])
    patch_stride = min(patch_stride_raw, patch_size)

    # --- Model width & depth ---
    dim        = trial.suggest_categorical('dim', [32, 64, 128, 256])
    ffn_ratio  = trial.suggest_int('ffn_ratio', 1, 4)
    large_size = trial.suggest_categorical('large_size', [13, 27, 31, 51])
    small_size = trial.suggest_categorical('small_size', [3, 5, 7])  # always <= large_size
    num_blocks = trial.suggest_int('num_blocks', 1, 3)

    # --- Regularization ---
    dropout      = trial.suggest_float('dropout',      0.0, 0.5)
    head_dropout = trial.suggest_float('head_dropout', 0.0, 0.5)

    # --- Optimisation ---
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size    = trial.suggest_categorical('batch_size', [64, 128, 256, 512])

    # --- Architecture flags ---
    # use_multi_scale is fixed to False: the forward_feature implementation never
    # calls the lat/smooth/upsample layers, so use_multi_scale=True produces a
    # head size mismatch at runtime. All official scripts also set this to False.
    revin = trial.suggest_categorical('revin', [0, 1])

    # --- News-event embedding width (events study only) ---
    # event_fusion / event_past / event_future are fixed in the base config;
    # only the embedding width is searched.
    use_events = getattr(base_cfg, 'use_events', False)
    if use_events:
        event_dim = trial.suggest_categorical('event_dim', [4, 8, 16])

    # Write sampled values into config
    cfg.seq_len         = seq_len
    cfg.patch_size      = patch_size
    cfg.patch_stride    = patch_stride
    cfg.dims            = [dim] * 4
    cfg.dw_dims         = [dim] * 4
    cfg.ffn_ratio       = ffn_ratio
    cfg.large_size      = [large_size] * 4
    cfg.small_size      = [small_size] * 4
    cfg.num_blocks      = [num_blocks] * 4
    cfg.dropout         = dropout
    cfg.head_dropout    = head_dropout
    cfg.learning_rate   = learning_rate
    cfg.batch_size      = batch_size
    cfg.use_multi_scale = False
    cfg.revin           = revin
    if use_events:
        cfg.event_dim = event_dim

    return cfg


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def objective(trial: optuna.Trial, base_cfg: argparse.Namespace) -> float:
    args = sample_hyperparameters(trial, base_cfg)
    set_seed(args.random_seed)

    setting = f'optuna_trial_{trial.number}'

    try:
        exp            = TunableExp(args)
        best_vali_loss = exp.train_with_trial(setting, trial)
    except optuna.TrialPruned:
        raise
    except Exception as exc:
        print(f'[Trial {trial.number}] failed with error: {exc}')
        raise optuna.TrialPruned()
    finally:
        torch.cuda.empty_cache()

    return best_vali_loss


# ---------------------------------------------------------------------------
# Base config builder (fixed settings that do not change across trials)
# ---------------------------------------------------------------------------

def build_base_config(tune_args: argparse.Namespace) -> argparse.Namespace:
    cfg = argparse.Namespace()

    # Identifiers
    cfg.model    = 'ModernTCN'
    cfg.model_id = 'optuna'
    cfg.des      = 'optuna'

    # Dataset
    cfg.data      = tune_args.data
    cfg.root_path = tune_args.root_path
    cfg.data_path = tune_args.data_path
    cfg.features  = tune_args.features
    cfg.target    = tune_args.target
    cfg.freq      = tune_args.freq
    cfg.embed     = tune_args.embed

    # Sequence
    cfg.seq_len   = tune_args.seq_len
    cfg.label_len = tune_args.label_len
    cfg.pred_len  = tune_args.pred_len
    cfg.enc_in    = tune_args.enc_in
    cfg.dec_in    = tune_args.enc_in
    cfg.c_out     = tune_args.enc_in

    # Training budget (reduced for speed during search)
    cfg.train_epochs = tune_args.train_epochs
    cfg.patience     = tune_args.patience
    cfg.num_workers  = tune_args.num_workers
    cfg.checkpoints  = tune_args.checkpoints

    # Aggregation mode
    cfg.aggregate_mean = tune_args.aggregate_mean

    # News-event conditioning (Study 2). event_fusion/past/future are fixed;
    # event_dim is searched (see sample_hyperparameters).
    cfg.use_events      = tune_args.use_events
    cfg.event_data_path = tune_args.event_data_path
    cfg.event_fusion    = tune_args.event_fusion
    cfg.event_past      = True
    cfg.event_future    = True
    cfg.event_dim       = tune_args.event_dim   # default; overwritten per-trial when searched
    cfg.event_in        = 0
    if cfg.use_events:
        if cfg.data == 'custom':
            cfg.data = 'custom_events'
        import pandas as pd
        ev_header = pd.read_csv(os.path.join(cfg.root_path, cfg.event_data_path), nrows=0)
        cfg.event_in = len([c for c in ev_header.columns if c != 'date'])
        print(f'news events: {cfg.event_in} feature columns from {cfg.event_data_path} '
              f'(fusion={cfg.event_fusion})')

    # Optimiser fixed settings
    cfg.pct_start = 0.3
    cfg.lradj     = 'TST'   # OneCycleLR — best for search
    cfg.use_amp   = False
    cfg.loss      = 'mse'
    cfg.itr       = 1

    # Fixed ModernTCN structural params
    cfg.stem_ratio            = 6
    cfg.downsample_ratio      = 2
    cfg.small_kernel_merged   = False
    cfg.call_structural_reparam = False
    cfg.affine                = 0
    cfg.subtract_last         = 0
    cfg.decomposition         = 0
    cfg.kernel_size           = 25
    cfg.individual            = 0

    # Legacy Transformer params (unused but required by data_provider / model_dict)
    cfg.embed_type     = 0
    cfg.d_model        = 512
    cfg.n_heads        = 8
    cfg.e_layers       = 2
    cfg.d_layers       = 1
    cfg.d_ff           = 2048
    cfg.moving_avg     = 25
    cfg.factor         = 1
    cfg.distil         = True
    cfg.activation     = 'gelu'
    cfg.output_attention = False
    cfg.do_predict     = False
    cfg.fc_dropout     = 0.05
    cfg.padding_patch  = 'end'
    cfg.patch_len      = 16
    cfg.stride         = 8
    cfg.test_flop      = False

    # GPU
    cfg.use_gpu      = torch.cuda.is_available()
    cfg.gpu          = 0
    cfg.use_multi_gpu= False
    cfg.devices      = '0'
    cfg.device_ids   = [0]

    cfg.random_seed  = tune_args.random_seed

    return cfg


# ---------------------------------------------------------------------------
# CLI argument parser for the tuning script itself
# ---------------------------------------------------------------------------

def parse_tune_args():
    p = argparse.ArgumentParser(description='Optuna HPO for ModernTCN')

    # Dataset (required)
    p.add_argument('--data',      type=str, default='custom', help='Dataset name, e.g. custom / custom_events')
    p.add_argument('--root_path', type=str, default='./data/', help='Root path to data directory')
    p.add_argument('--data_path', type=str, default='EURUSD-RV.csv', help='CSV filename')
    p.add_argument('--enc_in',    type=int, required=True, help='Number of input variables')

    # Task
    p.add_argument('--features',  type=str, default='M',    help='M / S / MS')
    p.add_argument('--target',    type=str, default='RV',
                   help='Target column for S/MS; a level RV column is log-transformed on load')
    p.add_argument('--freq',      type=str, default='h',    help='Time feature frequency')
    p.add_argument('--embed',     type=str, default='timeF',help='Time embedding type')
    p.add_argument('--seq_len',   type=int, default=22,     help='Fallback input length (seq_len is searched over {22,35,70,180})')
    p.add_argument('--label_len', type=int, default=0,      help='Label length (unused by ModernTCN; must be <= seq_len)')
    p.add_argument('--pred_len',  type=int, default=96,     help='Prediction horizon')

    # Tuning budget
    p.add_argument('--n_trials',     type=int, default=50,  help='Number of Optuna trials')
    p.add_argument('--train_epochs', type=int, default=30,  help='Max epochs per trial (keep small)')
    p.add_argument('--patience',     type=int, default=7,   help='Early-stop patience per trial')
    p.add_argument('--num_workers',  type=int, default=4,   help='DataLoader workers')
    p.add_argument('--checkpoints',  type=str, default='./optuna_checkpoints/', help='Checkpoint dir')

    # Study settings
    p.add_argument('--study_name',   type=str, default=None,  help='Optuna study name (auto-generated if omitted)')
    p.add_argument('--output_dir',   type=str, default='./optuna_results/', help='Results output directory')
    p.add_argument('--random_seed',  type=int, default=2021,  help='Random seed')
    p.add_argument('--n_startup_trials', type=int, default=10,
                   help='Random trials before TPE kicks in (exploration phase)')
    p.add_argument('--n_warmup_steps',   type=int, default=5,
                   help='Epochs before pruner starts evaluating a trial')
    p.add_argument('--reset', action='store_true',
                   help='Delete existing study DB and start fresh (use after changing search space)')
    p.add_argument('--aggregate_mean', '--aggregate_logsum', action='store_true',
                   dest='aggregate_mean', default=False,
                   help='Predict the single aggregated target ln((1/h) * sum_{k=1..h} RV_{t+k}) -- the log of the horizon-average variance -- instead of each step individually (single-value output). --aggregate_logsum is accepted as a deprecated alias.')

    # News events (Study 2)
    p.add_argument('--use_events', action='store_true', default=False,
                   help='Tune the news-event model: switches data->custom_events and adds '
                        'event_dim to the search space (event_fusion/past/future fixed)')
    p.add_argument('--event_data_path', type=str, default='events.csv',
                   help='Event calendar csv inside root_path')
    p.add_argument('--event_fusion', type=str, default='channel', choices=['inject', 'channel'],
                   help='How past events enter the backbone; fixed across the study')
    p.add_argument('--event_dim', type=int, default=8,
                   help='Fallback event embedding width (event_dim is searched when --use_events)')

    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    tune_args = parse_tune_args()
    set_seed(tune_args.random_seed)

    # Build the fixed base config
    base_cfg = build_base_config(tune_args)

    # Study name and persistence path
    study_name = tune_args.study_name or (
        f"moderntcn_{tune_args.data}_pl{tune_args.pred_len}"
        + (f"_ev{tune_args.event_fusion}" if tune_args.use_events else "")
    )
    out_dir    = os.path.join(tune_args.output_dir, study_name)
    os.makedirs(out_dir, exist_ok=True)
    db_path    = os.path.join(out_dir, 'study.db')

    if tune_args.reset and os.path.exists(db_path):
        os.remove(db_path)
        print(f'Removed existing study DB: {db_path}')

    print(f'\n{"="*60}')
    print(f'  Optuna HPO — ModernTCN')
    print(f'  Dataset  : {tune_args.data}  |  pred_len={tune_args.pred_len}')
    print(f'  Trials   : {tune_args.n_trials}')
    print(f'  Epochs/trial: {tune_args.train_epochs}  patience={tune_args.patience}')
    print(f'  Results  : {out_dir}')
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
        storage        = f'sqlite:///{db_path}',
        load_if_exists = True,   # resume if interrupted
    )

    study.optimize(
        lambda trial: objective(trial, base_cfg),
        n_trials     = tune_args.n_trials,
        show_progress_bar = True,
    )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    best = study.best_trial
    print(f'\n{"="*60}')
    print(f'  Best trial  : #{best.number}')
    print(f'  Vali MSE    : {best.value:.6f}')
    print(f'  Best params :')
    for k, v in best.params.items():
        print(f'    {k}: {v}')
    print(f'{"="*60}\n')

    # JSON — best params
    best_params_path = os.path.join(out_dir, 'best_params.json')
    with open(best_params_path, 'w') as f:
        json.dump({'best_value': best.value, 'params': best.params}, f, indent=2)
    print(f'Best params saved to: {best_params_path}')

    # CSV — all trials
    df = study.trials_dataframe()
    csv_path = os.path.join(out_dir, 'all_trials.csv')
    df.to_csv(csv_path, index=False)
    print(f'All trials saved to : {csv_path}')

    # Optuna visualisations (saved as HTML if plotly is available)
    try:
        import plotly  # noqa: F401
        fig_history = optuna.visualization.plot_optimization_history(study)
        fig_history.write_html(os.path.join(out_dir, 'optimization_history.html'))

        fig_importance = optuna.visualization.plot_param_importances(study)
        fig_importance.write_html(os.path.join(out_dir, 'param_importances.html'))

        fig_parallel = optuna.visualization.plot_parallel_coordinate(study)
        fig_parallel.write_html(os.path.join(out_dir, 'parallel_coordinate.html'))
        print(f'Visualisations saved to: {out_dir}')
    except ImportError:
        print('Install plotly to generate HTML visualisations: pip install plotly')


if __name__ == '__main__':
    main()
