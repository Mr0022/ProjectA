"""
Bayesian hyperparameter tuning for the LSTM forecaster using Optuna
(TPE sampler + Median pruner).

Usage:
    python LSTM_tune.py \
        --data custom \
        --root_path ./data/ \
        --data_path EURUSD-RV.csv \
        --features S \
        --target RV \
        --enc_in 1 \
        --seq_len 22 \
        --pred_len 5 \
        --aggregate_logsum \
        --n_trials 50 \
        --train_epochs 30 \
        --patience 7

Results are saved to 'LSTM results/<study_name>/' as a CSV and a best_params.json.
The Optuna study is persisted to 'LSTM results/<study_name>/study.db' (an SQLite
RDB), so interrupted runs resume automatically by re-running the same command.
Override the location with --output_dir.

Optuna Dashboard
----------------
The SQLite storage is fully compatible with optuna-dashboard. While (or after) a
run, inspect it live with:

    pip install optuna-dashboard
    optuna-dashboard sqlite:///"LSTM results/<study_name>/study.db"

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

from exp.exp_LSTM import Exp_LSTM
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

class TunableLSTM(Exp_LSTM):
    """Exp_LSTM with Optuna trial reporting and pruning hooks."""

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
        f_dim = -1 if self.args.features == 'MS' else 0

        for epoch in range(self.args.train_epochs):
            self.model.train()

            for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                outputs = self.model(batch_x)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                targets = self._get_target(batch_y, f_dim)

                loss = criterion(outputs, targets)
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

    # --- Look-back window ---
    # Same candidate set as ModernTCN's tune.py. The LSTM handles a variable
    # sequence length natively: it unrolls over seq_len steps and reads the last
    # hidden state, and the projection head is sized by hidden_size * pred_len
    # (not seq_len), so no head resizing is needed when seq_len changes.
    seq_len = trial.suggest_categorical('seq_len', [22, 35, 70, 180])

    # --- LSTM architecture ---
    hidden_size   = trial.suggest_categorical('hidden_size', [32, 64, 128, 256])
    num_layers    = trial.suggest_int('num_layers', 1, 3)
    dropout       = trial.suggest_float('dropout',      0.0, 0.5)
    head_dropout  = trial.suggest_float('head_dropout', 0.0, 0.5)
    bidirectional = trial.suggest_categorical('bidirectional', [False, True])

    # --- Normalisation ---
    revin = trial.suggest_categorical('revin', [0, 1])

    # --- Optimisation ---
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size    = trial.suggest_categorical('batch_size', base_cfg._batch_choices)
    pct_start     = trial.suggest_float('pct_start', 0.1, 0.5)

    # Write sampled values into config
    cfg.seq_len       = seq_len
    cfg.hidden_size   = hidden_size
    cfg.num_layers    = num_layers
    cfg.dropout       = dropout
    cfg.head_dropout  = head_dropout
    cfg.bidirectional = bidirectional
    cfg.revin         = revin
    cfg.learning_rate = learning_rate
    cfg.batch_size    = batch_size
    cfg.pct_start     = pct_start

    # Keep compat fields consistent with the run script.
    cfg.d_model  = hidden_size
    cfg.d_ff     = hidden_size * 4
    cfg.e_layers = num_layers

    return cfg


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def objective(trial: optuna.Trial, base_cfg: argparse.Namespace) -> float:
    args = sample_hyperparameters(trial, base_cfg)
    set_seed(args.random_seed)

    setting = f'lstm_optuna_trial_{trial.number}'
    trial.set_user_attr('setting', setting)

    try:
        exp            = TunableLSTM(args)
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
    cfg.model    = 'LSTM'
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
    cfg.c_out     = tune_args.enc_in if tune_args.features == 'M' else 1

    # Aggregation mode
    cfg.aggregate_logsum = tune_args.aggregate_logsum

    # Training budget (reduced for speed during search)
    cfg.train_epochs = tune_args.train_epochs
    cfg.patience     = tune_args.patience
    cfg.num_workers  = tune_args.num_workers
    cfg.checkpoints  = tune_args.checkpoints

    # Optimiser fixed settings
    cfg.lradj     = 'TST'   # OneCycleLR — best for search
    cfg.use_amp   = False
    cfg.loss      = 'mse'
    cfg.itr       = 1

    # RevIN sub-settings (revin flag itself is searched)
    cfg.affine        = 0
    cfg.subtract_last = 0

    # Misc fields required by data_provider / model even though LSTM ignores them
    cfg.embed_type     = 0
    cfg.n_heads        = 1
    cfg.d_layers       = 1
    cfg.distil         = True
    cfg.activation     = 'gelu'
    cfg.output_attention = False
    cfg.do_predict     = False
    cfg.decomposition  = 0
    cfg.kernel_size    = 25
    cfg.individual     = 0
    cfg.moving_avg     = 25
    cfg.factor         = 1

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
    p = argparse.ArgumentParser(description='Optuna HPO for the LSTM forecaster')

    # Dataset
    p.add_argument('--data',      type=str, default='custom', help='Dataset name, e.g. custom / custom_events')
    p.add_argument('--root_path', type=str, default='./data/', help='Root path to data directory')
    p.add_argument('--data_path', type=str, default='EURUSD-RV.csv', help='CSV filename')
    p.add_argument('--enc_in',    type=int, required=True, help='Number of input variables')

    # Task
    p.add_argument('--features',  type=str, default='S',     help='M / S / MS')
    p.add_argument('--target',    type=str, default='RV',
                   help='Target column for S/MS; a level RV column is log-transformed on load')
    p.add_argument('--freq',      type=str, default='h',     help='Time feature frequency')
    p.add_argument('--embed',     type=str, default='timeF', help='Time embedding type')
    p.add_argument('--seq_len',   type=int, default=22,      help='Fallback look-back window (seq_len is searched over {22,35,70,180})')
    p.add_argument('--label_len', type=int, default=0,       help='Label length (unused by LSTM; must be <= seq_len)')
    p.add_argument('--pred_len',  type=int, default=1,       help='Prediction horizon')
    p.add_argument('--aggregate_logsum', '--aggregate_mean', action='store_true',
                   dest='aggregate_logsum',  default=False,
                   help='Predict the mean of the next pred_len steps (single-value output)')

    # Search-space choice sets
    p.add_argument('--batch_choices',   type=_int_list, default=[64, 128, 256, 512],
                   help='Comma-separated batch-size candidates')

    # Tuning budget
    p.add_argument('--n_trials',     type=int, default=50,  help='Number of Optuna trials')
    p.add_argument('--train_epochs', type=int, default=30,  help='Max epochs per trial (keep small)')
    p.add_argument('--patience',     type=int, default=7,   help='Early-stop patience per trial')
    p.add_argument('--num_workers',  type=int, default=4,   help='DataLoader workers')
    p.add_argument('--checkpoints',  type=str, default='./optuna_checkpoints/', help='Checkpoint dir')

    # Study settings
    p.add_argument('--study_name',   type=str, default=None,  help='Optuna study name (auto-generated if omitted)')
    p.add_argument('--output_dir',   type=str, default='./LSTM results/', help='Results output directory')
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
        f"lstm_{tune_args.data}_pl{tune_args.pred_len}"
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
    print(f'  Optuna HPO — LSTM')
    print(f'  Dataset     : {tune_args.data}  |  pred_len={tune_args.pred_len}')
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
    study.set_user_attr('model', 'LSTM')
    study.set_user_attr('dataset', tune_args.data)
    study.set_user_attr('pred_len', tune_args.pred_len)
    study.set_user_attr('metric', 'val_mse')

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
    print(f'  Vali MSE   : {best.value:.6f}')
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
