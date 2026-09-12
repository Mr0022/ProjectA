"""
LSTM forecasting run script — mirrors run.py for ModernTCN.

Example (univariate, aggregate-mean over 5 days):

    python LSTM_run.py \
        --is_training 1 \
        --model_id forex_lstm \
        --data custom \
        --root_path ./data/ \
        --data_path forex_log_realized_volatility.csv \
        --features S \
        --target EURUSD \
        --seq_len 48 \
        --pred_len 5 \
        --enc_in 1 \
        --hidden_size 128 \
        --num_layers 2 \
        --dropout 0.1 \
        --learning_rate 0.001 \
        --aggregate_mean
"""

import argparse
import os
import random

import numpy as np
import torch

from exp.exp_LSTM import Exp_LSTM

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='LSTM Time-Series Forecaster')

# ── Random seed ─────────────────────────────────────────────────────────────
parser.add_argument('--random_seed', type=int, default=2021,
                    help='base random seed; training iteration ii uses (random_seed + ii), '
                         'so the default 2021 with --itr 5 sweeps seeds 2021..2025')

# ── Experiment identity ──────────────────────────────────────────────────────
parser.add_argument('--is_training', type=int,  required=True,  help='1=train  0=test')
parser.add_argument('--model_id',    type=str,  required=True,  help='experiment name')

# ── Dataset ──────────────────────────────────────────────────────────────────
parser.add_argument('--data',        type=str,  required=True,  help='dataset type, e.g. custom / ETTh1')
parser.add_argument('--root_path',   type=str,  default='./data/')
parser.add_argument('--data_path',   type=str,  default='forex_log_realized_volatility.csv')
parser.add_argument('--features',    type=str,  default='S',
                    help='M: multivariate→multivariate  S: univariate  MS: multivariate→univariate')
parser.add_argument('--target',      type=str,  default='OT',   help='target column for S/MS')
parser.add_argument('--freq',        type=str,  default='h',
                    help='time-feature frequency: s t h d b w m')
parser.add_argument('--embed',       type=str,  default='timeF')
parser.add_argument('--checkpoints', type=str,  default='./checkpoints/')

# ── Sequence ─────────────────────────────────────────────────────────────────
parser.add_argument('--seq_len',    type=int, default=22,   help='input (look-back) length')
parser.add_argument('--label_len',  type=int, default=0,    help='decoder overlap (unused by LSTM; must be <= seq_len)')
parser.add_argument('--pred_len',   type=int, default=1,    help='forecast horizon')

# ── LSTM architecture ────────────────────────────────────────────────────────
parser.add_argument('--hidden_size',   type=int,   default=128,
                    help='number of units in each LSTM layer')
parser.add_argument('--num_layers',    type=int,   default=2,
                    help='number of stacked LSTM layers')
parser.add_argument('--dropout',       type=float, default=0.1,
                    help='dropout between LSTM layers (ignored when num_layers=1)')
parser.add_argument('--head_dropout',  type=float, default=0.0,
                    help='dropout before the final projection head')
parser.add_argument('--bidirectional', action='store_true', default=False,
                    help='use bidirectional LSTM (doubles effective hidden size)')

# ── Normalisation ────────────────────────────────────────────────────────────
parser.add_argument('--revin',         type=int, default=1,  help='RevIN on/off (1/0)')
parser.add_argument('--affine',        type=int, default=0,  help='learnable RevIN scale/shift')
parser.add_argument('--subtract_last', type=int, default=0,  help='RevIN anchor: last value instead of mean')

# ── Input / output dims ──────────────────────────────────────────────────────
parser.add_argument('--enc_in', type=int, default=1,  help='number of input variables')
parser.add_argument('--c_out',  type=int, default=0,
                    help='output channels (0 = auto: 1 for S/MS, enc_in for M)')

# ── Aggregation mode ─────────────────────────────────────────────────────────
parser.add_argument('--aggregate_mean', action='store_true', default=False,
                    help='predict the mean of the next pred_len steps (single output)')

# ── Training ─────────────────────────────────────────────────────────────────
parser.add_argument('--train_epochs',  type=int,   default=100)
parser.add_argument('--batch_size',    type=int,   default=128)
parser.add_argument('--patience',      type=int,   default=10,   help='early-stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--lradj',         type=str,   default='TST',
                    help='LR schedule: TST (OneCycleLR) | type1 | type2 | type3 | constant')
parser.add_argument('--pct_start',     type=float, default=0.3,
                    help='OneCycleLR warm-up fraction (only used when lradj=TST)')
parser.add_argument('--itr',           type=int,   default=1,
                    help='number of training runs; each run ii uses seed (random_seed + ii). '
                         'e.g. --random_seed 2021 --itr 5 runs 5 seeds 2021..2025')
parser.add_argument('--num_workers',   type=int,   default=4)
parser.add_argument('--des',           type=str,   default='Exp')

# ── Hardware ─────────────────────────────────────────────────────────────────
parser.add_argument('--use_gpu',       type=bool,  default=True)
parser.add_argument('--gpu',           type=int,   default=0)
parser.add_argument('--use_multi_gpu', action='store_true', default=False)
parser.add_argument('--devices',       type=str,   default='0,1,2,3')

# ── Misc (kept for compatibility with data_provider) ─────────────────────────
parser.add_argument('--do_predict',   action='store_true', default=False)
parser.add_argument('--decomposition', type=int,  default=0)
parser.add_argument('--kernel_size',   type=int,  default=25)
parser.add_argument('--individual',    type=int,  default=0)
parser.add_argument('--moving_avg',    type=int,  default=25)
parser.add_argument('--embed_type',    type=int,  default=0)
parser.add_argument('--factor',        type=int,  default=1)

args = parser.parse_args()

# ---------------------------------------------------------------------------
# Derived / auto-computed fields
# ---------------------------------------------------------------------------

# Auto-set c_out based on task type
if args.c_out == 0:
    args.c_out = args.enc_in if args.features == 'M' else 1

# dec_in, d_model etc. — required by data_provider even though LSTM doesn't use them
args.dec_in    = args.enc_in
args.d_model   = args.hidden_size
args.d_ff      = args.hidden_size * 4
args.n_heads   = 1
args.e_layers  = args.num_layers
args.d_layers  = 1
args.distil    = True
args.activation = 'gelu'
args.output_attention = False
args.use_amp   = False
args.model     = 'LSTM'

# GPU setup
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
if args.use_gpu and args.use_multi_gpu:
    args.devices   = args.devices.replace(' ', '')
    device_ids     = args.devices.split(',')
    args.device_ids = [int(d) for d in device_ids]
    args.gpu        = args.device_ids[0]

# Reproducibility (base seed). In training, each --itr run re-seeds with
# random_seed + ii (see the loop below); this initial seeding also covers the
# is_training=0 path.
random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)

print('Args in experiment:')
print(args)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    Exp = Exp_LSTM

    if args.is_training:
        run_metrics = []  # (seed, {mse,mae,rse,qlike}) per --itr run, for the summary
        for ii in range(args.itr):
            # each run is a distinct seed = random_seed + ii, so e.g.
            # --random_seed 2021 --itr 5 sweeps seeds 2021, 2022, ..., 2025
            seed = args.random_seed + ii
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            if args.use_gpu:
                torch.cuda.manual_seed_all(seed)
            print('>>>>>>> run {}/{}  seed {} <<<<<<<'.format(ii + 1, args.itr, seed))

            setting = (
                '{model_id}_LSTM_{data}_ft{features}'
                '_sl{seq_len}_pl{pred_len}'
                '_hs{hidden_size}_nl{num_layers}'
                '_bi{bidirectional}_rv{revin}'
                '_agg{aggregate_mean}'
                '_{des}_{ii}'
            ).format(**vars(args), ii=ii)

            exp = Exp(args)
            print(f'>>>>>>> training : {setting} >>>>>>>')
            exp.train(setting)

            print(f'>>>>>>> testing  : {setting} <<<<<<<')
            m = exp.test(setting)
            if isinstance(m, dict):
                run_metrics.append((seed, m))

            torch.cuda.empty_cache()

        # ---- summary across the --itr runs (mean +/- std over seeds) ----
        if len(run_metrics) > 1:
            keys = ['mse', 'mae', 'rse', 'qlike']
            seeds = [s for s, _ in run_metrics]
            header = '{:>8}'.format('seed') + ''.join('{:>14}'.format(k) for k in keys)
            print('\n' + '=' * len(header))
            print('summary over {} runs  (seeds {}..{})'.format(len(run_metrics), min(seeds), max(seeds)))
            print(header)
            print('-' * len(header))
            for s, mm in run_metrics:
                print('{:>8}'.format(s) + ''.join('{:>14.6f}'.format(mm[k]) for k in keys))
            print('-' * len(header))
            arr = {k: np.array([mm[k] for _, mm in run_metrics], dtype=np.float64) for k in keys}
            print('{:>8}'.format('mean') + ''.join('{:>14.6f}'.format(arr[k].mean()) for k in keys))
            print('{:>8}'.format('std') + ''.join('{:>14.6f}'.format(arr[k].std(ddof=1)) for k in keys))
            print('=' * len(header))
            with open('result_lstm.txt', 'a') as f:
                f.write('summary over {} runs (seeds {}..{})\n'.format(len(run_metrics), min(seeds), max(seeds)))
                for k in keys:
                    f.write('{}: mean {:.6f} +/- {:.6f} (std)\n'.format(k, arr[k].mean(), arr[k].std(ddof=1)))
                f.write('\n')
    else:
        ii = 0
        setting = (
            '{model_id}_LSTM_{data}_ft{features}'
            '_sl{seq_len}_pl{pred_len}'
            '_hs{hidden_size}_nl{num_layers}'
            '_bi{bidirectional}_rv{revin}'
            '_agg{aggregate_mean}'
            '_{des}_{ii}'
        ).format(**vars(args), ii=ii)

        exp = Exp(args)
        print(f'>>>>>>> testing  : {setting} <<<<<<<')
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
