"""
HAR-LSTM run script -- hybrid of HAR-RV (Corsi, 2009) + LSTM residual learner.

The HAR-RV linear model is fit by OLS on the training window (2010-2021) and
its h-day-ahead forecast is removed; the LSTM then learns the HAR residual from
a look-back window of observed ln(RV). The final forecast is

    y_hat^(h) = HAR_pred^(h) + LSTM_residual_pred.

Horizon h is set by --pred_len. Run once per horizon -- the LSTM configuration
is expected to differ across h = 1, 5, 22. Suggested starting points:

    # ── h = 1 (daily) ─────────────────────────────────────────────────────────
    python HAR_LSTM_run.py --is_training 1 --model_id harlstm_h1 \
        --data har_residual --root_path ./data/ \
        --data_path realized_volatility.csv --target ln_RV \
        --features S --pred_len 1 \
        --seq_len 20 --hidden_size 64  --num_layers 1 --dropout 0.0 \
        --learning_rate 0.001 --revin 0

    # ── h = 5 (weekly) ────────────────────────────────────────────────────────
    python HAR_LSTM_run.py --is_training 1 --model_id harlstm_h5 \
        --data har_residual --root_path ./data/ \
        --data_path realized_volatility.csv --target ln_RV \
        --features S --pred_len 5 \
        --seq_len 40 --hidden_size 128 --num_layers 2 --dropout 0.1 \
        --learning_rate 0.001 --revin 0

    # ── h = 22 (monthly) ──────────────────────────────────────────────────────
    python HAR_LSTM_run.py --is_training 1 --model_id harlstm_h22 \
        --data har_residual --root_path ./data/ \
        --data_path realized_volatility.csv --target ln_RV \
        --features S --pred_len 22 \
        --seq_len 66 --hidden_size 128 --num_layers 2 --dropout 0.2 \
        --learning_rate 0.0005 --revin 0

Notes
-----
* --pred_len IS the horizon h (the target is the h-day forward-average ln RV,
  matching the aggregate-mean target of the deep-learning baselines). The LSTM
  head always emits a single value, so --aggregate_mean is not needed.
* Inputs and residual targets are standardised on the train split inside the
  dataset, so RevIN is redundant; keep --revin 0.
"""

import argparse
import os
import random

import numpy as np
import torch

from exp.exp_HAR_LSTM import Exp_HAR_LSTM

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='HAR-LSTM Hybrid Forecaster')

# ── Random seed ─────────────────────────────────────────────────────────────
parser.add_argument('--random_seed', type=int, default=2021)

# ── Experiment identity ──────────────────────────────────────────────────────
parser.add_argument('--is_training', type=int, required=True, help='1=train  0=test')
parser.add_argument('--model_id',    type=str, required=True, help='experiment name')

# ── Dataset ──────────────────────────────────────────────────────────────────
parser.add_argument('--data',        type=str, default='har_residual', help='dataset type')
parser.add_argument('--root_path',   type=str, default='./data/')
parser.add_argument('--data_path',   type=str, default='realized_volatility.csv')
parser.add_argument('--features',    type=str, default='S',
                    help='S: univariate (ln RV)')
parser.add_argument('--target',      type=str, default='ln_RV', help='value column')
parser.add_argument('--freq',        type=str, default='d')
parser.add_argument('--embed',       type=str, default='timeF')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

# ── Sequence ─────────────────────────────────────────────────────────────────
parser.add_argument('--seq_len',   type=int, default=22,  help='input (look-back) length')
parser.add_argument('--label_len', type=int, default=0,   help='unused by LSTM')
parser.add_argument('--pred_len',  type=int, default=1,   help='forecast horizon h (1/5/22)')

# ── LSTM architecture ────────────────────────────────────────────────────────
parser.add_argument('--hidden_size',   type=int,   default=64)
parser.add_argument('--num_layers',    type=int,   default=1)
parser.add_argument('--dropout',       type=float, default=0.0,
                    help='dropout between LSTM layers (ignored when num_layers=1)')
parser.add_argument('--head_dropout',  type=float, default=0.0)
parser.add_argument('--bidirectional', action='store_true', default=False)

# ── Normalisation (RevIN redundant -- dataset standardises internally) ───────
parser.add_argument('--revin',         type=int, default=0)
parser.add_argument('--affine',        type=int, default=0)
parser.add_argument('--subtract_last', type=int, default=0)

# ── Input / output dims ──────────────────────────────────────────────────────
parser.add_argument('--enc_in', type=int, default=1)
parser.add_argument('--c_out',  type=int, default=1)

# ── Training ─────────────────────────────────────────────────────────────────
parser.add_argument('--train_epochs',  type=int,   default=100)
parser.add_argument('--batch_size',    type=int,   default=128)
parser.add_argument('--patience',      type=int,   default=10)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--lradj',         type=str,   default='TST')
parser.add_argument('--pct_start',     type=float, default=0.3)
parser.add_argument('--itr',           type=int,   default=1)
parser.add_argument('--num_workers',   type=int,   default=4)
parser.add_argument('--des',           type=str,   default='Exp')

# ── Hardware ─────────────────────────────────────────────────────────────────
parser.add_argument('--use_gpu',       type=bool,  default=True)
parser.add_argument('--gpu',           type=int,   default=0)
parser.add_argument('--use_multi_gpu', action='store_true', default=False)
parser.add_argument('--devices',       type=str,   default='0,1,2,3')

# ── Misc (kept for compatibility with data_provider) ─────────────────────────
parser.add_argument('--do_predict',    action='store_true', default=False)
parser.add_argument('--aggregate_mean', action='store_true', default=False,
                    help='unused: HAR target is already the h-day mean')

args = parser.parse_args()

# ---------------------------------------------------------------------------
# Derived / auto-computed fields
# ---------------------------------------------------------------------------

if args.c_out == 0:
    args.c_out = 1

# Fields required by data_provider / model even though the LSTM ignores them.
args.dec_in     = args.enc_in
args.d_model    = args.hidden_size
args.d_ff       = args.hidden_size * 4
args.n_heads    = 1
args.e_layers   = args.num_layers
args.d_layers   = 1
args.distil     = True
args.activation = 'gelu'
args.output_attention = False
args.use_amp    = False
args.model      = 'HAR_LSTM'

# GPU setup
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
if args.use_gpu and args.use_multi_gpu:
    args.devices    = args.devices.replace(' ', '')
    device_ids      = args.devices.split(',')
    args.device_ids = [int(d) for d in device_ids]
    args.gpu        = args.device_ids[0]

# Reproducibility
random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)

print('Args in experiment:')
print(args)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    Exp = Exp_HAR_LSTM

    setting_fmt = (
        '{model_id}_HARLSTM_{data}_ft{features}'
        '_sl{seq_len}_h{pred_len}'
        '_hs{hidden_size}_nl{num_layers}'
        '_bi{bidirectional}_rv{revin}'
        '_{des}_{ii}'
    )

    if args.is_training:
        for ii in range(args.itr):
            setting = setting_fmt.format(**vars(args), ii=ii)
            exp = Exp(args)
            print(f'>>>>>>> training : {setting} >>>>>>>')
            exp.train(setting)

            print(f'>>>>>>> testing  : {setting} <<<<<<<')
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = setting_fmt.format(**vars(args), ii=ii)
        exp = Exp(args)
        print(f'>>>>>>> testing  : {setting} <<<<<<<')
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
