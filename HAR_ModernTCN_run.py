"""
HAR-ModernTCN run script -- hybrid of HAR-RV (Corsi, 2009) + ModernTCN residual
learner.

The HAR-RV linear model is fit by OLS on the training window (2010-2021) and its
h-day-ahead forecast is removed; ModernTCN then learns the HAR residual from a
look-back window of observed ln(RV). The final forecast is

    y_hat^(h) = HAR_pred^(h) + ModernTCN_residual_pred.

Horizon h is set by --pred_len. Run once per horizon -- the ModernTCN
configuration is expected to differ across h = 1, 5, 22. Suggested starts:

    # ── h = 1 (daily) ─────────────────────────────────────────────────────────
    python HAR_ModernTCN_run.py --is_training 1 --model_id harmtcn_h1 \
        --data har_residual --root_path ./data/ \
        --data_path realized_volatility.csv --target ln_RV --features S \
        --pred_len 1 --seq_len 32 \
        --patch_size 8 --patch_stride 4 \
        --num_blocks 1 1 1 1 --large_size 13 13 13 13 --small_size 5 5 5 5 \
        --dims 32 32 32 32 --dw_dims 32 32 32 32 \
        --ffn_ratio 1 --learning_rate 0.001 --revin 0

    # ── h = 5 (weekly) ────────────────────────────────────────────────────────
    python HAR_ModernTCN_run.py --is_training 1 --model_id harmtcn_h5 \
        --data har_residual --root_path ./data/ \
        --data_path realized_volatility.csv --target ln_RV --features S \
        --pred_len 5 --seq_len 48 \
        --patch_size 8 --patch_stride 4 \
        --num_blocks 1 1 1 1 --large_size 13 13 13 13 --small_size 5 5 5 5 \
        --dims 64 64 64 64 --dw_dims 64 64 64 64 \
        --ffn_ratio 1 --learning_rate 0.001 --revin 0

    # ── h = 22 (monthly) ──────────────────────────────────────────────────────
    python HAR_ModernTCN_run.py --is_training 1 --model_id harmtcn_h22 \
        --data har_residual --root_path ./data/ \
        --data_path realized_volatility.csv --target ln_RV --features S \
        --pred_len 22 --seq_len 66 --batch_size 32 \
        --patch_size 8 --patch_stride 4 \
        --num_blocks 1 1 1 1 --large_size 13 13 13 13 --small_size 5 5 5 5 \
        --dims 64 64 64 64 --dw_dims 64 64 64 64 \
        --ffn_ratio 1 --learning_rate 0.0005 --revin 0

Notes
-----
* --pred_len IS the horizon h (the target is the h-day forward-average ln RV,
  matching the deep-learning baselines). ModernTCN's head always emits a single
  value, so --aggregate_mean is not needed.
* Inputs and residual targets are standardised on the train split inside the
  dataset, so ModernTCN's RevIN is redundant; keep --revin 0.
* ModernTCN's backbone hard-codes a 4-stage downsampling stem, so num_blocks,
  large_size, small_size, dims and dw_dims must each have length 4 with equal
  dims (the head reads dims[-1]). Use small widths (32/64) for these short
  volatility windows. --freq is forced to 'h' purely to satisfy ModernTCN's
  (unused) time-embedding module; no time features are fed in.
"""

import argparse
import os
import random

import numpy as np
import torch

from exp.exp_HAR_ModernTCN import Exp_HAR_ModernTCN
from utils.str2bool import str2bool

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='HAR-ModernTCN Hybrid Forecaster')

# ── Random seed ─────────────────────────────────────────────────────────────
parser.add_argument('--random_seed', type=int, default=2021)

# ── Experiment identity ──────────────────────────────────────────────────────
parser.add_argument('--is_training', type=int, required=True, help='1=train  0=test')
parser.add_argument('--model_id',    type=str, required=True, help='experiment name')

# ── Dataset ──────────────────────────────────────────────────────────────────
parser.add_argument('--data',        type=str, default='har_residual')
parser.add_argument('--root_path',   type=str, default='./data/')
parser.add_argument('--data_path',   type=str, default='realized_volatility.csv')
parser.add_argument('--features',    type=str, default='S')
parser.add_argument('--target',      type=str, default='ln_RV')
parser.add_argument('--freq',        type=str, default='h',
                    help="kept 'h' for ModernTCN's (unused) time embedding")
parser.add_argument('--embed',       type=str, default='timeF')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

# ── Sequence ─────────────────────────────────────────────────────────────────
parser.add_argument('--seq_len',   type=int, default=22, help='input (look-back) length')
parser.add_argument('--label_len', type=int, default=0,  help='unused')
parser.add_argument('--pred_len',  type=int, default=1,  help='forecast horizon h (1/5/22)')

# ── ModernTCN architecture ───────────────────────────────────────────────────
parser.add_argument('--stem_ratio',       type=int, default=6)
parser.add_argument('--downsample_ratio', type=int, default=2)
parser.add_argument('--ffn_ratio',        type=int, default=1)
parser.add_argument('--patch_size',       type=int, default=8)
parser.add_argument('--patch_stride',     type=int, default=4)
parser.add_argument('--num_blocks', nargs='+', type=int, default=[1, 1, 1, 1], help='num_blocks per stage')
parser.add_argument('--large_size', nargs='+', type=int, default=[13, 13, 13, 13], help='big kernel per stage')
parser.add_argument('--small_size', nargs='+', type=int, default=[5, 5, 5, 5], help='small kernel per stage')
parser.add_argument('--dims',       nargs='+', type=int, default=[32, 32, 32, 32], help='dmodel per stage')
parser.add_argument('--dw_dims',    nargs='+', type=int, default=[32, 32, 32, 32])
parser.add_argument('--small_kernel_merged',     type=str2bool, default=False)
parser.add_argument('--call_structural_reparam', type=bool,     default=False)
parser.add_argument('--use_multi_scale',         type=str2bool, default=False,
                    help='kept False: this fork sizes the head for the downsampled '
                         'length (multi-scale fusion is not applied in forward)')

# ── Normalisation (RevIN redundant -- dataset standardises internally) ───────
parser.add_argument('--revin',         type=int, default=0)
parser.add_argument('--affine',        type=int, default=0)
parser.add_argument('--subtract_last', type=int, default=0)
parser.add_argument('--decomposition', type=int, default=0)
parser.add_argument('--kernel_size',   type=int, default=25)
parser.add_argument('--individual',    type=int, default=0)

# ── Input / output dims ──────────────────────────────────────────────────────
parser.add_argument('--enc_in',       type=int, default=1)
parser.add_argument('--c_out',        type=int, default=1)
parser.add_argument('--head_dropout', type=float, default=0.0)
parser.add_argument('--dropout',      type=float, default=0.05)

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
parser.add_argument('--use_amp',       action='store_true', default=False)
parser.add_argument('--test_flop',     action='store_true', default=False)

# ── Hardware ─────────────────────────────────────────────────────────────────
parser.add_argument('--use_gpu',       type=bool,  default=True)
parser.add_argument('--gpu',           type=int,   default=0)
parser.add_argument('--use_multi_gpu', action='store_true', default=False)
parser.add_argument('--devices',       type=str,   default='0,1,2,3')

# ── Misc (kept for compatibility) ────────────────────────────────────────────
parser.add_argument('--do_predict',     action='store_true', default=False)
parser.add_argument('--aggregate_mean', action='store_true', default=False,
                    help='unused: HAR target is already the h-day mean')
parser.add_argument('--output_attention', action='store_true', default=False)

args = parser.parse_args()

# ---------------------------------------------------------------------------
# Derived / auto-computed fields
# ---------------------------------------------------------------------------

if args.c_out == 0:
    args.c_out = 1

args.model = 'ModernTCN'   # exp builds ModernTCN.Model

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
    Exp = Exp_HAR_ModernTCN

    setting_fmt = (
        '{model_id}_HARMTCN_{data}_ft{features}'
        '_sl{seq_len}_h{pred_len}'
        '_dim{dim0}_nb{nb0}_lk{lk0}_ps{patch_size}_str{patch_stride}'
        '_rv{revin}_{des}_{ii}'
    )

    def make_setting(ii):
        return setting_fmt.format(
            ii=ii, dim0=args.dims[0], nb0=args.num_blocks[0],
            lk0=args.large_size[0], **vars(args))

    if args.is_training:
        for ii in range(args.itr):
            setting = make_setting(ii)
            exp = Exp(args)
            print(f'>>>>>>> training : {setting} >>>>>>>')
            exp.train(setting)

            print(f'>>>>>>> testing  : {setting} <<<<<<<')
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        setting = make_setting(0)
        exp = Exp(args)
        print(f'>>>>>>> testing  : {setting} <<<<<<<')
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
