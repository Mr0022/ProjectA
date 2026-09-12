#!/usr/bin/env bash
# =============================================================================
# ModernTCN (baseline, no news events) — realized-volatility forecasting
# Hyper-parameters are the best Optuna trials from:
#   tuningresults/ModernTCN1/best_params.json   (h = 1  day)
#   tuningresults/ModernTCN5/best_params.json   (h = 5  days)
#   tuningresults/ModernTCN22/best_params.json  (h = 22 days)
#
# tune.py expands single tuned values into 4-stage lists (dims == dw_dims ==
# [dim]*4, etc.), so that is reproduced below. --itr 5 sweeps seeds 2021..2025.
#
# !! Hyper-parameters below were tuned on the PREVIOUS dataset
#    (realized_volatility.csv) under the PREVIOUS target (mean of ln RV).
#    The data and target have since changed to data/EURUSD-RV.csv and
#    Y_t^(h) = ln((1/h) * sum_{k=1..h} RV_{t+k}); re-tune before publishing.
#
# Run from the ModernTCN-Long-term-forecasting/ directory:
#     bash scripts/moderntcn.sh
# In a notebook, prefix each block with '!' and run it as its own cell.
# =============================================================================
set -euo pipefail

# ---- h = 1  (ModernTCN1) ----------------------------------------------------
python run.py --is_training 1 --model_id ModernTCN_h1 --model ModernTCN \
  --data custom --root_path ./data/ --data_path EURUSD-RV.csv \
  --features S --target RV --enc_in 1 --dec_in 1 --c_out 1 \
  --aggregate_mean --seq_len 70 --pred_len 1 \
  --patch_size 16 --patch_stride 8 --ffn_ratio 2 \
  --num_blocks 2 2 2 2 --large_size 27 27 27 27 --small_size 5 5 5 5 \
  --dims 32 32 32 32 --dw_dims 32 32 32 32 \
  --dropout 0.33157505058759384 --head_dropout 0.13413677333143775 --revin 1 \
  --use_multi_scale False --lradj TST --pct_start 0.3 \
  --learning_rate 0.0063484758647924695 --batch_size 256 \
  --train_epochs 40 --patience 8 --num_workers 2 --itr 5

# ---- h = 5  (ModernTCN5) ----------------------------------------------------
python run.py --is_training 1 --model_id ModernTCN_h5 --model ModernTCN \
  --data custom --root_path ./data/ --data_path EURUSD-RV.csv \
  --features S --target RV --enc_in 1 --dec_in 1 --c_out 1 \
  --aggregate_mean --seq_len 22 --pred_len 5 \
  --patch_size 16 --patch_stride 8 --ffn_ratio 1 \
  --num_blocks 1 1 1 1 --large_size 13 13 13 13 --small_size 3 3 3 3 \
  --dims 128 128 128 128 --dw_dims 128 128 128 128 \
  --dropout 0.4744918542935045 --head_dropout 0.16435589854180058 --revin 1 \
  --use_multi_scale False --lradj TST --pct_start 0.3 \
  --learning_rate 9.048320833685613e-05 --batch_size 256 \
  --train_epochs 40 --patience 8 --num_workers 2 --itr 5

# ---- h = 22 (ModernTCN22) ---------------------------------------------------
python run.py --is_training 1 --model_id ModernTCN_h22 --model ModernTCN \
  --data custom --root_path ./data/ --data_path EURUSD-RV.csv \
  --features S --target RV --enc_in 1 --dec_in 1 --c_out 1 \
  --aggregate_mean --seq_len 22 --pred_len 22 \
  --patch_size 16 --patch_stride 2 --ffn_ratio 3 \
  --num_blocks 1 1 1 1 --large_size 51 51 51 51 --small_size 7 7 7 7 \
  --dims 256 256 256 256 --dw_dims 256 256 256 256 \
  --dropout 0.3222005482423507 --head_dropout 0.18550313066719137 --revin 1 \
  --use_multi_scale False --lradj TST --pct_start 0.3 \
  --learning_rate 0.0001385051157761346 --batch_size 256 \
  --train_epochs 40 --patience 8 --num_workers 2 --itr 5
