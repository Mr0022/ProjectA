#!/usr/bin/env bash
# =============================================================================
# EventTCN = ModernTCN + news-event conditioning (--use_events, channel fusion)
# Hyper-parameters are the best Optuna trials from:
#   tuningresults/EVENTTCN1/best_params.json    (h = 1  day)
#   tuningresults/EVENTTCN5/best_params.json    (h = 5  days)
#   tuningresults/EVENTTCN22/best_params.json   (h = 22 days)
#
# Events were tuned with --event_fusion channel (tune.py default) and event_dim
# in {4,8,16}; event_past/event_future stay on (defaults). --use_events auto-
# switches --data custom -> custom_events and reads data/events.csv.
#
# NOTE on patch_stride: tune.py clamps stride = min(patch_stride, patch_size).
# For h = 22 the JSON stores patch_stride 8 but patch_size 4, so the value that
# actually trained is 4 — that clamped value is used below.
#
# !! Hyper-parameters below were tuned on the PREVIOUS dataset
#    (realized_volatility.csv) under the PREVIOUS target (mean of ln RV).
#    The data and target have since changed to data/EURUSD-RV.csv and
#    Y_t^(h) = ln(sum_{k=1..h} RV_{t+k}); re-tune before publishing.
#
# Run from the ModernTCN-Long-term-forecasting/ directory:
#     bash scripts/eventtcn.sh
# In a notebook, prefix each block with '!' and run it as its own cell.
# =============================================================================
set -euo pipefail

# ---- h = 1  (EVENTTCN1, event_dim 16) ---------------------------------------
python run.py --is_training 1 --model_id EventTCN_h1 --model ModernTCN \
  --data custom --root_path ./data/ --data_path EURUSD-RV.csv \
  --features S --target RV --enc_in 1 --dec_in 1 --c_out 1 \
  --aggregate_logsum --seq_len 22 --pred_len 1 \
  --patch_size 32 --patch_stride 8 --ffn_ratio 3 \
  --num_blocks 3 3 3 3 --large_size 51 51 51 51 --small_size 3 3 3 3 \
  --dims 32 32 32 32 --dw_dims 32 32 32 32 \
  --dropout 0.41549663977882045 --head_dropout 0.2993197807081734 --revin 1 \
  --use_multi_scale False --lradj TST --pct_start 0.3 \
  --learning_rate 0.0055036472222018224 --batch_size 128 \
  --train_epochs 40 --patience 8 --num_workers 2 --itr 5 \
  --use_events --event_dim 16 --event_fusion channel

# ---- h = 5  (EVENTTCN5, event_dim 4) ----------------------------------------
python run.py --is_training 1 --model_id EventTCN_h5 --model ModernTCN \
  --data custom --root_path ./data/ --data_path EURUSD-RV.csv \
  --features S --target RV --enc_in 1 --dec_in 1 --c_out 1 \
  --aggregate_logsum --seq_len 35 --pred_len 5 \
  --patch_size 32 --patch_stride 2 --ffn_ratio 3 \
  --num_blocks 3 3 3 3 --large_size 13 13 13 13 --small_size 5 5 5 5 \
  --dims 32 32 32 32 --dw_dims 32 32 32 32 \
  --dropout 0.0750931679204987 --head_dropout 0.36903420636104134 --revin 1 \
  --use_multi_scale False --lradj TST --pct_start 0.3 \
  --learning_rate 0.004311298509350653 --batch_size 128 \
  --train_epochs 40 --patience 8 --num_workers 2 --itr 5 \
  --use_events --event_dim 4 --event_fusion channel

# ---- h = 22 (EVENTTCN22, event_dim 8; patch_stride clamped 8 -> 4) ----------
python run.py --is_training 1 --model_id EventTCN_h22 --model ModernTCN \
  --data custom --root_path ./data/ --data_path EURUSD-RV.csv \
  --features S --target RV --enc_in 1 --dec_in 1 --c_out 1 \
  --aggregate_logsum --seq_len 35 --pred_len 22 \
  --patch_size 4 --patch_stride 4 --ffn_ratio 2 \
  --num_blocks 1 1 1 1 --large_size 31 31 31 31 --small_size 5 5 5 5 \
  --dims 32 32 32 32 --dw_dims 32 32 32 32 \
  --dropout 0.2332348360983411 --head_dropout 0.32275773648706885 --revin 1 \
  --use_multi_scale False --lradj TST --pct_start 0.3 \
  --learning_rate 0.0077943332090161695 --batch_size 256 \
  --train_epochs 40 --patience 8 --num_workers 2 --itr 5 \
  --use_events --event_dim 8 --event_fusion channel
