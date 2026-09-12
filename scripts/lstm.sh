#!/usr/bin/env bash
# =============================================================================
# LSTM baseline — realized-volatility forecasting (uses LSTM_run.py, not run.py)
# Hyper-parameters are the best Optuna trials from:
#   tuningresults/LSTMM1/best_params.json    (h = 1  day)
#   tuningresults/LSTMM5/best_params.json    (h = 5  days)
#   tuningresults/LSTMM22/best_params.json   (h = 22 days)
#
# LSTM_tune.py fixed lradj=TST (OneCycleLR) and searched pct_start, so each
# horizon carries its own pct_start. All horizons chose bidirectional=false,
# so the --bidirectional flag is omitted. Note revin/batch_size differ per
# horizon (h=22 uses revin 0, batch 64). --itr 5 sweeps seeds 2021..2025.
#
# !! Hyper-parameters below were tuned on the PREVIOUS dataset
#    (realized_volatility.csv) under the PREVIOUS target (mean of ln RV).
#    The data and target have since changed to data/EURUSD-RV.csv and
#    Y_t^(h) = ln(sum_{k=1..h} RV_{t+k}); re-tune before publishing.
#
# Run from the ModernTCN-Long-term-forecasting/ directory:
#     bash scripts/lstm.sh
# In a notebook, prefix each block with '!' and run it as its own cell.
# =============================================================================
set -euo pipefail

# ---- h = 1  (LSTMM1) --------------------------------------------------------
python LSTM_run.py --is_training 1 --model_id LSTM_h1 \
  --data custom --root_path ./data/ --data_path EURUSD-RV.csv \
  --features S --target RV --enc_in 1 \
  --aggregate_logsum --seq_len 35 --pred_len 1 \
  --hidden_size 64 --num_layers 3 \
  --dropout 0.07676573564165186 --head_dropout 0.40397456078802163 --revin 1 \
  --lradj TST --pct_start 0.30908316790372015 \
  --learning_rate 0.000840108158564423 --batch_size 256 \
  --train_epochs 40 --patience 8 --num_workers 2 --itr 5

# ---- h = 5  (LSTMM5) --------------------------------------------------------
python LSTM_run.py --is_training 1 --model_id LSTM_h5 \
  --data custom --root_path ./data/ --data_path EURUSD-RV.csv \
  --features S --target RV --enc_in 1 \
  --aggregate_logsum --seq_len 22 --pred_len 5 \
  --hidden_size 64 --num_layers 2 \
  --dropout 0.23549950026331154 --head_dropout 0.4096358839061957 --revin 1 \
  --lradj TST --pct_start 0.26015477736906556 \
  --learning_rate 0.0019854378325347513 --batch_size 256 \
  --train_epochs 40 --patience 8 --num_workers 2 --itr 5

# ---- h = 22 (LSTMM22) -------------------------------------------------------
python LSTM_run.py --is_training 1 --model_id LSTM_h22 \
  --data custom --root_path ./data/ --data_path EURUSD-RV.csv \
  --features S --target RV --enc_in 1 \
  --aggregate_logsum --seq_len 35 --pred_len 22 \
  --hidden_size 64 --num_layers 1 \
  --dropout 0.00021377478950359723 --head_dropout 0.42441616746357586 --revin 0 \
  --lradj TST --pct_start 0.33053091331262563 \
  --learning_rate 0.009677926780016733 --batch_size 64 \
  --train_epochs 40 --patience 8 --num_workers 2 --itr 5
