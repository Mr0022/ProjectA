import torch
import torch.nn as nn
from layers.RevIN import RevIN


class Model(nn.Module):
    """
    LSTM forecaster.

    Input  : (B, seq_len, enc_in)
    Output : (B, pred_len, c_out)

    Configurable hyperparameters (all set via the configs namespace):
        hidden_size   – number of units in each LSTM layer
        num_layers    – number of stacked LSTM layers
        dropout       – dropout probability between LSTM layers
                        (ignored when num_layers == 1)
        head_dropout  – dropout applied before the projection head
        bidirectional – if True, use a bidirectional LSTM;
                        effective hidden dim becomes hidden_size * 2
        revin         – 1/0 to enable RevIN instance normalisation
        affine        – 1/0 for learnable RevIN scale/shift
        subtract_last – 1/0 RevIN anchor (mean vs last value)
    """

    def __init__(self, configs):
        super().__init__()

        self.seq_len  = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in   = configs.enc_in
        self.c_out    = configs.c_out

        # RevIN normalisation (operates on (B, L, C) directly)
        self.use_revin = bool(configs.revin)
        if self.use_revin:
            self.revin_layer = RevIN(
                configs.enc_in,
                affine=bool(configs.affine),
                subtract_last=bool(configs.subtract_last),
            )

        # LSTM backbone
        self.lstm = nn.LSTM(
            input_size  = configs.enc_in,
            hidden_size = configs.hidden_size,
            num_layers  = configs.num_layers,
            dropout     = configs.dropout if configs.num_layers > 1 else 0.0,
            batch_first = True,
            bidirectional = configs.bidirectional,
        )

        d_hidden = configs.hidden_size * (2 if configs.bidirectional else 1)

        # Projection head: last hidden state → (pred_len × c_out)
        self.head_dropout = nn.Dropout(configs.head_dropout)
        self.projection   = nn.Linear(d_hidden, configs.pred_len * configs.c_out)

    def forward(self, x):
        # x: (B, seq_len, enc_in)

        if self.use_revin:
            x = self.revin_layer(x, 'norm')

        lstm_out, _ = self.lstm(x)
        # lstm_out: (B, seq_len, hidden_size * num_directions)

        last = lstm_out[:, -1, :]                        # (B, d_hidden)
        last = self.head_dropout(last)
        out  = self.projection(last)                     # (B, pred_len * c_out)
        out  = out.view(x.shape[0], self.pred_len, self.c_out)  # (B, pred_len, c_out)

        if self.use_revin:
            out = self.revin_layer(out, 'denorm')

        return out
