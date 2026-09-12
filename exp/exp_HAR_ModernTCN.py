"""
HAR-ModernTCN experiment: a hybrid of the HAR-RV linear model (Corsi, 2009) and
a ModernTCN residual learner.

Pipeline (identical in spirit to exp_HAR_LSTM, only the residual learner differs)
-------------------------------------------------------------------------------
1. HAR-RV is fit by OLS on the TRAIN split (2010-2021) inside
   Dataset_HAR_Residual and removed from the h-day forward-average log-RV
   target, leaving residuals e_t = Y_t^(h) - HAR_pred_t.
2. ModernTCN (models/ModernTCN.py) is trained to predict the (standardised)
   HAR residual from a look-back window of observed ln(RV). Early stopping uses
   the genuine validation split (2022-2023).
3. At test time the hybrid forecast is reconstructed as
        y_hat^(h) = HAR_pred^(h) + ModernTCN_residual_pred
   and scored against the actual Y_t^(h) (ln-RV scale) with MSE / MAE / QLIKE.

The horizon h is taken from pred_len; run separately for h = 1, 5, 22 with the
per-horizon ModernTCN configuration of your choice.
"""

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import ModernTCN
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import os
import time
import warnings

warnings.filterwarnings('ignore')


class Exp_HAR_ModernTCN(Exp_Basic):

    def _build_model(self):
        # The residual learner emits a single value per horizon, so the model
        # head must output one step regardless of the horizon h (= pred_len).
        orig = self.args.pred_len
        self.args.pred_len = 1
        model = ModernTCN.Model(self.args).float()
        self.args.pred_len = orig

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    # ------------------------------------------------------------------
    # Validation (loss on standardised residuals)
    # ------------------------------------------------------------------

    def vali(self, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for seq_x, resid, har_pred, y_true in vali_loader:
                seq_x = seq_x.float().to(self.device)
                resid = resid.float()

                outputs = self.model(seq_x)                 # (B, 1, 1)
                outputs = outputs[:, -1:, :]
                loss = criterion(outputs.detach().cpu(), resid)
                total_loss.append(loss.item())
        self.model.train()
        return np.average(total_loss)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        _,          vali_loader  = self._get_data(flag='val')
        _,          test_loader  = self._get_data(flag='test')

        # Report the HAR-RV coefficients fit on the training window.
        b = train_data.har_beta
        print('HAR-RV (OLS, train 2010-2021)  '
              'const={:.4f}  b_d={:.4f}  b_w={:.4f}  b_m={:.4f}'.format(*b))

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        train_steps    = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim    = self._select_optimizer()
        criterion      = self._select_criterion()

        scheduler = lr_scheduler.OneCycleLR(
            optimizer       = model_optim,
            steps_per_epoch = train_steps,
            pct_start       = self.args.pct_start,
            epochs          = self.args.train_epochs,
            max_lr          = self.args.learning_rate,
        )

        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_loss = []
            epoch_start = time.time()

            for seq_x, resid, har_pred, y_true in train_loader:
                model_optim.zero_grad()

                seq_x = seq_x.float().to(self.device)
                resid = resid.float().to(self.device)

                outputs = self.model(seq_x)                 # (B, 1, 1)
                outputs = outputs[:, -1:, :]

                loss = criterion(outputs, resid)
                loss.backward()
                model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1,
                                         self.args, printout=False)
                    scheduler.step()

                train_loss.append(loss.item())

            train_loss = np.average(train_loss)
            vali_loss  = self.vali(vali_loader, criterion)
            test_loss  = self.vali(test_loader, criterion)

            print("Epoch: {0} | cost: {1:.1f}s | Train: {2:.7f}  Vali: {3:.7f}  Test: {4:.7f}".format(
                epoch + 1, time.time() - epoch_start,
                train_loss, vali_loss, test_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('LR ->', scheduler.get_last_lr()[0])

        self.model.load_state_dict(
            torch.load(os.path.join(path, 'checkpoint.pth')))
        return self.model

    # ------------------------------------------------------------------
    # Testing: reconstruct hybrid forecast and score on ln(RV) scale
    # ------------------------------------------------------------------

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            self.model.load_state_dict(
                torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        r_mean, r_std = test_data.r_mean, test_data.r_std

        hybrid_preds, har_preds, trues = [], [], []
        folder_path = './test_results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for seq_x, resid, har_pred, y_true in test_loader:
                seq_x = seq_x.float().to(self.device)

                outputs = self.model(seq_x)                 # (B, 1, 1)
                outputs = outputs[:, -1:, :].detach().cpu().numpy()

                # De-standardise the residual prediction, then add HAR linear part.
                resid_real = outputs * r_std + r_mean
                har_np     = har_pred.numpy()
                hybrid     = har_np + resid_real

                hybrid_preds.append(hybrid)
                har_preds.append(har_np)
                trues.append(y_true.numpy())

        hybrid_preds = np.concatenate(hybrid_preds, axis=0)
        har_preds    = np.concatenate(har_preds, axis=0)
        trues        = np.concatenate(trues, axis=0)

        folder_path = './results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)
        np.save(folder_path + 'pred.npy', hybrid_preds)
        np.save(folder_path + 'har_pred.npy', har_preds)
        np.save(folder_path + 'true.npy', trues)

        h_mae, h_mse, _, _, _, h_rse, _, h_qlike = metric(hybrid_preds, trues)
        l_mae, l_mse, _, _, _, l_rse, _, l_qlike = metric(har_preds, trues)

        print('HAR-only      mse:{:.4f}, mae:{:.4f}, rse:{:.4f}, qlike:{:.4f}'.format(
              l_mse, l_mae, l_rse, l_qlike))
        print('HAR-ModernTCN mse:{:.4f}, mae:{:.4f}, rse:{:.4f}, qlike:{:.4f}'.format(
              h_mse, h_mae, h_rse, h_qlike))

        with open('result_har_moderntcn.txt', 'a') as f:
            f.write(setting + '\n')
            f.write('HAR-only       mse:{:.4f}, mae:{:.4f}, rse:{:.4f}, qlike:{:.4f}\n'.format(
                    l_mse, l_mae, l_rse, l_qlike))
            f.write('HAR-ModernTCN  mse:{:.4f}, mae:{:.4f}, rse:{:.4f}, qlike:{:.4f}\n\n'.format(
                    h_mse, h_mae, h_rse, h_qlike))

        return {'mse': h_mse, 'mae': h_mae, 'qlike': h_qlike}
