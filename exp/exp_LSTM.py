from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import LSTM
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


class Exp_LSTM(Exp_Basic):

    def _build_model(self):
        # When aggregate_logsum is enabled the model head must output one value.
        # Temporarily set pred_len=1 so the projection layer is sized correctly,
        # then restore the original value for the data loader.
        if getattr(self.args, 'aggregate_logsum', False):
            orig = self.args.pred_len
            self.args.pred_len = 1
            model = LSTM.Model(self.args).float()
            self.args.pred_len = orig
        else:
            model = LSTM.Model(self.args).float()
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    # ------------------------------------------------------------------
    # Target helper: slice pred_len future steps and optionally aggregate
    # ------------------------------------------------------------------

    def _get_target(self, batch_y, f_dim):
        """
        Slice the future window and, when aggregating, reduce it to

            Y_t^(h) = ln( sum_{k=1..h} RV_{t+k} )

        The data channel holds ln(RV) (Dataset_Custom forbids scaling for
        exactly this reason), so the sum of levels is a log-sum-exp over the
        horizon axis. For h = 1 it reduces to ln(RV_{t+1}).
        """
        y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        if getattr(self.args, 'aggregate_logsum', False):
            y = torch.logsumexp(y, dim=1, keepdim=True)
        return y

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def vali(self, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                f_dim   = -1 if self.args.features == 'MS' else 0
                outputs = self.model(batch_x)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = self._get_target(batch_y, f_dim)

                loss = criterion(outputs.detach().cpu(), batch_y.detach().cpu())
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

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        train_steps   = len(train_loader)
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

            for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                f_dim   = -1 if self.args.features == 'MS' else 0
                outputs = self.model(batch_x)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = self._get_target(batch_y, f_dim)

                loss = criterion(outputs, batch_y)
                loss.backward()
                model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1,
                                         self.args, printout=False)
                    scheduler.step()

                train_loss.append(loss.item())

            train_loss = np.average(train_loss)
            vali_loss  = self.vali(vali_loader,  criterion)
            test_loss  = self.vali(test_loader,   criterion)

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
                print('LR →', scheduler.get_last_lr()[0])

        self.model.load_state_dict(
            torch.load(os.path.join(path, 'checkpoint.pth')))
        return self.model

    # ------------------------------------------------------------------
    # Testing
    # ------------------------------------------------------------------

    def test(self, setting, test=0):
        _, test_loader = self._get_data(flag='test')

        if test:
            self.model.load_state_dict(
                torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds, trues = [], []
        folder_path = './test_results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                f_dim   = -1 if self.args.features == 'MS' else 0
                outputs = self.model(batch_x)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = self._get_target(batch_y, f_dim)

                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        # Save results
        folder_path = './results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)
        np.save(folder_path + 'pred.npy', preds)

        mae, mse, rmse, mape, mspe, rse, corr, qlike = metric(preds, trues)
        print('mse:{:.4f}, mae:{:.4f}, rse:{:.4f}, qlike:{:.4f}'.format(
              mse, mae, rse, qlike))

        with open('result_lstm.txt', 'a') as f:
            f.write(setting + '\n')
            f.write('mse:{:.4f}, mae:{:.4f}, rse:{:.4f}, qlike:{:.4f}\n\n'.format(
                    mse, mae, rse, qlike))

        # return metrics so LSTM_run.py can aggregate mean/std across --itr seeds
        return {'mse': float(mse), 'mae': float(mae), 'rse': float(rse), 'qlike': float(qlike)}
