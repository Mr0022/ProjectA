from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import ModernTCN
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import math

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'ModernTCN': ModernTCN,
        }
        # When aggregate_mean is enabled the model head must output a single
        # value (the predicted ln(mean RV)).  Temporarily set pred_len=1 so that
        # ModernTCN builds with target_window=1, then restore the original
        # value so the data loader still loads the full pred_len future steps
        # (needed to build the ground-truth target inside _get_target).
        if getattr(self.args, 'aggregate_mean', False):
            orig_pred_len = self.args.pred_len
            self.args.pred_len = 1
            model = model_dict[self.args.model].Model(self.args).float()
            self.args.pred_len = orig_pred_len
        else:
            model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_target(self, batch_y, f_dim):
        """
        Slice the future window and, when aggregating, reduce it to ProjectC's
        target -- the log of the horizon-AVERAGE variance:

            Y_t^(h) = ln( (1/h) * sum_{k=1..h} RV_{t+k} )

        The data channel holds ln(RV) (Dataset_Custom forbids scaling for
        exactly this reason), so RV = exp(ln_RV) and

            log(mean(RV)) == logsumexp(ln_RV) - log(h)

        which aggregates in variance space (the additive, economically correct
        space) while staying numerically stable. For h = 1 it reduces to
        ln(RV_{t+1}).
        """
        y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        if getattr(self.args, 'aggregate_mean', False):
            h = y.shape[1]
            y = torch.logsumexp(y, dim=1, keepdim=True) - math.log(h)
        return y

    def _unpack_batch(self, batch):
        """Event datasets yield 6 tensors (past/future events last), plain ones 4."""
        if len(batch) == 6:
            batch_x, batch_y, batch_x_mark, batch_y_mark, event_x, event_y = batch
            event_x = event_x.float().to(self.device)
            event_y = event_y.float().to(self.device)
        else:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            event_x = event_y = None
        return batch_x, batch_y, batch_x_mark, batch_y_mark, event_x, event_y

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, event_x, event_y = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        elif 'TCN' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark, event_x=event_x, event_y=event_y)
                            # outputs = self.model(batch_x)   #if decide not to use time stamp, use this code
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    elif 'TCN' in self.args.model:
                        outputs = self.model(batch_x, batch_x_mark, event_x=event_x, event_y=event_y)
                        # outputs = self.model(batch_x)   #if decide not to use time stamp, use this code
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = self._get_target(batch_y, f_dim)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def _train_one_epoch(self, train_loader, model_optim, criterion, scheduler,
                         scaler, epoch, train_steps, time_now, total_epochs):
        """
        One pass over `train_loader`. Returns (mean train loss, time_now).

        Extracted so the two-stage refit (--refit_on_val) reuses this exact
        loop instead of a second copy: a divergence between the two copies
        would surface as an unexplained quality gap between a staged and an
        unstaged run, which is a miserable thing to debug.
        """
        iter_count = 0
        train_loss = []

        self.model.train()
        for i, batch in enumerate(train_loader):
            batch_x, batch_y, batch_x_mark, batch_y_mark, event_x, event_y = self._unpack_batch(batch)
            iter_count += 1
            model_optim.zero_grad()
            batch_x = batch_x.float().to(self.device)

            batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

            # encoder - decoder
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    elif 'TCN' in self.args.model:
                        outputs = self.model(batch_x, batch_x_mark, event_x=event_x, event_y=event_y)
                        #outputs = self.model(batch_x)   #if decide not to use time stamp, use this code
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = self._get_target(batch_y, f_dim)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
            else:
                if 'Linear' in self.args.model or 'TST' in self.args.model:
                    outputs = self.model(batch_x)
                elif 'TCN' in self.args.model:
                    outputs = self.model(batch_x, batch_x_mark, event_x=event_x, event_y=event_y)
                    # outputs = self.model(batch_x)   #if decide not to use time stamp, use this code
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                # print(outputs.shape,batch_y.shape)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = self._get_target(batch_y, f_dim)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

            if (i + 1) % 100 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((total_epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            if self.args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()
            else:
                loss.backward()
                model_optim.step()

            if self.args.lradj == 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                scheduler.step()

        return np.average(train_loss), time_now

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        # Which epoch produced the checkpoint. --refit_on_val needs it as the
        # stage-2 epoch budget, so it is tracked off EarlyStopping's own
        # best_score rather than inferred from the patience counter.
        best_epoch = 0

        for epoch in range(self.args.train_epochs):
            epoch_time = time.time()
            train_loss, time_now = self._train_one_epoch(
                train_loader, model_optim, criterion, scheduler, scaler,
                epoch, train_steps, time_now, self.args.train_epochs)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            # Monitoring only -- early stopping below keys on vali_loss alone.
            # Nothing may select on this number.
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            prev_best = early_stopping.best_score
            early_stopping(vali_loss, self.model, path)
            if early_stopping.best_score != prev_best:
                best_epoch = epoch + 1
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        print('Best validation epoch: {} of {} run'.format(best_epoch, epoch + 1))

        if getattr(self.args, 'refit_on_val', False):
            self._refit_on_train_val(path, best_epoch)

        return self.model

    def _refit_on_train_val(self, path, best_epoch):
        """
        Stage 2 of --refit_on_val: refit on train + validation for the epoch
        count stage 1 chose.

        Stage 1 spends the validation years on early stopping and then throws
        them away, so the shipped model never trains on the most recent ~20% of
        the pre-test sample -- precisely the rows closest in distribution to the
        test period. The HAR baselines DO fit on those years, so without this
        stage a HAR-vs-deep comparison is also comparing training-set sizes.

        Two invariants keep this a refit rather than a leak:

          * The model is RE-INITIALISED, not fine-tuned from the stage-1
            weights, so the validation rows influence the epoch count and
            nothing else.
          * There is NO early stopping here, because those rows are now inside
            the training set. Stopping on them would be selecting on data being
            trained on, which is the exact failure this method exists to avoid.

        The OneCycle schedule is rebuilt to span exactly `best_epoch` epochs so
        the cycle completes; in stage 1 early stopping cuts it off mid-schedule.
        train+val holds ~20% more batches, so the same epoch count is ~20% more
        gradient steps -- intended, since more data is the whole point.
        """
        if best_epoch < 1:
            print('[refit] stage 1 never improved on its first epoch; skipping refit')
            return self.model

        print('\n' + '=' * 78)
        print('[refit] stage 2: re-fitting on train+val for {} epoch(s)'.format(best_epoch))
        print('=' * 78)

        refit_data, refit_loader = self._get_data(flag='train_val')

        # Re-initialise. Fine-tuning the stage-1 weights would carry the
        # early-stopped model's state into a run that can no longer be stopped.
        self.model = self._build_model().to(self.device)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        train_steps = len(refit_loader)
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=best_epoch,
                                            max_lr=self.args.learning_rate)

        time_now = time.time()
        for epoch in range(best_epoch):
            epoch_time = time.time()
            train_loss, time_now = self._train_one_epoch(
                refit_loader, model_optim, criterion, scheduler, scaler,
                epoch, train_steps, time_now, best_epoch)
            print('[refit] Epoch: {}/{}, Steps: {} | Train Loss: {:.7f} | {:.1f}s'.format(
                epoch + 1, best_epoch, train_steps, train_loss, time.time() - epoch_time))

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)

        # Overwrite the checkpoint so test() loads the refitted weights. test()
        # reloads from disk when called with test=1, so leaving the stage-1
        # checkpoint here would silently score the wrong model.
        torch.save(self.model.state_dict(), path + '/' + 'checkpoint.pth')
        print('[refit] done -- checkpoint now holds the train+val model')
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        if self.args.call_structural_reparam and hasattr(self.model, 'structural_reparam'):
            self.model.structural_reparam()

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, event_x, event_y = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        elif 'TCN' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark, event_x=event_x, event_y=event_y)
                            # outputs = self.model(batch_x)   #if decide not to use time stamp, use this code
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    elif 'TCN' in self.args.model:
                        outputs = self.model(batch_x, batch_x_mark, event_x=event_x, event_y=event_y)
                        # outputs = self.model(batch_x)   #if decide not to use time stamp, use this code
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = self._get_target(batch_y, f_dim)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1], batch_x.shape[2]))
            exit()
        # concatenate (not np.array + reshape): the test loader keeps the final
        # partial batch, so per-batch first dimensions differ.
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr, qlike = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}, qlike:{}'.format(mse, mae, rse, qlike))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}, qlike:{}'.format(mse, mae, rse, qlike))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return {'mse': float(mse), 'mae': float(mae), 'rse': float(rse), 'qlike': float(qlike)}

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(pred_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, event_x, event_y = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(
                    batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        elif 'TCN' in self.args.model:
                            outputs = self.model(batch_x, batch_x_mark, event_x=event_x, event_y=event_y)
                            # outputs = self.model(batch_x)   #if decide not to use time stamp, use this code
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    elif 'TCN' in self.args.model:
                        outputs = self.model(batch_x, batch_x_mark, event_x=event_x, event_y=event_y)
                        # outputs = self.model(batch_x)   #if decide not to use time stamp, use this code
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.concatenate(preds, axis=0)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
