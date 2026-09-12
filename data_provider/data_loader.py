import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from utils.rv import prepare_rv_frame
import warnings

warnings.filterwarnings('ignore')





class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='EURUSD-RV.csv',
                 target='RV', scale=False, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        # Normalise the target to ln(RV): source files may store levels ('RV')
        # or logs ('ln_RV'). This also drops blank/incomplete rows (e.g. Excel
        # exports) and market holidays that leak in as RV = 0 -- NaN or -inf
        # targets would silently poison windows and test metrics.
        #
        # Everything downstream (the look-back window, and the ln(sum RV)
        # target built in Exp_Main._get_target) assumes this column is ln(RV),
        # which is also why `scale` must stay False: log-sum-exp over a
        # standardised series would not be ln(sum RV).
        df_raw = prepare_rv_frame(df_raw, self.target, verbose=(self.set_type == 0))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        # train: <= 2021, val: 2022-2023, test: >= 2024
        train_end = int((df_raw['date'].dt.year <= 2021).sum())
        val_end = int((df_raw['date'].dt.year <= 2023).sum())
        border1s = [0, train_end - self.seq_len, val_end - self.seq_len]
        border2s = [train_end, val_end, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            raise ValueError(
                'scale=True is incompatible with the ln(sum RV) target: the '
                'horizon aggregation is a log-sum-exp over raw ln(RV) values, '
                'which a StandardScaler would invalidate. Use RevIN (--revin 1) '
                'for input normalisation instead.'
            )
        data = df_data.values

        df_stamp = df_raw[['date']][border1:border2].copy()
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

class Dataset_Custom_Events(Dataset_Custom):
    """
    Dataset_Custom + a daily macro news-event calendar (data/events.csv).

    The event file must contain a 'date' column plus numeric per-day event
    features (multi-hot 'evt_*' indicator columns and 'n_events*' counts).
    Rows are aligned to the target CSV's trading dates by date; days missing
    from the event file are treated as no-event days (all zeros).

    'evt_*' indicator columns are kept raw (0/1). All other event columns
    (counts) are standardised with statistics from the TRAIN years only,
    mirroring how the target series is scaled.

    __getitem__ additionally returns:
        seq_x_events : (seq_len,  n_event_features)  events on the look-back days
        seq_y_events : (pred_len, n_event_features)  the KNOWN event schedule for
                                                     the pred_len forecast days
    Both are what is known at the forecast origin: the future window encodes
    only that an event is scheduled (release calendar), never its outcome.
    """

    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='EURUSD-RV.csv',
                 target='RV', scale=False, timeenc=0, freq='h',
                 event_path='events.csv'):
        self.event_path = event_path
        super().__init__(root_path=root_path, flag=flag, size=size,
                         features=features, data_path=data_path,
                         target=target, scale=scale, timeenc=timeenc, freq=freq)

    def __read_data__(self):
        super().__read_data__()

        # re-read and filter exactly like Dataset_Custom so row indices align
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_raw = prepare_rv_frame(df_raw, self.target, verbose=False)

        df_ev = pd.read_csv(os.path.join(self.root_path, self.event_path))
        df_ev['date'] = pd.to_datetime(df_ev['date'])
        ev_cols = [c for c in df_ev.columns if c != 'date']

        # align event rows to the trading dates of the target series;
        # dates absent from the event file become all-zero (no-event) days
        df_ev = df_ev.drop_duplicates(subset='date').set_index('date')
        # Days the calendar does not cover become all-zero rows, which is only
        # correct where the calendar genuinely spans the sample. Say so loudly
        # rather than silently labelling uncovered days as event-free.
        uncovered = int((~df_raw['date'].isin(df_ev.index)).sum())
        if uncovered and self.set_type == 0:
            lo, hi = df_ev.index.min(), df_ev.index.max()
            print(f"[events] WARNING: {uncovered} of {len(df_raw)} trading days are "
                  f"outside {self.event_path} ({lo.date()} .. {hi.date()}) and are "
                  f"treated as event-free.")
        events = df_ev.reindex(df_raw['date']).fillna(0.0)[ev_cols].values.astype(np.float32)

        # standardise count columns on the TRAIN years only; keep evt_* binary
        train_end = int((df_raw['date'].dt.year <= 2021).sum())
        count_idx = [i for i, c in enumerate(ev_cols) if not c.startswith('evt_')]
        if count_idx:
            tr = events[:train_end, count_idx]
            mean, std = tr.mean(axis=0), tr.std(axis=0) + 1e-8
            events[:, count_idx] = (events[:, count_idx] - mean) / std

        self.event_cols = ev_cols
        self.n_event_features = events.shape[1]
        # slice with the same borders as data_x/data_y so indices line up
        val_end = int((df_raw['date'].dt.year <= 2023).sum())
        border1s = [0, train_end - self.seq_len, val_end - self.seq_len]
        border2s = [train_end, val_end, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        self.data_events = events[border1:border2]

    def __getitem__(self, index):
        seq_x, seq_y, seq_x_mark, seq_y_mark = super().__getitem__(index)

        s_begin = index
        s_end = s_begin + self.seq_len
        seq_x_events = self.data_events[s_begin:s_end]
        seq_y_events = self.data_events[s_end:s_end + self.pred_len]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_x_events, seq_y_events


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='EURUSD-RV.csv',
                 target='RV', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        # same ln(RV) normalisation as Dataset_Custom so --do_predict sees the
        # series on the scale the model was trained on
        df_raw = prepare_rv_frame(df_raw, self.target, verbose=False)
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
