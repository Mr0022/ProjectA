"""
Turn the raw macro news calendar (data/events_daily.csv) into the wide daily
feature matrix the event models consume.

The raw file is one row per scheduled release::

    Date,Name,Impact,Currency
    2012-09-03,HCOB Manufacturing PMI,LOW,EUR

`Dataset_Custom_Events` instead expects one row per DAY: a 'date' column plus
numeric per-day features, split into two blocks by name::

    n_events*   count columns, standardised by the loader on the TRAIN years
    evt_*       0/1 indicators, kept raw by the loader (so they must be binary)

This script builds that file. It is the replacement for the hand-rolled
data/events.csv, which stops at 2025-04-07 and therefore leaves most of the
>= 2024 test period silently event-free.

Two properties are worth stating explicitly because they are easy to get wrong:

1. **No look-ahead in the column set.** Which events get their own `evt_*`
   column is decided from the TRAIN years alone (`--train-end-year`, default
   2021, matching the loader's split). Picking columns by full-sample frequency
   would let the test period choose the feature space.

2. **Impact is per occurrence, not per event name.** 237 of the 515
   (currency, name) pairs carry more than one impact label over the sample, so
   the impact counts are tallied from each row's own label rather than from a
   name -> impact lookup.

Usage
-----
    python build_event_features.py                      # defaults below
    python build_event_features.py --min-train-days 50  # fewer, denser columns
    python build_event_features.py --counts basic       # old 6-count block only
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.rv import prepare_rv_frame  # noqa: E402

IMPACTS = ['HIGH', 'MEDIUM', 'LOW']


def normalise_name(name):
    """'CPI Flash Estimate y/y' -> 'CPI_Flash_Estimate_y_y' (events.csv style)."""
    slug = re.sub(r'[^0-9A-Za-z]+', '_', str(name))
    return re.sub(r'_+', '_', slug).strip('_')


def trading_calendar(rv_path, target):
    """
    The trading days the event file must line up with.

    Reuses the loader's own cleaning (`prepare_rv_frame`) rather than
    re-implementing it, so the emitted index is exactly the index
    `Dataset_Custom_Events` will reindex onto -- including the dropped
    non-positive RV rows (market holidays that leak in as RV = 0).
    """
    df = pd.read_csv(rv_path)
    df = prepare_rv_frame(df, target, verbose=False)
    return pd.DatetimeIndex(df['date'].unique()).sort_values()


def load_raw(raw_path):
    df = pd.read_csv(raw_path)
    missing = {'Date', 'Name', 'Impact', 'Currency'} - set(df.columns)
    if missing:
        raise ValueError(f'{raw_path} is missing column(s): {sorted(missing)}')

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    bad = int(df['Date'].isna().sum())
    if bad:
        print(f'[raw] dropped {bad} row(s) with an unparseable Date')
        df = df[df['Date'].notna()]

    df['Impact'] = df['Impact'].astype(str).str.strip().str.upper()
    df['Currency'] = df['Currency'].astype(str).str.strip().str.upper()

    unknown = sorted(set(df['Impact']) - set(IMPACTS))
    if unknown:
        print(f'[raw] WARNING: unrecognised Impact label(s) {unknown}; '
              f'these rows count in n_events but in no per-impact column')
    return df.sort_values('Date').reset_index(drop=True)


def align_to_calendar(df, calendar, roll):
    """
    Restrict events to trading days.

    Weekend/holiday releases (~0.5% of rows) have no trading day of their own.
    Default is to drop them, which is what the loader's reindex would do anyway;
    `--roll-weekend-events` instead moves them to the next trading day, on the
    view that their information is impounded at the next open.
    """
    on_cal = df['Date'].isin(calendar)
    n_off = int((~on_cal).sum())
    if n_off == 0:
        return df

    if roll:
        pos = calendar.searchsorted(df.loc[~on_cal, 'Date'], side='left')
        inside = pos < len(calendar)
        rolled = df.loc[~on_cal].copy()
        rolled.loc[:, 'Date'] = np.where(
            inside, calendar[np.clip(pos, 0, len(calendar) - 1)], pd.NaT)
        rolled = rolled[rolled['Date'].notna()]
        print(f'[align] rolled {len(rolled)} of {n_off} off-calendar event(s) '
              f'forward to the next trading day '
              f'({n_off - len(rolled)} fell past the end of the sample)')
        df = pd.concat([df[on_cal], rolled], ignore_index=True)
    else:
        off = df.loc[~on_cal, 'Date']
        days = dict(off.dt.day_name().value_counts())
        print(f'[align] dropped {n_off} event(s) on non-trading days {days}; '
              f'pass --roll-weekend-events to keep them')
        df = df[on_cal]
    return df.sort_values('Date').reset_index(drop=True)


def build_counts(df, calendar, mode):
    """Dense per-day tallies. Standardised by the loader (no 'evt_' prefix)."""
    out = pd.DataFrame(index=calendar)
    out.index.name = 'date'

    out['n_events'] = df.groupby('Date').size()
    for imp in IMPACTS:
        out[f'n_events_{imp.lower()}'] = df[df['Impact'] == imp].groupby('Date').size()
    for cur in ['EUR', 'USD']:
        out[f'n_events_{cur.lower()}'] = df[df['Currency'] == cur].groupby('Date').size()

    if mode == 'cross':
        # A HIGH-impact USD print and a HIGH-impact EUR print are not the same
        # shock to EUR/USD, and the marginal counts above cannot tell them
        # apart. The raw file labels every row, so the cross is free.
        for cur in ['EUR', 'USD']:
            for imp in IMPACTS:
                sel = df[(df['Currency'] == cur) & (df['Impact'] == imp)]
                out[f'n_events_{cur.lower()}_{imp.lower()}'] = sel.groupby('Date').size()

    return out.fillna(0.0).astype('float64')


def build_indicators(df, calendar, min_train_days, train_end_year):
    """
    One 0/1 column per frequent (currency, event name) pair.

    Selection uses the TRAIN years only -- see the module docstring. Names that
    normalise to the same column (e.g. 'Current Account n.s.a' and
    '... n.s.a.') are merged, and the result is clamped to 0/1 because the
    loader passes 'evt_*' columns through unscaled.
    """
    df = df.copy()
    df['col'] = 'evt_' + df['Currency'] + '_' + df['Name'].map(normalise_name)

    train = df[df['Date'].dt.year <= train_end_year]
    freq = train.groupby('col')['Date'].nunique()
    keep = sorted(freq[freq >= min_train_days].index)

    print(f'[indicators] {df["col"].nunique()} distinct (currency, name) columns; '
          f'{len(keep)} occur on >= {min_train_days} distinct train days '
          f'(<= {train_end_year}) and get a column')
    dropped = df['col'].nunique() - len(keep)
    if dropped:
        rare = int(df.loc[~df['col'].isin(keep)].shape[0])
        print(f'[indicators] {dropped} rare column(s) dropped, covering '
              f'{rare} raw row(s) ({rare / len(df) * 100:.1f}% of the calendar); '
              f'they still count in the n_events* block')

    if not keep:
        return pd.DataFrame(index=calendar).rename_axis('date')

    wide = (df[df['col'].isin(keep)]
            .assign(v=1.0)
            .pivot_table(index='Date', columns='col', values='v', aggfunc='max')
            .reindex(index=calendar, columns=keep)
            .fillna(0.0)
            .astype('float64'))
    wide.index.name = 'date'
    return wide


def report(out, calendar, raw, train_end_year, val_end_year, out_path):
    ev_cols = [c for c in out.columns if c != 'date']
    evt = [c for c in ev_cols if c.startswith('evt_')]
    cnt = [c for c in ev_cols if not c.startswith('evt_')]

    years = out['date'].dt.year
    n_tr = int((years <= train_end_year).sum())
    n_va = int(((years > train_end_year) & (years <= val_end_year)).sum())

    covered = calendar.isin(raw['Date'].unique())
    uncovered = calendar[~covered]

    print()
    print(f'[out] {out_path}')
    print(f'[out] {len(out)} rows x {len(ev_cols)} event features '
          f'({len(cnt)} count + {len(evt)} indicator)')
    print(f'[out] dates {out["date"].min().date()} .. {out["date"].max().date()} '
          f'| train {n_tr} / val {n_va} / test {len(out) - n_tr - n_va}')
    if evt:
        print(f'[out] indicator density: {out[evt].to_numpy().mean() * 100:.2f}% '
              f'of cells are 1; mean {out[evt].sum(axis=1).mean():.1f} '
              f'named events/day')
    print(f'[out] n_events/day: mean {out["n_events"].mean():.1f}, '
          f'max {out["n_events"].max():.0f}')

    if len(uncovered):
        by_year = dict(uncovered.to_series().dt.year.value_counts().sort_index())
        print(f'[out] WARNING: {len(uncovered)} of {len(calendar)} trading days are '
              f'outside the raw calendar and are emitted as all-zero (event-free) '
              f'rows: {by_year}')
    else:
        print('[out] every trading day is covered by the raw calendar')


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--raw-path', default='data/events_daily.csv',
                   help='raw long-format calendar (Date,Name,Impact,Currency)')
    p.add_argument('--rv-path', default='data/EURUSD-RV.csv',
                   help="RV file whose trading days set the output index; "
                        "'none' emits a row per date present in the raw file")
    p.add_argument('--target', default='RV',
                   help='RV column name in --rv-path (matches run.py --target)')
    p.add_argument('--out', default='data/events_daily_features.csv',
                   help='destination for the wide daily feature file')
    p.add_argument('--min-train-days', type=int, default=30,
                   help='an event needs this many distinct TRAIN days to earn '
                        'its own evt_* column (default: 30)')
    p.add_argument('--train-end-year', type=int, default=2021,
                   help='last train year; must match the loader split')
    p.add_argument('--val-end-year', type=int, default=2023,
                   help='last validation year; must match the loader split')
    p.add_argument('--counts', choices=['basic', 'cross'], default='cross',
                   help="'basic' = the 6 count columns events.csv had; "
                        "'cross' also adds currency x impact counts (default)")
    p.add_argument('--roll-weekend-events', action='store_true',
                   help='move non-trading-day releases to the next trading day '
                        'instead of dropping them')
    args = p.parse_args(argv)

    raw = load_raw(args.raw_path)
    print(f'[raw] {len(raw)} rows, {raw["Date"].nunique()} distinct dates, '
          f'{raw["Name"].nunique()} distinct names, '
          f'{raw["Date"].min().date()} .. {raw["Date"].max().date()}')

    if str(args.rv_path).lower() == 'none':
        calendar = pd.DatetimeIndex(sorted(raw['Date'].unique()))
        print(f'[cal] no RV file: using the {len(calendar)} dates in the raw file')
    else:
        calendar = trading_calendar(args.rv_path, args.target)
        print(f'[cal] {len(calendar)} trading days from {args.rv_path} '
              f'({calendar.min().date()} .. {calendar.max().date()})')
        raw = align_to_calendar(raw, calendar, args.roll_weekend_events)

    counts = build_counts(raw, calendar, args.counts)
    indicators = build_indicators(raw, calendar, args.min_train_days,
                                  args.train_end_year)

    out = pd.concat([counts, indicators], axis=1).reset_index()

    # The loader trusts these invariants; assert them here rather than let a
    # malformed file surface as a silently wrong training run.
    assert out['date'].is_monotonic_increasing, 'output dates are not sorted'
    assert not out['date'].duplicated().any(), 'output has duplicate dates'
    assert not out.isna().to_numpy().any(), 'output contains NaN'
    evt = [c for c in out.columns if c.startswith('evt_')]
    if evt:
        vals = np.unique(out[evt].to_numpy())
        assert set(vals) <= {0.0, 1.0}, f'evt_* columns are not binary: {vals[:5]}'

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out, index=False, date_format='%Y-%m-%d')

    report(out, calendar, raw, args.train_end_year, args.val_end_year, args.out)
    return out


if __name__ == '__main__':
    main()
