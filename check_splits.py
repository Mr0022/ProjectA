"""
Split-consistency check.

Reports train / val / test sizes for every model at every horizon and asserts
that the TEST count agrees across models sharing a data file.

Split definition (copied from ProjectC, data_provider/data_loader.py):
    deep models : train year <= 2021 | val 2022-2023 | test year >= 2024,
                  with the test slice starting at val_end - seq_len so the
                  first forecast origin is the last trading day of 2023
    HAR (OLS)   : train year <= 2023 (val folded in, no hyperparameters)
                  test  year >= 2024

Both index a forecast by the FIRST day its target window covers, so the two
rules select the same forecasts and the same number of them.

Run:  python check_splits.py
"""
import sys
import numpy as np
import pandas as pd

from utils.rv import prepare_rv_frame, pick_rv_column, to_log_rv

HORIZONS = (1, 5, 22)
SEQ_LENS = (22, 35, 70)          # every look-back used in scripts/
TRAIN_END, VAL_END = 2021, 2023  # deep-model boundaries
HAR_TRAIN_END = 2023             # HAR folds val into train


def deep_counts(path, target, seq_len, h):
    """Dataset_Custom window counts, from its border arithmetic."""
    df = prepare_rv_frame(pd.read_csv(path), target, verbose=False)
    yrs = df['date'].dt.year.values
    N = len(df)
    train_end = int((yrs <= TRAIN_END).sum())
    val_end = int((yrs <= VAL_END).sum())
    border1s = [0, train_end - seq_len, val_end - seq_len]
    border2s = [train_end, val_end, N]
    return tuple((b2 - b1) - seq_len - h + 1
                 for b1, b2 in zip(border1s, border2s))


def har_counts(path, h):
    """HAR row counts under the lagged-regressor / [t..t+h-1] convention."""
    raw = pd.read_csv(path, index_col=0, parse_dates=True)
    col = pick_rv_column(raw.columns)
    s = raw[col].sort_index().dropna().astype(float)
    ln_rv, keep = to_log_rv(s.values, col, verbose=False)
    s = pd.Series(ln_rv, index=s.index)[keep]

    rv = np.exp(s)
    y = s if h == 1 else np.log(rv.rolling(h).mean()).shift(-h).shift(1)
    feat = s.shift(1).rolling(22).mean()          # RV_m is the binding lag
    ok = y.notna() & feat.notna()
    yr = s.index.year
    return int((ok & (yr <= HAR_TRAIN_END)).sum()), 0, int((ok & (yr > HAR_TRAIN_END)).sum())


def main():
    rows, failures = [], []
    for h in HORIZONS:
        deep = {sl: deep_counts('data/EURUSD-RV.csv', 'RV', sl, h) for sl in SEQ_LENS}
        har = har_counts('data/EURUSD-RV.csv', h)
        harq = har_counts('data/realized_volatility_with_rqq.csv', h)

        for sl, c in deep.items():
            rows.append((f'LSTM / ModernTCN (seq_len={sl})', h, *c, 'EURUSD-RV.csv'))
        rows.append(('HAR-RV', h, *har, 'EURUSD-RV.csv'))
        rows.append(('HAR-Q', h, *harq, 'realized_volatility_with_rqq.csv'))

        # every model on EURUSD-RV.csv must agree on the test count
        expect = har[2]
        for sl, c in deep.items():
            if c[2] != expect:
                failures.append(f'h={h} seq_len={sl}: deep test {c[2]} != HAR-RV {expect}')
        # and the deep count must not depend on seq_len
        if len({c[2] for c in deep.values()}) != 1:
            failures.append(f'h={h}: deep test count varies with seq_len')

    w = max(len(r[0]) for r in rows)
    print(f"{'model':<{w}}  {'h':>3}  {'train':>6}  {'val':>5}  {'test':>5}   data")
    print('-' * (w + 45))
    last_h = None
    for name, h, tr, va, te, src in rows:
        if last_h is not None and h != last_h:
            print()
        print(f'{name:<{w}}  {h:>3}  {tr:>6}  {va:>5}  {te:>5}   {src}')
        last_h = h

    print()
    if failures:
        print('FAIL')
        for f in failures:
            print('  -', f)
        return 1
    print('PASS: every model on EURUSD-RV.csv reports the same test count at each')
    print('      horizon, and the deep count is independent of seq_len.')
    print()
    print('HAR-Q is listed on its own file (it needs realized quarticity, which')
    print('EURUSD-RV.csv does not carry), so its counts differ by construction.')
    print('Its split RULE is identical; point it at EURUSD-RV.csv once that file')
    print('has an RQ column and its counts fall in line with the rest.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
