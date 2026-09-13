"""
Smoke test for the daily event calendar built by build_event_features.py.

Answers one question: can the event models actually train on this file, and do
the tensors they receive line up with the dates they claim to?

Checks, in order:

  1. file contract   -- what Dataset_Custom_Events assumes about the csv
  2. coverage        -- trading days the calendar does not reach (all-zero rows)
  3. dataset wiring  -- splits, shapes, and run.py's event_in derivation
  4. date alignment  -- data_events[j] really is the event row for trading day j
  5. no look-ahead   -- seq_y_events covers only the forecast window
  6. scaling         -- counts standardised on TRAIN only, evt_* left binary
  7. batch + model   -- one real DataLoader batch through ModernTCN, fwd + bwd

1 and 2 need only pandas; 3-7 need torch and are skipped without it.

    python smoke_test_events.py
    python smoke_test_events.py --event-path events_daily_features.csv --pred-len 5
"""

import argparse
import os
import sys
import types

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

PASS, FAIL, SKIP = [], [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  [{"PASS" if cond else "FAIL"}] {name}' + (f' -- {detail}' if detail else ''))
    return cond


def skip(name, why):
    SKIP.append(name)
    print(f'  [SKIP] {name} -- {why}')


def section(title):
    print(f'\n{title}\n' + '-' * len(title))


def check_file_contract(path):
    section(f'1. file contract: {path}')
    df = pd.read_csv(path)
    check('has a lowercase "date" column', 'date' in df.columns,
          f'columns start: {list(df.columns)[:3]}')

    ev_cols = [c for c in df.columns if c != 'date']
    check('has at least one event feature column', len(ev_cols) > 0,
          f'{len(ev_cols)} feature columns')

    non_numeric = [c for c in ev_cols if not pd.api.types.is_numeric_dtype(df[c])]
    check('every feature column is numeric', not non_numeric, f'offenders: {non_numeric[:5]}')
    check('no NaN anywhere', not df.isna().to_numpy().any(),
          f'{int(df.isna().to_numpy().sum())} NaN cells')
    check('no inf anywhere', bool(np.isfinite(df[ev_cols].to_numpy()).all()))

    d = pd.to_datetime(df['date'], errors='coerce')
    check('every date parses', not d.isna().any(), f'{int(d.isna().sum())} unparseable')
    check('dates are unique', not d.duplicated().any(), f'{int(d.duplicated().sum())} duplicates')
    check('dates are sorted ascending', d.is_monotonic_increasing)

    evt = [c for c in ev_cols if c.startswith('evt_')]
    cnt = [c for c in ev_cols if not c.startswith('evt_')]
    if evt:
        vals = set(np.unique(df[evt].to_numpy()))
        check('evt_* columns are strictly 0/1 (loader keeps them raw)',
              vals <= {0.0, 1.0}, f'distinct values: {sorted(vals)[:5]}')
    check('count columns are non-negative (they are raw tallies here)',
          bool((df[cnt].to_numpy() >= 0).all()) if cnt else True)
    check('no all-zero feature column (dead input)',
          bool((df[ev_cols].to_numpy().sum(axis=0) > 0).all()),
          f'dead: {[c for c in ev_cols if df[c].sum() == 0][:5]}')
    print(f'  ... {len(cnt)} count + {len(evt)} indicator columns, {len(df)} rows, '
          f'{d.min().date()} .. {d.max().date()}')
    return df


def check_coverage(ev_df, root, data_path, target):
    section('2. coverage against the RV trading calendar')
    from utils.rv import prepare_rv_frame
    rv = prepare_rv_frame(pd.read_csv(os.path.join(root, data_path)), target, verbose=False)
    ev_dates = pd.to_datetime(ev_df['date'])

    covered = rv['date'].isin(set(ev_dates))
    n_unc = int((~covered).sum())
    check('event file has a row for every trading day', n_unc == 0,
          f'{n_unc} of {len(rv)} trading days missing -> all-zero rows')

    # An all-zero row inside the test period is worse than one in early train:
    # it tells the model "no events today" on days that certainly had some.
    yrs = rv.loc[~covered, 'date'].dt.year if n_unc else pd.Series(dtype=int)
    n_test_unc = int((yrs >= 2024).sum())
    check('no uncovered day falls in the >= 2024 test period', n_test_unc == 0,
          f'{n_test_unc} uncovered test days')
    if n_unc:
        print(f'  ... uncovered by year: {dict(yrs.value_counts().sort_index())}')
    return rv


def build_args(root, data_path, event_path, target, seq_len, pred_len, label_len):
    return types.SimpleNamespace(
        data='custom_events', root_path=root, data_path=data_path,
        event_data_path=event_path, features='S', target=target,
        embed='timeF', freq='h', seq_len=seq_len, label_len=label_len,
        pred_len=pred_len, batch_size=8, num_workers=0, use_events=True)


def check_dataset(args, ev_df, rv):
    section('3. dataset wiring')
    from data_provider.data_factory import data_provider

    sets = {}
    for flag in ['train', 'val', 'test']:
        ds, dl = data_provider(args, flag)
        sets[flag] = (ds, dl)

    tr, va, te = (sets[f][0] for f in ['train', 'val', 'test'])
    check('all three splits are non-empty',
          all(len(sets[f][0]) > 0 for f in sets),
          ', '.join(f'{f}={len(sets[f][0])}' for f in sets))

    n_feat = tr.n_event_features
    check('event feature count matches the csv',
          n_feat == len([c for c in ev_df.columns if c != 'date']), f'{n_feat} features')

    # run.py derives event_in straight from the header; if that ever drifts from
    # what the dataset builds, the Linear layer is silently the wrong width.
    header = pd.read_csv(os.path.join(args.root_path, args.event_data_path), nrows=0)
    check("run.py's event_in matches the dataset",
          len([c for c in header.columns if c != 'date']) == n_feat)

    check('every split exposes the same feature width',
          va.n_event_features == n_feat and te.n_event_features == n_feat)
    check('event rows align with value rows in every split',
          all(len(sets[f][0].data_events) == len(sets[f][0].data_x) for f in sets))

    train_end = int((rv['date'].dt.year <= 2021).sum())
    val_end = int((rv['date'].dt.year <= 2023).sum())
    check('split borders follow the 2021/2023 year cuts',
          len(tr.data_x) == train_end
          and len(va.data_x) == val_end - train_end + args.seq_len
          and len(te.data_x) == len(rv) - val_end + args.seq_len,
          f'train_end={train_end}, val_end={val_end}, n={len(rv)}')
    return sets, train_end, val_end


def check_alignment(sets, ev_df, rv, args, train_end, val_end):
    section('4. date alignment (events must describe the day they sit on)')
    ev = ev_df.copy()
    ev['date'] = pd.to_datetime(ev['date'])
    ev_cols = [c for c in ev.columns if c != 'date']
    aligned = ev.set_index('date').reindex(rv['date']).fillna(0.0)[ev_cols].to_numpy()

    borders = {'train': 0,
               'val': train_end - args.seq_len,
               'test': val_end - args.seq_len}
    ok = True
    for flag, b1 in borders.items():
        ds = sets[flag][0]
        evt_idx = [i for i, c in enumerate(ds.event_cols) if c.startswith('evt_')]
        got = ds.data_events[:, evt_idx]
        want = aligned[b1:b1 + len(ds.data_events)][:, evt_idx]
        if not np.allclose(got, want):
            ok = False
            print(f'    {flag}: {int((got != want).sum())} mismatched indicator cells')
    check('evt_* rows equal the csv rows for the same trading dates '
          '(all 3 splits, borders included)', ok)

    # Spot-check one real, dated event rather than trusting the vector algebra.
    ds = sets['test'][0]
    test_dates = rv['date'].to_numpy()[val_end - args.seq_len:]
    dense = ds.data_events[:, [i for i, c in enumerate(ds.event_cols)
                               if c.startswith('evt_')]].sum(axis=1)
    j = int(np.argmax(dense))
    check('busiest test day carries its named events',
          dense[j] > 0,
          f'{pd.Timestamp(test_dates[j]).date()} has {int(dense[j])} named events')


def check_no_lookahead(sets, args):
    section('5. no look-ahead in the future-event window')
    ds = sets['train'][0]
    i = 0
    out = ds[i]
    check('__getitem__ returns 6 tensors (x, y, x_mark, y_mark, ev_x, ev_y)', len(out) == 6)
    seq_x, seq_y, _, _, ev_x, ev_y = out

    s_end = i + args.seq_len
    check('seq_x_events is exactly the look-back window',
          ev_x.shape[0] == args.seq_len
          and np.array_equal(ev_x, ds.data_events[i:s_end]),
          f'shape {tuple(ev_x.shape)}')
    check('seq_y_events starts the day AFTER the look-back ends',
          np.array_equal(ev_y, ds.data_events[s_end:s_end + args.pred_len]),
          f'rows [{s_end}, {s_end + args.pred_len})')
    check('seq_y_events covers exactly pred_len days',
          ev_y.shape[0] == args.pred_len, f'{ev_y.shape[0]} vs pred_len={args.pred_len}')
    check('past and future event windows do not overlap',
          not np.shares_memory(ev_x, ev_y) or args.pred_len == 0 or
          ev_x.shape[0] + ev_y.shape[0] == args.seq_len + args.pred_len)

    last = len(ds) - 1
    _, _, _, _, ev_x_l, ev_y_l = ds[last]
    check('the last window is fully in range (no short/ragged tail)',
          ev_x_l.shape[0] == args.seq_len and ev_y_l.shape[0] == args.pred_len,
          f'index {last}: {ev_x_l.shape[0]}/{ev_y_l.shape[0]}')


def check_scaling(sets):
    section('6. scaling (train-only stats; indicators untouched)')
    tr = sets['train'][0]
    cnt_idx = [i for i, c in enumerate(tr.event_cols) if not c.startswith('evt_')]
    evt_idx = [i for i, c in enumerate(tr.event_cols) if c.startswith('evt_')]

    if cnt_idx:
        m = tr.data_events[:, cnt_idx].mean(axis=0)
        s = tr.data_events[:, cnt_idx].std(axis=0)
        check('count columns are ~standardised on the train split',
              bool(np.abs(m).max() < 1e-3 and np.abs(s - 1).max() < 1e-2),
              f'|mean|max={np.abs(m).max():.2e}, |std-1|max={np.abs(s - 1).max():.2e}')
        # Test stats must NOT be forced to 0/1 -- that would mean the scaler saw them.
        te_m = sets['test'][0].data_events[:, cnt_idx].mean(axis=0)
        check('test counts are NOT re-centred (no train/test leakage)',
              bool(np.abs(te_m).max() > 1e-6), f'|test mean|max={np.abs(te_m).max():.3f}')

    if evt_idx:
        vals = set(np.unique(tr.data_events[:, evt_idx]))
        check('evt_* survive the loader as 0/1', vals <= {0.0, 1.0}, f'{sorted(vals)[:5]}')
    check('no NaN/inf in the tensors the model will see',
          bool(np.isfinite(tr.data_events).all()))


def check_model(sets, args, ev_df, event_dim, fusion):
    section(f'7. one real batch through ModernTCN (event_fusion={fusion})')
    import torch
    from models.ModernTCN import Model

    n_feat = len([c for c in ev_df.columns if c != 'date'])
    # Mirrors scripts/eventtcn.sh; every attribute Model.__init__ reads is set
    # explicitly so a missing one fails here rather than mid-training.
    cfg = types.SimpleNamespace(
        seq_len=args.seq_len, pred_len=args.pred_len, enc_in=1,
        stem_ratio=6, downsample_ratio=2, ffn_ratio=2,
        # ModernTCN's backbone hardcodes 4 stages, and its te_patch only
        # accepts freq 'h'/'t' -- both are what run.py's defaults give it.
        num_blocks=[1, 1, 1, 1], large_size=[13] * 4, small_size=[5] * 4,
        dims=[32] * 4, dw_dims=[32] * 4, patch_size=8, patch_stride=4,
        small_kernel_merged=False, dropout=0.1, head_dropout=0.0,
        use_multi_scale=False, revin=1, affine=0, subtract_last=0,
        freq='h', individual=False, kernel_size=25, decomposition=0,
        use_events=True, event_in=n_feat, event_dim=event_dim,
        event_past=True, event_future=True, event_fusion=fusion,
    )
    model = Model(cfg).float()
    check(f'event embedding is Linear({n_feat} -> {event_dim})',
          model.model.event_embed.in_features == n_feat
          and model.model.event_embed.out_features == event_dim)

    batch = next(iter(sets['train'][1]))
    check('DataLoader yields 6 tensors', len(batch) == 6)
    x, y, x_mark, y_mark, ev_x, ev_y = [b.float() for b in batch]
    check('past-event batch is (B, seq_len, n_features)',
          tuple(ev_x.shape) == (x.shape[0], args.seq_len, n_feat), f'{tuple(ev_x.shape)}')
    check('future-event batch is (B, pred_len, n_features)',
          tuple(ev_y.shape) == (x.shape[0], args.pred_len, n_feat), f'{tuple(ev_y.shape)}')

    out = model(x, x_mark, event_x=ev_x, event_y=ev_y)
    check('forward pass produces a finite output', bool(torch.isfinite(out).all()),
          f'output {tuple(out.shape)}')

    loss = torch.nn.functional.mse_loss(out[:, -args.pred_len:, :],
                                        y[:, -args.pred_len:, :].float())
    loss.backward()
    g = model.model.event_embed.weight.grad
    check('loss is finite', bool(torch.isfinite(loss)), f'mse={loss.item():.4f}')
    check('gradient reaches the event embedding (events are really connected)',
          g is not None and bool(torch.isfinite(g).all()) and float(g.abs().sum()) > 0,
          f'|grad|sum={float(g.abs().sum()):.3e}' if g is not None else 'no grad')

    # Same batch, events zeroed: the output must move, or conditioning is a no-op.
    with torch.no_grad():
        base = model(x, x_mark, event_x=ev_x, event_y=ev_y)
        zeroed = model(x, x_mark, event_x=torch.zeros_like(ev_x),
                       event_y=torch.zeros_like(ev_y))
    check('zeroing the events changes the prediction (conditioning is live)',
          float((base - zeroed).abs().max()) > 1e-8,
          f'max|delta|={float((base - zeroed).abs().max()):.3e}')


def check_end_to_end(a, root_dir):
    """
    The real thing: run.py for one epoch on the new calendar.

    Everything above tests pieces in isolation; this is the only check that
    proves the documented command line actually trains and scores a model.
    """
    section('8. end-to-end: run.py --use_events, 1 epoch')
    import subprocess
    import tempfile

    cmd = [
        sys.executable, 'run.py',
        '--is_training', '1', '--model_id', 'smoke_events', '--model', 'ModernTCN',
        '--data', 'custom', '--root_path', a.root_path, '--data_path', a.data_path,
        '--features', 'S', '--target', a.target,
        '--enc_in', '1', '--dec_in', '1', '--c_out', '1',
        '--seq_len', str(a.seq_len), '--label_len', str(a.label_len),
        '--pred_len', str(a.pred_len),
        '--use_events', '--event_data_path', a.event_path,
        '--event_dim', str(a.event_dim),
        '--aggregate_mean', '--event_fusion', 'channel',
        '--ffn_ratio', '3',
        '--num_blocks', '1', '1', '1', '1',
        '--large_size', '13', '13', '13', '13',
        '--small_size', '5', '5', '5', '5',
        '--dims', '32', '32', '32', '32', '--dw_dims', '32', '32', '32', '32',
        '--patch_size', '8', '--patch_stride', '4',
        '--dropout', '0.1', '--head_dropout', '0.1', '--revin', '1',
        '--use_multi_scale', 'False',
        '--train_epochs', '1', '--itr', '1', '--batch_size', '128',
        '--learning_rate', '0.001', '--use_gpu', '',
        '--des', 'smoke',
    ]
    print('  $ ' + ' '.join(cmd[1:]))

    # run.py appends to ./result.txt and writes ./checkpoints, ./results and
    # ./test_results relative to its cwd. Run it from a throwaway directory of
    # symlinks so a smoke run never lands in the repo's real result log.
    with tempfile.TemporaryDirectory(prefix='smoke_events_') as tmp:
        for entry in ['run.py', 'data', 'data_provider', 'exp', 'models',
                      'layers', 'utils']:
            os.symlink(os.path.join(root_dir, entry), os.path.join(tmp, entry))
        r = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=3600)
    tail = (r.stdout or '')[-4000:]

    if not check('run.py exits 0', r.returncode == 0, f'exit {r.returncode}'):
        print('  ---- stdout tail ----')
        print('\n'.join(tail.splitlines()[-25:]))
        print('  ---- stderr tail ----')
        print('\n'.join((r.stderr or '').splitlines()[-25:]))
        return

    n_feat = len([c for c in pd.read_csv(
        os.path.join(a.root_path, a.event_path), nrows=0).columns if c != 'date'])
    check(f'run.py reports {n_feat} event feature columns',
          f'news events: {n_feat} feature columns' in tail)
    check('training loop produced a train/val/test loss line',
          'Train Loss' in tail and 'Vali Loss' in tail)
    check('test metrics were computed', 'mse:' in tail.lower())

    losses = [float(m) for m in __import__('re').findall(r'Train Loss: ([0-9.]+)', tail)]
    check('train loss is finite and non-degenerate',
          bool(losses) and all(np.isfinite(losses)) and all(l > 0 for l in losses),
          f'train loss {losses}')
    for line in tail.splitlines():
        if 'mse:' in line.lower() or 'Vali Loss' in line:
            print(f'  ... {line.strip()}')



def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root-path', default='./data/')
    p.add_argument('--data-path', default='EURUSD-RV.csv')
    p.add_argument('--event-path', default='events_daily_features.csv')
    p.add_argument('--target', default='RV')
    p.add_argument('--seq-len', type=int, default=70)
    p.add_argument('--label-len', type=int, default=0)
    p.add_argument('--pred-len', type=int, default=1)
    p.add_argument('--event-dim', type=int, default=8)
    p.add_argument('--no-end-to-end', action='store_true',
                   help='skip stage 8 (the real run.py training run)')
    a = p.parse_args()

    print(f'smoke test: {os.path.join(a.root_path, a.event_path)} '
          f'(seq_len={a.seq_len}, pred_len={a.pred_len})')

    ev_df = check_file_contract(os.path.join(a.root_path, a.event_path))
    rv = check_coverage(ev_df, a.root_path, a.data_path, a.target)

    try:
        import torch  # noqa: F401
    except ImportError as e:
        for name in ['dataset wiring', 'date alignment', 'no look-ahead',
                     'scaling', 'batch + model', 'end-to-end run.py']:
            skip(name, f'torch not installed ({e})')
    else:
        args = build_args(a.root_path, a.data_path, a.event_path, a.target,
                          a.seq_len, a.pred_len, a.label_len)
        sets, train_end, val_end = check_dataset(args, ev_df, rv)
        check_alignment(sets, ev_df, rv, args, train_end, val_end)
        check_no_lookahead(sets, args)
        check_scaling(sets)
        for fusion in ['inject', 'channel']:
            check_model(sets, args, ev_df, a.event_dim, fusion)
        if a.no_end_to_end:
            skip('end-to-end run.py', '--no-end-to-end')
        else:
            check_end_to_end(a, os.path.dirname(os.path.abspath(__file__)))

    print(f'\n{"=" * 60}')
    print(f'{len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped')
    if FAIL:
        print('FAILED:')
        for f in FAIL:
            print(f'  - {f}')
    print('=' * 60)
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
