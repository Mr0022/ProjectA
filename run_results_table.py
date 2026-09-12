"""
Train+test every model at its Optuna-best hyperparameters (h = 1, 5, 22),
parse the test-set MSE / MAE / QLIKE, add the HAR-RV baseline, and emit a
comparison table (CSV + Markdown).

Forecast target (all models): Y_t^(h) = ln( (1/h) * sum_{k=1..h} RV_{t+k} ), on
data/EURUSD-RV.csv.

!! The hyperparameters below were tuned on the PREVIOUS dataset
   (realized_volatility.csv) under the PREVIOUS target (mean of ln RV).
   They are kept as a starting point only -- re-run tune.py / LSTM_tune.py
   against the current data and target before treating this table as final.

HAR-Q is deliberately absent: it needs realized quarticity, which
EURUSD-RV.csv does not carry, so it runs on a different series and its
numbers are not comparable here (see HAR_Q_run.py's header).
"""
import os, re, subprocess, sys, json
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT, CSV = './data/', 'EURUSD-RV.csv'
EPOCHS, PATIENCE, NW = '100', '20', '0'

common_dl = ['--root_path', ROOT, '--data_path', CSV, '--features', 'S', '--target', 'RV']
common_tr = ['--lradj', 'TST', '--train_epochs', EPOCHS, '--patience', PATIENCE,
             '--num_workers', NW, '--itr', '1']

# ---- best hyperparameters per (model, horizon) ----------------------------
CONFIGS = {
 # ----- LSTM (custom, aggregated log-mean target) -----
 ('LSTM', 1): ['python','LSTM_run.py','--is_training','1','--model_id','LSTM_best_h1',
   '--data','custom','--enc_in','1','--aggregate_mean','--seq_len','22','--pred_len','1',
   '--hidden_size','128','--num_layers','3','--dropout','0.07608933759269505',
   '--head_dropout','0.29636364107380875','--revin','1',
   '--learning_rate','0.006732273106345163','--batch_size','256','--pct_start','0.12174563991351338'],
 ('LSTM', 5): ['python','LSTM_run.py','--is_training','1','--model_id','LSTM_best_h5',
   '--data','custom','--enc_in','1','--aggregate_mean','--seq_len','22','--pred_len','5',
   '--hidden_size','128','--num_layers','3','--dropout','0.06286059033500357',
   '--head_dropout','0.46748063314790966','--revin','1',
   '--learning_rate','0.00665668210175974','--batch_size','256','--pct_start','0.26239813663907'],
 ('LSTM', 22): ['python','LSTM_run.py','--is_training','1','--model_id','LSTM_best_h22',
   '--data','custom','--enc_in','1','--aggregate_mean','--seq_len','22','--pred_len','22',
   '--hidden_size','256','--num_layers','3','--dropout','0.1384357769833385',
   '--head_dropout','0.32313582077506997','--bidirectional','--revin','0',
   '--learning_rate','0.0033776477597474404','--batch_size','256','--pct_start','0.19707901510276543'],
 # ----- ModernTCN (custom, aggregated log-mean target) -----
 ('ModernTCN', 1): ['python','run.py','--is_training','1','--model_id','ModernTCN_best_h1','--model','ModernTCN',
   '--data','custom','--enc_in','1','--dec_in','1','--c_out','1','--aggregate_mean','--seq_len','22','--pred_len','1',
   '--patch_size','16','--patch_stride','2','--ffn_ratio','3',
   '--num_blocks','3','3','3','3','--large_size','31','31','31','31','--small_size','5','5','5','5',
   '--dims','256','256','256','256','--dw_dims','256','256','256','256',
   '--dropout','0.0905459364171444','--head_dropout','0.13523654131380805','--revin','1',
   '--use_multi_scale','False','--pct_start','0.3','--learning_rate','4.0990314788088356e-05','--batch_size','256'],
 ('ModernTCN', 5): ['python','run.py','--is_training','1','--model_id','ModernTCN_best_h5','--model','ModernTCN',
   '--data','custom','--enc_in','1','--dec_in','1','--c_out','1','--aggregate_mean','--seq_len','22','--pred_len','5',
   '--patch_size','16','--patch_stride','4','--ffn_ratio','4',
   '--num_blocks','2','2','2','2','--large_size','51','51','51','51','--small_size','5','5','5','5',
   '--dims','256','256','256','256','--dw_dims','256','256','256','256',
   '--dropout','0.4085579485889118','--head_dropout','0.3912055316326212','--revin','1',
   '--use_multi_scale','False','--pct_start','0.3','--learning_rate','4.208053742775343e-05','--batch_size','256'],
 ('ModernTCN', 22): ['python','run.py','--is_training','1','--model_id','ModernTCN_best_h22','--model','ModernTCN',
   '--data','custom','--enc_in','1','--dec_in','1','--c_out','1','--aggregate_mean','--seq_len','22','--pred_len','22',
   '--patch_size','16','--patch_stride','4','--ffn_ratio','4',
   '--num_blocks','2','2','2','2','--large_size','51','51','51','51','--small_size','5','5','5','5',
   '--dims','32','32','32','32','--dw_dims','32','32','32','32',
   '--dropout','0.12202910269242769','--head_dropout','0.025825336475530227','--revin','1',
   '--use_multi_scale','False','--pct_start','0.3','--learning_rate','0.00017992298020199686','--batch_size','256'],
}

METRIC_RE = re.compile(r'mse:\s*([\d.eE+-]+),\s*mae:\s*([\d.eE+-]+),\s*rse:\s*[\d.eE+-]+,\s*qlike:\s*([\d.eE+-]+)')

def parse_metrics(model, out):
    """Return (mse, mae, qlike) from a run's stdout."""
    lines = out.splitlines()
    cand = [ln for ln in lines if ln.strip().startswith('mse:')]
    if not cand:
        return None
    m = METRIC_RE.search(cand[-1])
    return tuple(float(x) for x in m.groups()) if m else None

def main():
    rows = []
    for (model, h), cmd in CONFIGS.items():
        full = cmd + common_dl + common_tr
        print(f'\n===== {model}  h={h} =====', flush=True)
        res = subprocess.run(full, cwd=HERE, capture_output=True, text=True)
        out = res.stdout + '\n' + res.stderr
        met = parse_metrics(model, out)
        if met is None:
            print('PARSE FAIL — tail of output:')
            print('\n'.join(out.splitlines()[-25:]))
            sys.exit(1)
        mse, mae, ql = met
        print(f'{model} h={h}: MSE={mse:.4f} MAE={mae:.4f} QLIKE={ql:.4f}', flush=True)
        rows.append(dict(model=model, horizon=h, MSE=mse, MAE=mae, QLIKE=ql))

    # ---- HAR-RV baseline from existing results ----
    har = pd.read_csv(os.path.join(HERE, 'HAR-RV results', 'har_rv_all_metrics.csv'))
    har = har[har['split'] == 'test']
    for _, r in har.iterrows():
        rows.append(dict(model='HAR-RV', horizon=int(r['horizon']),
                         MSE=float(r['MSE']), MAE=float(r['MAE']), QLIKE=float(r['QLIKE'])))

    df = pd.DataFrame(rows)
    order = ['HAR-RV', 'LSTM', 'ModernTCN']
    df['model'] = pd.Categorical(df['model'], categories=order, ordered=True)
    df = df.sort_values(['horizon', 'model']).reset_index(drop=True)
    df.to_csv(os.path.join(HERE, 'model_comparison_metrics.csv'), index=False)

    # ---- Markdown table (models as rows, metric×horizon as columns) ----
    md = ['| Model | ' + ' | '.join(f'h={h} {m}' for h in (1,5,22) for m in ('MSE','MAE','QLIKE')) + ' |',
          '|' + '---|'*(1+9)]
    for model in order:
        cells = [model]
        for h in (1,5,22):
            r = df[(df.model==model) & (df.horizon==h)]
            if len(r):
                cells += [f'{r.MSE.iloc[0]:.4f}', f'{r.MAE.iloc[0]:.4f}', f'{r.QLIKE.iloc[0]:.4f}']
            else:
                cells += ['-','-','-']
        md.append('| ' + ' | '.join(cells) + ' |')
    md_txt = '\n'.join(md)
    with open(os.path.join(HERE, 'model_comparison_table.md'), 'w') as f:
        f.write(md_txt + '\n')
    print('\n\n' + md_txt)

if __name__ == '__main__':
    main()
