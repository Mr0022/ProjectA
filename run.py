import argparse
import os

import torch
from exp.exp_ModernTCN import Exp_Main
import random
import numpy as np
from utils.str2bool import str2bool

parser = argparse.ArgumentParser(description='ModernTCN')

# random seed
parser.add_argument('--random_seed', type=int, default=2021,
                    help='base random seed; training iteration ii uses (random_seed + ii), '
                         'so the default 2021 with --itr 5 sweeps seeds 2021..2025')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='ModernTCN',
                    help='model name, options: [ModernTCN]')

# data loader
parser.add_argument('--data', type=str, default='custom', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='EURUSD-RV.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='RV',
                    help="target feature in S or MS task; a level RV column is log-transformed on load")
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')


# forecasting task
parser.add_argument('--seq_len', type=int, default=22, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length (unused by ModernTCN; must be <= seq_len)')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')




#ModernTCN
parser.add_argument('--stem_ratio', type=int, default=6, help='stem ratio')
parser.add_argument('--downsample_ratio', type=int, default=2, help='downsample_ratio')
parser.add_argument('--ffn_ratio', type=int, default=2, help='ffn_ratio')
parser.add_argument('--patch_size', type=int, default=16, help='the patch size')
parser.add_argument('--patch_stride', type=int, default=8, help='the patch stride')

parser.add_argument('--num_blocks', nargs='+',type=int, default=[1,1,1,1], help='num_blocks in each stage')
parser.add_argument('--large_size', nargs='+',type=int, default=[31,29,27,13], help='big kernel size')
parser.add_argument('--small_size', nargs='+',type=int, default=[5,5,5,5], help='small kernel size for structral reparam')
parser.add_argument('--dims', nargs='+',type=int, default=[256,256,256,256], help='dmodels in each stage')
parser.add_argument('--dw_dims', nargs='+',type=int, default=[256,256,256,256])

parser.add_argument('--small_kernel_merged', type=str2bool, default=False, help='small_kernel has already merged or not')
parser.add_argument('--call_structural_reparam', type=bool, default=False, help='structural_reparam after training')
parser.add_argument('--use_multi_scale', type=str2bool, default=True, help='use_multi_scale fusion')


# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

# Formers
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')

parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2,
                    help='number of training runs; each run ii uses seed (random_seed + ii). '
                         'e.g. --random_seed 2021 --itr 5 runs 5 seeds 2021..2025')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--aggregate_logsum', '--aggregate_mean', action='store_true',
                    dest='aggregate_logsum', default=False,
                    help='when pred_len>1, predict the single aggregated target ln(sum_{k=1..h} RV_{t+k}) instead of each step individually (single-value output). --aggregate_mean is kept as a deprecated alias for this flag.')

# news events
parser.add_argument('--use_events', action='store_true', default=False,
                    help='condition on the daily macro news-event calendar: past events are '
                         'embedded and injected at the stem; the KNOWN future event schedule '
                         '(release calendar over the pred_len horizon) FiLM-conditions the head')
parser.add_argument('--event_data_path', type=str, default='events.csv',
                    help='event calendar csv inside root_path (a date column + numeric event columns)')
parser.add_argument('--event_dim', type=int, default=8,
                    help='dimension of the learned event-type embedding (keep small: 4-16)')
parser.add_argument('--event_past', type=str2bool, default=True,
                    help='ablation switch: inject look-back events into the stem')
parser.add_argument('--event_future', type=str2bool, default=True,
                    help='ablation switch: FiLM-condition the head on the future event schedule')
parser.add_argument('--event_fusion', type=str, default='inject', choices=['inject', 'channel'],
                    help="how PAST events enter the backbone: 'inject' (default) adds the shared "
                         "event embedding onto the stem feature map; 'channel' makes past events a "
                         "separate variable that flows through the whole backbone, so ModernTCN's "
                         "cross-variable ConvFFN mixes value<->events at every stage (events keep "
                         "their own patch stem and stay outside RevIN). Future events use FiLM in "
                         "both modes.")

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args = parser.parse_args()



# random seed (base). In training, each --itr run re-seeds with random_seed + ii
# (see the loop below); this initial seeding also covers the is_training=0 path.
fix_seed = args.random_seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_events:
    if args.data == 'custom':
        args.data = 'custom_events'
    import pandas as pd
    ev_header = pd.read_csv(os.path.join(args.root_path, args.event_data_path), nrows=0)
    args.event_in = len([c for c in ev_header.columns if c != 'date'])
    print('news events: {} feature columns from {}'.format(args.event_in, args.event_data_path))

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)
if __name__ == '__main__':

    Exp = Exp_Main

    if args.is_training:
        run_metrics = []  # (seed, {mse,mae,rse,qlike}) per --itr run, for the summary
        for ii in range(args.itr):
            # each run is a distinct seed = random_seed + ii, so e.g.
            # --random_seed 2021 --itr 5 sweeps seeds 2021, 2022, ..., 2025
            seed = args.random_seed + ii
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            if args.use_gpu:
                torch.cuda.manual_seed_all(seed)
            print('>>>>>>> run {}/{}  seed {} <<<<<<<'.format(ii + 1, args.itr, seed))
            # setting record of experiments
            setting = '{}_{}_{}_ft{}_sl{}_pl{}_dim{}_nb{}_lk{}_sk{}_ffr{}_ps{}_str{}_multi{}_merged{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.pred_len,
                args.dims[0],
                args.num_blocks[0],
                args.large_size[0],
                args.small_size[0],
                args.ffn_ratio,
                args.patch_size,
                args.patch_stride,
                args.use_multi_scale,
                args.small_kernel_merged,
                args.des,
                ii)
            if args.use_events:
                setting += '_ev{}d{}p{:d}f{:d}'.format(args.event_fusion[:3], args.event_dim, args.event_past, args.event_future)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            m = exp.test(setting)
            if isinstance(m, dict):
                run_metrics.append((seed, m))

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()

        # ---- summary across the --itr runs (mean +/- std over seeds) ----
        if len(run_metrics) > 1:
            keys = ['mse', 'mae', 'rse', 'qlike']
            seeds = [s for s, _ in run_metrics]
            header = '{:>8}'.format('seed') + ''.join('{:>14}'.format(k) for k in keys)
            print('\n' + '=' * len(header))
            print('summary over {} runs  (seeds {}..{})'.format(len(run_metrics), min(seeds), max(seeds)))
            print(header)
            print('-' * len(header))
            for s, mm in run_metrics:
                print('{:>8}'.format(s) + ''.join('{:>14.6f}'.format(mm[k]) for k in keys))
            print('-' * len(header))
            arr = {k: np.array([mm[k] for _, mm in run_metrics], dtype=np.float64) for k in keys}
            print('{:>8}'.format('mean') + ''.join('{:>14.6f}'.format(arr[k].mean()) for k in keys))
            print('{:>8}'.format('std') + ''.join('{:>14.6f}'.format(arr[k].std(ddof=1)) for k in keys))
            print('=' * len(header))
            with open('result.txt', 'a') as f:
                f.write('summary over {} runs (seeds {}..{})\n'.format(len(run_metrics), min(seeds), max(seeds)))
                for k in keys:
                    f.write('{}: mean {:.6f} +/- {:.6f} (std)\n'.format(k, arr[k].mean(), arr[k].std(ddof=1)))
                f.write('\n')
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                      args.model,
                                                                                                      args.data,
                                                                                                      args.features,
                                                                                                      args.seq_len,
                                                                                                      args.label_len,
                                                                                                      args.pred_len,
                                                                                                      args.d_model,
                                                                                                      args.n_heads,
                                                                                                      args.e_layers,
                                                                                                      args.d_layers,
                                                                                                      args.d_ff,
                                                                                                      args.factor,
                                                                                                      args.embed,
                                                                                                      args.distil,
                                                                                                      args.des, ii)
        if args.use_events:
            setting += '_ev{}d{}p{:d}f{:d}'.format(args.event_fusion[:3], args.event_dim, args.event_past, args.event_future)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
