import os
from datetime import datetime
import time
import argparse
import json
import pickle
import logging
import numpy as np

import hyperopt
from hyperopt import fmin, tpe, hp, Trials, partial, STATUS_OK
from logging_util import init_logger
from train4tune import main

sane_space ={'model': 'SANE',
         'hidden_size': 64,
         'learning_rate': 0.0022353347994672973,
         'weight_decay': 1.5262562842068167e-05,
         'optimizer': 'adam',
         'in_dropout': 0.4,
         'out_dropout': 0.2,
         'activation': 'relu'
         }

def get_args():
    parser = argparse.ArgumentParser("sane")
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
    parser.add_argument('--arch_filename', type=str, default='', help='given the location of searched res')
    parser.add_argument('--arch', type=str, default='', help='given the specific of searched res')
    parser.add_argument('--num_layers', type=int, default=2, help='num of GNN layers in SANE')
    parser.add_argument('--tune_topK', action='store_true', default=False, help='whether to tune topK archs')
    parser.add_argument('--record_time', action='store_true', default=False, help='whether to tune topK archs')
    parser.add_argument('--transductive', action='store_true', help='use transductive settings in train_search.')
    parser.add_argument('--with_linear', action='store_true', default=False, help='whether to use linear in NaOp')
    parser.add_argument('--with_layernorm', action='store_true', default=False, help='whether to use layer norm')
    parser.add_argument('--hyper_epoch', type=int, default=50, help='epoch in hyperopt.')
    parser.add_argument('--epochs', type=int, default=400, help='epoch in train GNNs.')
    parser.add_argument('--cos_lr', action='store_true', default=False, help='using lr decay in training GNNs.')
    parser.add_argument('--fix_last', type=bool, default=True, help='fix last layer in design architectures.')
    parser.add_argument('--model', type=str, default='SANE', help='gpu device id')
    parser.add_argument('--hidden_size', type=int, default=64, help='gpu device id')
    parser.add_argument('--learning_rate', type=int, default=0.0022353347994672973, help='gpu device id')
    parser.add_argument('--weight_decay', type=int, default=1.5262562842068167e-05, help='gpu device id')
    parser.add_argument('--optimizer', type=str, default='adam', help='gpu device id')
    parser.add_argument('--in_dropout', type=int, default=0.4, help='gpu device id')
    parser.add_argument('--out_dropout', type=int, default=0.2, help='gpu device id')
    parser.add_argument('--activation', type=str, default='relu', help='gpu device id')
    parser.add_argument('--seed', type=int, default=2, help='gpu device id')
    parser.add_argument('--grad_clip', type=int, default=5, help='gpu device id')
    parser.add_argument('--momentum', type=int, default=0.9, help='gpu device id')

    global args1
    args1 = parser.parse_args()

class ARGS(object):

    def __init__(self):
        super(ARGS, self).__init__()

def generate_args(arg_map):
    args = ARGS()
    for k, v in arg_map.items():
        setattr(args, k, v)
    for k, v in args1.__dict__.items():
        setattr(args, k, v)
    setattr(args, 'rnd_num', 1)

    args.learning_rate = 10**args.learning_rate
    args.weight_decay = 10**args.weight_decay
    args.in_dropout = args.in_dropout / 10.0
    args.out_dropout = args.out_dropout / 10.0
    args.save = '{}_{}'.format(args.data, datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S'))
    args1.save = '../logs/tune-{}'.format(args.save)
    args.seed = 2
    args.grad_clip = 5
    args.momentum = 0.9
    return args

def generate_args_1():
    args = ARGS()
    args.learning_rate = 0.0022353347994672973
    args.weight_decay = 1.5262562842068167e-05
    args.in_dropout = 0.6
    args.out_dropout = 0.5
    args.save = '{}_{}'.format(args.data, datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S'))
    args1.save = '../logs/tune-{}'.format(args.save)
    args.seed = 2
    args.grad_clip = 5
    args.momentum = 0.9
    return args


def objective(args):
    args = generate_args(args)
    vali_auc, test_auc, args,test_f1,test_recall = main(args)
    return {
        'loss': -vali_auc,
        'test_auc': test_auc,
        'test_f1': test_f1,
        'test_recall': test_recall,

        'status': STATUS_OK,
        'eval_time': round(time.time(), 2),
        }

def run_fine_tune():

    tune_str = time.strftime('%Y%m%d-%H%M%S')
    path = 'logs/tune-%s_%s' % (args1.data, tune_str)
    if not os.path.exists(path):
      os.mkdir(path)
    log_filename = os.path.join(path, 'log.txt')
    init_logger('fine-tune', log_filename, logging.INFO, False)

    lines = open(args1.arch_filename, 'r').readlines()

    suffix = args1.arch_filename.split('_')[-1][:-4] # need to re-write the suffix?

    test_res = []
    arch_set = set()
    if args1.data in ['small_Reddit', 'PubMed']:
        sane_space['hidden_size'] = hp.choice('hidden_size', [16, 32, 64])
    if args1.data == 'PPI':
        sane_space['learning_rate'] = hp.uniform("lr", -3, -1.6)
        sane_space['in_dropout'] = hp.choice('in_dropout', [0, 1])
        sane_space['out_dropout'] = hp.choice('out_dropout', [0, 1])
        sane_space['hidden_size'] = hp.choice('hidden_size', [64, 128, 256, 512, 1024])
    if args1.data == 'CiteSeer':
        sane_space['learning_rate'] = hp.uniform("lr", -2.5, -1.6)
        sane_space['weight_decay'] = hp.choice('wr', [-8])
        sane_space['in_dropout'] = hp.choice('in_dropout', [5])
        sane_space['out_dropout'] = hp.choice('out_dropout', [0])

    for ind, l in enumerate(lines):
        try:
            print('**********process {}-th/{}, logfilename={}**************'.format(ind+1, len(lines), log_filename))
            logging.info('**********process {}-th/{}**************'.format(ind+1, len(lines)))
            res = {}
            #iterate each searched architecture
            parts = l.strip().split(',')
            arch = parts[1].split('=')[1]
            args1.arch = arch
            if arch in arch_set:
                logging.info('the %s-th arch %s already searched....info=%s', ind+1, arch, l.strip())
                continue
            else:
                arch_set.add(arch)
            res['searched_info'] = l.strip()

            start = time.time()
            trials = Trials()
            record_time_res = []
            c_vali_auc, c_test_auc , c_test_f1, c_test_recall= 0, 0,0,0
            for d in trials.results:
                if -d['loss'] > c_vali_auc:
                    c_vali_auc = -d['loss']
                    c_test_auc = d['test_auc']
                    c_test_f1=d['test_f1']
                    c_test_recall=d['test_recall']
                    record_time_res.append('%s,%s,%s' % (d['eval_time'] - start, c_vali_auc, c_test_auc))
            res['test_auc'] = c_test_auc
            res['test_f1'] = c_test_f1
            res['test_recall'] = c_test_recall
            print('test_auc={}'.format(c_test_auc))
            print('test_res=', res)

            test_aucs=[]
            test_f1s=[]
            test_recalls=[]
            for i in range(5):
                vali_auc, t_auc, test_args, test_f1,test_recall= main(args1)
                print('cal std: times:{}, valid_Auc:{}, test_auc:{}'.format(i,vali_auc,t_auc))
                test_aucs.append(t_auc)
                test_f1s.append(test_f1)
                test_recalls.append(test_recall)
            test_aucs = np.array(test_aucs)
            test_f1s = np.array(test_f1s)
            test_recalls = np.array(test_recalls)
            print('test_aucs_5_times:{:.04f}+-{:.04f},test_f1s_5_times:{:.04f}+-{:.04f},test_recalls_5_times:{:.04f}+-{:.04f}'.format(np.mean(test_aucs), np.std(test_aucs),np.mean(test_f1s), np.std(test_f1s),np.mean(test_recalls), np.std(test_recalls)))
            test_res.append(res)

            test_res.append(res)
            with open('tuned_res/%s_res_%s_%s.pkl' % (args1.data, tune_str, suffix), 'wb+') as fw:
                pickle.dump(test_res, fw)
            logging.info('**********finish {}-th/{}**************8'.format(ind+1, len(lines)))
        except Exception as e:
            logging.info('errror occured for %s-th, arch_info=%s, error=%s', ind+1, l.strip(), e)
            import traceback
            traceback.print_exc()
    print('finsh tunining {} archs, saved in {}'.format(len(arch_set), 'tuned_res/%s_res_%s_%s.pkl' % (args1.data, tune_str, suffix)))


if __name__ == '__main__':
    get_args()
    if args1.arch_filename:
        run_fine_tune()


