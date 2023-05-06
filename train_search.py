import os
import os.path as osp
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
from utils import FraudYelpDataset, FraudAmazonDataset,normalize,FraudTeleminiDataset,FraudTelemaxDataset
from utils import index_to_mask,compute_priors
from torch.autograd import Variable
from model_search import Network
from architect import Architect
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from logging_util import init_logger

parser = argparse.ArgumentParser("sane-train-search")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--k', type=int, default=1, help='sample probility')
parser.add_argument('--record_time', action='store_true', default=False, help='used for run_with_record_time func')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.005, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=10, help='num of training epochs')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--epsilon', type=float, default=0.0, help='the explore rate in the gradient descent process')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-3, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--transductive', action='store_true', help='use transductive settings in train_search.')
parser.add_argument('--with_conv_linear', type=bool, default=True, help=' in NAMixOp with linear op')
parser.add_argument('--fix_last', type=bool, default=True, help='fix last layer in design architectures.')
parser.add_argument('--num_layers', type=int, default=2, help='num of aggregation layers')

args = parser.parse_args()

def main():
    global device
    device = torch.device('cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu')
    args.save = 'logs/search-{}'.format(args.save)
    if not os.path.exists(args.save):
        utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    log_filename = os.path.join(args.save, 'log.txt')
    init_logger('', log_filename, logging.INFO, False)
    print('*************log_filename=%s************' % log_filename)

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args.__dict__)

    if args.data == 'yelp':
        edge_index,labels,feat_data = FraudYelpDataset()
    elif args.data == 'amazon':
        edge_index,labels,feat_data = FraudAmazonDataset()

    elif args.data == 'tele_mini':
        edge_index,labels,feat_data = FraudTeleminiDataset()

    elif args.data == 'tele_max':
        edge_index,labels,feat_data = FraudTelemaxDataset()

    data = Data()
    if args.data == 'yelp' or args.data == 'tele_mini' or args.data == 'tele_max':
        index = list(range(len(labels)))
        idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels, stratify=labels,
                                                                    train_size=0.4,
                                                                    random_state=2, shuffle=True)
        idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                                    test_size=0.67,
                                                                    random_state=2, shuffle=True)
    elif args.data == 'amazon':
        index = list(range(3305, len(labels)))
        idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[3305:], stratify=labels[3305:],
                                                                    train_size=0.4,
                                                                    random_state=2, shuffle=True)
        idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                                    test_size=0.67,
                                                                    random_state=2, shuffle=True)
    data.train_mask = index_to_mask(torch.tensor(idx_train), len(labels))
    data.val_mask = index_to_mask(torch.tensor(idx_valid), len(labels))
    data.test_mask = index_to_mask(torch.tensor(idx_test), len(labels))
    data.y = torch.tensor(labels, dtype=int)

    # data.x = torch.tensor(normalize(feat_data), dtype=torch.float)
    data.x = torch.tensor(feat_data, dtype=torch.float)
    edge_index = edge_index

    train_num_y_0 = data.y[data.train_mask].tolist().count(0)
    train_num_y_1 = data.y[data.train_mask].tolist().count(1)
    train_label_priors = compute_priors(train_num_y_0, train_num_y_1, device)


    valid_num_y_0 = data.y[data.val_mask].tolist().count(0)
    valid_num_y_1 = data.y[data.val_mask].tolist().count(1)
    valid_label_priors = compute_priors(valid_num_y_0, valid_num_y_1, device)

    hidden_size = 64
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(criterion, feat_data.shape[1], int(max(labels)) + 1, hidden_size, k=args.k, num_layers=args.num_layers,epsilon=args.epsilon, with_conv_linear=args.with_conv_linear, args=args)

    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)
    architect = Architect(model, args)# send model to compute validation loss
    search_cost = 0
    max_valid_auc = 0
    for epoch in range(args.epochs):
        t1 = time.time()
        lr = scheduler.get_lr()[0]
        train_auc, train_obj = train(data, edge_index,train_label_priors, valid_label_priors,model, architect, criterion, optimizer, lr)
        scheduler.step()
        t2 = time.time()
        search_cost += (t2 - t1)
        valid_auc, valid_obj = infer(data,edge_index, model, criterion)
        test_auc,  test_obj = infer(data,edge_index, model, criterion, test=True)

        if epoch % 1 == 0:
            if valid_auc> max_valid_auc:
                logging.info('epoch %d lr %e', epoch, lr)
                genotype = model.genotype()
                max_valid_auc=valid_auc
                logging.info('genotype = %s', genotype)
            logging.info('epoch=%s, train_auc=%f, valid_auc=%f, test_auc=%f, explore_num=%s', epoch, train_auc, valid_auc,test_auc, model.explore_num)
            print('epoch={}, train_auc={:.04f}, valid_auc={:.04f}, test_auc={:.04f},explore_num={}'.format(epoch, train_auc, valid_auc, test_auc, model.explore_num))
        utils.save(model, os.path.join(args.save, 'weights.pt'))
    logging.info('The search process costs %.2fs', search_cost)
    return genotype

def train(data, edge_index, train_label_priors, valid_label_priors,model, architect, criterion, optimizer, lr):
    return train_trans(data, edge_index,train_label_priors,valid_label_priors,model, architect, criterion, optimizer, lr)

def infer(data,edge_index, model, criterion, test=False):
    return infer_trans(data, edge_index, model, criterion, test=test)

def train_trans(data, edge_index,train_label_priors,valid_label_priors,model, architect, criterion, optimizer, lr):

    model.train()
    mask = data.train_mask
    target = Variable(data.y[mask], requires_grad=False).to(device)

    #architecture send input or send logits, which are important for computation in architecture
    architect.step(data.to(device), edge_index,valid_label_priors,lr, optimizer, unrolled=args.unrolled)
    #train loss
    logits = model(data.to(device),edge_index)
    input = logits[mask].to(device)

    optimizer.zero_grad()
    loss = criterion(input+ train_label_priors, target)
    loss.backward()
    # nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    print('loss = %s', loss.item())
    logits = torch.sigmoid(logits)
    auc=roc_auc_score(np.array(data.y[mask].data.cpu()), np.array(logits[mask].data.cpu().numpy()[:, 1]))
    return auc, loss/mask.sum().item()


def infer_trans(data, edge_index, model, criterion, test=False):
    model.eval()
    with torch.no_grad():
        logits = model(data.to(device),edge_index)
    if test:
        mask = data.test_mask
    else:
        mask = data.val_mask
    input = logits[mask].to(device)
    target = data.y[mask].to(device)
    loss = criterion(input, target)
    input = torch.sigmoid(input)
    auc = roc_auc_score(np.array(target.data.cpu()), np.array(input.data.cpu().numpy()[:, 1]))
    return auc, loss/mask.sum().item()

def run_by_seed():
    res = []
    for i in range(5):
        print('searched {}-th for {}...'.format(i+1, args.data))
        args.save = '{}-{}'.format(args.data, time.strftime("%Y%m%d-%H%M%S"))
        seed = np.random.randint(0, 10000)
        args.seed = seed
        genotype = main()
        res.append('seed={},genotype={},saved_dir={}'.format(seed, genotype, args.save))
    filename = 'exp_res/%s-searched_res-%s-eps%s-reg%s.txt' % (args.data, time.strftime('%Y%m%d-%H%M%S'), args.epsilon, args.weight_decay)
    fw = open(filename, 'w+')
    fw.write('\n'.join(res))
    fw.close()
    print('searched res for {} saved in {}'.format(args.data, filename))


if __name__ == '__main__':
    run_by_seed()


