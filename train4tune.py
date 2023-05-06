import os
import os.path as osp
import sys
import time
import glob
import pickle
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch import cat
from sklearn.metrics import f1_score, recall_score,roc_auc_score
from utils import FraudYelpDataset, FraudAmazonDataset,normalize,FraudTeleminiDataset,FraudTelemaxDataset
from torch.autograd import Variable
from model import NetworkGNN as Network
from utils import index_to_mask,compute_priors
from utils import gen_uniform_60_20_20_split, save_load_split
# from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
# from sklearn.metrics import roc_auc_score
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, CoraFull, Reddit, PPI
from sklearn.model_selection import StratifiedKFold
from torch_geometric.utils import add_self_loops
from logging_util import init_logger
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T

def main(exp_args):
    global train_args
    train_args = exp_args

    tune_str = time.strftime('%Y%m%d-%H%M%S')
    train_args.save = 'logs/tune-{}-{}'.format(train_args.data, tune_str)
    if not os.path.exists(train_args.save):
        os.mkdir(train_args.save)

    global device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda' if train_args.gpu < 0 else 'cuda:{}'.format(train_args.gpu))

    # if not torch.cuda.is_available():
    #     logging.info('no gpu device available')
    #     sys.exit(1)

    #np.random.seed(train_args.seed)
    # torch.cuda.set_device(train_args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(train_args.seed)
    cudnn.enabled=True
    # torch.cuda.manual_seed(train_args.seed)

    if train_args.data == 'Amazon_Computers':
        dataset = Amazon('../data/Amazon_Computers', 'Computers')
    elif train_args.data == 'Coauthor_Physics':
        dataset = Coauthor('../data/Coauthor_Physics', 'Physics')
    elif train_args.data == 'Coauthor_CS':
        dataset = Coauthor('../data/Coauthor_CS', 'CS')
    elif train_args.data == 'Cora_Full':
        dataset = CoraFull('../data/Cora_Full')
    elif train_args.data == 'PubMed':
        dataset = Planetoid('../data/', 'PubMed')
    elif train_args.data == 'Cora':
        dataset = Planetoid('../data/', 'Cora')
    elif train_args.data == 'yelp':
        edge_index,labels,feat_data = FraudYelpDataset()
    elif train_args.data == 'amazon':
        edge_index,labels,feat_data = FraudAmazonDataset()
    elif train_args.data == 'tele_mini':
        edge_index,labels,feat_data = FraudTeleminiDataset()

    elif train_args.data == 'tele_max':
        edge_index,labels,feat_data = FraudTelemaxDataset()

    elif train_args.data == 'CiteSeer':
        dataset = Planetoid('../data/', 'CiteSeer')
    elif train_args.data == 'PPI':
        train_dataset = PPI('../data/PPI', split='train')
        val_dataset = PPI('../data/PPI', split='val')
        test_dataset = PPI('../data/PPI', split='test')
        ppi_train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        ppi_val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
        ppi_test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
        # print('load PPI done!')
        data = [ppi_train_loader, ppi_val_loader, ppi_test_loader]

    if train_args.data == 'small_Reddit':
        dataset = Reddit('../data/Reddit/')
        with open('../data/small_Reddit/sampled_reddit.obj', 'rb') as f:
            data = pickle.load(f)
            raw_dir = '../data/small_Reddit/raw/'
    genotype = train_args.arch
    hidden_size = train_args.hidden_size

    if train_args.data != 'PPI':
        data = Data()
        if train_args.data == 'yelp' or train_args.data == 'tele_max' or train_args.data == 'tele_mini':
            index = list(range(len(labels)))
            idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels, stratify=labels,
                                                                    train_size=0.4,
                                                                    random_state=2, shuffle=True)
            idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                                    test_size=0.67,
                                                                    random_state=2, shuffle=True)
        elif train_args.data == 'amazon':
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
        data.x = torch.tensor(normalize(feat_data), dtype=torch.float)
        edge_index = edge_index
        train_num_y_0 = data.y[data.train_mask].tolist().count(0)
        train_num_y_1 = data.y[data.train_mask].tolist().count(1)
        train_label_priors = compute_priors(train_num_y_0, train_num_y_1, device)
        valid_num_y_0 = data.y[data.val_mask].tolist().count(0)
        valid_num_y_1 = data.y[data.val_mask].tolist().count(1)
        valid_label_priors = compute_priors(valid_num_y_0, valid_num_y_1, device)

        num_features = data.x.shape[1]
        num_classes = int(max(labels))+1
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
        num_features = train_dataset.num_features
        num_classes = train_dataset.num_classes
    criterion = criterion.to(device)
    model = Network(genotype, criterion, num_features, num_classes, hidden_size,
                    num_layers=train_args.num_layers, in_dropout=train_args.in_dropout,
                    out_dropout=train_args.out_dropout, act=train_args.activation,
                    is_mlp=False, args=train_args)
    model = model.to(device)

    logging.info("genotype=%s, param size = %fMB, args=%s", genotype, utils.count_parameters_in_MB(model), train_args.__dict__)

    if train_args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            train_args.learning_rate,
            weight_decay=train_args.weight_decay
            )
    elif train_args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            train_args.learning_rate,
            momentum=train_args.momentum,
            weight_decay=train_args.weight_decay
            )
    elif train_args.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(
            model.parameters(),
            train_args.learning_rate,
            weight_decay=train_args.weight_decay
            )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(train_args.epochs))

    val_res = 0
    best_val_auc = best_test_auc = 0
    for epoch in range(train_args.epochs):
        train_auc, train_obj = train(train_args.data, data, edge_index,train_label_priors,valid_label_priors,model, criterion, optimizer)
        if train_args.cos_lr:
            scheduler.step()

        valid_auc, valid_obj,valid_f1,valid_recall,embedding = infer(train_args.data, data, edge_index,model, criterion)
        test_auc, test_obj,test_f1,test_recall,embedding = infer(train_args.data, data, edge_index,model, criterion, test=True)

        if valid_auc > best_val_auc:
            best_val_auc = valid_auc
            best_test_auc = test_auc
            best_test_f1 = test_f1
            best_test_recall = test_recall

        if epoch % 10 == 0:
            logging.info('epoch=%s, lr=%s, train_obj=%s, train_auc=%f, valid_auc=%s, test_auc=%s,test_f1=%s,test_recall=%s', epoch, scheduler.get_lr()[0], train_obj, train_auc, best_val_auc, best_test_auc, best_test_f1,best_test_recall)

        utils.save(model, os.path.join(train_args.save, 'weights.pt'))

    model.eval()
    logits,embedding = model(data.to(device), edge_index)

    return best_val_auc, best_test_auc, train_args,best_test_f1,best_test_recall,embedding[data.test_mask],data.y[data.test_mask]

def train(dataset_name, data, edge_index,train_label_priors,valid_label_priors,model, criterion, optimizer):
    if dataset_name == 'PPI':
        return train_ppi(data, edge_index,train_label_priors,valid_label_priors,model,criterion, optimizer)
    else:
        return train_trans(data, edge_index,train_label_priors,valid_label_priors,model,criterion, optimizer)

def infer(dataset_name, data, edge_index,model, criterion, test=False):
    if dataset_name == 'PPI':
        return infer_ppi(data, edge_index,model, criterion, test=test)
    else:
        return infer_trans(data, edge_index,model, criterion, test=test)

def train_trans(data, edge_index,train_label_priors,valid_label_priors,model, criterion, optimizer):

    mask = data.train_mask
    model.train()
    target = data.y[mask].to(device)

    optimizer.zero_grad()
    logits,embedding = model(data.to(device),edge_index)

    input = logits[mask].to(device)

    loss = criterion(input+train_label_priors, target)
    loss.backward()
    # nn.utils.clip_grad_norm_(model.parameters(), train_args.grad_clip)
    optimizer.step()

    logits = torch.sigmoid(logits)
    auc = roc_auc_score(np.array(data.y[mask].data.cpu()), np.array(logits[mask].data.cpu().numpy()[:, 1]))

    # acc = logits[mask].max(1)[1].eq(data.y[mask]).sum().item() / mask.sum().item()
    return auc, loss/mask.sum().item()



def infer_trans(data, edge_index,model, criterion, test=False):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        logits,embedding = model(data.to(device),edge_index)
    if test:
        mask = data.test_mask
    else:
        mask = data.val_mask
    input = logits[mask].to(device)
    target = data.y[mask].to(device)
    loss = criterion(input, target)

    input = torch.sigmoid(input)
    auc = roc_auc_score(np.array(target.data.cpu()), np.array(input.data.cpu().numpy()[:, 1]))
    f1 = f1_score(np.array(data.y[mask].data.cpu()), np.array(logits[mask].data.cpu().numpy().argmax(axis=1)),average="macro")
    recall = recall_score(np.array(data.y[mask].data.cpu()), np.array(logits[mask].data.cpu().numpy().argmax(axis=1)),average="macro")
    # acc = logits[mask].max(1)[1].eq(data.y[mask]).sum().item() / mask.sum().item()
    return auc, loss/mask.sum().item(),f1,recall,embedding



def train_ppi(data, edge_index,model, criterion, optimizer):
    model.train()
    preds, ys = [], []
    total_loss = 0
    # input all data

    for train_data in data[0]:
        train_data = train_data.to(device)
        target = Variable(train_data.y).to(device)

        # train loss
        optimizer.zero_grad()
        input = model(train_data).to(device)
        loss = criterion(input, target)
        total_loss += loss.item()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), train_args.grad_clip)
        optimizer.step()

        preds.append((input > 0).float().cpu())
        ys.append(train_data.y.cpu())
    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    prec1 = f1_score(y, pred, average='micro')
    # print('train_loss:', total_loss / len(data[0].dataset))
    return prec1, total_loss / len(data[0].dataset)

def infer_ppi(data, edge_index,model, criterion, test=False):
    model.eval()
    total_loss = 0
    preds, ys = [], []
    if test:
        infer_data = data[2]
    else:
        infer_data = data[1]

    for val_data in infer_data:
        val_data = val_data.to(device)
        with torch.no_grad():
            logits,embedding = model(val_data).to(device)

        loss = criterion(logits, val_data.y.to(device))
        total_loss += loss.item()

        preds.append((logits > 0).float().cpu())
        ys.append(val_data.y.cpu())
    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    prec1 = f1_score(y, pred, average='micro')
    return prec1, total_loss / len(infer_data.dataset),embedding

if __name__ == '__main__':
  main()



