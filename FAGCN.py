import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
import numpy as np


class FALayer(nn.Module):
    def __init__(self, in_dim, dropout):
        super(FALayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(2 * in_dim, 1)
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)

    def edge_applying(self, edges):
        h2 = torch.cat([edges.dst['h'], edges.src['h']], dim=1)
        g = torch.tanh(self.gate(h2)).squeeze()
        e = g * edges.dst['d'] * edges.src['d']
        e = self.dropout(e)
        return {'e': e, 'm': g}

    def forward(self, h,g):
        self.g=g
        self.g.ndata['h'] = h
        self.g.apply_edges(self.edge_applying)
        self.g.update_all(fn.u_mul_e('h', 'e', '_'), fn.sum('_', 'z'))

        return self.g.ndata['z']


class FAGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.5, eps=0.3, layer_num=1):
        super(FAGCN, self).__init__()
        self.eps = eps
        self.layer_num = layer_num
        self.dropout = dropout

        self.layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.layers.append(FALayer(hidden_dim, dropout))

        self.t1 = nn.Linear(in_dim, hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)

    def forward(self, x,edge_index):
        g=dgl.graph((edge_index[0].cpu(),edge_index[1].cpu()))
        g = dgl.to_simple(g)
        g = dgl.remove_self_loop(g)
        g=g.to('cuda')
        deg = g.in_degrees().cuda().float().clamp(min=1)
        norm = torch.pow(deg, -0.5)
        g.ndata['d'] = norm
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = torch.relu(self.t1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        raw = h
        for i in range(self.layer_num):
            h = self.layers[i](h,g)
            h = self.eps * raw + h

        return h

