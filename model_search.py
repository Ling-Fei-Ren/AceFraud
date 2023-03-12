import numpy as np
from operations import *
from torch.autograd import Variable
from genotypes import NA_PRIMITIVES,  SC_PRIMITIVES, LA_PRIMITIVES, LATION_PRIMITIVES
import pyro
import random

def act_map(act):
    if act == "linear":
        return lambda x: x
    elif act == "elu":
        return torch.nn.functional.elu
    elif act == "sigmoid":
        return torch.sigmoid
    elif act == "tanh":
        return torch.tanh
    elif act == "relu":
        return torch.nn.functional.relu
    elif act == "relu6":
        return torch.nn.functional.relu6
    elif act == "softplus":
        return torch.nn.functional.softplus
    elif act == "leaky_relu":
        return torch.nn.functional.leaky_relu
    else:
        raise Exception("wrong activate function")

class NaMixedOp(nn.Module):

  def __init__(self, in_dim, out_dim, k,with_linear):
    super(NaMixedOp, self).__init__()
    self._ops_0 = nn.ModuleList()
    self._ops_1 = nn.ModuleList()
    self._ops_2 = nn.ModuleList()
    self.with_linear = with_linear
    self.k=k

    for primitive in NA_PRIMITIVES:
      op_0 = NA_OPS[primitive](in_dim, out_dim)
      self._ops_0.append(op_0)

      op_1 = NA_OPS[primitive](in_dim, out_dim)
      self._ops_1.append(op_1)

      op_2 = NA_OPS[primitive](in_dim, out_dim)
      self._ops_2.append(op_2)

      if with_linear:
        self._ops_linear_0 = nn.ModuleList()
        op_linear_0 = torch.nn.Linear(in_dim, out_dim)
        self._ops_linear_0.append(op_linear_0)

        self._ops_linear_1 = nn.ModuleList()
        op_linear_1 = torch.nn.Linear(in_dim, out_dim)
        self._ops_linear_1.append(op_linear_1)

        self._ops_linear_2 = nn.ModuleList()
        op_linear_2 = torch.nn.Linear(in_dim, out_dim)
        self._ops_linear_2.append(op_linear_2)

  def forward(self, x, weights, edge_index, ):
    mixed_res_0 = []
    mixed_res_1 = []
    mixed_res_2 = []
    # p_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=1, probs=weights).rsample()

    if self.with_linear:
      p_sampled = random.choices([0, 1], weights=[1-self.k, self.k], k=weights[0].shape[0])
      for p,w, op, linear in zip(p_sampled,weights[0], self._ops_0, self._ops_linear_0):
        mixed_res_0.append(p*w * F.elu(op(x, edge_index[0].cuda())+linear(x)))

      p_sampled = random.choices([0, 1], weights=[1-self.k, self.k], k=weights[1].shape[0])
      for p,w, op, linear in zip(p_sampled,weights[1], self._ops_1, self._ops_linear_1):
        mixed_res_1.append(p*w * F.elu(op(x, edge_index[1].cuda())+linear(x)))

      p_sampled = random.choices([0, 1], weights=[1-self.k, self.k], k=weights[2].shape[0])
      for p, w, op, linear in zip(p_sampled,weights[2], self._ops_2, self._ops_linear_2):
        mixed_res_2.append(p*w * F.elu(op(x, edge_index[2].cuda())+linear(x)))
    else:
      for w, op in zip(weights[0], self._ops_0):
        mixed_res_0.append(w * F.elu(op(x, edge_index[0].cuda())))

      for w, op in zip(weights[1], self._ops_1):
        mixed_res_1.append(w * F.elu(op(x, edge_index[1].cuda())))

      for w, op in zip(weights[2], self._ops_2):
        mixed_res_2.append(w * F.elu(op(x, edge_index[2].cuda())))

    return sum(mixed_res_0),sum(mixed_res_1),sum(mixed_res_2)

class ScMixedOp(nn.Module):

  def __init__(self,k):
    super(ScMixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in SC_PRIMITIVES:
      op = SC_OPS[primitive]()
      self._ops.append(op)
    self.k=k

  def forward(self, x, weights):
    mixed_res = []
    p_sampled = random.choices([0, 1], weights=[1-self.k, self.k], k=weights.shape[0])
    for p,w, op in zip(p_sampled,weights, self._ops):
      mixed_res.append(w * op(x))
    return sum(mixed_res)

class LaMixedOp(nn.Module):

  def __init__(self, hidden_size, k,num_layers=None):
    super(LaMixedOp, self).__init__()
    self._ops = nn.ModuleList()
    self.k=k
    for primitive in LA_PRIMITIVES:
      op = LA_OPS[primitive](hidden_size, num_layers)
      self._ops.append(op)

  def forward(self, x, weights):
    mixed_res = []
    p_sampled = random.choices([0, 1], weights=[1-self.k, self.k], k=weights.shape[0])
    for p,w, op in zip(p_sampled,weights, self._ops):
      mixed_res.append(w * F.relu(op(x)))
    return sum(mixed_res)

class LationMixedOp(nn.Module):

  def __init__(self, in_dim, out_dim):
    super(LationMixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in LATION_PRIMITIVES:
      op = LATION_OPS[primitive](in_dim, out_dim)
      self._ops.append(op)

  def forward(self, x, weights):
    mixed_res = []
    for w, op in zip(weights, self._ops):
      mixed_res.append(w * F.relu(op(x)))
    return sum(mixed_res)



class Network(nn.Module):
  '''
      implement this for sane.
      Actually, sane can be seen as the combination of three cells, node aggregator, skip connection, and layer aggregator
      for sane, we dont need cell, since the DAG is the whole search space, and what we need to do is implement the DAG.
  '''

  def __init__(self, criterion, in_dim, out_dim, hidden_size, k,num_layers=3, dropout=0.5, epsilon=0.0, with_conv_linear=False, args=None):
    super(Network, self).__init__()
    self.in_dim = in_dim
    self.out_dim = out_dim
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self._criterion = criterion
    self.dropout = dropout
    self.epsilon = epsilon
    self.explore_num = 0
    self.with_linear = with_conv_linear
    self.args = args
    self.k=k

    #node aggregator op
    self.lin1 = nn.Linear(in_dim, hidden_size)
    self.lin2 = nn.Linear(hidden_size * 3, hidden_size)
    self.layers = nn.ModuleList()
    for i in range(self.num_layers):
        self.layers.append(NaMixedOp(hidden_size, hidden_size,self.k,self.with_linear))

    self.relations=nn.ModuleList()
    for i in range(self.num_layers):
        self.relations.append(LationMixedOp(hidden_size, hidden_size))
    #skip op
    self.scops = nn.ModuleList()
    for i in range(self.num_layers-1):
        self.scops.append(ScMixedOp(self.k))
    if not self.args.fix_last:
        self.scops.append(ScMixedOp(self.k))

    self.laop = LaMixedOp(hidden_size,self.k, num_layers)

    # self.classifier = nn.Linear(hidden_size, out_dim)
    self.classifier = nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, out_dim))
    self._initialize_alphas()

  def new(self):
    model_new = Network(self._criterion, self.in_dim, self.out_dim, self.hidden_size).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, data,edge_index, discrete=False):
    x, edge_index = data.x, edge_index

    
    self.na_weights = F.softmax(self.na_alphas, dim=-1)
    self.sc_weights = F.softmax(self.sc_alphas, dim=-1)
    self.la_weights = F.softmax(self.la_alphas, dim=-1)


    #generate weights by softmax
    x = self.lin1(x)
    jk = []
    for i in range(self.num_layers):
        x0, x1, x2 = self.layers[i](x, self.na_weights[i], edge_index)
        x0 = F.dropout(x0, p=self.dropout, training=self.training)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x = torch.cat([x0, x1, x2], 1)
        x=self.lin2(x)
        x=torch.relu(x)
        if self.args.fix_last and i == self.num_layers-1:
            jk += [x]
        else:
            jk += [self.scops[i](x, self.sc_weights[i])]

    merge_feature = self.laop(jk, self.la_weights[0])
    merge_feature = F.dropout(merge_feature, p=self.dropout, training=self.training)
    merge_feature=torch.relu(merge_feature)
    logits = self.classifier(merge_feature)
    return logits

  def _loss(self, data,valid_label_priors, edge_index,is_valid=True):
      logits = self(data,edge_index)
      if is_valid:
          input = logits[data.val_mask]
          target = data.y[data.val_mask]
      else:
          input = logits[data.train_mask]
          target = data.y[data.train_mask]
      return self._criterion(input+valid_label_priors, target)

  def _loss_ppi(self, data, is_valid=True):
      input = self(data)
      target = data.y
      return self._criterion(input, target)

  def _initialize_alphas(self):
    #k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_na_ops = len(NA_PRIMITIVES)
    num_sc_ops = len(SC_PRIMITIVES)
    num_la_ops = len(LA_PRIMITIVES)

    #self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).cuda(), requires_grad=True)
    self.na_alphas = Variable(1e-3*torch.randn(self.num_layers, 3,num_na_ops), requires_grad=True)
    if self.args.fix_last:
        self.sc_alphas = Variable(1e-3*torch.randn(self.num_layers-1, num_sc_ops), requires_grad=True)
    else:
        self.sc_alphas = Variable(1e-3*torch.randn(self.num_layers, num_sc_ops), requires_grad=True)

    self.la_alphas = Variable(1e-3*torch.randn(1, num_la_ops), requires_grad=True)
    self._arch_parameters = [
      self.na_alphas,
      self.sc_alphas,
      self.la_alphas,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):

    def _parse(na_weights, sc_weights, la_weights):
      gene = []
      na_indices = torch.argmax(na_weights, dim=-1)
      for k in na_indices:
          for s in k:
            gene.append(NA_PRIMITIVES[s])
      #sc_indices = sc_weights.argmax(dim=-1)
      sc_indices = torch.argmax(sc_weights, dim=-1)
      for k in sc_indices:
          gene.append(SC_PRIMITIVES[k])
      #la_indices = la_weights.argmax(dim=-1)
      la_indices = torch.argmax(la_weights, dim=-1)
      for k in la_indices:
          gene.append(LA_PRIMITIVES[k])

      return '||'.join(gene)

    gene = _parse(F.softmax(self.na_alphas, dim=-1).data.cpu(), F.softmax(self.sc_alphas, dim=-1).data.cpu(), F.softmax(self.la_alphas, dim=-1).data.cpu())


    return gene

  def sample_arch(self):

    num_na_ops = len(NA_PRIMITIVES)
    num_sc_ops = len(SC_PRIMITIVES)
    num_la_ops = len(LA_PRIMITIVES)

    gene = []
    for i in range(3):
        op = np.random.choice(NA_PRIMITIVES, 1)[0]
        gene.append(op)
    for i in range(2):
        op = np.random.choice(SC_PRIMITIVES, 1)[0]
        gene.append(op)
    op = np.random.choice(LA_PRIMITIVES, 1)[0]
    gene.append(op)
    return '||'.join(gene)

  def get_weights_from_arch(self, arch):
    arch_ops = arch.split('||')
    #print('arch=%s' % arch)
    num_na_ops = len(NA_PRIMITIVES)
    num_sc_ops = len(SC_PRIMITIVES)
    num_la_ops = len(LA_PRIMITIVES)


    na_alphas = Variable(torch.zeros(self.num_layers, num_na_ops).cuda(), requires_grad=True)
    sc_alphas = Variable(torch.zeros(self.num_layers-1, num_sc_ops).cuda(), requires_grad=True)
    la_alphas = Variable(torch.zeros(1, num_la_ops).cuda(), requires_grad=True)

    for i in range(self.num_layers):
        ind = NA_PRIMITIVES.index(arch_ops[i])
        na_alphas[i][ind] = 1

    for i in range(self.num_layers, self.num_layers * 2 - 1):
        ind = SC_PRIMITIVES.index(arch_ops[i])
        sc_alphas[i-3][ind] = 1

    ind = LA_PRIMITIVES.index(arch_ops[self.num_layers * 2 - 1])
    la_alphas[0][ind] = 1

    arch_parameters = [na_alphas, sc_alphas, la_alphas]
    return arch_parameters

  def set_model_weights(self, weights):
    self.na_weights = weights[0]
    self.sc_weights = weights[1]
    self.la_weights = weights[2]
    #self._arch_parameters = [self.na_alphas, self.sc_alphas, self.la_alphas]



