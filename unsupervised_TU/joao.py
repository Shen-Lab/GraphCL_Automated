import os.path as osp
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json

from aug import TUDataset_aug as TUDataset
from torch_geometric.data import DataLoader
import sys
import json
from torch import optim

from gin import Encoder
from evaluate_embedding import evaluate_embedding
from model import *

from arguments import arg_parse
from torch_geometric.transforms import Constant
import pdb


class GcnInfomax(nn.Module):
  def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
    super(GcnInfomax, self).__init__()

    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.prior = args.prior

    self.embedding_dim = mi_units = hidden_dim * num_gc_layers
    self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)

    self.local_d = FF(self.embedding_dim)
    self.global_d = FF(self.embedding_dim)

    if self.prior:
        self.prior_d = PriorDiscriminator(self.embedding_dim)

    self.init_emb()

  def init_emb(self):
    initrange = -1.5 / self.embedding_dim
    for m in self.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


  def forward(self, x, edge_index, batch, num_graphs):

    if x is None:
        x = torch.ones(batch.shape[0]).to(device)

    y, M = self.encoder(x, edge_index, batch)
    
    g_enc = self.global_d(y)
    l_enc = self.local_d(M)

    mode='fd'
    measure='JSD'
    local_global_loss = local_global_loss_(l_enc, g_enc, edge_index, batch, measure)
 
    if self.prior:
        prior = torch.rand_like(y)
        term_a = torch.log(self.prior_d(prior)).mean()
        term_b = torch.log(1.0 - self.prior_d(y)).mean()
        PRIOR = - (term_a + term_b) * self.gamma
    else:
        PRIOR = 0
    
    return local_global_loss + PRIOR


class simclr(nn.Module):
  def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
    super(simclr, self).__init__()

    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.prior = args.prior

    self.embedding_dim = mi_units = hidden_dim * num_gc_layers
    self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)

    self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True), nn.Linear(self.embedding_dim, self.embedding_dim))

    self.init_emb()

  def init_emb(self):
    initrange = -1.5 / self.embedding_dim
    for m in self.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


  def forward(self, x, edge_index, batch, num_graphs):

    if x is None:
        x = torch.ones(batch.shape[0]).to(device)

    y, M = self.encoder(x, edge_index, batch)
    
    y = self.proj_head(y)
    
    return y

  def loss_cal(self, x, x_aug):

    T = 0.2
    batch_size, _ = x.size()
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()

    return loss


import random
def setup_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    
    args = arg_parse()
    setup_seed(args.seed)

    accuracies = {'val':[], 'test':[]}
    epochs = 20
    log_interval = 20
    batch_size = 128
    # batch_size = 512
    lr = args.lr
    DS = args.DS
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS)
    # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

    dataset = TUDataset(path, name=DS, aug=args.aug).shuffle()
    dataset_eval = TUDataset(path, name=DS, aug='none').shuffle()
    print(len(dataset))
    print(dataset.get_num_feature())
    try:
        dataset_num_features = dataset.get_num_feature()
    except:
        dataset_num_features = 1

    dataloader = DataLoader(dataset, batch_size=batch_size)
    dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = simclr(args.hidden_dim, args.num_gc_layers).to(device)
    # print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print('================')
    print('lr: {}'.format(lr))
    print('num_features: {}'.format(dataset_num_features))
    print('hidden_dim: {}'.format(args.hidden_dim))
    print('num_gc_layers: {}'.format(args.num_gc_layers))
    print('================')

    model.eval()
    emb, y = model.encoder.get_embeddings(dataloader_eval)

    aug_P = np.ones(5) / 5
    for epoch in range(1, epochs+1):
        dataloader.dataset.aug_P = aug_P
        loss_all = 0
        model.train()
        for data in dataloader:

            data, data_aug = data
            optimizer.zero_grad()
            
            node_num, _ = data.x.size()
            data = data.to(device)
            x = model(data.x, data.edge_index, data.batch, data.num_graphs)

            if args.aug == 'dnodes' or args.aug == 'subgraph' or args.aug == 'random2' or args.aug == 'random3' or args.aug == 'random4' or args.aug == 'minmax':
                edge_idx = data_aug.edge_index.numpy()
                _, edge_num = edge_idx.shape
                idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]

                node_num_aug = len(idx_not_missing)
                data_aug.x = data_aug.x[idx_not_missing]

                data_aug.batch = data.batch[idx_not_missing]
                idx_dict = {idx_not_missing[n]:n for n in range(node_num_aug)}
                edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if not edge_idx[0, n] == edge_idx[1, n]]
                data_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1)

            data_aug = data_aug.to(device)
            x_aug = model(data_aug.x, data_aug.edge_index, data_aug.batch, data_aug.num_graphs)

            loss = model.loss_cal(x, x_aug)
            print(loss)
            loss_all += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()
        print('Epoch {}, Loss {}'.format(epoch, loss_all / len(dataloader)))

        if epoch % log_interval == 0:
            model.eval()
            emb, y = model.encoder.get_embeddings(dataloader_eval)
            acc_val, acc = evaluate_embedding(emb, y)
            accuracies['val'].append(acc_val)
            accuracies['test'].append(acc)

        # minmax
        loss_aug = np.zeros(5)
        for n in range(5):
            _aug_P = np.zeros(5)
            _aug_P[n] = 1
            dataloader.dataset.aug_P = _aug_P
            count, count_stop = 0, len(dataloader)//5+1
            with torch.no_grad():
                for data in dataloader:

                    data, data_aug = data
                    node_num, _ = data.x.size()
                    data = data.to(device)
                    x = model(data.x, data.edge_index, data.batch, data.num_graphs)

                    if args.aug == 'dnodes' or args.aug == 'subgraph' or args.aug == 'random2' or args.aug == 'random3' or args.aug == 'random4' or args.aug == 'minmax':
                        edge_idx = data_aug.edge_index.numpy()
                        _, edge_num = edge_idx.shape
                        idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]

                        node_num_aug = len(idx_not_missing)
                        data_aug.x = data_aug.x[idx_not_missing]

                        data_aug.batch = data.batch[idx_not_missing]
                        idx_dict = {idx_not_missing[n]:n for n in range(node_num_aug)}
                        edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if not edge_idx[0, n] == edge_idx[1, n]]
                        data_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1)

                    data_aug = data_aug.to(device)
                    x_aug = model(data_aug.x, data_aug.edge_index, data_aug.batch, data_aug.num_graphs)

                    loss = model.loss_cal(x, x_aug)
                    loss_aug[n] += loss.item() * data.num_graphs
                    if args.mode == 'fast':
                        count += 1
                        if count == count_stop:
                            break

            if args.mode == 'fast':
                loss_aug[n] /= (count_stop*batch_size)
            else:
                loss_aug[n] /= len(dataloader.dataset)

        gamma = float(args.gamma)
        beta = 1
        b = aug_P + beta * (loss_aug - gamma * (aug_P - 1/5))

        mu_min, mu_max = b.min()-1/5, b.max()-1/5
        mu = (mu_min + mu_max) / 2
        # bisection method
        while abs(np.maximum(b-mu, 0).sum() - 1) > 1e-2:
            if np.maximum(b-mu, 0).sum() > 1:
                mu_min = mu
            else:
                mu_max = mu
            mu = (mu_min + mu_max) / 2

        aug_P = np.maximum(b-mu, 0)
        aug_P /= aug_P.sum()
        print(loss_aug, aug_P)

    tpe  = ('local' if args.local else '') + ('prior' if args.prior else '')
    with open('results/log_' + args.DS + '_' + args.aug + '_' + args.gamma, 'a+') as f:
        s = json.dumps(accuracies)
        f.write('{},{},{},{},{},{},{}\n'.format(args.DS, tpe, args.num_gc_layers, epochs, log_interval, lr, s))
        f.write('\n')
