import argparse

from loader import BioDataset_graphcl, DataLoader
# from torch_geometric.data import DataLoader
from torch_geometric.nn.inits import uniform
from torch_geometric.nn import global_mean_pool, global_add_pool

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np
from model import GNN
from sklearn.metrics import roc_auc_score
import pandas as pd
from copy import deepcopy


class graphcl(nn.Module):
    def __init__(self, gnn, emb_dim):
        super(graphcl, self).__init__()
        self.gnn = gnn
        self.pool = global_mean_pool
        self.projection_head = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU(inplace=True), nn.Linear(emb_dim, emb_dim))

    def forward_cl(self, x, edge_index, edge_attr, batch):
        x = self.gnn(x, edge_index, edge_attr)
        x = self.pool(x, batch)
        x = self.projection_head(x)
        return x

    def loss_cl(self, x1, x2, mean=True):
        T = 0.1
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[np.arange(batch_size), np.arange(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss)
        if mean:
            loss = loss.mean()
        return loss


#reference: https://github.com/tkipf/gae; https://github.com/DaehanKim/vgae_pytorch
class vgae(nn.Module):
    def __init__(self, gnn, emb_dim):
        super(vgae, self).__init__()
        self.encoder = gnn
        self.encoder_mean = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU(inplace=True), nn.Linear(emb_dim, emb_dim))
        # make sure std is positive
        self.encoder_std = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU(inplace=True), nn.Linear(emb_dim, emb_dim), nn.Softplus())
        # only reconstruct first 7-dim, please refer to https://github.com/snap-stanford/pretrain-gnns/issues/30
        self.decoder = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(emb_dim, emb_dim), nn.ReLU(inplace=True), nn.Linear(emb_dim, 7), nn.Sigmoid())
        self.decoder_edge = nn.Sequential(nn.ReLU(), nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, 1))

        self.bceloss = nn.BCELoss(reduction='none')
        self.pool = global_mean_pool
        self.add_pool = global_add_pool
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

    def forward_encoder(self, x, edge_index, edge_attr):
        x = self.encoder(x, edge_index, edge_attr)
        x_mean = self.encoder_mean(x)
        x_std = self.encoder_std(x)
        gaussian_noise = torch.randn(x_mean.shape).to(x.device)
        x = gaussian_noise * x_std + x_mean
        return x, x_mean, x_std

    def forward_decoder(self, x, edge_index, edge_index_neg):
        eleWise_mul = x[edge_index[0]] * x[edge_index[1]]
        edge_attr_pred = self.decoder(eleWise_mul)

        edge_pos = self.sigmoid( self.decoder_edge(eleWise_mul) ).squeeze()
        edge_neg = self.sigmoid( self.decoder_edge(x[edge_index_neg[0]] * x[edge_index_neg[1]]) ).squeeze()
        return edge_attr_pred, edge_pos, edge_neg

    def loss_vgae(self, edge_attr_pred, edge_attr, edge_pos_pred, edge_neg_pred, edge_index_batch, edge_index_neg_batch, x_mean, x_std, batch, reward=None):
        # evaluate p(A|Z)
        num_edge, _ = edge_attr_pred.shape
        loss_rec = self.bceloss(edge_attr_pred.reshape(-1), edge_attr[:, :7].reshape(-1))
        loss_rec = loss_rec.reshape((num_edge, -1)).sum(dim=1)

        loss_edge_pos = self.bceloss(edge_pos_pred, torch.ones(edge_pos_pred.shape).to(edge_pos_pred.device))
        loss_edge_neg = self.bceloss(edge_neg_pred, torch.zeros(edge_neg_pred.shape).to(edge_neg_pred.device))
        loss_pos = loss_rec + loss_edge_pos
        loss_pos = self.pool(loss_pos, edge_index_batch)
        loss_neg = self.pool(loss_edge_neg, edge_index_neg_batch)
        loss_rec = loss_pos + loss_neg
        if not reward is None:
            loss_rec = loss_rec * reward

        # evaluate p(Z|X,A)
        kl_divergence = - 0.5 * (1 + 2 * torch.log(x_std) - x_mean**2 - x_std**2).sum(dim=1)
        kl_ones = torch.ones(kl_divergence.shape).to(kl_divergence.device)
        kl_divergence = self.pool(kl_divergence, batch)
        kl_double_norm = 1 / self.add_pool(kl_ones, batch)
        kl_divergence = kl_divergence * kl_double_norm

        loss = (loss_rec + kl_divergence).mean()

        '''
        # link prediction for sanity check
        from sklearn.metrics import roc_auc_score
        from sklearn.metrics import average_precision_score
        print(roc_auc_score(edge_attr.cpu().numpy(), edge_attr_pred.detach().cpu().numpy()), average_precision_score(edge_attr.cpu().numpy(), edge_attr_pred.detach().cpu().numpy()))
        '''
        return loss, (loss_edge_pos.mean()+loss_edge_neg.mean()).item()/2

    def generate(self, data):
        x, _, _ = self.forward_encoder(data.x, data.edge_index, data.edge_attr)
        eleWise_mul = torch.einsum('nd,md->nmd', x, x)

        # calculate softmax probability
        prob = self.decoder_edge(eleWise_mul).squeeze()
        prob = torch.exp(prob)
        prob[torch.isinf(prob)] = 1e10
        prob[list(range(x.shape[0])), list(range(x.shape[0]))] = 0
        prob = torch.einsum('nm,n->nm', prob, 1/prob.sum(dim=1))

        # sparsify
        prob[prob<1e-1] = 0
        prob[prob.sum(dim=1)==0] = 1
        prob[list(range(x.shape[0])), list(range(x.shape[0]))] = 0
        prob = torch.einsum('nm,n->nm', prob, 1/prob.sum(dim=1))

        edge_attr_prob = self.decoder(eleWise_mul)
        edge_attr_rand = torch.rand(edge_attr_prob.shape)
        edge_attr_pred = torch.zeros(edge_attr_prob.shape)

        edge_attr_pred[edge_attr_rand<edge_attr_prob] = 1
        edge_attr_pred = torch.cat((edge_attr_pred, torch.zeros((edge_attr_pred.shape[0], edge_attr_pred.shape[1], 2))), dim=2)
        return prob, edge_attr_pred


def train(args, loader, model_cl, optimizer_cl, model_1, optimizer_1, model_2, optimizer_2, device):
    pretrain_loss, generative_loss = 0, 0
    link_loss = 0
    # for step, batch in enumerate(tqdm(loader, desc="Iteration")):
    for step, batch in enumerate(loader):
        batch, batch1, batch2 = batch
        batch, batch1, batch2 = batch.to(device), batch1.to(device), batch2.to(device)

        # 1. graphcl
        optimizer_cl.zero_grad()

        x1 = model_cl.forward_cl(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch)
        x2 = model_cl.forward_cl(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch)
        loss_cl = model_cl.loss_cl(x1, x2, mean=False)

        loss = loss_cl.mean()

        loss.backward()
        optimizer_cl.step()
        pretrain_loss += float(loss.item())

        # reward for joao
        loss_cl = loss_cl.detach()
        loss_cl = loss_cl - loss_cl.mean()
        loss_cl[loss_cl>0] = 1
        loss_cl[loss_cl<=0] = 0.01 # weaken the reward for low cl loss

        # 2. joao
        optimizer_1.zero_grad()
        optimizer_2.zero_grad()

        x, x_mean, x_std = model_1.forward_encoder(batch.x, batch.edge_index, batch.edge_attr)
        edge_attr_pred, edge_pos_pred, edge_neg_pred = model_1.forward_decoder(x, batch.edge_index, batch.edge_index_neg)
        loss_1, link_loss_1 = model_1.loss_vgae(edge_attr_pred, batch.edge_attr, edge_pos_pred, edge_neg_pred, batch.edge_index_batch, batch.edge_index_neg_batch, x_mean, x_std, batch.batch, reward=loss_cl)

        x, x_mean, x_std = model_2.forward_encoder(batch.x, batch.edge_index, batch.edge_attr)
        edge_attr_pred, edge_pos_pred, edge_neg_pred = model_2.forward_decoder(x, batch.edge_index, batch.edge_index_neg)
        loss_2, link_loss_2 = model_2.loss_vgae(edge_attr_pred, batch.edge_attr, edge_pos_pred, edge_neg_pred, batch.edge_index_batch, batch.edge_index_neg_batch, x_mean, x_std, batch.batch, reward=loss_cl)

        loss = loss_1 + loss_2

        loss.backward()
        optimizer_1.step()
        optimizer_2.step()
        generative_loss += float(loss.item())

        link_loss += (link_loss_1+link_loss_2)/2

    return pretrain_loss/(step+1), generative_loss/(step+1), link_loss/(step+1)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--num_workers', type=int, default = 16, help='number of workers for dataset loading')
    parser.add_argument('--aug_mode', type=str, default = 'generative') 
    parser.add_argument('--aug_strength', type=float, default = 0.2)
    parser.add_argument('--resume', type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    #set up dataset
    root_unsupervised = 'dataset/unsupervised'
    dataset = BioDataset_graphcl(root_unsupervised, data_type='unsupervised')
    dataset.set_augMode(args.aug_mode)
    dataset.set_augStrength(args.aug_strength)
    print(dataset)

    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    # set up graphcl model
    gnn = GNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type)
    model = graphcl(gnn, args.emb_dim)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    # set up vgae model 1
    gnn_generative_1 = GNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type)
    model_generative_1 = vgae(gnn_generative_1, args.emb_dim)
    model_generative_1.to(device)
    optimizer_generative_1 = optim.Adam(model_generative_1.parameters(), lr=args.lr, weight_decay=args.decay)

    # set up vgae model 2
    gnn_generative_2 = GNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type)
    model_generative_2 = vgae(gnn_generative_2, args.emb_dim)
    model_generative_2.to(device)
    optimizer_generative_2 = optim.Adam(model_generative_2.parameters(), lr=args.lr, weight_decay=args.decay)

    # start training
    model.train(), model_generative_1.train(), model_generative_2.train()
    if args.resume == 0:
        torch.save({'graphcl':model.state_dict(), 'graphcl_opt': optimizer.state_dict(), 'generator_1':model_generative_1.state_dict(), 'generator_1_opt':optimizer_generative_1.state_dict(), 'generator_2':model_generative_2.state_dict(), 'generator_2_opt':optimizer_generative_2.state_dict()}, './weights_generative/checkpoint_joao_0.pth')
    else:
        checkpoint = torch.load('./weights_generative/checkpoint_joao_'+str(args.resume)+'.pth')
        model.load_state_dict(checkpoint['graphcl'])
        model_generative_1.load_state_dict(checkpoint['generator_1'])
        model_generative_2.load_state_dict(checkpoint['generator_2'])
        optimizer.load_state_dict(checkpoint['graphcl_opt'])
        optimizer_generative_1.load_state_dict(checkpoint['generator_1_opt'])
        optimizer_generative_2.load_state_dict(checkpoint['generator_2_opt'])

    for epoch in range(args.resume+1, args.epochs+1):
        loader.dataset.set_generator(deepcopy(model_generative_1).cpu(), deepcopy(model_generative_2).cpu())
        pretrain_loss, generative_loss, link_loss = train(args, loader, model, optimizer, model_generative_1, optimizer_generative_1, model_generative_2, optimizer_generative_2, device)

        print(epoch, pretrain_loss, generative_loss, link_loss)

        torch.save({'graphcl':model.state_dict(), 'graphcl_opt': optimizer.state_dict(), 'generator_1':model_generative_1.state_dict(), 'generator_1_opt':optimizer_generative_1.state_dict(), 'generator_2':model_generative_2.state_dict(), 'generator_2_opt':optimizer_generative_2.state_dict()}, './weights_generative/checkpoint_joao_'+str(epoch)+'.pth')

if __name__ == "__main__":
    main()

