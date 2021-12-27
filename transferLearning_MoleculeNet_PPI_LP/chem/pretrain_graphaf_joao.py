import argparse

from loader import MoleculeDataset_graphcl_graphaf, DataLoader
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
from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd
from copy import deepcopy

from graphaf_utils.model import GraphFlowModel
from graphaf_utils.dataloader import PretrainZinkDataset
from torch.utils.data import DataLoader as DataLoader_GraphAF


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


def train(args, loader_graphaf, loader, model_cl, optimizer_cl, model_1, optimizer_1, model_2, optimizer_2, device, epoch):
    pretrain_loss, generative_loss = 0, 0
    for i_batch, batch_data in enumerate(tqdm(loader_graphaf)):
    # for i_batch, batch_data in enumerate(loader_graphaf):
        batch, idxs = batch_data
        inp_node_features = batch['node'].to(device)
        inp_adj_features = batch['adj'].to(device)

        optimizer_1.zero_grad()
        # optimizer_2.zero_grad()
        # calculate latent representations
        out_z_1, out_logdet_1, _ = model_1(inp_node_features, inp_adj_features)
        # out_z_2, out_logdet_2, _ = model_2(inp_node_features, inp_adj_features)

        # set latent representations
        loader.dataset.set_idxs(idxs.numpy().tolist())
        # loader.dataset.set_nl_el((out_z_1[0].detach().cpu(), out_z_1[1].detach().cpu()),
        #                          (out_z_2[0].detach().cpu(), out_z_2[1].detach().cpu()))
        loader.dataset.set_nl_el((out_z_1[0].detach().cpu(), out_z_1[1].detach().cpu()), None)

        # graphcl
        x1, x2 = [], []
        optimizer_cl.zero_grad()
        for step, batch in enumerate(loader):
            _, batch1, batch2 = batch
            batch1, batch2 = batch1.to(device), batch2.to(device)
            x1.append(model_cl.forward_cl(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch))
            x2.append(model_cl.forward_cl(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch))
        x1, x2 = torch.cat(x1, dim=0), torch.cat(x2, dim=0)
        loss_cl = model_cl.loss_cl(x1, x2, mean=False)
        loss = loss_cl.mean()
        pretrain_loss += loss.item()
        loss.backward()
        optimizer_cl.step()

        # reward for joao
        loss_cl = loss_cl.detach()
        loss_cl = loss_cl - loss_cl.mean()
        loss_cl[loss_cl>0] = 1
        loss_cl[loss_cl<=0] = 0.01 # weaken the reward for low cl loss

        # joao
        loss_1 = model_1.log_prob(out_z_1, out_logdet_1)
        # loss_2 = model_2.log_prob(out_z_2, out_logdet_2)
        loss_1 = (loss_1 * loss_cl).mean()
        # loss_2 = (loss_2 * loss_cl).mean()
        # generative_loss += (loss_1.item() + loss_2.item())
        generative_loss += loss_1.item()
        loss_1.backward()
        # loss_2.backward()
        optimizer_1.step()
        # optimizer_2.step()

        # update generator
        # loader.dataset.set_generator(deepcopy(model_1).cpu(), deepcopy(model_2).cpu())
        loader.dataset.set_generator(deepcopy(model_1).cpu(), None)

        step_size = 5000
        if i_batch % step_size == 0:
            print(pretrain_loss/step_size, generative_loss/step_size)
            pretrain_loss, generative_loss = 0, 0
            torch.save({'graphcl':model_cl.state_dict(), 'graphcl_opt': optimizer_cl.state_dict(), 'generator_1':model_1.state_dict(), 'generator_1_opt':optimizer_1.state_dict(), 'generator_2':model_2.state_dict(), 'generator_2_opt':optimizer_2.state_dict()}, './weights_graphaf/checkpoint_joao_epoch'+str(epoch)+'_step'+str(i_batch//step_size)+'.pth')


def read_molecules(path='./GraphAF/zinc_pregnn/zinc_pregnn'):
    print('reading data from %s' % path)
    node_features = np.load(path + '_node_features.npy')
    adj_features = np.load(path + '_adj_features.npy')
    mol_sizes = np.load(path + '_mol_sizes.npy')

    f = open(path + '_config.txt', 'r')
    data_config = eval(f.read())
    f.close()

    return node_features, adj_features, mol_sizes, data_config


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=128,
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
    parser.add_argument('--dataset', type=str, default = 'zinc_standard_agent', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--num_workers', type=int, default = 32, help='number of workers for dataset loading')
    parser.add_argument('--aug_mode', type=str, default = 'generative') 
    parser.add_argument('--aug_strength', type=float, default = 0.2)
    parser.add_argument('--resume', type=int, default=0)

    # graphaf args
    parser.add_argument('--name', type=str, default='base', help='model name, crucial for test and checkpoint initialization')
    parser.add_argument('--deq_type', type=str, default='random', help='dequantization methods.')
    parser.add_argument('--deq_coeff', type=float, default=0.9, help='dequantization coefficient.(only for deq_type random)')
    parser.add_argument('--num_flow_layer', type=int, default=6, help='num of affine transformation layer in each timestep')
    parser.add_argument('--gcn_layer', type=int, default=3, help='num of rgcn layers')
    parser.add_argument('--nhid', type=int, default=128, help='num of hidden units of gcn')
    parser.add_argument('--nout', type=int, default=128, help='num of out units of gcn')
    parser.add_argument('--st_type', type=str, default='sigmoid', help='architecture of st net, choice: [sigmoid, exp, softplus, spine]')
    parser.add_argument('--sigmoid_shift', type=float, default=2.0, help='sigmoid shift on s.')
    parser.add_argument('--all_save_prefix', type=str, default='./', help='path of save prefix')
    parser.add_argument('--learn_prior', action='store_true', default=False, help='learn log-var of gaussian prior.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--is_bn', action='store_true', default=False, help='batch norm on node embeddings.')
    parser.add_argument('--is_bn_before', action='store_true', default=False, help='batch norm on node embeddings on st-net input.')
    parser.add_argument('--scale_weight_norm', action='store_true', default=False, help='apply weight norm on scale factor.')
    parser.add_argument('--divide_loss', action='store_true', default=False, help='divide loss by length of latent.')
    parser.add_argument('--temperature', type=float, default=0.75, help='temperature for normal distribution')
    parser.add_argument('--min_atoms', type=int, default=5, help='minimum #atoms of generated mol, otherwise the mol is simply discarded')
    parser.add_argument('--max_atoms', type=int, default=48, help='maximum #atoms of generated mol')    

    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    #set up dataset
    dataset = MoleculeDataset_graphcl_graphaf("dataset/" + args.dataset, dataset=args.dataset)
    dataset.set_augMode(args.aug_mode)
    dataset.set_augStrength(args.aug_strength)
    print(dataset)

    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=args.num_workers)

    # set up graphcl model
    gnn = GNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type)
    model = graphcl(gnn, args.emb_dim)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)



    # set up graphaf model 1
    model_generative_1 = GraphFlowModel(38, 9, 4, 12, args)
    model_generative_1.to(device)
    optimizer_generative_1 = optim.Adam(model_generative_1.parameters(), lr=args.lr, weight_decay=args.decay)
    # set up graphaf model 2
    model_generative_2 = GraphFlowModel(38, 9, 4, 12, args)
    model_generative_2.to(device)
    optimizer_generative_2 = optim.Adam(model_generative_2.parameters(), lr=args.lr, weight_decay=args.decay)

    # dataloader for graphaf
    node_features, adj_features, mol_sizes, _ = read_molecules()
    dataset_graphaf = PretrainZinkDataset(node_features, adj_features, mol_sizes)
    loader_graphaf = DataLoader_GraphAF(dataset_graphaf, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)



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

    loader.dataset.set_generator(deepcopy(model_generative_1).cpu(), deepcopy(model_generative_2).cpu())
    for epoch in range(args.resume+1, args.epochs+1):
        train(args, loader_graphaf, loader, model, optimizer, model_generative_1, optimizer_generative_1, model_generative_2, optimizer_generative_2, device, epoch)

if __name__ == "__main__":
    main()

