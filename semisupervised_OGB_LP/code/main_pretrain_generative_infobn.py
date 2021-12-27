import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from gnn import GNN, vgae

from tqdm import tqdm
import argparse
import time
import numpy as np
import pandas as pd
import os

### importing OGB
from dataset_graphcl import PygGraphPropPredDataset, collate
from ogb.graphproppred import Evaluator

### importing utils
from utils import ASTNodeEncoder, get_vocab_mapping
### for data transform
from utils import augment_edge, encode_y_to_arr, decode_arr_to_seq
from copy import deepcopy


multicls_criterion = torch.nn.CrossEntropyLoss()

def train(model, optimizer, generator1, optimizer_gen1, generator2, optimizer_gen2, model_ib, optimizer_ib, device, loader):
    model.train()
    loss_pretrain, loss_generator = 0, 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch, batch1, batch2 = batch
        batch, batch1, batch2 = batch.to(device), batch1.to(device), batch2.to(device)

        #1. graphcl
        if batch1.x.shape[0] == 1 or batch1.batch[-1] == 0:
            pass
        else:
            x1, x2 = model.forward_cl(batch1), model.forward_cl(batch2)
            optimizer.zero_grad()
            loss_cl = model.loss_cl(x1, x2, mean=False)
            loss = loss_cl.mean()

            loss.backward()
            optimizer.step()
        loss_pretrain += loss.item()

        # information bottleneck
        optimizer_ib.zero_grad()
        _x1 = model_ib.forward_cl(batch1)
        _x2 = model_ib.forward_cl(batch2)
        loss_ib = model_ib.loss_cl(_x1, x1.detach(), mean=False) + model_ib.loss_cl(_x2, x2.detach(), mean=False)
        loss = loss_ib.mean()
        loss.backward()
        optimizer_ib.step()

        # reward for joao
        loss_cl = loss_ib.detach()
        loss_cl = loss_cl - loss_cl.mean()
        loss_cl[loss_cl>0] = 1
        loss_cl[loss_cl<=0] = 0.01 # weaken the reward for low cl loss

        # 2. joao
        optimizer_gen1.zero_grad()
        optimizer_gen2.zero_grad()

        _batch = deepcopy(batch)
        x, x_mean, x_std = generator1.forward_encoder(batch)
        edge_pos_pred, edge_neg_pred = generator1.forward_decoder(x, batch._edge_index, batch.edge_index_neg)
        loss_1 = generator1.loss_vgae(edge_pos_pred, edge_neg_pred, batch._edge_index_batch, batch.edge_index_neg_batch, x_mean, x_std, batch.batch, reward=loss_cl)
 
        x, x_mean, x_std = generator2.forward_encoder(_batch)
        edge_pos_pred, edge_neg_pred = generator2.forward_decoder(x, _batch._edge_index, _batch.edge_index_neg)
        loss_2 = generator2.loss_vgae(edge_pos_pred, edge_neg_pred, _batch._edge_index_batch, _batch.edge_index_neg_batch, x_mean, x_std, _batch.batch, reward=loss_cl)

        loss = loss_1 + loss_2

        loss.backward()
        optimizer_gen1.step()
        optimizer_gen2.step()
        loss_generator += loss.item()

    print(loss_pretrain/step, loss_generator/step)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbg-code data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--drop_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--max_seq_len', type=int, default=5,
                        help='maximum sequence length to predict (default: 5)')
    parser.add_argument('--num_vocab', type=int, default=5000,
                        help='the number of vocabulary used for sequence prediction (default: 5000)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-code",
                        help='dataset name (default: ogbg-code)')

    parser.add_argument('--aug_mode', type=str, default='generative')
    parser.add_argument('--aug_strength', type=float, default=0.2)

    parser.add_argument('--resume', type=int, default=17)
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name=args.dataset)
    dataset.set_augMode(args.aug_mode)
    dataset.set_augStrength(args.aug_strength)

    split_idx = dataset.get_idx_split()

    vocab2idx, idx2vocab = get_vocab_mapping([dataset.data.y[i] for i in split_idx['train']], args.num_vocab)

    loader = torch.utils.data.DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate)

    nodetypes_mapping = pd.read_csv(os.path.join(dataset.root, 'mapping', 'typeidx2type.csv.gz'))
    nodeattributes_mapping = pd.read_csv(os.path.join(dataset.root, 'mapping', 'attridx2attr.csv.gz'))

    ### Encoding node features into emb_dim vectors.
    ### The following three node features are used.
    # 1. node type
    # 2. node attribute
    # 3. node depth
    node_encoder = ASTNodeEncoder(args.emb_dim, num_nodetypes = len(nodetypes_mapping['type']), num_nodeattributes = len(nodeattributes_mapping['attr']), max_depth = 20)
    model = GNN(num_vocab = len(vocab2idx), max_seq_len = args.max_seq_len, node_encoder = node_encoder, num_layer = args.num_layer, gnn_type = 'gin', emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # generators
    node_encoder1 = ASTNodeEncoder(args.emb_dim, num_nodetypes = len(nodetypes_mapping['type']), num_nodeattributes = len(nodeattributes_mapping['attr']), max_depth = 20)
    generator1 = vgae(num_vocab = len(vocab2idx), max_seq_len = args.max_seq_len, node_encoder = node_encoder1, num_layer = args.num_layer, gnn_type = 'gin', emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    optimizer_gen1 = optim.Adam(generator1.parameters(), lr=0.001)

    node_encoder2 = ASTNodeEncoder(args.emb_dim, num_nodetypes = len(nodetypes_mapping['type']), num_nodeattributes = len(nodeattributes_mapping['attr']), max_depth = 20)
    generator2 = vgae(num_vocab = len(vocab2idx), max_seq_len = args.max_seq_len, node_encoder = node_encoder2, num_layer = args.num_layer, gnn_type = 'gin', emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    optimizer_gen2 = optim.Adam(generator2.parameters(), lr=0.001)

    # ib estimator
    node_encoder_ib = ASTNodeEncoder(args.emb_dim, num_nodetypes = len(nodetypes_mapping['type']), num_nodeattributes = len(nodeattributes_mapping['attr']), max_depth = 20)
    model_ib = GNN(num_vocab = len(vocab2idx), max_seq_len = args.max_seq_len, node_encoder = node_encoder_ib, num_layer = args.num_layer, gnn_type = 'gin', emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    optimizer_ib = optim.Adam(model_ib.parameters(), lr=0.001)

    if not args.resume == 0:
        checkpoint = torch.load('./weights_generative_ibalone/checkpoint_'+str(args.resume)+'.pt')
        model.load_state_dict(checkpoint['graphcl'])
        optimizer.load_state_dict(checkpoint['graphcl_optimizer'])
        generator1.load_state_dict(checkpoint['generator1'])
        optimizer_gen1.load_state_dict(checkpoint['generator1_optimizer'])
        generator2.load_state_dict(checkpoint['generator2'])
        optimizer_gen2.load_state_dict(checkpoint['generator2_optimizer'])
        model_ib.load_state_dict(checkpoint['model_ib'])
        optimizer_ib.load_state_dict(checkpoint['ib_optimizer'])

    for epoch in range(args.resume+1, args.epochs + 1):
        loader.dataset.set_generator(deepcopy(generator1).cpu(), deepcopy(generator2).cpu())
        train(model, optimizer, generator1, optimizer_gen1, generator2, optimizer_gen2, model_ib, optimizer_ib, device, loader)

        torch.save({'graphcl':model.state_dict(), 'graphcl_optimizer': optimizer.state_dict(), 'generator1':generator1.state_dict(), 'generator1_optimizer':optimizer_gen1.state_dict(), 'generator2':generator2.state_dict(), 'generator2_optimizer':optimizer_gen2.state_dict(), 'model_ib':model_ib.state_dict(), 'ib_optimizer':optimizer_ib.state_dict()}, './weights_generative_ibalone/checkpoint_'+str(epoch)+'.pt')


if __name__ == "__main__":
    main()

