import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from gnn import GNN, vgae

from tqdm import tqdm
import argparse
import time
import numpy as np

### importing OGB
from dataset_graphcl import PygGraphPropPredDataset, collate
from ogb.graphproppred import Evaluator
from copy import deepcopy


multicls_criterion = torch.nn.CrossEntropyLoss()

def train(model, optimizer, generator1, optimizer_gen1, generator2, optimizer_gen2, device, loader):
    model.train()
    loss_pretrain, loss_generator = 0, 0
    # for step, batch in enumerate(tqdm(loader, desc="Iteration")):
    for step, batch in enumerate(loader):
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

        # reward for joao
        loss_cl = loss_cl.detach()
        loss_cl = loss_cl - loss_cl.mean()
        loss_cl[loss_cl>0] = 1
        loss_cl[loss_cl<=0] = 0.01 # weaken the reward for low cl loss

        # 2. joao
        optimizer_gen1.zero_grad()
        optimizer_gen2.zero_grad()

        x, x_mean, x_std = generator1.forward_encoder(batch)
        edge_attr_pred, edge_pos_pred, edge_neg_pred = generator1.forward_decoder(x, batch.edge_index, batch.edge_index_neg)
        loss_1 = generator1.loss_vgae(edge_attr_pred, batch.edge_attr, edge_pos_pred, edge_neg_pred, batch.edge_index_batch, batch.edge_index_neg_batch, x_mean, x_std, batch.batch, reward=loss_cl)
 
        x, x_mean, x_std = generator2.forward_encoder(batch)
        edge_attr_pred, edge_pos_pred, edge_neg_pred = generator2.forward_decoder(x, batch.edge_index, batch.edge_index_neg)
        loss_2 = generator2.loss_vgae(edge_attr_pred, batch.edge_attr, edge_pos_pred, edge_neg_pred, batch.edge_index_batch, batch.edge_index_neg_batch, x_mean, x_std, batch.batch, reward=loss_cl)

        loss = loss_1 + loss_2

        loss.backward()
        optimizer_gen1.step()
        optimizer_gen2.step()
        loss_generator += loss.item()

    print(loss_pretrain/step, loss_generator/step)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbg-ppa data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=24,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-ppa",
                        help='dataset name (default: ogbg-ppa)')

    parser.add_argument('--aug_mode', type=str, default='generative')
    parser.add_argument('--aug_strength', type=float, default=0.2)

    parser.add_argument('--resume', type=int, default=44)
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting

    dataset = PygGraphPropPredDataset(name=args.dataset)
    dataset.set_augMode(args.aug_mode)
    dataset.set_augStrength(args.aug_strength)

    split_idx = dataset.get_idx_split()

    loader = torch.utils.data.DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers, collate_fn=collate)

    model = GNN(gnn_type = 'gin', num_class = dataset.num_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # generators
    generator1 = vgae(gnn_type = 'gin', num_class = dataset.num_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    optimizer_gen1 = optim.Adam(generator1.parameters(), lr=0.001)
    generator2 = vgae(gnn_type = 'gin', num_class = dataset.num_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    optimizer_gen2 = optim.Adam(generator2.parameters(), lr=0.001)

    if not args.resume == 0:
        checkpoint = torch.load('./weights_generative_joao/checkpoint_'+str(args.resume)+'.pt')
        model.load_state_dict(checkpoint['graphcl'])
        optimizer.load_state_dict(checkpoint['graphcl_optimizer'])
        generator1.load_state_dict(checkpoint['generator1'])
        optimizer_gen1.load_state_dict(checkpoint['generator1_optimizer'])
        generator2.load_state_dict(checkpoint['generator2'])
        optimizer_gen2.load_state_dict(checkpoint['generator2_optimizer'])

    for epoch in range(args.resume+1, args.epochs + 1):
        loader.dataset.set_generator(deepcopy(generator1).cpu(), deepcopy(generator2).cpu())
        train(model, optimizer, generator1, optimizer_gen1, generator2, optimizer_gen2, device, loader)

        torch.save({'graphcl':model.state_dict(), 'graphcl_optimizer': optimizer.state_dict(), 'generator1':generator1.state_dict(), 'generator1_optimizer':optimizer_gen1.state_dict(), 'generator2':generator2.state_dict(), 'generator2_optimizer':optimizer_gen2.state_dict()}, './weights_generative_joao/checkpoint_'+str(epoch)+'.pt')
    

if __name__ == "__main__":
    main()

