import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from gnn_proj import GNN

from tqdm import tqdm
import argparse
import time
import numpy as np

### importing OGB
from dataset_aug import PygGraphPropPredDataset, collate
from ogb.graphproppred import Evaluator


multicls_criterion = torch.nn.CrossEntropyLoss()

def train(model, device, loader, optimizer, aug_P, gamma):
    model.train()
    loader.dataset.aug_P = aug_P

    n_proj = np.random.choice(16, 1, p=aug_P)[0]
    n1_proj, n2_proj = n_proj//4, n_proj%4
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch1, batch2 = batch
        batch1, batch2 = batch1.to(device), batch2.to(device)

        if batch1.x.shape[0] == 1 or batch1.batch[-1] == 0:
            pass
        else:
            x1, x2 = model.forward_cl(batch1, n1_proj), model.forward_cl(batch2, n2_proj)
            optimizer.zero_grad()
            loss = model.loss_cl(x1, x2)
            loss.backward()
            optimizer.step()

    # joint augmentation optimization
    loss_aug = np.zeros(16)
    for n in range(16):
        _aug_P = np.zeros(16)
        _aug_P[n] = 1
        loader.dataset.aug_P = _aug_P
        n1_proj, n2_proj = n//4, n%4
        for batch in loader:
            batch1, batch2 = batch
            batch1, batch2 = batch1.to(device), batch2.to(device)
            
            if batch1.x.shape[0] == 1 or batch1.batch[-1] == 0:
                pass
            else:
                x1, x2 = model.forward_cl(batch1, n1_proj), model.forward_cl(batch2, n2_proj)
                loss = model.loss_cl(x1, x2)
                loss_aug[n] = loss.item()
            break
    
    gamma = gamma
    beta = 1
    b = aug_P + beta * (loss_aug - gamma * (aug_P - 1/16))
    
    mu_min, mu_max = b.min()-1/16, b.max()-1/16
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
    print(loss_aug.reshape((4, 4)))
    print(aug_P.reshape((4, 4)))
    
    return aug_P


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbg-ppa data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
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
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-ppa",
                        help='dataset name (default: ogbg-ppa)')

    parser.add_argument('--aug_ratio', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting

    dataset = PygGraphPropPredDataset(name=args.dataset, mode='sampling', aug_ratio=args.aug_ratio)

    split_idx = dataset.get_idx_split()

    train_loader = torch.utils.data.DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers, collate_fn=collate)

    if args.gnn == 'gin':
        model = GNN(gnn_type = 'gin', num_class = dataset.num_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type = 'gin', num_class = dataset.num_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type = 'gcn', num_class = dataset.num_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type = 'gcn', num_class = dataset.num_classes, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    else:
        raise ValueError('Invalid GNN type')

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    valid_curve = []
    test_curve = []
    train_curve = []

    aug_P = np.ones(16) / 16
    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        aug_P = train(model, device, train_loader, optimizer, aug_P, args.gamma)

        if epoch % 20 == 0:
            torch.save(model.state_dict(), './weights/joaov2_' + str(args.aug_ratio) + '_' + str(args.gamma) + '_' + str(epoch) + '.pt')
    
    print('Finished training!')


if __name__ == "__main__":
    main()
