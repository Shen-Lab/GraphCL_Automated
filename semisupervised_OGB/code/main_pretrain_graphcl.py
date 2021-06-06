import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from gnn import GNN

from tqdm import tqdm
import argparse
import time
import numpy as np
import pandas as pd
import os

### importing OGB
from dataset_aug import PygGraphPropPredDataset, collate
from ogb.graphproppred import Evaluator

### importing utils
from utils import ASTNodeEncoder, get_vocab_mapping
### for data transform
from utils import augment_edge, encode_y_to_arr, decode_arr_to_seq


multicls_criterion = torch.nn.CrossEntropyLoss()

def train(model, device, loader, optimizer):
    model.train()

    loss_accum = 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch1, batch2 = batch
        batch1, batch2 = batch1.to(device), batch2.to(device)

        if batch1.x.shape[0] == 1 or batch1.batch[-1] == 0:
            pass
        else:
            x1, x2 = model.forward_cl(batch1), model.forward_cl(batch2)
            optimizer.zero_grad()
            loss = model.loss_cl(x1, x2)
            loss.backward()
            optimizer.step()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbg-code data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gcn-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gcn-virtual)')
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
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-code",
                        help='dataset name (default: ogbg-code)')

    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name=args.dataset, mode='graphcl')

    split_idx = dataset.get_idx_split()

    vocab2idx, idx2vocab = get_vocab_mapping([dataset.data.y[i] for i in split_idx['train']], args.num_vocab)

    train_loader = torch.utils.data.DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate)

    nodetypes_mapping = pd.read_csv(os.path.join(dataset.root, 'mapping', 'typeidx2type.csv.gz'))
    nodeattributes_mapping = pd.read_csv(os.path.join(dataset.root, 'mapping', 'attridx2attr.csv.gz'))

    ### Encoding node features into emb_dim vectors.
    ### The following three node features are used.
    # 1. node type
    # 2. node attribute
    # 3. node depth
    node_encoder = ASTNodeEncoder(args.emb_dim, num_nodetypes = len(nodetypes_mapping['type']), num_nodeattributes = len(nodeattributes_mapping['attr']), max_depth = 20)

    if args.gnn == 'gin':
        model = GNN(num_vocab = len(vocab2idx), max_seq_len = args.max_seq_len, node_encoder = node_encoder, num_layer = args.num_layer, gnn_type = 'gin', emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(num_vocab = len(vocab2idx), max_seq_len = args.max_seq_len, node_encoder = node_encoder, num_layer = args.num_layer, gnn_type = 'gin', emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    elif args.gnn == 'gcn':
        model = GNN(num_vocab = len(vocab2idx), max_seq_len = args.max_seq_len, node_encoder = node_encoder, num_layer = args.num_layer, gnn_type = 'gcn', emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(num_vocab = len(vocab2idx), max_seq_len = args.max_seq_len, node_encoder = node_encoder, num_layer = args.num_layer, gnn_type = 'gcn', emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    else:
        raise ValueError('Invalid GNN type')

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train(model, device, train_loader, optimizer)

        if epoch % 10 == 0:
            torch.save(model.state_dict(), './weights/' + args.dataset + '_graphcl_' + str(epoch) + '.pt')

    print('Finished training!')


if __name__ == "__main__":
    main()
