import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from gnn_proj import GNN

from tqdm import tqdm
import argparse
import time
import numpy as np
import pandas as pd
import os

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

### importing utils
from utils import ASTNodeEncoder, get_vocab_mapping
### for data transform
from utils import augment_edge, encode_y_to_arr, decode_arr_to_seq


multicls_criterion = torch.nn.CrossEntropyLoss()

def train(model, device, loader, optimizer):
    model.train()

    loss_accum = 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred_list = model(batch)
            optimizer.zero_grad()

            loss = 0
            for i in range(len(pred_list)):
                loss += multicls_criterion(pred_list[i].to(torch.float32), batch.y_arr[:,i])

            loss = loss / len(pred_list)
            
            loss.backward()
            optimizer.step()

            loss_accum += loss.item()

    print('Average training loss: {}'.format(loss_accum / (step + 1)))

def eval(model, device, loader, evaluator, arr_to_seq):
    model.eval()
    seq_ref_list = []
    seq_pred_list = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred_list = model(batch)

            mat = []
            for i in range(len(pred_list)):
                mat.append(torch.argmax(pred_list[i], dim = 1).view(-1,1))
            mat = torch.cat(mat, dim = 1)
            
            seq_pred = [arr_to_seq(arr) for arr in mat]
            
            # PyG = 1.4.3
            # seq_ref = [batch.y[i][0] for i in range(len(batch.y))]

            # PyG >= 1.5.0
            seq_ref = [batch.y[i] for i in range(len(batch.y))]

            seq_ref_list.extend(seq_ref)
            seq_pred_list.extend(seq_pred)

    input_dict = {"seq_ref": seq_ref_list, "seq_pred": seq_pred_list}

    return evaluator.eval(input_dict)

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

    parser.add_argument('--n_splits', type=int, default=0)
    parser.add_argument('--pretrain', type=str, default='')
    parser.add_argument('--pretrain_weight', type=str, default='')
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name = args.dataset)

    seq_len_list = np.array([len(seq) for seq in dataset.data.y])
    print('Target seqence less or equal to {} is {}%.'.format(args.max_seq_len, np.sum(seq_len_list <= args.max_seq_len) / len(seq_len_list)))

    split_idx = dataset.get_idx_split()

    # print(split_idx['train'])
    # print(split_idx['valid'])
    # print(split_idx['test'])

    # train_method_name = [' '.join(dataset.data.y[i]) for i in split_idx['train']]
    # valid_method_name = [' '.join(dataset.data.y[i]) for i in split_idx['valid']]
    # test_method_name = [' '.join(dataset.data.y[i]) for i in split_idx['test']]
    # print('#train')
    # print(len(train_method_name))
    # print('#valid')
    # print(len(valid_method_name))
    # print('#test')
    # print(len(test_method_name))

    # train_method_name_set = set(train_method_name)
    # valid_method_name_set = set(valid_method_name)
    # test_method_name_set = set(test_method_name)

    # # unique method name
    # print('#unique train')
    # print(len(train_method_name_set))
    # print('#unique valid')
    # print(len(valid_method_name_set))
    # print('#unique test')
    # print(len(test_method_name_set))

    # # unique valid/test method name
    # print('#valid unseen during training')
    # print(len(valid_method_name_set - train_method_name_set))
    # print('#test unseen during training')
    # print(len(test_method_name_set - train_method_name_set))


    ### building vocabulary for sequence predition. Only use training data.

    vocab2idx, idx2vocab = get_vocab_mapping([dataset.data.y[i] for i in split_idx['train']], args.num_vocab)

    # test encoder and decoder
    # for data in dataset:
    #     # PyG >= 1.5.0
    #     print(data.y)
    #
    #     # PyG 1.4.3
    #     # print(data.y[0])
    #     data = encode_y_to_arr(data, vocab2idx, args.max_seq_len)
    #     print(data.y_arr[0])
    #     decoded_seq = decode_arr_to_seq(data.y_arr[0], idx2vocab)
    #     print(decoded_seq)
    #     print('')

    ## test augment_edge
    # data = dataset[2]
    # print(data)
    # data_augmented = augment_edge(data)
    # print(data_augmented)

    ### set the transform function
    # augment_edge: add next-token edge as well as inverse edges. add edge attributes.
    # encode_y_to_arr: add y_arr to PyG data object, indicating the array representation of a sequence.
    dataset.transform = transforms.Compose([augment_edge, lambda data: encode_y_to_arr(data, vocab2idx, args.max_seq_len)])

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    idx = np.random.choice(split_idx["train"].size()[0], split_idx["train"].size()[0]//args.n_splits, replace=False)
    split_idx["train"] = split_idx["train"][idx]

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

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

    if not args.pretrain == '':
        model.load_state_dict(torch.load('./weights/' + args.pretrain_weight))

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    valid_curve = []
    test_curve = []
    # train_curve = []

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train(model, device, train_loader, optimizer)

        print('Evaluating...')
        # train_perf = eval(model, device, train_loader, evaluator, arr_to_seq = lambda arr: decode_arr_to_seq(arr, idx2vocab))
        valid_perf = eval(model, device, valid_loader, evaluator, arr_to_seq = lambda arr: decode_arr_to_seq(arr, idx2vocab))
        test_perf = eval(model, device, test_loader, evaluator, arr_to_seq = lambda arr: decode_arr_to_seq(arr, idx2vocab))


        # print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})
        print({'Validation': valid_perf, 'Test': test_perf})

        # train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])

    print('F1')
    best_val_epoch = np.argmax(np.array(valid_curve))
    # best_train = max(train_curve)
    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))

    with open('./results_' + args.dataset + '_' + args.pretrain + '_' + str(args.n_splits) + '.log', 'a+') as f:
        f.write(str(valid_curve[best_val_epoch]) + ' ' + str(test_curve[best_val_epoch]) + '\n')


if __name__ == "__main__":
    main()
