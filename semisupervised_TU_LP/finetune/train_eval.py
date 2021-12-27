import sys
import time

import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader

from utils import print_weights
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def cross_validation_with_val_set(dataset,
                                  model_func,
                                  epochs,
                                  batch_size,
                                  lr,
                                  weight_decay,
                                  epoch_select,
                                  with_eval_mode=True,
                                  logger=None,
                                  model_PATH=None, n_splits=None, result_PATH=None, result_feat=None):
    assert epoch_select in ['val_max', 'test_max'], epoch_select
   
    folds=10
    # pbar = tqdm(total=folds)
    val_losses, train_accs, test_accs, durations = [], [], [], []
    for fold, (train_idx, test_idx, val_idx) in enumerate(
            zip(*k_fold(dataset, folds, epoch_select, n_splits))):

        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        val_dataset = dataset[val_idx]

        train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=16)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=16)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=16)

        model = model_func(dataset).to(device)
        # # train from scratch
        model.load_state_dict(torch.load(model_PATH))

        if fold == 0:
            print_weights(model)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train(
                model, optimizer, train_loader, device)
            train_accs.append(train_acc)
            val_losses.append(eval_loss(
                model, val_loader, device, with_eval_mode))
            test_accs.append(eval_acc(
                model, test_loader, device, with_eval_mode))
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_accs[-1],
                'val_loss': val_losses[-1],
                'test_acc': test_accs[-1],
            }

            if logger is not None:
                logger(eval_info)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)
        # pbar.update(1)

    duration = tensor(durations)
    train_acc, test_acc = tensor(train_accs), tensor(test_accs)
    val_loss = tensor(val_losses)
    train_acc = train_acc.view(folds, epochs)
    test_acc = test_acc.view(folds, epochs)
    val_loss = val_loss.view(folds, epochs)
    if epoch_select == 'test_max':  # take epoch that yields best test results.
        _, selected_epoch = test_acc.mean(dim=0).max(dim=0)
        selected_epoch = selected_epoch.repeat(folds)
    else:  # take epoch that yields min val loss for each fold individually.
        _, selected_epoch = val_loss.min(dim=1)
    test_acc = test_acc[torch.arange(folds, dtype=torch.long), selected_epoch]
    train_acc_mean = train_acc[:, -1].mean().item()
    test_acc_mean = test_acc.mean().item()
    test_acc_std = test_acc.std().item()
    duration_mean = duration.mean().item()

    print(train_acc_mean, test_acc_mean, test_acc_std, duration_mean)
    sys.stdout.flush()

    with open(result_PATH, 'a+') as f:
        f.write(result_feat + ' ' + str(test_acc_mean) + '\n')


def k_fold(dataset, folds, epoch_select, n_splits):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx))

    if epoch_select == 'test_max':
        val_indices = [test_indices[i] for i in range(folds)]
    else:
        val_indices = [test_indices[i - 1] for i in range(folds)]

    skf_semi = StratifiedKFold(n_splits, shuffle=True, random_state=12345)
    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i].long()] = 0
        train_mask[val_indices[i].long()] = 0
        idx_train = train_mask.nonzero(as_tuple=False).view(-1)

        for _, idx in skf_semi.split(torch.zeros(idx_train.size()[0]), dataset.data.y[idx_train]):
            idx_train = idx_train[idx]
            break

        train_indices.append(idx_train)

    return train_indices, test_indices, val_indices


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train(model, optimizer, loader, device):
    model.train()

    total_loss = 0
    correct = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        loss = F.nll_loss(out, data.y.long().view(-1))
        pred = out.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def eval_acc(model, loader, device, with_eval_mode):
    if with_eval_mode:
        model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


def eval_loss(model, loader, device, with_eval_mode):
    if with_eval_mode:
        model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        loss += F.nll_loss(out, data.y.long().view(-1), reduction='sum').item()
    return loss / len(loader.dataset)

