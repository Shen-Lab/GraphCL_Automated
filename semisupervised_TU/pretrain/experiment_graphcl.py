import torch
from torch.optim import Adam
from tu_dataset import DataLoader

from utils import print_weights
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def experiment(dataset, model_func, epochs, batch_size, lr, weight_decay,
                dataset_name=None, aug_mode='uniform', aug_ratio=0.2, suffix=0):
    model = model_func(dataset).to(device)
    print_weights(model)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    dataset.set_aug_mode(aug_mode)
    dataset.set_aug_ratio(aug_ratio)
    loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=16)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # for epoch in tqdm(range(1, epochs+1)):
    for epoch in range(1, epochs+1):
        pretrain_loss = train(loader, model, optimizer, device)
        print(pretrain_loss)

        if epoch % 20 == 0:
            weight_path = './weights_graphcl/' + dataset_name + '_' + str(lr) + '_' + str(epoch) + '_' + str(suffix)  + '.pt'
            torch.save(model.state_dict(), weight_path)


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train(loader, model, optimizer, device):
    model.train()
    total_loss = 0
    for _, data1, data2 in loader:
        # print(data1, data2)
        optimizer.zero_grad()
        data1 = data1.to(device)
        data2 = data2.to(device)
        out1 = model.forward_graphcl(data1)
        out2 = model.forward_graphcl(data2)
        loss = model.loss_graphcl(out1, out2)
        loss.backward()
        total_loss += loss.item() * num_graphs(data1)
        optimizer.step()

    return total_loss/len(loader.dataset)

