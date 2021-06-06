import torch
from torch.optim import Adam
from tu_dataset import DataLoader
import numpy as np

from utils import print_weights
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def experiment(dataset, model_func, epochs, batch_size, lr, weight_decay,
                dataset_name=None, aug_mode='uniform', aug_ratio=0.2, suffix=0, gamma_joao=0.1):
    model = model_func(dataset).to(device)
    print_weights(model)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    dataset.set_aug_mode('sample')
    dataset.set_aug_ratio(aug_ratio)
    aug_prob = np.ones(25) / 25
    dataset.set_aug_prob(aug_prob)

    loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=16)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # for epoch in tqdm(range(1, epochs+1)):
    for epoch in range(1, epochs+1):
        pretrain_loss, aug_prob = train(loader, model, optimizer, device, gamma_joao)
        print(pretrain_loss, aug_prob)
        loader.dataset.set_aug_prob(aug_prob)

        if epoch % 20 == 0:
            weight_path = './weights_joao/' + dataset_name + '_' + str(lr) + '_' + str(epoch) + '_' + str(gamma_joao) + '_' + str(suffix)  + '.pt'
            torch.save(model.state_dict(), weight_path)


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train(loader, model, optimizer, device, gamma_joao):
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

    aug_prob = joao(loader, model, gamma_joao)
    return total_loss/len(loader.dataset), aug_prob


def joao(loader, model, gamma_joao):
    aug_prob = loader.dataset.aug_prob
    # calculate augmentation loss
    loss_aug = np.zeros(25)
    for n in range(25):
        _aug_prob = np.zeros(25)
        _aug_prob[n] = 1
        loader.dataset.set_aug_prob(_aug_prob)

        count, count_stop = 0, len(loader.dataset)//(loader.batch_size*10)+1 # for efficiency, we only use around 10% of data to estimate the loss
        with torch.no_grad():
            for _, data1, data2 in loader:
                data1 = data1.to(device)
                data2 = data2.to(device)
                out1 = model.forward_graphcl(data1)
                out2 = model.forward_graphcl(data2)
                loss = model.loss_graphcl(out1, out2)
                loss_aug[n] += loss.item() * num_graphs(data1)
                count += 1
                if count == count_stop:
                    break
        loss_aug[n] /= (count*loader.batch_size)

    # view selection, projected gradient descent, reference: https://arxiv.org/abs/1906.03563
    beta = 1
    gamma = gamma_joao

    b = aug_prob + beta * (loss_aug - gamma * (aug_prob - 1/25))
    mu_min, mu_max = b.min()-1/25, b.max()-1/25
    mu = (mu_min + mu_max) / 2

    # bisection method
    while abs(np.maximum(b-mu, 0).sum() - 1) > 1e-2:
        if np.maximum(b-mu, 0).sum() > 1:
            mu_min = mu
        else:
            mu_max = mu
        mu = (mu_min + mu_max) / 2

    aug_prob = np.maximum(b-mu, 0)
    aug_prob /= aug_prob.sum()

    return aug_prob

