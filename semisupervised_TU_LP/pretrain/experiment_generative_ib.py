import torch
from torch.optim import Adam
from tu_dataset import DataLoader

from utils import print_weights
from tqdm import tqdm
from copy import deepcopy
from res_gcn import vgae

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def experiment(dataset, model_func, epochs, batch_size, lr, weight_decay,
                dataset_name=None, aug_mode='uniform', aug_ratio=0.2, suffix=0):
    model, encoder, decoder = model_func(dataset)
    generator_1 = vgae(encoder, decoder)
    model_ib, _encoder, _decoder = model_func(dataset)
    generator_2 = vgae(_encoder, _decoder)

    model, generator_1, generator_2, model_ib = model.to(device), generator_1.to(device), generator_2.to(device), model_ib.to(device)
    print_weights(model)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    dataset.set_aug_mode('generative')
    loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=16)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_generator_1 = Adam(generator_1.parameters(), lr=lr)
    optimizer_generator_2 = Adam(generator_2.parameters(), lr=lr)
    optimizer_ib = Adam(model_ib.parameters(), lr=lr)

    # for epoch in tqdm(range(1, epochs+1)):
    for epoch in range(1, epochs+1):
        loader.dataset.set_generator(deepcopy(generator_1).cpu(), deepcopy(generator_2).cpu())
        pretrain_loss, generative_loss = train(loader, model, optimizer, generator_1, optimizer_generator_1, generator_2, optimizer_generator_2, model_ib, optimizer_ib, device)
        print(pretrain_loss, generative_loss)

        if epoch % 20 == 0:
            weight_path = './weights_infominbn/' + dataset_name + '_' + str(lr) + '_' + str(epoch) + '_' + str(suffix)  + '.pt'
            torch.save(model.state_dict(), weight_path)

    torch.save({'graphcl':model.state_dict(), 'graphcl_opt': optimizer.state_dict(), 'model_ib':model_ib.state_dict(), 'model_ib_opt': optimizer_ib.state_dict(), 'generator_1':generator_1.state_dict(), 'generator_1_opt':optimizer_generator_1.state_dict(), 'generator_2':generator_2.state_dict(), 'generator_2_opt':optimizer_generator_2.state_dict()}, './weights_generative_joao/checkpoint_' + dataset_name + '_' + str(lr) + '_' + str(suffix)  + '.pt')


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train(loader, model, optimizer, generator_1, optimizer_generator_1, generator_2, optimizer_generator_2, model_ib, optimizer_ib, device):
    model.train()
    generator_1.train()
    generator_2.train()
    model_ib.train()
    total_loss, generative_loss = 0, 0
    for data, data1, data2 in loader:
        optimizer.zero_grad()
        data1 = data1.to(device)
        data2 = data2.to(device)
        out1 = model.forward_graphcl(data1)
        out2 = model.forward_graphcl(data2)
        loss_cl = model.loss_graphcl(out1, out2, mean=False)
        loss = loss_cl.mean()
        loss.backward()
        total_loss += loss.item() * num_graphs(data1)
        optimizer.step()

        # information bottleneck
        optimizer_ib.zero_grad()
        _out1 = model_ib.forward_graphcl(data1)
        _out2 = model_ib.forward_graphcl(data2)
        loss_ib = model_ib.loss_graphcl(_out1, out1.detach(), mean=False) + model_ib.loss_graphcl(_out2, out2.detach(), mean=False)
        loss = loss_ib.mean()
        loss.backward()
        optimizer_ib.step()

        # reward for joao
        loss_cl = loss_cl.detach() + loss_ib.detach()
        loss_cl = loss_cl - loss_cl.mean()
        loss_cl[loss_cl>0] = 1
        loss_cl[loss_cl<=0] = 0.01 # weaken the reward for low cl loss

        # joao
        optimizer_generator_1.zero_grad()
        optimizer_generator_2.zero_grad()
        data = data.to(device)

        loss_1 = generator_1(data, reward=loss_cl)
        loss_2 = generator_2(data, reward=loss_cl)

        loss = loss_1 + loss_2
        loss.backward()
        optimizer_generator_1.step()
        optimizer_generator_2.step()
        generative_loss += loss.item() * num_graphs(data)

    return total_loss/len(loader.dataset), generative_loss/len(loader.dataset)

