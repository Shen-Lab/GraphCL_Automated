from torch_geometric.data import InMemoryDataset
import pandas as pd
import shutil, os
import os.path as osp
import torch
import numpy as np
from ogb.utils.url import decide_download, download_url, extract_zip
from ogb.io.read_graph_pyg import read_graph_pyg

from itertools import repeat
import torch_geometric


class PygGraphPropPredDataset(InMemoryDataset):
    def __init__(self, name, mode=None, aug_ratio=0.2, aug_prob=None, root = 'dataset', transform=None, pre_transform = None, meta_dict = None):
        '''
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - transform, pre_transform (optional): transform/pre-transform graph objects

            - meta_dict: dictionary that stores all the meta-information about data. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        ''' 

        self.name = name ## original name, e.g., ogbg-molhiv
        self.mode, self.aug_ratio, self.aug_prob = mode, aug_ratio, aug_prob
        
        if meta_dict is None:
            self.dir_name = '_'.join(name.split('-')) 
            
            # check if previously-downloaded folder exists.
            # If so, use that one.
            if osp.exists(osp.join(root, self.dir_name + '_pyg')):
                self.dir_name = self.dir_name + '_pyg'

            self.original_root = root
            self.root = osp.join(root, self.dir_name)
            
            # master = pd.read_csv(os.path.join(os.path.dirname(__file__), 'master.csv'), index_col = 0)
            master = pd.read_csv(os.path.join('/home/wuzhenyu_sjtu/anaconda3/envs/ogb/lib/python3.6/site-packages/ogb/graphproppred', 'master.csv'), index_col = 0)
            if not self.name in master:
                error_mssg = 'Invalid dataset name {}.\n'.format(self.name)
                error_mssg += 'Available datasets are as follows:\n'
                error_mssg += '\n'.join(master.keys())
                raise ValueError(error_mssg)
            self.meta_info = master[self.name]
            
        else:
            self.dir_name = meta_dict['dir_path']
            self.original_root = ''
            self.root = meta_dict['dir_path']
            self.meta_info = meta_dict
        
        # check version
        # First check whether the dataset has been already downloaded or not.
        # If so, check whether the dataset version is the newest or not.
        # If the dataset is not the newest version, notify this to the user. 
        if osp.isdir(self.root) and (not osp.exists(osp.join(self.root, 'RELEASE_v' + str(self.meta_info['version']) + '.txt'))):
            print(self.name + ' has been updated.')
            if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
                shutil.rmtree(self.root)

        self.download_name = self.meta_info['download_name'] ## name of downloaded file, e.g., tox21

        self.num_tasks = int(self.meta_info['num tasks'])
        self.eval_metric = self.meta_info['eval metric']
        self.task_type = self.meta_info['task type']
        self.__num_classes__ = int(self.meta_info['num classes'])
        self.binary = self.meta_info['binary'] == 'True'

        super(PygGraphPropPredDataset, self).__init__(self.root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    def get_idx_split(self, split_type = None):
        if split_type is None:
            split_type = self.meta_info['split']
            
        path = osp.join(self.root, 'split', split_type)

        # short-cut if split_dict.pt exists
        if os.path.isfile(os.path.join(path, 'split_dict.pt')):
            return torch.load(os.path.join(path, 'split_dict.pt'))

        train_idx = pd.read_csv(osp.join(path, 'train.csv.gz'), compression='gzip', header = None).values.T[0]
        valid_idx = pd.read_csv(osp.join(path, 'valid.csv.gz'), compression='gzip', header = None).values.T[0]
        test_idx = pd.read_csv(osp.join(path, 'test.csv.gz'), compression='gzip', header = None).values.T[0]

        return {'train': torch.tensor(train_idx, dtype = torch.long), 'valid': torch.tensor(valid_idx, dtype = torch.long), 'test': torch.tensor(test_idx, dtype = torch.long)}

    @property
    def num_classes(self):
        return self.__num_classes__

    @property
    def raw_file_names(self):
        if self.binary:
            return ['data.npz']
        else:
            file_names = ['edge']
            if self.meta_info['has_node_attr'] == 'True':
                file_names.append('node-feat')
            if self.meta_info['has_edge_attr'] == 'True':
                file_names.append('edge-feat')
            return [file_name + '.csv.gz' for file_name in file_names]

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        url = self.meta_info['url']
        if decide_download(url):
            path = download_url(url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
            shutil.rmtree(self.root)
            shutil.move(osp.join(self.original_root, self.download_name), self.root)

        else:
            print('Stop downloading.')
            shutil.rmtree(self.root)
            exit(-1)

    def process(self):
        ### read pyg graph list
        add_inverse_edge = self.meta_info['add_inverse_edge'] == 'True'

        if self.meta_info['additional node files'] == 'None':
            additional_node_files = []
        else:
            additional_node_files = self.meta_info['additional node files'].split(',')

        if self.meta_info['additional edge files'] == 'None':
            additional_edge_files = []
        else:
            additional_edge_files = self.meta_info['additional edge files'].split(',')

        data_list = read_graph_pyg(self.raw_dir, add_inverse_edge = add_inverse_edge, additional_node_files = additional_node_files, additional_edge_files = additional_edge_files, binary=self.binary)

        if self.task_type == 'subtoken prediction':
            graph_label_notparsed = pd.read_csv(osp.join(self.raw_dir, 'graph-label.csv.gz'), compression='gzip', header = None).values
            graph_label = [str(graph_label_notparsed[i][0]).split(' ') for i in range(len(graph_label_notparsed))]

            for i, g in enumerate(data_list):
                g.y = graph_label[i]

        else:
            if self.binary:
                graph_label = np.load(osp.join(self.raw_dir, 'graph-label.npz'))['graph_label']
            else:
                graph_label = pd.read_csv(osp.join(self.raw_dir, 'graph-label.csv.gz'), compression='gzip', header = None).values

            has_nan = np.isnan(graph_label).any()

            for i, g in enumerate(data_list):
                if 'classification' in self.task_type:
                    if has_nan:
                        g.y = torch.from_numpy(graph_label[i]).view(1,-1).to(torch.float32)
                    else:
                        g.y = torch.from_numpy(graph_label[i]).view(1,-1).to(torch.long)
                else:
                    g.y = torch.from_numpy(graph_label[i]).view(1,-1).to(torch.float32)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get(self, idx):
        data = self.data.__class__()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            start, end = slices[idx].item(), slices[idx + 1].item()
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key, item)] = slice(start, end)
            elif start + 1 == end:
                s = slices[start]
            else:
                s = slice(start, end)
            data[key] = item[s]
        data.num_nodes = self.data.__num_nodes__[idx]
        data.x = torch.zeros(data.num_nodes, dtype=torch.long)

        data2 = self.data.__class__()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            start, end = slices[idx].item(), slices[idx + 1].item()
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key, item)] = slice(start, end)
            elif start + 1 == end:
                s = slices[start]
            else:
                s = slice(start, end)
            data2[key] = item[s]
        data2.num_nodes = self.data.__num_nodes__[idx]
        data2.x = torch.zeros(data2.num_nodes, dtype=torch.long)

        if self.mode == None:
            data.num_nodes = data.x.size()[0]

        elif self.mode == 'graphcl':
            n1 = np.random.choice(3, 1)[0]
            n2 = np.random.choice(3, 1)[0]
            if n1 == 0:
                data = node_drop(data, self.aug_ratio)
            elif n1 == 1:
                data = subgraph(data, self.aug_ratio)
            elif n1 == 2:
                data = edge_pert(data, self.aug_ratio)
            data.num_nodes = data.x.size()[0]
            if n2 == 0:
                data2 = node_drop(data2, self.aug_ratio)
            elif n2 == 1:
                data2 = subgraph(data2, self.aug_ratio)
            elif n2 == 2:
                data2 = edge_pert(data2, self.aug_ratio)
            data2.num_nodes = data2.x.size()[0]

        elif self.mode == 'sampling':
            n = np.random.choice(16, 1, p=self.aug_P)[0]
            n1, n2 = n//4, n%4
            if n1 == 0:
                data = node_drop(data, self.aug_ratio)
            elif n1 == 1:
                data = subgraph(data, self.aug_ratio)
            elif n1 == 2:
                data = edge_pert(data, self.aug_ratio)
            elif n1 == 3:
                None
            data.num_nodes = data.x.size()[0]
            if n2 == 0:
                data2 = node_drop(data2, self.aug_ratio)
            elif n2 == 1:
                data2 = subgraph(data2, self.aug_ratio)
            if n2 == 2:
                data2 = edge_pert(data2, self.aug_ratio)
            if n2 == 3:
                None
            data2.num_nodes = data2.x.size()[0]

        return data, data2


def node_drop(data, aug_ratio=0.2):
    node_num = data.x.size()[0]
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num  * aug_ratio)

    idx_perm = np.random.permutation(node_num)

    idx_nondrop = idx_perm[drop_num:]
    idx_nondrop.sort()
    edge_index, edge_attr = torch_geometric.utils.subgraph(torch.tensor(idx_nondrop), data.edge_index, data.edge_attr, relabel_nodes=True, num_nodes=data.num_nodes)

    data.edge_index = edge_index
    data.edge_attr = edge_attr
    data.x = data.x[idx_nondrop]
    return data


def subgraph(data, aug_ratio=0.2):
    node_num = data.x.size()[0]
    _, edge_num = data.edge_index.size()
    sub_num = int(node_num * (1-aug_ratio))

    edge_index = data.edge_index.numpy()

    idx_sub = [np.random.randint(node_num, size=1)[0]]
    idx_neigh = set([n for n in edge_index[1][edge_index[0]==idx_sub[0]] if not n == idx_sub[0]])

    # subgraph
    while len(idx_sub) <= sub_num:
        if len(idx_neigh) == 0:
            break
        sample_node = np.random.choice(list(idx_neigh))
        idx_sub.append(sample_node)
        idx_neigh = idx_neigh.union(set([n for n in edge_index[1][edge_index[0]==idx_sub[-1]]])).difference(idx_sub)

    idx_nondrop = idx_sub
    idx_nondrop.sort()
    edge_index, edge_attr = torch_geometric.utils.subgraph(torch.tensor(idx_nondrop), data.edge_index, data.edge_attr, relabel_nodes=True, num_nodes=data.num_nodes)

    data.edge_index = edge_index
    data.edge_attr = edge_attr
    data.x = data.x[idx_nondrop]
    return data


def edge_pert(data, aug_ratio=0.2):
    node_num = data.x.size()[0]
    _, edge_num = data.edge_index.size()
    pert_num = int(edge_num * aug_ratio)

    idx_pert = np.random.choice(edge_num, (edge_num - pert_num), replace=False)

    data.edge_index = data.edge_index[:, idx_pert]
    data.edge_attr = data.edge_attr[idx_pert, :]
    return data


# no node feature so no attr_mask
def attr_mask(data, aug_ratio=0.2):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    mask_num = int(node_num * aug_ratio)

    token = data.x.float().mean(dim=0).long()
    idx_mask = np.random.choice(node_num, mask_num, replace=False)

    data.x[idx_mask] = token
    return data


from torch_geometric.data import Batch
def collate(data_list):
    batch1 = Batch.from_data_list([data[0] for data in data_list])
    batch2 = Batch.from_data_list([data[1] for data in data_list])
    return batch1, batch2

