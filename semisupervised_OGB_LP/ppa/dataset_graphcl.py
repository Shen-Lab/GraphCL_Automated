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
import torch_geometric.utils as tg_utils


class PygGraphPropPredDataset(InMemoryDataset):
    def __init__(self, name, root = 'dataset', transform=None, pre_transform = None, meta_dict = None):
        '''
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - transform, pre_transform (optional): transform/pre-transform graph objects

            - meta_dict: dictionary that stores all the meta-information about data. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        ''' 

        self.name = name ## original name, e.g., ogbg-molhiv
        self.set_augMode('none')
        self.set_augStrength(0.2)
        self.augmentations = [self.node_drop, self.subgraph, self.edge_pert, lambda x:x]
        self.set_generator(None, None)
        
        if meta_dict is None:
            self.dir_name = '_'.join(name.split('-')) 
            
            # check if previously-downloaded folder exists.
            # If so, use that one.
            if osp.exists(osp.join(root, self.dir_name + '_pyg')):
                self.dir_name = self.dir_name + '_pyg'

            self.original_root = root
            self.root = osp.join(root, self.dir_name)
            
            # master = pd.read_csv(os.path.join(os.path.dirname(__file__), 'master.csv'), index_col = 0)
            master = pd.read_csv(os.path.join('/scratch/user/yuning.you/.conda/envs/graphcl/lib/python3.7/site-packages/ogb/graphproppred', 'master.csv'), index_col = 0)
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

    def set_augMode(self, aug_mode):
        self.aug_mode = aug_mode
    
    def set_augStrength(self, aug_strength):
        self.aug_strength = aug_strength

    def node_drop(self, data):
        node_num = data.x.size()[0]
        _, edge_num = data.edge_index.size()
        drop_num = int(node_num  * self.aug_strength)

        idx_perm = np.random.permutation(node_num)

        idx_nondrop = idx_perm[drop_num:]
        idx_nondrop.sort()
        edge_index, edge_attr = torch_geometric.utils.subgraph(torch.tensor(idx_nondrop), data.edge_index, data.edge_attr, relabel_nodes=True, num_nodes=data.num_nodes)

        data.edge_index = edge_index
        data.edge_attr = edge_attr
        data.x = data.x[idx_nondrop]
        data.__num_nodes__ = data.x.shape[0]
        return data

    def subgraph(self, data):
        node_num = data.x.size()[0]
        _, edge_num = data.edge_index.size()
        sub_num = int(node_num * (1-self.aug_strength))

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
        data.__num_nodes__ = data.x.shape[0]
        return data

    def edge_pert(self, data):
        node_num = data.x.size()[0]
        _, edge_num = data.edge_index.size()
        pert_num = int(edge_num * self.aug_strength)
        
        # delete edges
        idx_drop = np.random.choice(edge_num, (edge_num - pert_num), replace=False)
        edge_index = data.edge_index[:, idx_drop]
        edge_attr = data.edge_attr[idx_drop]

        # add edges
        adj = torch.ones((node_num, node_num))
        adj[edge_index[0], edge_index[1]] = 0
        edge_index_nonexist = adj.nonzero(as_tuple=False).t()
        idx_add = np.random.choice(edge_index_nonexist.shape[1], pert_num, replace=False)
        edge_index_add = edge_index_nonexist[:, idx_add]
        # random 7-dim edge_attr in [0,1], for details please refer to https://github.com/snap-stanford/pretrain-gnns/issues/30
        edge_attr_add = torch.tensor( np.random.rand(edge_index_add.shape[1], 7), dtype=torch.float32 )
        edge_index = torch.cat((edge_index, edge_index_add), dim=1)
        edge_attr = torch.cat((edge_attr, edge_attr_add), dim=0)

        data.edge_index = edge_index
        data.edge_attr = edge_attr
        return data

    # no node feature so no attr_mask
    def attr_mask(data):
        pass

    def set_generator(self, generator1, generator2):
        self.generators = [generator1, generator2]

    def generator_generate(self, data, n_generator=1):
        with torch.no_grad():
            prob, edge_attr_pred = self.generators[n_generator-1].generate(data)
            prob = prob.numpy()

        node_num = data.x.size()[0]
        idx_sub = np.random.choice(node_num, 8, replace=False).tolist()
        idx_neigh = idx_sub
        idx_edge = []
        # weighted random walk
        for _ in range(10): # limit the walk within 10 steps
            _idx_neigh = []
            # online sampling based on p(v|v_c)
            for n in idx_neigh:
                try:
                    idx_neigh_n = np.random.choice(node_num, 1, p=prob[n])
                except:
                    idx_neigh_n = np.random.choice(node_num, 1)
                _idx_neigh += [idx_neigh_n[0]]
                idx_edge += [(n, idx_neigh_n[0]), (idx_neigh_n[0], n)]
            # get the new neighbors
            idx_neigh = _idx_neigh
            idx_sub += idx_neigh
        idx_sub = list(set(idx_sub))

        idx_sub.sort()
        idx_edge = list(set(idx_edge))
        idx_edge = torch.tensor(idx_edge).t()
        edge_attr = edge_attr_pred[idx_edge[0], idx_edge[1]]
        edge_index, edge_attr = tg_utils.subgraph(idx_sub, idx_edge, edge_attr=edge_attr, relabel_nodes=True, num_nodes=node_num)
        # edge_index, edge_attr = tg_utils.subgraph(idx_sub, idx_edge, edge_attr=edge_attr, relabel_nodes=False, num_nodes=node_num) # for visualization
        # data.idx_sub = idx_sub  # for visualization

        data.x = data.x[idx_sub]
        data.edge_index = edge_index
        data.edge_attr = edge_attr
        data.__num_nodes__ = data.x.shape[0]
        return data

    def get(self, idx):
        data, data1, data2 = self.data.__class__(), self.data.__class__(), self.data.__class__()
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
            data[key], data1[key], data2[key] = item[s], item[s], item[s]
        data.num_nodes, data1.num_nodes, data2.num_nodes = self.data.__num_nodes__[idx], self.data.__num_nodes__[idx], self.data.__num_nodes__[idx]
        data.x, data1.x, data2.x = torch.zeros(data.num_nodes, dtype=torch.long), torch.zeros(data.num_nodes, dtype=torch.long), torch.zeros(data.num_nodes, dtype=torch.long)

        if self.aug_mode == 'none':
            n_aug1, n_aug2 = 3, 3
            data1 = self.augmentations[n_aug1](data1)
            data2 = self.augmentations[n_aug2](data2)
        elif self.aug_mode == 'uniform':
            n_aug = np.random.choice(16, 1)[0]
            n_aug1, n_aug2 = n_aug//4, n_aug%4
            data1 = self.augmentations[n_aug1](data1)
            data2 = self.augmentations[n_aug2](data2)

        # generative model
        elif self.aug_mode == 'generative':
            data1 = self.generator_generate(data1, 1)
            data2 = self.generator_generate(data2, 2)
            edge_index_neg = tg_utils.negative_sampling(data.edge_index, num_nodes=data.x.shape[0])
            data.edge_index_neg = edge_index_neg

        return data, data1, data2


from torch_geometric.data import Batch
def collate(data_list):
    batch = Batch.from_data_list([data[0] for data in data_list], follow_batch=['edge_index', 'edge_index_neg'])
    batch1 = Batch.from_data_list([data[1] for data in data_list])
    batch2 = Batch.from_data_list([data[2] for data in data_list])
    return batch, batch1, batch2

