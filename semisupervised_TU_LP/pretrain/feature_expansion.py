import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add
from torch_geometric.utils import degree, remove_self_loops, add_self_loops

class FeatureExpander(MessagePassing):

    def __init__(self, degree=True, onehot_maxdeg=0, AK=1,
                 centrality=False, remove_edges="none",
                 edge_noises_add=0, edge_noises_delete=0, group_degree=0):
        super().__init__(aggr='add')
        self.degree = degree
        self.onehot_maxdeg = onehot_maxdeg
        self.AK = AK
        self.centrality = centrality
        self.remove_edges = remove_edges
        self.edge_noises_add = edge_noises_add
        self.edge_noises_delete = edge_noises_delete
        self.group_degree = group_degree
        assert remove_edges in ["none", "nonself", "all"], remove_edges
        self.edge_norm_diag = 1e-8

    def transform(self, data):
        if data.x is None:
            data.x = torch.ones(data.num_nodes, 1)

        self.add_noise_to_edges(data)

        deg, deg_onehot = self.compute_degree(data.edge_index, data.num_nodes)
        akx = self.compute_akx(data.num_nodes, data.x, data.edge_index)
        cent = self.compute_centrality(data)
        data.x = torch.cat([data.x, deg, deg_onehot, akx, cent], dim=-1)

        self.remove_specified_edges(data)
        self.group_nodes_by_degree(data)

        return data

    def add_noise_to_edges(self, data):
        if self.edge_noises_delete > 0:
            num_edges_new = data.num_edges - int(data.num_edges * self.edge_noises_delete)
            idxs = torch.randperm(data.num_edges)[:num_edges_new]
            data.edge_index = data.edge_index[:, idxs]
        if self.edge_noises_add > 0:
            num_new_edges = int(data.num_edges * self.edge_noises_add)
            idx = torch.randint(0, data.num_nodes, (2, num_new_edges))
            data.edge_index = torch.cat([data.edge_index, idx], dim=1)

    def remove_specified_edges(self, data):
        if self.remove_edges == "all":
            data.edge_index = None
        elif self.remove_edges == "nonself":
            self_edges = torch.arange(data.num_nodes).unsqueeze(0).repeat(2, 1)
            data.edge_index = self_edges

    def group_nodes_by_degree(self, data):
        if self.group_degree > 0:
            deg, _ = self.compute_degree(data.edge_index, data.num_nodes)
            x_base = data.x
            deg_base = deg.view(-1)
            super_nodes = []

            for k in range(1, self.group_degree + 1):
                idxs = deg_base == k
                if torch.any(idxs):
                    x_grouped = x_base[idxs].mean(dim=0, keepdim=True)
                    super_nodes.append(torch.cat([torch.tensor([[idxs.sum().item()]]), x_grouped], dim=-1))
                x_base = x_base[~idxs]
                deg_base = deg_base[~idxs]

            data.x = torch.cat(super_nodes + [x_base], dim=0) if super_nodes else x_base

    def compute_degree(self, edge_index, num_nodes):
        row, _ = edge_index
        deg = degree(row, num_nodes).view(-1, 1)

        if self.onehot_maxdeg > 0:
            deg_onehot = F.one_hot(deg.clamp(max=self.onehot_maxdeg).long().view(-1), num_classes=self.onehot_maxdeg + 1).float()
        else:
            deg_onehot = torch.zeros(num_nodes, 0)

        if not self.degree:
            deg = torch.zeros(num_nodes, 1)

        return deg, deg_onehot

    def compute_centrality(self, data):
        if not self.centrality:
            return torch.zeros(data.num_nodes, 0)

        G = nx.Graph(data.edge_index.cpu().numpy().T)
        centrality_measures = [
            nx.closeness_centrality(G),
            nx.betweenness_centrality(G),
            nx.pagerank_numpy(G)
        ]
        centrality = torch.tensor([[cent[i] for cent in centrality_measures] for i in range(data.num_nodes)])
        return centrality

    def compute_akx(self, num_nodes, x, edge_index, edge_weight=None):
        if self.AK <= 0:
            return torch.zeros(num_nodes, 0)

        edge_index, norm = self.norm(edge_index, num_nodes, edge_weight, self.edge_norm_diag)

        xs = []
        for _ in range(self.AK):
            x = self.propagate(edge_index, x=x, norm=norm)
            xs.append(x)
        return torch.cat(xs, dim=-1)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, diag_val=1e-8):
        edge_weight = edge_weight if edge_weight is not None else torch.ones(edge_index.size(1), device=edge_index.device)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        loop_weight = torch.full((num_nodes,), diag_val, device=edge_weight.device)
        edge_weight = torch.cat([edge_weight, loop_weight])

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
