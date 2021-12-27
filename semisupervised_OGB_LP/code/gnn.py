import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform
import torch.nn as nn

from conv import GNN_node, GNN_node_Virtualnode

from torch_scatter import scatter_mean


class GNN(torch.nn.Module):

    def __init__(self, num_vocab, max_seq_len, node_encoder, num_layer = 5, emb_dim = 300, 
                    gnn_type = 'gin', virtual_node = True, residual = False, drop_ratio = 0.5, JK = "last", graph_pooling = "mean"):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_vocab = num_vocab
        self.max_seq_len = max_seq_len
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, node_encoder, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)
        else:
            self.gnn_node = GNN_node(num_layer, emb_dim, node_encoder, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)


        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        self.graph_pred_linear_list = torch.nn.ModuleList()

        if graph_pooling == "set2set":
            for i in range(max_seq_len):
                 self.graph_pred_linear_list.append(torch.nn.Linear(2*emb_dim, self.num_vocab))

        else:
            for i in range(max_seq_len):
                 self.graph_pred_linear_list.append(torch.nn.Linear(emb_dim, self.num_vocab))

        self.proj_head = torch.nn.Sequential(torch.nn.Linear(self.emb_dim, self.emb_dim), torch.nn.ReLU(inplace=True), torch.nn.Linear(self.emb_dim, self.emb_dim))

    def forward(self, batched_data):
        '''
            Return:
                A list of predictions.
                i-th element represents prediction at i-th position of the sequence.
        '''

        h_node = self.gnn_node(batched_data)

        h_graph = self.pool(h_node, batched_data.batch)

        pred_list = []
        # for i in range(self.max_seq_len):
        #     pred_list.append(self.graph_pred_mlp_list[i](h_graph))

        for i in range(self.max_seq_len):
            pred_list.append(self.graph_pred_linear_list[i](h_graph))

        return pred_list

    def forward_cl(self, batched_data):
        h_node = self.gnn_node(batched_data)

        h_graph = self.pool(h_node, batched_data.batch)
        z = self.proj_head(h_graph)
        return z

    def loss_cl(self, x1, x2, mean=True):
        T = 0.5
        batch_size, _ = x1.size()

        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss)
        if mean:
            loss = loss.mean()
        return loss


class vgae(nn.Module):
    def __init__(self, num_vocab, max_seq_len, node_encoder, num_layer = 5, emb_dim = 300, 
                    gnn_type = 'gin', virtual_node = True, residual = False, drop_ratio = 0.5, JK = "last", graph_pooling = "mean"):
        super(vgae, self).__init__()
        self.encoder = GNN_node(num_layer, emb_dim, node_encoder, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)
        self.encoder_mean = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU(inplace=True), nn.Linear(emb_dim, emb_dim))
        # make sure std is positive
        self.encoder_std = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU(inplace=True), nn.Linear(emb_dim, emb_dim), nn.Softplus())
        self.decoder_edge = nn.Sequential(nn.ReLU(), nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, 1))

        self.bceloss = nn.BCELoss(reduction='none')
        self.pool = global_mean_pool
        self.add_pool = global_add_pool
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

    def forward_encoder(self, batched_data):
        x = self.encoder(batched_data)
        x_mean = self.encoder_mean(x)
        x_std = self.encoder_std(x)
        gaussian_noise = torch.randn(x_mean.shape).to(x.device)
        x = gaussian_noise * x_std + x_mean
        return x, x_mean, x_std

    def forward_decoder(self, x, edge_index, edge_index_neg):
        eleWise_mul = x[edge_index[0]] * x[edge_index[1]]

        edge_pos = self.sigmoid( self.decoder_edge(eleWise_mul) ).squeeze()
        edge_neg = self.sigmoid( self.decoder_edge(x[edge_index_neg[0]] * x[edge_index_neg[1]]) ).squeeze()
        return edge_pos, edge_neg

    def loss_vgae(self, edge_pos_pred, edge_neg_pred, edge_index_batch, edge_index_neg_batch, x_mean, x_std, batch, reward=None):
        # evaluate p(A|Z)
        num_edge = edge_pos_pred.shape[0]
        x_std = x_std + 1e-10 # in case of inf

        loss_edge_pos = self.bceloss(edge_pos_pred, torch.ones(edge_pos_pred.shape).to(edge_pos_pred.device))
        loss_edge_neg = self.bceloss(edge_neg_pred, torch.zeros(edge_neg_pred.shape).to(edge_neg_pred.device))

        loss_pos = self.pool(loss_edge_pos, edge_index_batch)
        loss_neg = self.pool(loss_edge_neg, edge_index_neg_batch)
        loss_rec = loss_pos + loss_neg
        if not reward is None:
            loss_rec = loss_rec * reward

        # evaluate p(Z|X,A)
        kl_divergence = - 0.5 * (1 + 2 * torch.log(x_std) - x_mean**2 - x_std**2).sum(dim=1)
        kl_ones = torch.ones(kl_divergence.shape).to(kl_divergence.device)
        kl_divergence = self.pool(kl_divergence, batch)
        kl_double_norm = 1 / self.add_pool(kl_ones, batch)
        kl_divergence = kl_divergence * kl_double_norm

        loss = (loss_rec + kl_divergence).mean()

        '''
        # link prediction for sanity check
        from sklearn.metrics import roc_auc_score
        from sklearn.metrics import average_precision_score
        print(roc_auc_score(edge_attr.cpu().numpy(), edge_attr_pred.detach().cpu().numpy()), average_precision_score(edge_attr.cpu().numpy(), edge_attr_pred.detach().cpu().numpy()))
        '''
        return loss

    def generate(self, data):
        x, _, _ = self.forward_encoder(data)
        eleWise_mul = torch.einsum('nd,md->nmd', x, x)

        # calculate softmax probability
        prob = self.decoder_edge(eleWise_mul).squeeze()
        prob = torch.exp(prob)
        prob[torch.isinf(prob)] = 1e10
        prob[list(range(x.shape[0])), list(range(x.shape[0]))] = 0
        prob = torch.einsum('nm,n->nm', prob, 1/prob.sum(dim=1))

        # sparsify
        prob[prob<1e-1] = 0
        prob[prob.sum(dim=1)==0] = 1
        prob[list(range(x.shape[0])), list(range(x.shape[0]))] = 0
        prob = torch.einsum('nm,n->nm', prob, 1/prob.sum(dim=1))

        return prob


if __name__ == '__main__':
    pass

