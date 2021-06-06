import re
import argparse
from datasets import get_dataset
from res_gcn import ResGCN_graphcl, vgae_encoder, vgae_decoder

import experiment_joao


str2bool = lambda x: x.lower() == "true"
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default="datasets")
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--n_layers_feat', type=int, default=1)
parser.add_argument('--n_layers_conv', type=int, default=3)
parser.add_argument('--n_layers_fc', type=int, default=2)
parser.add_argument('--hidden', type=int, default=128)
parser.add_argument('--global_pool', type=str, default="sum")
parser.add_argument('--skip_connection', type=str2bool, default=False)
parser.add_argument('--res_branch', type=str, default="BNConvReLU")
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--edge_norm', type=str2bool, default=True)

parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=100)

parser.add_argument('--dataset', type=str, default="NCI1")
parser.add_argument('--aug_mode', type=str, default="sample")
parser.add_argument('--aug_ratio', type=float, default=0.2)
parser.add_argument('--suffix', type=int, default=0)

parser.add_argument('--model', type=str, default='joao')
parser.add_argument('--gamma_joao', type=float, default=0.1)
args = parser.parse_args()


def create_n_filter_triple(dataset, feat_str, net, gfn_add_ak3=False,
                            gfn_reall=True, reddit_odeg10=False,
                            dd_odeg10_ak1=False):
    # Add ak3 for GFN.
    if gfn_add_ak3 and 'GFN' in net:
        feat_str += '+ak3'
    # Remove edges for GFN.
    if gfn_reall and 'GFN' in net:
        feat_str += '+reall'
    # Replace degree feats for REDDIT datasets (less redundancy, faster).
    if reddit_odeg10 and dataset in [
            'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K']:
        feat_str = feat_str.replace('odeg100', 'odeg10')
    # Replace degree and akx feats for dd (less redundancy, faster).
    if dd_odeg10_ak1 and dataset in ['DD']:
        feat_str = feat_str.replace('odeg100', 'odeg10')
        feat_str = feat_str.replace('ak3', 'ak1')
    return dataset, feat_str, net


def get_model_with_default_configs(model_name,
                                   num_feat_layers=args.n_layers_feat,
                                   num_conv_layers=args.n_layers_conv,
                                   num_fc_layers=args.n_layers_fc,
                                   residual=args.skip_connection,
                                   hidden=args.hidden):
    # More default settings.
    res_branch = args.res_branch
    global_pool = args.global_pool
    dropout = args.dropout
    edge_norm = args.edge_norm

    # modify default architecture when needed
    if model_name.find('_') > 0:
        num_conv_layers_ = re.findall('_conv(\d+)', model_name)
        if len(num_conv_layers_) == 1:
            num_conv_layers = int(num_conv_layers_[0])
            print('[INFO] num_conv_layers set to {} as in {}'.format(
                num_conv_layers, model_name))
        num_fc_layers_ = re.findall('_fc(\d+)', model_name)
        if len(num_fc_layers_) == 1:
            num_fc_layers = int(num_fc_layers_[0])
            print('[INFO] num_fc_layers set to {} as in {}'.format(
                num_fc_layers, model_name))
        residual_ = re.findall('_res(\d+)', model_name)
        if len(residual_) == 1:
            residual = bool(int(residual_[0]))
            print('[INFO] residual set to {} as in {}'.format(
                residual, model_name))
        gating = re.findall('_gating', model_name)
        if len(gating) == 1:
            global_pool += "_gating"
            print('[INFO] add gating to global_pool {} as in {}'.format(
                global_pool, model_name))
        dropout_ = re.findall('_drop([\.\d]+)', model_name)
        if len(dropout_) == 1:
            dropout = float(dropout_[0])
            print('[INFO] dropout set to {} as in {}'.format(
                dropout, model_name))
        hidden_ = re.findall('_dim(\d+)', model_name)
        if len(hidden_) == 1:
            hidden = int(hidden_[0])
            print('[INFO] hidden set to {} as in {}'.format(
                hidden, model_name))

    if model_name == 'ResGCN_graphcl':
        def foo(dataset):
            return ResGCN_graphcl(dataset=dataset, hidden=hidden, num_feat_layers=num_feat_layers, num_conv_layers=num_conv_layers,
                          num_fc_layers=num_fc_layers, gfn=False, collapse=False,
                          residual=residual, res_branch=res_branch,
                          global_pool=global_pool, dropout=dropout,
                          edge_norm=edge_norm)

    else:
        raise ValueError("Unknown model {}".format(model_name))
    return foo


def run_experiment_graphcl(dataset_feat_net_triple
                               =create_n_filter_triple(args.dataset, 'deg+odeg100', 'ResGCN_graphcl', gfn_add_ak3=True, reddit_odeg10=True, dd_odeg10_ak1=True),
                           get_model=get_model_with_default_configs):

    dataset_name, feat_str, net = dataset_feat_net_triple
    dataset = get_dataset(
        dataset_name, sparse=True, feat_str=feat_str, root=args.data_root)
    model_func = get_model(net)

    experiment_joao.experiment(dataset, model_func, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, weight_decay=0, dataset_name=dataset_name, aug_mode=args.aug_mode, aug_ratio=args.aug_ratio, suffix=args.suffix, gamma_joao=args.gamma_joao)


if __name__ == '__main__':
   print(args)

   run_experiment_graphcl()

