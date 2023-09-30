import itertools
import math
import os
import os.path as osp
import random

import torch
import pandas as pd
import networkx as nx
from dotmap import DotMap
from torch_geometric import seed_everything
from torch_geometric.nn import SignedGCN
from tqdm import tqdm
from data_proc import load_graph, collect_results
import numpy as np
from balanceDegree import edgesBalanceDegree_sp

network='Slashdot'

args = DotMap(
    train_file_folder='{}/trains_Augmentation'.format(network), # path_to_the_train_file_folder
    test_file_path='{0}/tests/{0}-test-5.csv'.format(network), # path_to_the_test_csv_file
    num_layers=2,
    channels=64,
    lr=1e-2,
    epochs=300,
    seed=2023,
    divisions_number=30,
    initial=0.5
)

device = 'cuda:5'


def run(train_file_name: str):
    # load graph
    g_train = load_graph(
        osp.join(args.train_file_folder, train_file_name)).to(device)
    g_test = load_graph(args.test_file_path).to(device)

    num_nodes = (torch.concat(
        [g_train[:2], g_test[:2]], dim=-1).max() + 1).item()  # +1

    # 计算平衡度
    edges = g_train.cpu().numpy()
    m = len(edges[0])
    balanceDegree = edgesBalanceDegree_sp(edges)
    balanceDegree=torch.tensor(balanceDegree)
    # probabilities=torch.softmax(balanceDegree,dim=0).numpy()
    # chooseOrder=np.random.choice(np.arange(m), size=m, replace=False, p=probabilities)
    chooseOrder = (-balanceDegree).argsort()

    # pos & neg edge index
    train_pos_edge_index, train_neg_edge_index = g_train[:2,
                                                 g_train[2] > 0], g_train[:2, g_train[2] < 0]
    test_pos_edge_index, test_neg_edge_index = g_test[:2,
                                               g_test[2] > 0], g_test[:2, g_test[2] < 0]

    # model
    seed_everything(args.seed)
    model = SignedGCN(args.channels, args.channels,
                      num_layers=args.num_layers, lamb=5).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=5e-4)
    x = torch.from_numpy(np.random.rand(num_nodes, 64).astype(np.float32)).to(device)

    initial_num = round(m * args.initial)
    epoch_div=args.epochs // args.divisions_number

    def train(epoch):
        # 获取当前epoch的训练集
        epoch_num=round((epoch//epoch_div+1)*((m-initial_num)/args.divisions_number))
        num=epoch_num+initial_num
        train_edge_now=torch.tensor(edges[:,chooseOrder[:num]]).to(device)
        train_pos_edge_now = train_edge_now[:2, train_edge_now[2] > 0]
        train_neg_edge_now = train_edge_now[:2, train_edge_now[2] < 0]
        model.train()
        optimizer.zero_grad()
        z = model(x, train_pos_edge_index, train_neg_edge_index)
        loss = model.loss(z, train_pos_edge_now, train_neg_edge_now)
        loss.backward()
        optimizer.step()
        return loss.item()

    def test(z: torch.Tensor,
             pos_edge_index: torch.Tensor,
             neg_edge_index: torch.Tensor) -> dict:
        from sklearn.metrics import f1_score, roc_auc_score

        with torch.no_grad():
            pos_p = model.discriminate(z, pos_edge_index)[:, :2].max(dim=1)[1]
            neg_p = model.discriminate(z, neg_edge_index)[:, :2].max(dim=1)[1]
        pred = (1 - torch.cat([pos_p, neg_p])).cpu()
        y = torch.cat(
            [pred.new_ones((pos_p.size(0))),
             pred.new_zeros(neg_p.size(0))])
        pred, y = pred.numpy(), y.numpy()

        return {
            'auc': roc_auc_score(y, pred),
            'f1': f1_score(y, pred, average='binary') if pred.sum() > 0 else 0,
            'f1micro': f1_score(y, pred, average='micro') if pred.sum() > 0 else 0,
            'f1macro': f1_score(y, pred, average='macro') if pred.sum() > 0 else 0
        }

    res_dict = {}

    for epoch in tqdm(range(args.epochs)):
        loss = train(epoch)
    # model.load_state_dict(torch.load('bitcoin-otc1/bitcoin-otc1.pt'))
    # torch.save(model.state_dict(), 'slashdot.pt')
    # model evaluation
    model.eval()
    with torch.no_grad():
        z = model(x, train_pos_edge_index, train_neg_edge_index)
    # np.savetxt('slashdot-embedding.csv', z.detach().cpu().numpy(), fmt='%.2f', delimiter=',')
    res_dict.update(test(z, test_pos_edge_index, test_neg_edge_index))

    res_dict['train_file'] = train_file_name
    return res_dict


if __name__ == '__main__':
    train_files = os.listdir(args.train_file_folder)
    results = [run(train_file) for train_file in train_files]
    print(results)
    collect_results(results, save_res=True, suffix='sgcn_random_initial')
