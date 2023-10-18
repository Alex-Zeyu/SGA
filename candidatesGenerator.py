import os
import torch
from torch_geometric import seed_everything
from torch_geometric.nn import SignedGCN
from data_proc import load_graph
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d','--dataset', type=str, default='bitcoin-alpha',
                      choices=['bitcoin-alpha', 'Epinions', 'bitcoin-otc', 'Slashdot', 'wiki-elec', 'wiki-RfA'])
parser.add_argument('-n','--num', type=int, default=1, choices=[1,2,3,4,5])
parser.add_argument('-la','--num_layers', type=int, default=2)
parser.add_argument('-ch','--channels', type=int, default=64)
parser.add_argument('-s','--seed', type=float, default=2023)

args = parser.parse_args()

network = args.dataset
num=args.num
train_file_path='{0}/trains/{0}-train-{1}.csv'.format(network,num)  # path_to_the_train_file_folder
test_file_path='{0}/tests/{0}-test-{1}.csv'.format(network,num)  # path_to_the_test_csv_file

if not os.path.exists('{0}/candidates_{1}'.format(network,num)):
    os.makedirs('{0}/candidates_{1}'.format(network,num))

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def run(train_file_name: str):
    # load graph
    g_train = load_graph(train_file_path).to(device)
    g_test = load_graph(test_file_path).to(device)

    num_nodes = (torch.concat(
        [g_train[:2], g_test[:2]], dim=-1).max() + 1).item()  # +1

    # pos & neg edge index
    train_pos_edge_index, train_neg_edge_index = g_train[:2,
                                                 g_train[2] > 0], g_train[:2, g_train[2] < 0]

    # model
    seed_everything(args.seed)
    model = SignedGCN(args.channels, args.channels,
                      num_layers=args.num_layers, lamb=5).to(device)
    model.load_state_dict(torch.load('{0}/models/{1}.pt'.format(network, train_file_name[:-4])))
    model = model.to(device)

    x = model.create_spectral_features(
        train_pos_edge_index, train_neg_edge_index, num_nodes=num_nodes)

    # model evaluation
    model.eval()
    with torch.no_grad():
        z = model(x, train_pos_edge_index, train_neg_edge_index)
        size = len(g_train[0])
        candidates_score_pos = torch.empty(0, 4).to(device)
        candidates_score_neg = torch.empty(0, 4).to(device)
        for i in range(0,num_nodes,100):
            candidate_now = torch.triu_indices(100, num_nodes, offset=i + 1).to(device)
            candidate_now[0] += i

            candidate_now_score = model.discriminate(z, candidate_now)[:, :2]
            candidate_now_score = torch.concat([candidate_now.T, candidate_now_score], dim=1)
            candidates_score_pos = torch.concat([candidates_score_pos, candidate_now_score], dim=0)
            candidates_score_neg = torch.concat([candidates_score_neg, candidate_now_score], dim=0)
            candidates_score_pos = candidates_score_pos[torch.argsort(-candidates_score_pos[:, 2])][:size]
            candidates_score_neg = candidates_score_neg[torch.argsort(-candidates_score_neg[:, 3])][:size]
        del_score_pos = model.discriminate(z, train_pos_edge_index)
        del_score_pos = torch.concat([train_pos_edge_index.T, del_score_pos], dim=1)
        del_score_neg = model.discriminate(z, train_neg_edge_index)
        del_score_neg = torch.concat([train_neg_edge_index.T, del_score_neg], dim=1)
        
    candidates_score_pos = candidates_score_pos.detach().cpu().numpy()
    candidates_score_neg = candidates_score_neg.detach().cpu().numpy()
    del_score_pos = del_score_pos.detach().cpu().numpy()
    del_score_neg = del_score_neg.detach().cpu().numpy()
    np.savetxt('{0}/candidates_{1}/pos_add.csv'.format(network,num), candidates_score_pos, fmt='%.2f', delimiter=',')
    np.savetxt('{0}/candidates_{1}/neg_add.csv'.format(network,num), candidates_score_neg, fmt='%.2f', delimiter=',')

    del_score_pos = del_score_pos[np.argsort(del_score_pos[:, 2])]
    np.savetxt('{0}/candidates_{1}/pos_del.csv'.format(network,num), del_score_pos, fmt='%.2f', delimiter=',')

    del_score_neg = del_score_neg[np.argsort(del_score_neg[:, 3])]
    np.savetxt('{0}/candidates_{1}/neg_del.csv'.format(network,num), del_score_neg, fmt='%.2f', delimiter=',')


if __name__ == '__main__':
    train_file_name='{0}-train-{1}.csv'.format(network,num)
    run(train_file_name)

