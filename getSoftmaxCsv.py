import os
import os.path as osp
import torch
import pandas as pd
from dotmap import DotMap
from torch_geometric import seed_everything
from torch_geometric.nn import SignedGCN
from tqdm import tqdm
from data_proc import load_graph, collect_results
import numpy as np

network = 'Slashdot'
args = DotMap(
    train_file_folder='{}/trains'.format(network),  # path_to_the_train_file_folder
    test_file_path='{0}/tests/{0}-test-6.csv'.format(network),  # path_to_the_test_csv_file
    num_layers=2,
    channels=64,
    lr=1e-2,
    epochs=300,
    seed=2023
)

device = 'cuda:0'
# device = 'cpu'


def run(train_file_name: str):
    # load graph
    g_train = load_graph(
        osp.join(args.train_file_folder, train_file_name)).to(device)
    g_test = load_graph(args.test_file_path).to(device)

    num_nodes = (torch.concat(
        [g_train[:2], g_test[:2]], dim=-1).max() + 1).item()  # +1

    # pos & neg edge index
    train_pos_edge_index, train_neg_edge_index = g_train[:2,
                                                 g_train[2] > 0], g_train[:2, g_train[2] < 0]
    test_pos_edge_index, test_neg_edge_index = g_test[:2,
                                               g_test[2] > 0], g_test[:2, g_test[2] < 0]

    # model
    seed_everything(args.seed)
    model = SignedGCN(args.channels, args.channels,
                      num_layers=args.num_layers, lamb=5).to(device)
    model.load_state_dict(torch.load('{0}/models/{1}.pt'.format(network, train_file_name[:-4])))

    x = model.create_spectral_features(
        train_pos_edge_index, train_neg_edge_index, num_nodes=num_nodes)

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
    np.savetxt('{0}/pos_add.csv'.format(network), candidates_score_pos, fmt='%.2f', delimiter=',')
    np.savetxt('{0}/neg_add.csv'.format(network), candidates_score_neg, fmt='%.2f', delimiter=',')

    del_score_pos = del_score_pos[np.argsort(del_score_pos[:, 2])]
    np.savetxt('{0}/pos_del.csv'.format(network), del_score_pos, fmt='%.2f', delimiter=',')
    # del_score_pos = del_score_pos[np.argsort(-del_score_pos[:, 3])]
    # np.savetxt('{0}/pos_neg.csv'.format(network), del_score_pos, fmt='%.2f', delimiter=',')
    # del_score_pos = del_score_pos[np.argsort(del_score_pos[:, 2])]
    # np.savetxt('{0}/pos_r.csv'.format(network), del_score_pos, fmt='%.2f', delimiter=',')
    del_score_neg = del_score_neg[np.argsort(del_score_neg[:, 3])]
    np.savetxt('{0}/neg_del.csv'.format(network), del_score_neg, fmt='%.2f', delimiter=',')
    # del_score_neg = del_score_neg[np.argsort(-del_score_neg[:, 2])]
    # np.savetxt('{0}/neg_pos.csv'.format(network), del_score_neg, fmt='%.2f', delimiter=',')
    # del_score_neg = del_score_neg[np.argsort(del_score_neg[:, 3])]
    # np.savetxt('{0}/neg_r.csv'.format(network), del_score_neg, fmt='%.2f', delimiter=',')

    res_dict.update(test(z, test_pos_edge_index, test_neg_edge_index))

    res_dict['train_file'] = train_file_name
    return res_dict


if __name__ == '__main__':
    train_files = os.listdir(args.train_file_folder)
    results = [run(train_file) for train_file in train_files]
    print(results)
    collect_results(results, save_res=False, suffix='dataset_4_bitcoin-otc_aug')
