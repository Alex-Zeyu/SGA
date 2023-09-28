import itertools
import math
import os
import os.path as osp
import torch
from dotmap import DotMap
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from modules import MyGSGNN
from tqdm import tqdm
from data_proc import load_graph, collect_results
import numpy as np
import utils
from torch_geometric import seed_everything

network='Epinions'

args = DotMap(
    train_file_folder='{}/trains_Augmentation'.format(network), # path_to_the_train_file_folder
    test_file_path='{0}/tests/{0}-test-5.csv'.format(network), # path_to_the_test_csv_file
    num_layers=2,
    channels=64,
    lr=1e-2,
    epochs=3000,
    seed=2023,
    k=4,
    weight_decay=0.0001,
    L=0.01,
    step_size=20,
    gamma=0.99
)

device = 'cuda:6'
print(device)

# device = 'cpu'


def run(train_file_name: str):
    # load graph
    g_train = load_graph(
        osp.join(args.train_file_folder, train_file_name)).to(device)
    g_test = load_graph(args.test_file_path).to(device)

    N = (torch.concat(
        [g_train[:2], g_test[:2]], dim=-1).max() + 1).item() #+1

    # pos & neg edge index
    train_pos_edge_index, train_neg_edge_index = g_train[:2,
                                                         g_train[2] > 0], g_train[:2, g_train[2] < 0]
    test_pos_edge_index, test_neg_edge_index = g_test[:2,
                                                      g_test[2] > 0], g_test[:2, g_test[2] < 0]
    seed_everything(args.seed)
    # model
    model = MyGSGNN(64,64,16,k=args.k).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    x = utils.arg_func.create_features(args, train_pos_edge_index, train_neg_edge_index, N).to(device)
    utils.data.setup_seed(args.seed)

    def train():
        model.train()
        optimizer.zero_grad()
        z = model(x, train_pos_edge_index, train_neg_edge_index)
        loss = model.loss(z, train_pos_edge_index , train_neg_edge_index)
        loss.backward()
        optimizer.step()
        scheduler.step()
        return loss.item()

    def test(z: torch.Tensor,
             pos_edge_index: torch.Tensor,
             neg_edge_index: torch.Tensor) -> dict:
        from sklearn.metrics import f1_score, roc_auc_score

        with torch.no_grad():
            pos_p = F.softmax(model.discriminate(z, test_pos_edge_index),dim=1)
            neg_p = F.softmax(model.discriminate(z, test_neg_edge_index),dim=1)
        pred_p = torch.cat([pos_p[:, 0], neg_p[:, 0]], dim=0).cpu()
        pos_p = pos_p[:, :2].max(dim=1)[1]
        neg_p = neg_p[:, :2].max(dim=1)[1]
        pred = (1 - torch.cat([pos_p, neg_p])).cpu()
        y = torch.cat(
            [pred.new_ones((pos_p.size(0))),
             pred.new_zeros(neg_p.size(0))])

        pred, y = pred.numpy(), y.numpy()

        return {
            # 'auc_1':roc_auc_score(y, pred)
            'auc': roc_auc_score(y, pred_p),
            'f1': f1_score(y, pred, average='binary') if pred.sum() > 0 else 0,
            'f1micro': f1_score(y, pred, average='micro') if pred.sum() > 0 else 0,
            'f1macro': f1_score(y, pred, average='macro') if pred.sum() > 0 else 0
        }

    res_dict = {}

    for epoch in tqdm(range(args.epochs)):
        loss = train()
    # model.load_state_dict(torch.load('{0}/models/{1}.pt'.format(network,train_file_name[:-4])))

    # torch.save(model.state_dict(), '{0}/models/{1}.pt'.format(network,train_file_name[:-4]))

    model.eval()
    with torch.no_grad():
        z = model(x, train_pos_edge_index, train_neg_edge_index)

    # np.savetxt('{0}/embeddings/{1}-embedding.csv'.format(network,train_file_name[:-4]), z.detach().cpu().numpy(), fmt='%.2f', delimiter=',')

    res_dict.update(test(z, test_pos_edge_index, test_neg_edge_index))
    res_dict['train_file'] = train_file_name
    return res_dict


if __name__ == '__main__':
    train_files = os.listdir(args.train_file_folder)
    results = [run(train_file) for train_file in train_files]
    print(results)
    collect_results(results, save_res=False, suffix='gsgnn_aug')
