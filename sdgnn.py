import itertools
import math
import os
import os.path as osp
import torch
from dotmap import DotMap
from torch_geometric import seed_everything
from torch_geometric_signed_directed.nn.signed import SDGNN
from torch_geometric_signed_directed.utils.signed import link_sign_prediction_logistic_function
from tqdm import tqdm
from data_proc import load_graph, collect_results
import numpy as np

network='Epinions'

args = DotMap(
    train_file_folder='{}/trains_Augmentation'.format(network), # path_to_the_train_file_folder
    test_file_path='{0}/tests/{0}-test-1.csv'.format(network), # path_to_the_test_csv_file
    num_layers=2,
    channels=20,
    lr=1e-2,
    epochs=100,
    seed=2023
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def run(train_file_name: str):
    # load graph
    g_train = load_graph(
        osp.join(args.train_file_folder, train_file_name)).to(device)
    g_test = load_graph(args.test_file_path).to(device)

    num_nodes = (torch.concat(
        [g_train[:2], g_test[:2]], dim=-1).max() + 1).item() #+1


    # model
    seed_everything(args.seed)
    model = SDGNN(num_nodes,g_train.T,args.channels, args.channels).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=5e-4)

    def test(model):
        model.eval()
        with torch.no_grad():
            z = model()

        embeddings = z.cpu().numpy()
        train_X = g_train[:2].T.cpu().numpy()
        test_X = g_test[:2].T.cpu().numpy()
        train_y = g_train[2].cpu().numpy()
        test_y = g_test[2].cpu().numpy()
        accuracy, f1, f1_macro, f1_micro, auc_score = link_sign_prediction_logistic_function(
            embeddings, train_X, train_y, test_X, test_y)
        # return auc_score, f1, f1_macro, f1_micro, accuracy
        return {
            # 'auc_1': roc_auc_score(y, pred),
            'auc': auc_score,
            'f1': f1,
            'f1micro': f1_micro,
            'f1macro': f1_macro
        }

    def train(model, optimizer):
        model.train()
        optimizer.zero_grad()
        loss = model.loss()
        loss.backward()
        optimizer.step()

    res_dict = {}

    for epoch in tqdm(range(args.epochs)):
        loss = train(model, optimizer)
    # model.load_state_dict(torch.load('{0}/models/{1}.pt'.format(network,train_file_name[:-4])))
    # torch.save(model.state_dict(), '{0}/models/{1}.pt'.format(network,train_file_name[:-4]))

    model.eval()
    # np.savetxt('{0}/embeddings/{1}-embedding.csv'.format(network,train_file_name[:-4]), z.detach().cpu().numpy(), fmt='%.2f', delimiter=',')
    res_dict.update(test(model))

    res_dict['train_file'] = train_file_name
    return res_dict


if __name__ == '__main__':
    train_files = os.listdir(args.train_file_folder)
    results = [run(train_file) for train_file in train_files]
    print(results)
    collect_results(results, save_res=True, suffix='dataset_aug')
