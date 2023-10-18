import os
import torch
from torch_geometric import seed_everything
from torch_geometric.nn import SignedGCN
from tqdm import tqdm
from data_proc import load_graph
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d','--dataset', type=str, default='bitcoin-alpha',
                      choices=['bitcoin-alpha', 'Epinions', 'bitcoin-otc', 'Slashdot', 'wiki-elec', 'wiki-RfA'])
parser.add_argument('-n','--num', type=int, default=1, choices=[1,2,3,4,5])
parser.add_argument('-la','--num_layers', type=int, default=2)
parser.add_argument('-ch','--channels', type=int, default=64)
parser.add_argument('-lr','--lr', type=float, default=1e-2)
parser.add_argument('-e','--epochs', type=int, default=300)
parser.add_argument('-s','--seed', type=float, default=2023)

args = parser.parse_args()

network = args.dataset
num=args.num
train_file_path='{0}/trains/{0}-train-{1}.csv'.format(network,num)  # path_to_the_train_file_folder
test_file_path='{0}/tests/{0}-test-{1}.csv'.format(network,num)  # path_to_the_test_csv_file

if not os.path.exists('{}/trains'.format(network)):
    os.makedirs('{}/trains'.format(network))
if not os.path.exists('{}/tests'.format(network)):
    os.makedirs('{}/tests'.format(network))
if not os.path.exists('{}/models'.format(network)):
    os.makedirs('{}/models'.format(network))
    if not os.path.exists('{}/embeddings'.format(network)):
        os.makedirs('{}/embeddings'.format(network))
if not os.path.exists('results'):
    os.makedirs('results')


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


def run(train_file_name: str):
    # load graph
    g_train = load_graph(train_file_path).to(device)
    g_test = load_graph(test_file_path).to(device)

    num_nodes = (torch.concat(
        [g_train[:2], g_test[:2]], dim=-1).max() + 1).item() #+1

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
    x = model.create_spectral_features(
        train_pos_edge_index, train_neg_edge_index, num_nodes=num_nodes)
    
    def train():
        model.train()
        optimizer.zero_grad()
        z = model(x, train_pos_edge_index, train_neg_edge_index)
        loss = model.loss(z, train_pos_edge_index , train_neg_edge_index)
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
        loss = train()
    
    # save model and embeddings
    torch.save(model.state_dict(), '{0}/models/{1}.pt'.format(network,train_file_name[:-4]))

    model.eval()
    with torch.no_grad():
        z = model(x, train_pos_edge_index, train_neg_edge_index)
    np.savetxt('{0}/embeddings/{1}-embeddings.csv'.format(network,train_file_name[:-4]), z.detach().cpu().numpy(), fmt='%.2f', delimiter=',')


if __name__ == '__main__':
    train_file_name='{0}-train-{1}.csv'.format(network,num)
    run(train_file_name)
