import os
import os.path as osp
import torch
from dotmap import DotMap
from torch_geometric import seed_everything
from torch_geometric_signed_directed.nn.signed import SiGAT
from torch_geometric_signed_directed.utils.signed import link_sign_prediction_logistic_function
from tqdm import tqdm
from data_proc import load_graph, collect_results
import numpy as np
from balanceDegree import edgesBalanceDegree_sp

network='bitcoin-otc'
num=1
baseline='SiGAT'

args = DotMap(
    train_file_folder='{0}/augset_{1}'.format(network,num),
    test_file_path='{0}/tests/{0}-test-{1}.csv'.format(network,num), # path_to_the_test_csv_file
    num_layers=2,
    channels=20,
    lr=5e-3,
    epochs=300,
    seed=2023,
    divisions_number=30,
    initial=0.25
)


if not os.path.exists('results'):
    os.makedirs('results')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


def run(train_file_name: str):
    # load graph
    g_train = load_graph(
        osp.join(args.train_file_folder, train_file_name)).to(device)
    g_test = load_graph(args.test_file_path).to(device)

    num_nodes = (torch.concat(
        [g_train[:2], g_test[:2]], dim=-1).max() + 1).item()  # +1

    edges = g_train.cpu().numpy()
    m = len(edges[0])
    balanceDegree = edgesBalanceDegree_sp(edges)
    balanceDegree=torch.tensor(balanceDegree)
    chooseOrder = (-balanceDegree).argsort()

    # model
    seed_everything(args.seed)
    x = torch.from_numpy(np.random.rand(num_nodes, args.channels).astype(np.float32)).to(device)
    model = SiGAT(num_nodes, g_train.T, args.channels, args.channels, init_emb=x).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=5e-4)

    initial_num = round(m * args.initial)
    epoch_div=args.epochs // args.divisions_number

    def train(model, optimizer, epoch):
        # get train set for this epoch
        epoch_num=round((epoch//epoch_div+1)*((m-initial_num)/args.divisions_number))
        num=epoch_num+initial_num
        train_edge_now=torch.tensor(edges[:,chooseOrder[:num]]).to(device)
        train_pos_edge_now = train_edge_now[:2, train_edge_now[2] > 0]
        train_neg_edge_now = train_edge_now[:2, train_edge_now[2] < 0]
        model.pos_edge_index = train_pos_edge_now
        model.neg_edge_index = train_neg_edge_now
        model.train()
        optimizer.zero_grad()
        loss = model.loss()
        loss.backward()
        optimizer.step()

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
        return {
            'auc': auc_score,
            'f1': f1,
            'f1micro': f1_micro,
            'f1macro': f1_macro
        }

    res_dict = {}
    for epoch in tqdm(range(args.epochs)):
        loss = train(model, optimizer, epoch)
    model.eval()
    res_dict.update(test(model))

    res_dict['train_file'] = train_file_name
    return res_dict


if __name__ == '__main__':
    train_files = os.listdir(args.train_file_folder)
    results = [run(train_file) for train_file in train_files]
    print(results)
    collect_results(results, save_res=True, suffix='{0}_{1}_{2}_SGA'.format(baseline,network,num))
