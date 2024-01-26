import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument('-d','--dataset', type=str, default='bitcoin-alpha',
                      choices=['bitcoin-alpha', 'Epinions', 'bitcoin-otc', 'Slashdot', 'wiki-elec', 'wiki-RfA'])
parser.add_argument('-n','--num', type=int, default=1, choices=[1,2,3,4,5])
parser.add_argument('-a', '--useDataAugmentation', type=int, default=1, choices=[0,1])
parser.add_argument('-t', '--useTrainingPlan', type=int, default=1, choices=[0,1])
parser.add_argument('-T','--T', type=int, default=30)
parser.add_argument('-la','--lambda0', type=float, default=0.25)

parser.add_argument('-ly','--num_layers', type=int, default=2)
parser.add_argument('-ch','--channels', type=int, default=20)
parser.add_argument('-lr','--lr', type=float, default=5e-3)
parser.add_argument('-e','--epochs', type=int, default=1500)
parser.add_argument('-s','--seed', type=float, default=2023)

args = parser.parse_args()
network = args.dataset
num=args.num
baseline='SiGAT'
flag_a=args.useDataAugmentation
flag_t=args.useTrainingPlan

train_file_folder='{0}/augset_{1}'.format(network,num)  # path_to_the_train_file_folder
test_file_path='{0}/tests/{0}-test-{1}.csv'.format(network,num)  # path_to_the_test_csv_file

if not os.path.exists('results'):
    os.makedirs('results')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


def run(train_file_name: str):
    # load graph
    g_train = load_graph(
        osp.join(train_file_folder, train_file_name)).to(device)
    g_test = load_graph(test_file_path).to(device)

    num_nodes = (torch.concat(
        [g_train[:2], g_test[:2]], dim=-1).max() + 1).item()  # +1

    edges = g_train.cpu().numpy()
    m = len(edges[0])
    if flag_t:
        balanceDegree = edgesBalanceDegree_sp(edges)
        difficultyScore=torch.tensor((1-balanceDegree)/2)
        chooseOrder = difficultyScore.argsort()

    # model
    seed_everything(args.seed)
    x = torch.from_numpy(np.random.rand(num_nodes, args.channels).astype(np.float32)).to(device)
    model = SiGAT(num_nodes, g_train.T, args.channels, args.channels, init_emb=x).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=5e-4)

    if flag_t:
        initial_num = round(m * args.lambda0)
        epoch_div=args.epochs // args.T

    def train(model, optimizer, epoch):
        if flag_t:
            # get train set for this epoch
            epoch_num=round((epoch//epoch_div+1)*((m-initial_num)/args.T))
            num=epoch_num+initial_num
            train_edge_now=torch.tensor(edges[:,chooseOrder[:num]]).to(device)
        else:
            train_edge_now=g_train
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
    if flag_a:
        train_files = os.listdir(train_file_folder)
        results = [run(train_file) for train_file in train_files]
    else:
        args.train_file_folder='{}/trains/'.format(network)
        results=[run('{0}-train-{1}.csv'.format(network,num))]
    print(results)
    if flag_a and flag_t:
        result_name='{0}_{1}_{2}_SGA'.format(baseline,network,num)
    elif flag_a:
        result_name = '{0}_{1}_{2}_dataAugmentation'.format(baseline, network, num)
    elif flag_t:
        result_name = '{0}_{1}_{2}_TrainingPlan'.format(baseline, network, num)
    else:
        result_name = '{0}_{1}_{2}_baseline'.format(baseline, network, num)
    collect_results(results, save_res=True, suffix=result_name)
