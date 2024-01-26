import argparse
import os
import os.path as osp
import torch
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from modules import MyGSGNN
from tqdm import tqdm
from data_proc import load_graph, collect_results, setup_seed
import numpy as np
from torch_geometric import seed_everything
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
parser.add_argument('-ch','--channels', type=int, default=64)
parser.add_argument('-lr','--lr', type=float, default=1e-2)
parser.add_argument('-e','--epochs', type=int, default=3000)
parser.add_argument('-s','--seed', type=float, default=2023)

args = parser.parse_args()
network = args.dataset
num=args.num
baseline='GSGNN'
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
        [g_train[:2], g_test[:2]], dim=-1).max() + 1).item() #+1
    edges = g_train.cpu().numpy()
    m = len(edges[0])
    if flag_t:
        balanceDegree = edgesBalanceDegree_sp(edges)
        difficultyScore=torch.tensor((1-balanceDegree)/2)
        chooseOrder = difficultyScore.argsort()
    # pos & neg edge index
    train_pos_edge_index, train_neg_edge_index = g_train[:2,
                                                         g_train[2] > 0], g_train[:2, g_train[2] < 0]
    test_pos_edge_index, test_neg_edge_index = g_test[:2,
                                                      g_test[2] > 0], g_test[:2, g_test[2] < 0]
    seed_everything(args.seed)
    # model
    model = MyGSGNN(args.channels,args.channels,16,k=4).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.99)
    setup_seed(args.seed)
    x = torch.from_numpy(np.random.rand(num_nodes, args.channels).astype(np.float32)).to(device)
    
    if flag_t:
        initial_num = round(m * args.lambda0)
        epoch_div=args.epochs // args.T
    
    def train(epoch):
        if flag_t:
            # get train set for this epoch
            epoch_num=round((epoch//epoch_div+1)*((m-initial_num)/args.T))
            num=epoch_num+initial_num
            train_edge_now=torch.tensor(edges[:,chooseOrder[:num]]).to(device)
        else:
            train_edge_now=g_train
        train_pos_edge_now = train_edge_now[:2, train_edge_now[2] > 0]
        train_neg_edge_now = train_edge_now[:2, train_edge_now[2] < 0]
        model.train()
        optimizer.zero_grad()
        z = model(x, train_pos_edge_index, train_neg_edge_index)
        loss = model.loss(z, train_pos_edge_now, train_neg_edge_now)
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
            'auc': roc_auc_score(y, pred_p),
            'f1': f1_score(y, pred, average='binary') if pred.sum() > 0 else 0,
            'f1micro': f1_score(y, pred, average='micro') if pred.sum() > 0 else 0,
            'f1macro': f1_score(y, pred, average='macro') if pred.sum() > 0 else 0
        }

    res_dict = {}

    for epoch in tqdm(range(args.epochs)):
        loss = train(epoch)

    model.eval()
    with torch.no_grad():
        z = model(x, train_pos_edge_index, train_neg_edge_index)

    res_dict.update(test(z, test_pos_edge_index, test_neg_edge_index))
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
