# %%
import os
from glob import glob
import pandas as pd
import joblib

from balanceDegree import edgesBalanceDegree_sp

import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling, structured_negative_sampling
from torch_geometric.nn import GCNConv, GATConv

from sklearn.metrics import f1_score, roc_auc_score


# %%
# define models
class GATSign(torch.nn.Module):
    def __init__(self, em_dim, num_layers, lamb=5):
        super(GATSign, self).__init__()
        self.em_dim = em_dim
        self.num_layers = num_layers
        self.lamb = lamb

        self.lin = torch.nn.Linear(2 * self.em_dim, out_features=3)

        self.convs = torch.nn.ModuleList(
            [
                GATConv(in_channels=self.em_dim, out_channels=self.em_dim)
                for _ in range(self.num_layers)
            ]
        )
        self.reset_parameters()

    def forward(self, x, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        z = self.convs[0](x, edge_index)
        for i in range(1, self.num_layers):
            z = self.convs[i](z, edge_index)
        return z

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin.reset_parameters()

    def discriminate(self, z, edge_index):
        """
        Given node embedding z, classified the link relation between node pairs
        :param z: node features
        :param edge_index: edge indicies
        :return:
        """
        value = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        value = self.lin(value)
        return torch.log_softmax(value, dim=1)

    def nll_loss(self, z, pos_edge_index, neg_edge_index):
        """Computes the discriminator loss based on node embeddings :obj:`z`,
        and positive edges :obj:`pos_edge_index` and negative edges
        :obj:`neg_edge_index`.

        Args:
            z (Tensor): The node embeddings.
            pos_edge_index (LongTensor): The positive edge indices.
            neg_edge_index (LongTensor): The negative edge indices.
        """

        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        none_edge_index = negative_sampling(edge_index, z.size(0))

        nll_loss = 0
        nll_loss += F.nll_loss(
            self.discriminate(z, pos_edge_index),
            pos_edge_index.new_full((pos_edge_index.size(1),), 0),
        )
        nll_loss += F.nll_loss(
            self.discriminate(z, neg_edge_index),
            neg_edge_index.new_full((neg_edge_index.size(1),), 1),
        )
        nll_loss += F.nll_loss(
            self.discriminate(z, none_edge_index),
            none_edge_index.new_full((none_edge_index.size(1),), 2),
        )
        return nll_loss / 3.0

    def pos_embedding_loss(self, z, pos_edge_index):
        """Computes the triplet loss between positive node pairs and sampled
        non-node pairs.

        Args:
            z (Tensor): The node embeddings.
            pos_edge_index (LongTensor): The positive edge indices.
        """
        i, j, k = structured_negative_sampling(pos_edge_index, z.size(0))

        out = (z[i] - z[j]).pow(2).sum(dim=1) - (z[i] - z[k]).pow(2).sum(dim=1)
        return torch.clamp(out, min=0).mean()

    def neg_embedding_loss(self, z, neg_edge_index):
        """Computes the triplet loss between negative node pairs and sampled
        non-node pairs.

        Args:
            z (Tensor): The node embeddings.
            neg_edge_index (LongTensor): The negative edge indices.
        """
        i, j, k = structured_negative_sampling(neg_edge_index, z.size(0))

        out = (z[i] - z[k]).pow(2).sum(dim=1) - (z[i] - z[j]).pow(2).sum(dim=1)
        return torch.clamp(out, min=0).mean()

    def loss(self, z, pos_edge_index, neg_edge_index):
        """Computes the overall objective.

        Args:
            z (Tensor): The node embeddings.
            pos_edge_index (LongTensor): The positive edge indices.
            neg_edge_index (LongTensor): The negative edge indices.
        """
        nll_loss = self.nll_loss(z, pos_edge_index, neg_edge_index)
        loss_1 = self.pos_embedding_loss(z, pos_edge_index)
        loss_2 = self.neg_embedding_loss(z, neg_edge_index)
        return nll_loss + self.lamb * (loss_1 + loss_2)


class GCNSign(torch.nn.Module):
    def __init__(self, em_dim, num_layers, lamb=5):
        super(GCNSign, self).__init__()
        self.em_dim = em_dim
        self.num_layers = num_layers
        self.lamb = lamb

        self.lin = torch.nn.Linear(2 * self.em_dim, 3)

        self.convs = torch.nn.ModuleList(
            [
                GCNConv(in_channels=self.em_dim, out_channels=self.em_dim)
                for _ in range(self.num_layers)
            ]
        )
        self.reset_parameters()

    def forward(self, x, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        z = self.convs[0](x, edge_index)
        for i in range(1, self.num_layers):
            z = self.convs[i](z, edge_index)
        return z

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin.reset_parameters()

    def discriminate(self, z, edge_index):
        """
        Given node embedding z, classified the link relation between node pairs
        :param z: node features
        :param edge_index: edge indicies
        :return:
        """
        value = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        value = self.lin(value)
        return torch.log_softmax(value, dim=1)

    def nll_loss(self, z, pos_edge_index, neg_edge_index):
        """Computes the discriminator loss based on node embeddings :obj:`z`,
        and positive edges :obj:`pos_edge_index` and negative edges
        :obj:`neg_edge_index`.

        Args:
            z (Tensor): The node embeddings.
            pos_edge_index (LongTensor): The positive edge indices.
            neg_edge_index (LongTensor): The negative edge indices.
        """

        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        none_edge_index = negative_sampling(edge_index, z.size(0))

        nll_loss = 0
        nll_loss += F.nll_loss(
            self.discriminate(z, pos_edge_index),
            pos_edge_index.new_full((pos_edge_index.size(1),), 0),
        )
        nll_loss += F.nll_loss(
            self.discriminate(z, neg_edge_index),
            neg_edge_index.new_full((neg_edge_index.size(1),), 1),
        )
        nll_loss += F.nll_loss(
            self.discriminate(z, none_edge_index),
            none_edge_index.new_full((none_edge_index.size(1),), 2),
        )
        return nll_loss / 3.0

    def pos_embedding_loss(self, z, pos_edge_index):
        """Computes the triplet loss between positive node pairs and sampled
        non-node pairs.

        Args:
            z (Tensor): The node embeddings.
            pos_edge_index (LongTensor): The positive edge indices.
        """
        i, j, k = structured_negative_sampling(pos_edge_index, z.size(0))

        out = (z[i] - z[j]).pow(2).sum(dim=1) - (z[i] - z[k]).pow(2).sum(dim=1)
        return torch.clamp(out, min=0).mean()

    def neg_embedding_loss(self, z, neg_edge_index):
        """Computes the triplet loss between negative node pairs and sampled
        non-node pairs.

        Args:
            z (Tensor): The node embeddings.
            neg_edge_index (LongTensor): The negative edge indices.
        """
        i, j, k = structured_negative_sampling(neg_edge_index, z.size(0))

        out = (z[i] - z[k]).pow(2).sum(dim=1) - (z[i] - z[j]).pow(2).sum(dim=1)
        return torch.clamp(out, min=0).mean()

    def loss(self, z, pos_edge_index, neg_edge_index):
        """Computes the overall objective.

        Args:
            z (Tensor): The node embeddings.
            pos_edge_index (LongTensor): The positive edge indices.
            neg_edge_index (LongTensor): The negative edge indices.
        """
        nll_loss = self.nll_loss(z, pos_edge_index, neg_edge_index)
        loss_1 = self.pos_embedding_loss(z, pos_edge_index)
        loss_2 = self.neg_embedding_loss(z, neg_edge_index)
        return nll_loss + self.lamb * (loss_1 + loss_2)


# %%
# utility functions
def load_data(args, round: int) -> dict:
    """Load data from csv file and return as dictionary
    Returns:
        dict: {
            "train_ds": torch.Tensor, shape (3, n_edges),
            "test_ds": torch.Tensor, shape (3, n_edges),
            "n": int, number of nodes,
            "m": int, number of edges
        }
    """
    dataset = args.dataset
    path_test = os.path.join(dataset, "tests", f"{dataset}-test-{round}.csv")

    if args.augment:
        path_train = glob(os.path.join(dataset, f"augset_{round}", "*.csv"))[0]
    else:
        path_train = os.path.join(dataset, "trains", f"{dataset}-train-{round}.csv")
    # read data in csv format
    train_ds = pd.read_csv(path_train, names=["src", "dst", "sign"]).values
    test_ds = pd.read_csv(path_test, names=["src", "dst", "sign"]).values
    # convert to 0-based index
    base = min(train_ds[:, :2].min(), test_ds[:, :2].min())
    train_ds -= base
    test_ds -= base
    # convert shape to (3, n_edges)
    train_ds = torch.from_numpy(train_ds).t()
    test_ds = torch.from_numpy(test_ds).t()

    return {
        "train_ds": train_ds,
        "test_ds": test_ds,
        "n": torch.max(train_ds[:2].max(), test_ds[:2].max()).item() + 1,
        "m": train_ds.shape[1] + test_ds.shape[1],
    }


def convert_metric_to_tabular(metrics: dict, console: bool, save_path: str="") -> pd.DataFrame:
    """Convert metrics dictionary to tabular format and save to csv file
    Args:
        metrics (dict): dictionary of metrics
        save_path (str): path to save the csv file
        console (bool): whether to print the tabular format to console
    Returns:
        pd.DataFrame: tabular format of the metrics
    """
    df = pd.DataFrame(metrics)
    # add std for each metric
    # df.loc["mean"] = df.mean() * 100
    # df.loc["std"] = df.loc["mean"].std()
    # df.loc["repr"] = df.loc["mean"].apply(lambda x: f"{x * 100:1f}") + "±" + df.loc["std"].apply(lambda x: f"{x:.2f}")

    df_summary = (df * 100).describe().loc[["mean", "std"]].T.round(1)
    df_summary["repr"] = df_summary["mean"].astype(str) + "$\pm$" + df_summary["std"].astype(str)
    
    if console:
        print(df_summary)
    if save_path:
        df_summary.to_csv(save_path, index=False)
    return df_summary


# %%
# run script
if __name__ == "__main__":
    from torch_geometric import seed_everything
    import copy
    import argparse

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="bitcoin-alpha")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--round", type=int, default=5)
    parser.add_argument("--model", type=str, choices=["gat", "gcn"], default="gcn")
    # SGA arguments
    parser.add_argument("-a", "--augment", type=int, default=1, choices=[0, 1])
    parser.add_argument("-t", "--train_plan", type=int, default=1, choices=[0, 1])
    parser.add_argument("-T", "--T", type=int, default=30)
    parser.add_argument("-la", "--lambda0", type=float, default=0.25)
    args = parser.parse_known_args()[0]
    print(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    flag_a = args.augment
    flag_t = args.train_plan

    # Build GCN model
    seed_everything(args.seed)
    if args.model == "gcn":
        model = GCNSign(em_dim=args.channels, num_layers=args.n_layers).to(device)
    elif args.model == "gat":
        model = GATSign(em_dim=args.channels, num_layers=args.n_layers).to(device)
    else:
        raise NotImplementedError
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    model_init_params = copy.deepcopy(model.state_dict())

    # store the results
    metrics_dict = {"auc": [], "f1": [], "f1mi": [], "f1ma": []}
    
    for rnd in range(1, args.round + 1):
        # load the graph data
        train_test = load_data(args, rnd)
        train_ds = train_test["train_ds"].to(device)
        test_ds = train_test["test_ds"].to(device)
        n = train_test["n"]
        m = train_test["m"]
        # split pos and neg edges
        train_pos_edge_index = train_ds[:2, train_ds[2] > 0]
        train_neg_edge_index = train_ds[:2, train_ds[2] < 0]
        test_pos_edge_index = test_ds[:2, test_ds[2] > 0]
        test_neg_edge_index = test_ds[:2, test_ds[2] < 0]

        # training plan
        if flag_t:
            balanceDegree = edgesBalanceDegree_sp(train_ds.cpu().numpy())
            difficultyScore = torch.tensor((1 - balanceDegree) / 2)
            chooseOrder = difficultyScore.argsort()

            initial_num = round(m * args.lambda0)
            epoch_div = args.epochs // args.T

        # model
        model.load_state_dict(model_init_params)
        # create node features
        x = torch.randn(n, args.channels).to(device)

        def train(epoch):
            # prepare epoch data
            if flag_t:
                # get train set for this epoch
                epoch_num = round(
                    (epoch // epoch_div + 1) * ((m - initial_num) / args.T)
                )
                num = epoch_num + initial_num
                train_edge_now = train_ds[:, chooseOrder[:num]]
            else:
                train_edge_now = train_ds
            train_pos_edge_index = train_edge_now[:2, train_edge_now[2] > 0]
            train_neg_edge_index = train_edge_now[:2, train_edge_now[2] < 0]

            # start training
            model.train()
            optimiser.zero_grad()
            z = model(x, train_pos_edge_index, train_neg_edge_index)
            loss = model.loss(z, train_pos_edge_index, train_neg_edge_index)
            loss.backward()
            optimiser.step()
            return loss.item()

        def test():
            with torch.no_grad():
                z = model(x, train_pos_edge_index, train_neg_edge_index)
                pos_p = model.discriminate(z, test_pos_edge_index)[:, :2].max(dim=1)[1]
                neg_p = model.discriminate(z, test_neg_edge_index)[:, :2].max(dim=1)[1]
            pred = (1 - torch.cat([pos_p, neg_p])).cpu()
            y = torch.cat(
                [pred.new_ones((pos_p.size(0))), pred.new_zeros(neg_p.size(0))]
            )
            pred, y = pred.numpy(), y.numpy()

            return {
                "auc": roc_auc_score(y, pred),
                "f1": f1_score(y, pred, average="binary"),
                "f1mi": f1_score(y, pred, average="micro"),
                "f1ma": f1_score(y, pred, average="macro"),
            }

        for epoch in range(args.epochs):
            loss = train(epoch)
        
        model.eval()
        round_metrics = test()
        metrics_dict["auc"].append(round_metrics["auc"])
        metrics_dict["f1"].append(round_metrics["f1"])
        metrics_dict["f1mi"].append(round_metrics["f1mi"])
        metrics_dict["f1ma"].append(round_metrics["f1ma"])

    convert_metric_to_tabular(metrics_dict, console=True)

# %%
