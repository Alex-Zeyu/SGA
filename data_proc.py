import torch
import pandas as pd
# import os.path as osp
import joblib


def load_graph(path: str):
    df = pd.read_csv(path, names=['src', 'tgt', 'weight', 'time'], header=None)[
        ['src', 'tgt', 'weight']]
    graph = torch.tensor(df.values, dtype=torch.int64).t()
    graph[2] = (graph[2] > 0) * 2 - 1
    return graph


def collect_results(res_list: list, save_res: bool = True, suffix: str = ''):
    res_df = pd.DataFrame.from_dict(res_list)
    print(res_df)
    if save_res:
        res_df.to_csv(f'results/result_{suffix}.csv')
    return res_df
