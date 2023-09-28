import numpy as np
import networkx as nx
import random
import math
import csv
import os

dataname = 'Slashdot'

num = 1

train_p = 0.8

if not os.path.exists('{}/trains'.format(dataname)):
    os.makedirs('{}/trains'.format(dataname))
if not os.path.exists('{}/tests'.format(dataname)):
    os.makedirs('{}/tests'.format(dataname))
if not os.path.exists('{}/trains_Augmentation'.format(dataname)):
    os.makedirs('{}/trains_Augmentation'.format(dataname))
if not os.path.exists('{}/models'.format(dataname)):
    os.makedirs('{}/models'.format(dataname))
    if not os.path.exists('{}/embeddings'.format(dataname)):
        os.makedirs('{}/embeddings'.format(dataname))

pos_edges,neg_edges=[],[]
with open('networks/{}.csv'.format(dataname), 'r') as data, \
        open('{0}/trains/{0}-train-{1}.csv'.format(dataname, num), 'w', newline='') as set_train, \
        open('{0}/tests/{0}-test-{1}.csv'.format(dataname, num), 'w', newline='') as set_test:
    reader = csv.reader(data)
    writer_train = csv.writer(set_train)
    writer_test = csv.writer(set_test)

    for row in reader:
        # row[2] = str(int(row[2]) / abs(int(row[2])))
        if int(row[2])>0:
            pos_edges.append(row)
        else:
            neg_edges.append(row)

    # 随机打乱数据集
    np.random.shuffle(pos_edges)
    np.random.shuffle(neg_edges)

    # 划分数据集
    split_index_pos = round(len(pos_edges) * 0.8)
    split_index_neg = round(len(neg_edges) * 0.8)
    train_data, test_data = pos_edges[:split_index_pos], pos_edges[split_index_pos:]
    train_data += neg_edges[:split_index_neg]
    test_data += neg_edges[split_index_neg:]

    for e in train_data:
        writer_train.writerow(e)
    for e in test_data:
        writer_test.writerow(e)