import csv
import math
import os
import argparse
import numpy as np
from scipy import sparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default='bitcoin-alpha',
                    choices=['bitcoin-alpha', 'Epinions', 'bitcoin-otc', 'Slashdot', 'wiki-elec', 'wiki-RfA'])
parser.add_argument('-n', '--num', type=int, default=1, choices=[1, 2, 3, 4, 5])
parser.add_argument('-pd', '--pos_del', type=float, default=0)
parser.add_argument('-nd', '--neg_del', type=float, default=0)
parser.add_argument('-pa', '--pos_add', type=float, default=1)
parser.add_argument('-na', '--neg_add', type=float, default=1)

args = parser.parse_args()

dataset = args.dataset
num = args.num
p1, p2, p3, p4 = args.pos_del, args.neg_del, args.pos_add, args.neg_add

pos_add, neg_add, pos_del, neg_del = [], [], [], []
with open('{0}/candidates_{1}/pos_add.csv'.format(dataset, num), 'r') as data:
    reader = csv.reader(data)
    for row in reader:
        pos_add.append([float(r) for r in row])
with open('{0}/candidates_{1}/neg_add.csv'.format(dataset, num), 'r') as data:
    reader = csv.reader(data)
    for row in reader:
        neg_add.append([float(r) for r in row])
with open('{0}/candidates_{1}/pos_del.csv'.format(dataset, num), 'r') as data:
    reader = csv.reader(data)
    for row in reader:
        pos_del.append([float(r) for r in row])
with open('{0}/candidates_{1}/neg_del.csv'.format(dataset, num), 'r') as data:
    reader = csv.reader(data)
    for row in reader:
        neg_del.append([float(r) for r in row])

node_num = int(max(max([max(pa[:2]) for pa in pos_add]), max([max(na[:2]) for na in neg_add]),
                   max([max(na[:2]) for na in pos_del]), max([max(na[:2]) for na in neg_del]))) + 1
A = sparse.dok_matrix((node_num, node_num), dtype=np.int16)

with open('{0}/trains/{0}-train-{1}.csv'.format(dataset, num), 'r') as data:
    reader = csv.reader(data)
    for row in reader:
        i, j, weight = int(row[0]), int(row[1]), int(row[2])
        A[i, j] = weight

abs_A = A.multiply(A)
abs_A = abs_A + abs_A.T
sym_A = A + A.T
path_pos_minus_neg = sym_A.dot(sym_A)
path_pos_plus_neg = abs_A.dot(abs_A)


def canAddEdge(i, j, sign):
    delta_pos_minus_neg = path_pos_minus_neg[i, j] * sign
    delta_pos_plus_neg = path_pos_plus_neg[i, j]
    delta_neg=delta_pos_plus_neg - delta_pos_minus_neg
    return delta_neg<=0


for d in pos_del:
    i, j, p_pos = int(d[0]), int(d[1]), float(d[2])
    if (math.e ** p_pos) < p1:
        A[i, j] = 0
    else:
        break
for d in neg_del:
    i, j, p_neg = int(d[0]), int(d[1]), float(d[3])
    if (math.e ** p_neg) < p2:
        A[i, j] = 0
    else:
        break

for d in pos_add:
    if math.e ** d[2] < p3:
        break
    i, j = int(d[0]), int(d[1])
    if not canAddEdge(i,j,1):
        continue
    if A[i, j] == 0:
        A[i, j] = 1
    if A[j, i] == 0:
        A[j, i] = 1
for d in neg_add:
    if math.e ** d[3] < p4:
        break
    i, j = int(d[0]), int(d[1])
    if not canAddEdge(i,j,-1):
        continue
    if A[i, j] == 0:
        A[i, j] = -1
    if A[j, i] == 0:
        A[j, i] = -1

if not os.path.exists('{0}/augset_{1}'.format(dataset, num)):
    os.makedirs('{0}/augset_{1}'.format(dataset, num))

with open('{0}/augset_{5}/{0}-{5}-{1}-{2}-{3}-{4}.csv'.format(dataset, p1, p2, p3, p4, num),
          'w',
          newline='') as aug_set:
    writer = csv.writer(aug_set)
    for (row, col), value in A.items():
        writer.writerow([row, col, value])
