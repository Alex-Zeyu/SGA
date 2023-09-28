import csv
import math
import numpy as np
from scipy import sparse

network = 'Epinions'
num = 1
p1,p2,p3,p4=0,0.1,0.95,1

pos_add, neg_add, pos_del, neg_del = [], [], [], []
with open('{}/pos_add.csv'.format(network), 'r') as data:
    reader = csv.reader(data)
    for row in reader:
        pos_add.append([float(r) for r in row])
with open('{}/neg_add.csv'.format(network), 'r') as data:
    reader = csv.reader(data)
    for row in reader:
        neg_add.append([float(r) for r in row])
with open('{}/pos_del.csv'.format(network), 'r') as data:
    reader = csv.reader(data)
    for row in reader:
        pos_del.append([float(r) for r in row])
with open('{}/neg_del.csv'.format(network), 'r') as data:
    reader = csv.reader(data)
    for row in reader:
        neg_del.append([float(r) for r in row])

node_num=int(max(max([max(pa[:2]) for pa in pos_add]),max([max(na[:2]) for na in neg_add]),max([max(na[:2]) for na in pos_del]),max([max(na[:2]) for na in neg_del])))+1
A = sparse.dok_matrix((node_num, node_num), dtype=np.int16)

pos, neg = 0, 0
with open('{0}/trains/{0}-train-{1}.csv'.format(network, num), 'r') as data:
    reader = csv.reader(data)
    for row in reader:
        i,j,weight=int(row[0]),int(row[1]),int(row[2])
        if weight > 0:
            pos += 1
        else:
            neg += 1
        A[i,j]=weight

pos_del.sort(key=lambda x:x[2])
neg_del.sort(key=lambda x:x[3])

pos2none, neg2none = 0, 0
for d in pos_del:
    i,j,p_pos=int(d[0]),int(d[1]),float(d[2])
    if (math.e**p_pos)<p1:
        A[i,j] = 0
        pos2none += 1
    else:
        break
for d in neg_del:
    i,j,p_neg= int(d[0]),int(d[1]),float(d[3])
    if (math.e**p_neg)<p2:
        A[i,j] = 0
        neg2none += 1
    else:
        break

none2pos, none2neg = 0, 0
for d in pos_add:
    if math.e**d[2]<p3:
        break
    i,j=int(d[0]),int(d[1])
    if A[i,j] == 0:
        A[i,j] = 1
        none2pos += 1
    if A[j,i] == 0:
        A[j,i] = 1
        none2pos += 1
for d in neg_add:
    if math.e**d[3]<p4:
        break
    i,j=int(d[0]),int(d[1])
    if A[i,j] == 0:
        A[i,j] = -1
        none2neg += 1
    if A[j,i] == 0:
        A[j,i] = -1
        none2neg += 1

with open('{0}/trains_Augmentation/{0}-{5}-softmax-{1}-{2}-{3}-{4}.csv'.format(network, p1, p2, p3, p4, num),
          'w',
          newline='') as aug_set:
    writer = csv.writer(aug_set)
    for (row, col), value in A.items():
        writer.writerow([row, col, value])
