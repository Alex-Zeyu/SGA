import numpy as np
from scipy import sparse as sp


def edgesBalanceDegree_sp(edges):
    node_num = int(max([max(e) for e in edges[:2]])) + 1
    edges_num=len(edges[0])
    A = sp.coo_matrix((edges[2], (edges[0], edges[1])), shape=[node_num, node_num])
    A=A.tocsr()
    abs_A = A.multiply(A)
    abs_A = abs_A + abs_A.T
    A1 = A + A.T
    x = A1.dot(A1)
    y = abs_A.dot(abs_A)
    x = x.multiply(A)
    bds=np.ones(edges_num)
    for i in range(edges_num):
        pos_plus_neg=y[edges[0][i],edges[1][i]]
        pos_minus_neg=x[edges[0][i],edges[1][i]]
        if pos_plus_neg>0:
            bds[i]=pos_minus_neg/pos_plus_neg
    return bds
