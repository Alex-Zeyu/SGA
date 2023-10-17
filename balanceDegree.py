import numpy as np
from scipy import sparse as sp


def balanceDegree(A):
    abs_A = A * A
    abs_A = abs_A + abs_A.T
    A = A + A.T
    x = np.trace(np.linalg.matrix_power(A, 3)) // 6
    y = np.trace(np.linalg.matrix_power(abs_A, 3)) // 6
    return (x + y) / (2 * y)


def edgesBalanceDegree(A):
    abs_A = A * A
    abs_A = abs_A + abs_A.T
    A1 = A + A.T
    x = np.linalg.matrix_power(A1, 2)  #
    y = np.linalg.matrix_power(abs_A, 2)  #
    x = x * A
    x[y == 0] = 1
    y[y == 0] = 1
    res = (x + y) / (y * 2)
    res[A == 0] = -1
    return res


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
        pos_plus_add=y[edges[0][i],edges[1][i]]
        pos_minus_add=x[edges[0][i],edges[1][i]]
        if pos_plus_add>0:
            bds[i]=(pos_minus_add+pos_plus_add)/(2*pos_plus_add)
    return bds


def findAndDel(e, edges):
    for i,edge in enumerate(edges):
        if int(float(e[0]))==int(float(edge[0])) and int(float(e[1]))==int(float(edge[1])):
            del edges[i]
            return edges

