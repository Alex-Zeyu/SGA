import numpy as np
import networkx as nx
import random
import math
import csv
import os


def balanceDegree(A):
    abs_A = A * A
    abs_A = abs_A + abs_A.T
    A = A + A.T
    x = np.trace(np.linalg.matrix_power(A, 3)) // 6
    y = np.trace(np.linalg.matrix_power(abs_A, 3)) // 6
    return (x + y) / (2 * y)


# def edgeBalanceDegree(i, j, A):
#     abs_A = A * A
#     abs_A = abs_A + abs_A.T
#     A = A + A.T
#     x = np.linalg.matrix_power(A, 2)[i][j]  # 正-负
#     y = np.linalg.matrix_power(abs_A, 2)[i][j]  # 总三角形
#     if y == 0:
#         return -1


def edgesBalanceDegree(A):
    abs_A = A * A
    abs_A = abs_A + abs_A.T
    A1 = A + A.T
    x = np.linalg.matrix_power(A1, 2)  # 二跳 正-负
    y = np.linalg.matrix_power(abs_A, 2)  # 二跳
    x = x * A
    x[y == 0] = 1
    y[y == 0] = 1
    res = (x + y) / (y * 2)
    res[A == 0] = -1
    return res


# dataname = 'bitcoin-otc'
# num=2
# edges = []
# with open('{0}/unused/{0}-{1}-softmax-0.2-0.2-0.97-0.99.csv'.format(dataname,num), 'r') as data:
# with open('bitcoin-otc/unused/Bitcoin-OTC-1_train.csv', 'r') as data:
#     reader = csv.reader(data)
#     for row in reader:
#         edges.append([float(r) for r in row])
# edges=[[0,1,1],[0,2,-1],[0,3,1],[2,0,1],[2,3,-1],[3,1,1]]
#
# node_num = int(max([max(e[:2]) for e in edges]))+1
# A=np.zeros([node_num,node_num])
# for e in edges:
#     i,j,weight=int(e[0]),int(e[1]),int(e[2])
#     A[i][j]=weight
# # B=balanceDegree(A)
# # print(B)
# # print(edgesBalanceDegree(np.array([[]])))
# balanceDegree = edgesBalanceDegree(A)
# balanceDegree = np.array([balanceDegree[int(e[0])][int(e[1])] for e in edges])
# chooseOrder = (-balanceDegree).argsort()
# for i in range(len(edges)):
#     print(edges[chooseOrder[i]],balanceDegree[chooseOrder[i]])
