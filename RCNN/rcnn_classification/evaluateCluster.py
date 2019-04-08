import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

def bestmap(l1, l2):
    if np.shape(l1) != np.shape(l2):
        print('长度不匹配')
    Label1 = np.unique(l1)
    Label2 = np.unique(l2)
    nclass1, nclass2 = np.shape(Label1)[0], np.shape(Label2)[0]
    nclass = max(nclass1, nclass2)
    G = np.zeros((nclass, nclass))
    for i in range(nclass1):
        for j in range(nclass2):
            tmp = np.where(l1 == Label1[i])
            tmp1 = np.where(l2 == Label2[j])
            a = tmp[0].tolist()
            b = tmp1[0].tolist()
            G[i][j] = len([val for val in a if val in b])
    row_ind, col_ind = linear_sum_assignment(-G)
    for i in range(nclass2):
        # print(row_ind[i])
        tmp = np.where(col_ind == row_ind[i])
        row_ind[i] = tmp[0][0]
    newL2 = np.zeros(np.shape(l2))
    for i in range(nclass2):
        tmp = np.where(l2 == Label2[i])[0].tolist()
        newL2[tmp] = Label1[row_ind[i]]
    return newL2


def evaluateCluster(out_all, labels_all):
    repeatNum = 1
    for i in range(repeatNum):
        model = KMeans(n_clusters=7, n_jobs=5)  # 分为k类, 并发数4
        model.fit(out_all)  # 开始聚类
        result = model.labels_
        result = result.astype(int)
        labels_all = labels_all.astype(int)
        result = bestmap(labels_all, result)
        result_acc = np.sum(labels_all == result)/result.shape[0]
        print(result_acc)
        '''labels_all = labels_all.astype(int)
        labels_all1 = np.where(labels_all == 1, labels_all, 0)
        labels_all2 = labels_all1[0:result.shape[0]]
        # labels_all2 = labels_all1[6000:9500]
        print(labels_all)
        print(labels_all2)
        print(result)
        print(np.sum(labels_all2 == result))
        print(np.sum(labels_all2 == result)/result.shape[0])'''
    return result_acc
