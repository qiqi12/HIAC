# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 20:04:58 2022

@author: 佡儁
"""

import numpy as np
import scipy.spatial.distance as dis
import matplotlib.pyplot as plt
import sklearn.cluster as sc
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import load_digits
from sklearn import decomposition as dec
import matplotlib as mpl
mpl.use('TkAgg')

global num
num = 1


def Plot(data, labels):
    global num
    global photoPath
    global sign
    plt.figure(num)
    num += 1
    sortNumMax = np.max(labels)
    sortNumMin = np.min(labels)
    color = '#606470' #无需颜色代表类别的数据集
    for i in range(sortNumMin, sortNumMax + 1):
        Together = []
        for j in range(data.shape[0]):
            if labels[j] == i:
                Together.append(data[j])
        Together = np.array(Together)
        Together = Together.reshape(-1, data.shape[1])  # 在不区分类时，整理矩阵形状
        font_size = 10
        plt.scatter(Together[:, 0], Together[:, 1], font_size, color, 'o')
    plt.xticks(fontsize=20)#不显示x刻度
    plt.yticks(fontsize=20)#不显示y刻度
    plt.title(sign, fontstyle='italic', size=20)
    plt.savefig(photoPath, dpi=300, bbox_inches='tight')  ####改改


def TGP(data, k, photoPath, threshold):
    global num
    pointNum = data.shape[0]
    probality = np.zeros([int(pointNum / 10), 2])
    distance = dis.pdist(data)
    distance_matrix = -dis.squareform(distance) + np.max(distance)
    distance_sort = -np.sort(-distance_matrix, axis=1)

    pointWeight = np.mean(distance_sort[:, 1:k + 1], axis=1)
    dc = (max(pointWeight) - min(pointWeight)) * 10 / pointNum
    for i in range(probality.shape[0]):
        location = min(pointWeight) + dc * i
        probality[i, 0] = location
    for i in range(pointNum):
        j = (int)((pointWeight[i] - min(pointWeight)) / dc)
        if j < int(pointNum / 10):
            probality[j, 1] += 1
    probality[:, 1] = probality[:, 1] / pointNum
    plt.figure(num)
    num += 1
    plt.plot(probality[:, 0], probality[:, 1])
    plt.axvline(x=threshold, c="r", ls="--", lw=1.5)
    plt.title('Decision graph', fontstyle='italic', size=20)
    plt.xlabel('wight', fontsize=20)
    plt.ylabel('probability', fontsize=20)
    plt.savefig(photoPath, dpi=150, bbox_inches='tight')  ####改改
    plt.show()
    return distance_sort

def shrink(data, knn, T, threshold, distanceTGP, iterationTime):
    global disIndex
    bata = data.copy()
    pointNum = data.shape[0]
    distance = dis.pdist(data)
    distance_matrix = dis.squareform(distance)
    distance_sort = np.sort(distance_matrix, axis=1)
    area = np.mean(distance_sort[:, round(pointNum * 0.015)])
    density = np.zeros(pointNum)
    if iterationTime == 0:
        disIndex = np.argsort(distance_matrix, axis=1)  # 记录第一次迭代时knn
    # 计算密度
    for i in range(pointNum):
        num = 1
        while (distance_sort[i, num] < area):
            num += 1
        density[i] = num
    densityThreshold = np.mean(density)
    G = np.mean(distance_sort[:, 1])

    for i in range(pointNum):
        if density[i] < densityThreshold:
            displacement = np.zeros(data.shape[1], dtype=np.float32)
            for j in range(knn + 1):  # 第j个邻居
                if (data[disIndex[i][j]] == data[i]).all():
                    continue
                else:
                    if iterationTime == 0:  # 第一次迭代，从knn中选择有引力的邻居，无引力的标记为-1
                        # 后续迭代中，用此次迭代的邻居信息
                        if distanceTGP[i, j] < threshold:
                            disIndex[i][j] = -1  # 消除引力
                    if disIndex[i][j] != -1:
                        ff = (data[disIndex[i][j]] - data[i])
                        fff = (distance_sort[i, 1] / (
                                    distance_matrix[i, disIndex[i, j]] * distance_matrix[i, disIndex[i, j]]))
                        displacement += G * ff * fff
            bata[i] = data[i] + displacement * T
    return bata

if __name__ == "__main__":
    ######################初始化####################################
    global disIndex
    global photoPath
    global sign
    filePath = "./a1.txt"  # 改改改
    labelPath = ""
    with_label = False
    iterationTime = 40  # 改改改
    k = 10  # 改改改
    Tset = [0.7]  # 改改改
    threshold = 1.18617  # 改改改
    isIter = 1  # 改改改
    pca = dec.PCA(n_components=2)
    DecisionPath = "./decision" + str(k) + "_" + str(threshold) + ".png"  #决策图的路径

    #####################提取数据#################
    data = np.loadtxt(filePath, dtype=np.float32)
    if with_label:
        labels = data[:, -1]
        labels = np.array(labels, dtype=np.int32())
        data = data[:, :-1]
    else:
        labels = np.zeros(data.shape[0], dtype=np.int32)  #如果不需要用颜色区分，则全置0，若需要区分，则注释掉  改改改
    dim = data.shape[1]  # 数据维度（不含label）
    cluster_num = max(labels) - min(labels) + 1
    ##################################################################################

    disIndex = np.zeros([data.shape[0], data.shape[0]], dtype=np.int32)
    #sign = 'original'
    #photoPath = "./oringnal.png"
    #Plot(data, labels)  # 原数据分布图

    ########################归一化###################################
    data_without_nml = data.copy()
    for j in range(data.shape[1]):
        max_ = max(data[:, j])
        min_ = min(data[:, j])
        if max_ == min_:
            continue
        for i in range(data.shape[0]):
            data[i][j] = (data[i][j] - min_) / (max_ - min_)
    #np.savetxt("./a1-normal.txt", data)  # 如果需要保留归一化后的数据
    sign = '(A)'
    photoPath = "./normal-original.png"
    Plot(data, labels)

    distanceTGP = TGP(data, k,  DecisionPath, threshold)  # data after prune,we can determine the threshold

    ###########################迭代收缩###########################
    if isIter:
        for T in Tset:
            ddata = data.copy()
            for i in range(iterationTime):
                dataAfterStrink = shrink(ddata, k, T, threshold, distanceTGP, i)
                ddata = dataAfterStrink
            photoPath = "./" + str(threshold) + "_" + str(k) + "_" + str(T) + "_" + str(i + 1) + "_shrink.png"
            sign = '(D)'
            Plot(ddata, labels)
