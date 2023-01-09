# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 20:04:58 2022

@author: 佡儁
"""
import os.path

import numpy as np
import pandas
import scipy.spatial.distance as dis
import matplotlib.pyplot as plt
import matplotlib
import sklearn.cluster as sc
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import load_digits
from sklearn import decomposition as dec
from sklearn import cluster

global num
num = 1


def Plot(data, labels, title):
    global num
    global photoPath
    plt.figure(num)
    num += 1
    sortNumMax = np.max(labels)
    sortNumMin = np.min(labels)
    color = ['#125B50', '#4D96FF', '#FFD93D', '#FF6363', '#CE49BF', '#22577E', '#4700D8', '#F900DF', '#95CD41',
             '#FF5F00', '#40DFEF', '#8E3200', '#001E6C', '#C36A2D', '#B91646']  # 正常数据集
    lineform = ['o']
    for i in range(sortNumMin, sortNumMax + 1):
        Together = []
        flag = 0
        for j in range(data.shape[0]):
            if labels[j] == i:
                flag += 1
                Together.append(data[j])
        Together = np.array(Together)
        Together = Together.reshape(-1, data.shape[1])  # 在不区分类时，整理矩阵形状
        fontSize = 15
        colorNum = (i - sortNumMin) % len(color)# 正常数据集
        formNum = 0
        plt.scatter(Together[:, 0], Together[:, 1], fontSize, color[colorNum], lineform[formNum])
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.title(title, fontsize=20)
    plt.savefig(photoPath, dpi=300, bbox_inches='tight')
    plt.show()


def TGP(data, k, photo_path, threshold):
    '''

    Parameters
    ----------
    data
    k
    photo_path
    threshold

    Returns:the edge weight matrix that record the edge weight of each object i and its k-nearest-neighbors
    -------

    '''
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
    plt.savefig(photo_path, dpi=300)
    plt.show()
    return distance_sort

def prune(data, knn, threshold, distanceTGP):
    '''

    Parameters
    ----------
    data:
    knn:the number of neighbor
    threshold:to clip invalid-edge
    distanceTGP:the weight matrix for each object, we only need distanceTGP[:,:k+1], i.e. the k-nearest-neighbors

    Returns: the index matrix which records the valid-neighbors index of object i
    -------

    '''
    pointNum = data.shape[0]
    distance = dis.pdist(data)
    distance_matrix = dis.squareform(distance)
    disIndex = np.argsort(distance_matrix, axis=1)
    distance_sort = np.sort(distance_matrix, axis=1)
    area = np.mean(distance_sort[:, round(pointNum * 0.015)])
    density = np.zeros(pointNum)
    for i in range(pointNum):
        num = 1
        while (distance_sort[i, num] < area):
            num += 1
        density[i] = num
    densityThreshold = np.mean(density)
    for i in range(pointNum):
        if density[i] < densityThreshold:
            for j in range(knn + 1):  # 第j个邻居
                if (data[disIndex[i][j]] == data[i]).all():
                    continue
                else:
                    # 后续迭代中，用此次迭代的邻居信息
                    if distanceTGP[i, j] < threshold:
                        disIndex[i][j] = -1  # 消除引力
    return disIndex

def shrink(data, knn, T, disIndex):
    '''

    Parameters
    ----------
    data
    knn
    T:
    disIndex:i.e. the index matrix which records the valid-neighbors index of each object i
            for object i, if j is invalid-neighbor of i, neighbor_index[i][j] = -1,
            else neighbor_index[i][j] is the index of object j
    Returns:dataset after ameliorating
    -------

    '''
    bata = data.copy()
    pointNum = data.shape[0]
    distance = dis.pdist(data)
    distance_matrix = dis.squareform(distance)
    distance_sort = np.sort(distance_matrix, axis=1)
    area = np.mean(distance_sort[:, round(pointNum * 0.015)])
    density = np.zeros(pointNum)

    # 计算密度
    for i in range(pointNum):
        num = 1
        while (distance_sort[i, num] < area):
            num += 1
        density[i] = num
    densityThreshold = np.mean(density)
    G = np.mean(distance_sort[:, 1])
    # 收缩样本点
    for i in range(pointNum):
        if density[i] < densityThreshold:
            displacement = np.zeros(data.shape[1], dtype=np.float32)
            for j in range(knn + 1):  # 第j个邻居
                if (data[disIndex[i][j]] == data[i]).all():
                    continue
                else:
                    if disIndex[i][j] != -1:
                        ff = (data[disIndex[i][j]] - data[i])
                        fff = (distance_sort[i, 1] / (
                                    distance_matrix[i, disIndex[i, j]] * distance_matrix[i, disIndex[i, j]]))
                        displacement += G * ff * fff
            bata[i] = data[i] + displacement * T
    return bata


if __name__ == "__main__":
    ######################初始化####################################
    global photoPath
    filePath = "data-sets/shapes/Aggregation.txt"  # 改改改
    file_name, _ = filePath.split("/")[-1].split(".")
    labelPath = ""
    save_dir = os.path.join(".", file_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    iterationTime = 29  # 改改改，the d in paper HIAC
    with_label = True #标签和数据是否在一个txt文件里
    k = 35 # 改改改，邻居数量
    T = 1.8
    threshold = 1.1822 #改改改，即用于剪切无效边的权重阈值
    pca = dec.PCA(n_components=2) #对高维数据，可用PCA降维展示

    #####################read data and label#################
    data = np.loadtxt(filePath, dtype=np.float32)
    if with_label:
        labels = data[:, -1]
        labels = np.array(labels, dtype=np.int32)
        data = data[:, :-1]
    else:
        labels = np.loadtxt(labelPath, dtype=np.int32)
    dim = data.shape[1]#数据维度（不含label）
    cluster_num = max(labels) - min(labels) + 1

    ########################normalization###################################
    data_without_nml = data.copy()
    for j in range(data.shape[1]):
        max_ = max(data[:, j])
        min_ = min(data[:, j])
        if max_ == min_:
            continue
        for i in range(data.shape[0]):
            data[i][j] = (data[i][j] - min_) / (max_ - min_)
    np.savetxt(os.path.join(save_dir, file_name + "_normalization.txt"), data)  # 改改改

    ########################PCA to visaulize###################################
    dataPCA = data.copy()
    if dim > 2:
        dataPCA = pca.fit_transform(data)
        photoPath = os.path.join(save_dir, file_name + "_after_pca.png")
        Plot(dataPCA, labels, "original")
    else:
        photoPath = os.path.join(save_dir, file_name + "_original.png")
        Plot(data, labels, "original")

    ######################call HIAC##############################
    photoPath = os.path.join(save_dir, "decision_" + str(k) + "_" + str(threshold) + ".png")
    distanceTGP = TGP(data, k, photoPath, threshold)  # we can determine the threshold，and return the weight matrix
    neighbor_index = prune(data, k, threshold, distanceTGP) # clip invalid-neighbors based on the weight threshold and the decision-graph,
                                                            # and then return the index matrix which records the valid-neighbors index of object i
                                                            # for object i, if j is invalid-neighbor of i, neighbor_index[i][j] = -1,
                                                            # else neighbor_index[i][j] is the index of object j
                                                            # its necessary for you to know that we only need K-nearest-neighbor of each object,
                                                            # so,

    for i in range(iterationTime): # ameliorated the dataset by d time-segments
        bata = shrink(data, k, T, neighbor_index)
        data = bata
    np.savetxt(os.path.join(save_dir, file_name + '_ameliorated_by_HIAC.txt'), data)

    ########################PCA to visaulize after ameliorating###################################
    dataPCA = data.copy()
    if dim > 2:
        dataPCA = pca.fit_transform(data)
        photoPath = os.path.join(save_dir, file_name + "_after_pca.png")
        Plot(dataPCA, labels, "original")
    else:
        photoPath = os.path.join(save_dir, file_name + "_ameliorated_by_HIAC.png")
        Plot(data, labels, "ameliorated")
