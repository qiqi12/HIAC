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

from DPC import DPC

global num
num = 1

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
    if photoPath != "":
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
    #density = -np.sort(-density)
    densityThreshold = np.mean(density)
    G = np.mean(distance_sort[:, 1])

    # print("G  ",G)

    for i in range(pointNum):
        if density[i] < densityThreshold:
            displacement = np.zeros(data.shape[1], dtype=np.float32)
            for j in range(knn + 1):  # 第j个邻居
                if (data[disIndex[i][j]] == data[i]).all():
                    continue
                else:
                    if iterationTime == 0:  # 第一次迭代，从knn中选择有效邻居，除有效邻居外，其他样本点标记为-1
                        # 后续迭代中，用此次迭代的邻居信息
                        if distanceTGP[i, j] < threshold:
                            disIndex[i][j] = -1  # 只保留有效邻居的引力
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
    filePath = "./wifi_localization.txt"  # 改改改
    file_name, _ = filePath.split("/")[-1].split(".")
    labelPath = ""
    with_label = True

    #####################提取数据#################
    data = np.loadtxt(filePath, dtype=np.float32)
    if with_label:
        labels = data[:, -1]
        labels = np.array(labels, dtype=np.int32)
        data = data[:, :-1]
    else:
        labels = np.loadtxt(labelPath, dtype=np.int32)
    dim = data.shape[1]#数据维度（不含label）
    cluster_num = max(labels) - min(labels) + 1

    ########################归一化###################################
    data_without_nml = data.copy()
    for j in range(data.shape[1]):
        max_ = max(data[:, j])
        min_ = min(data[:, j])
        if max_ == min_:
            continue
        for i in range(data.shape[0]):
            data[i][j] = (data[i][j] - min_) / (max_ - min_)
    np.savetxt(file_name + "_normalization.txt", data)  # 改改改


    #######################KMeans####################
    ddata = data.copy()
    k = 10
    T = 0.8
    threshold = 1.4286
    iterationTime = 29
    ######################计算裁剪阈值，以确定有效邻居##############################
    photoPath = "decision_" + str(k) + "_" + str(threshold) + ".png"
    distanceTGP = TGP(ddata, k, photoPath, threshold)  # we can determine the threshold，并返回权重矩阵，根据矩阵和阈值确定有效邻居
    ##########如果只需要简单调用HIAC，在已知最优参数情况下############
    global disIndex
    disIndex = np.zeros([data.shape[0], data.shape[0]], dtype=np.int32) #disIndex后续用于保存该点与有效邻居以及其他样本点的关系
    for i in range(iterationTime):
        bata = shrink(ddata, k, T, threshold, distanceTGP, i)
        ddata = bata
    np.savetxt("./dataset-after-HIAC-kmeans.txt", ddata)
    res = sc.KMeans(n_clusters=cluster_num).fit(ddata)
    jieguo1 = metrics.adjusted_mutual_info_score(labels, res.labels_, average_method='max')


    #######################AGG####################
    ddata = data.copy()
    k = 5
    T = 0.5
    threshold = 1.5176
    iterationTime = 1
    ######################计算裁剪阈值，以确定有效邻居##############################
    photoPath = "decision_" + str(k) + "_" + str(threshold) + ".png"
    distanceTGP = TGP(ddata, k, photoPath, threshold)  # we can determine the threshold，并返回权重矩阵，根据矩阵和阈值确定有效邻居
    ##########如果只需要简单调用HIAC，在已知最优参数情况下############
    disIndex = np.zeros([data.shape[0], data.shape[0]], dtype=np.int32)  # disIndex后续用于保存该点与有效邻居以及其他样本点的关系
    for i in range(iterationTime):
        bata = shrink(ddata, k, T, threshold, distanceTGP, i)
        ddata = bata
    np.savetxt("./dataset-after-HIAC-agg.txt", ddata)
    res = sc.AgglomerativeClustering(n_clusters=cluster_num).fit(ddata)
    jieguo2 = metrics.adjusted_mutual_info_score(labels, res.labels_, average_method='max')


    #######################DPC####################
    ddata = data.copy()
    k = 6
    T=0.3
    threshold = 1.514
    iterationTime = 4
    ######################计算裁剪阈值，以确定有效邻居##############################
    photoPath = "decision_" + str(k) + "_" + str(threshold) + ".png"
    distanceTGP = TGP(ddata, k, photoPath, threshold)  # we can determine the threshold，并返回权重矩阵，根据矩阵和阈值确定有效邻居
    ##########如果只需要简单调用HIAC，在已知最优参数情况下############
    disIndex = np.zeros([data.shape[0], data.shape[0]], dtype=np.int32)  # disIndex后续用于保存该点与有效邻居以及其他样本点的关系
    for i in range(iterationTime):
        bata = shrink(ddata, k, T, threshold, distanceTGP, i)
        ddata = bata
    np.savetxt("./dataset-after-HIAC-DPC.txt", ddata)
    res = DPC(ddata, cluster_num)
    jieguo3 = metrics.adjusted_mutual_info_score(labels, res, average_method='max')

    print("KMeans: {:.4f}; AGG: {:.4f}; DPC: {:.4f}".format(jieguo1, jieguo2, jieguo3))