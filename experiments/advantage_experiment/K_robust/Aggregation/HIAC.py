# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 20:04:58 2022

@author: 佡儁
"""

import numpy as np
import scipy.spatial.distance as dis
import matplotlib.pyplot as plt
import matplotlib
import sklearn.cluster as sc
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import load_digits
from sklearn import decomposition as dec

global num
num = 1


def Plot(data, labels):
    global num
    global photoPath
    plt.figure(num)
    num += 1
    sortNumMax = np.max(labels)
    sortNumMin = np.min(labels)
    # print(sortNumMin,sortNumMax)
    # print(data)
    color = ['#125B50', '#4D96FF', '#FFD93D', '#FF6363', '#CE49BF', '#22577E', '#4700D8', '#F900DF', '#95CD41',
             '#FF5F00', '#40DFEF', '#8E3200', '#001E6C', '#C36A2D', '#B91646']  # 正常数据集
    lineform = ['*', 'o', '+', 'x']  # 正常数据集
    # color = ['#db6400','#606470']#噪声数据集
    #color = ['#606470']#无需颜色代表类别的数据集
    lineform = ['o']  # 噪声数据集、无需颜色代表类别的数据集
    for i in range(sortNumMin, sortNumMax + 1):
        Together = []
        flag = 0
        for j in range(data.shape[0]):
            if labels[j] == i:
                flag += 1
                Together.append(data[j])
        Together = np.array(Together)
        Together = Together.reshape(-1, data.shape[1])  # 在不区分类时，整理矩阵形状
        colorNum = 1  # 噪声数据集
        formNum = 0  # 噪声数据集
        fontSize = 15
        # if i==0:#消融点的标签为0！！！
        #    colorNum = 0
        #    fontSize = 45
        colorNum = i - sortNumMin  # 正常数据集
        # formNum = np.mod(i+1,4)#正常数据集
        #colorNum = 0#无需颜色代表类别的数据集
        formNum = 0  # 无需颜色代表类别的数据集
        plt.scatter(Together[:, 0], Together[:, 1], fontSize, color[colorNum], lineform[formNum])
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    # plt.xlabel('attribute 1', fontsize=20)
    # plt.ylabel('attribute 2', fontsize=20)
    # plt.xticks(())#不显示x刻度
    # plt.yticks(())#不显示y刻度
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

    # print("G  ",G)

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
    filePath = "./data-sets/noise-datasets/Asymmetric-noise.txt"  # 改改改
    labelPath = "./data-sets/noise-datasets/Asymmetric-noise-label.txt"
    iterationTime = 40  # 改改改
    k = 20  # 改改改
    Tset = [0.6]  # 改改改
    threshold = 1.13094  # 改改改
    isIter = 0  # 改改改
    answer = 1  # 改改改
    draw_clustering = 0  # 是否对原始数据集进行聚类并保存可视化图  改改改
    cluster_num = 3  # 改改改
    pca = dec.PCA(n_components=2)

    '''################################data############################
    iris = load_s3()
    data = iris.data
    labels = iris.target
    dim = data.shape[1]#数据的维度
    print(labels)
    print(data.shape[1])
    '''  #####################当不是用sklearn自带数据集时，用以下程序提取数据#################
    data = np.loadtxt(filePath, dtype=np.float32)
    dim = data.shape[1]# 数据维度（不含label）  改改改
    if dim == data.shape[1] - 1:
        labels = data[:, -1]
        labels = np.array(labels, dtype=np.int32())
        print(labels[:20])
    else:
        labels = np.loadtxt(labelPath, dtype=np.int32)
        labels = np.zeros(data.shape[0], dtype=np.int32)
    data = data[:, :dim]
    print(len(data))
    print(max(labels), min(labels))
    ##################################################################################

    disIndex = np.zeros([data.shape[0], data.shape[0]], dtype=np.int32)
    photoPath = "./data-process/s3/s3-oringnal.png"
    Plot(data, labels)  # 原数据分布图

    if draw_clustering == 1:
        ###########对原始数据集聚类##############
        result_Agg = sc.AgglomerativeClustering(n_clusters=cluster_num).fit(data)
        result_KMeans = sc.KMeans(n_clusters=cluster_num).fit(data)
        photoPath = "./data-process/s3/s3-Agg.png"
        Plot(data, result_Agg.labels_)
        photoPath = "./data-process/s3/s3-KMeans.png"
        Plot(data, result_KMeans.labels_)
    ########################归一化###################################
    data_without_nml = data.copy()
    for j in range(data.shape[1]):
        max_ = max(data[:, j])
        min_ = min(data[:, j])
        if max_ == min_:
            continue
        for i in range(data.shape[0]):
            data[i][j] = (data[i][j] - min_) / (max_ - min_)
    np.savetxt("./data-process/s3/compound-1.txt", data)  # 改改改

    photoPath = "./data-process/s3/decision" + str(k) + "_" + str(threshold) + ".png"
    distanceTGP = TGP(data, k, photoPath, threshold)  # data after prune,we can determine the threshold
    # print(distanceTGP)

    ###########################################################
    dataDim = dim
    dataPCA = data.copy()
    if dim > 2:
        dataPCA = pca.fit_transform(data)
    photoPath = "./data-process/s3/s3.png"
    Plot(dataPCA, labels)
    ###########################################################

    # 迭代收缩
    if isIter:
        matplotlib.use('AGG')
        for T in Tset:
            ddata = data.copy()
            maxScore1 = -1.
            maxScore2 = -1.
            iterTime = 0
            for i in range(iterationTime):
                dataAfterStrink = shrink(ddata, k, T, threshold, distanceTGP, i)
                ddata = dataAfterStrink

                if i >= 0 and answer == 0:  # 只得到收缩后的图
                    ddataPCA = ddata.copy()
                    photoPath = "./data-process/s3/" + str(threshold) + "_" + str(k) + "_" + str(T) + "_" + str(
                        i + 1) + "_shrink.png"
                    Plot(ddataPCA, labels)

                    result1 = sc.AgglomerativeClustering(n_clusters=cluster_num).fit(ddataPCA)
                    result2 = sc.KMeans(n_clusters=cluster_num).fit(ddataPCA)
                    jieguo1 = metrics.normalized_mutual_info_score(labels, result1.labels_, average_method='max')
                    jieguo2 = metrics.normalized_mutual_info_score(labels, result2.labels_, average_method='max')
                    if maxScore1 < jieguo1:
                        maxScore1 = jieguo1
                        iterTime = i + 1
                    if maxScore2 < jieguo2:
                        maxScore2 = jieguo2
                        iterTime = i + 1

                    if dim > 2:
                        ddataPCA = pca.fit_transform(ddataPCA)
                    photoPath = "./data-process/s3/" + str(threshold) + "_" + str(k) + "_" + str(T) + "_" + str(
                        i + 1) + "_AGG.png"
                    Plot(data, result1.labels_)
                    photoPath = "./data-process/s3/" + str(threshold) + "_" + str(k) + "_" + str(T) + "_" + str(
                        i + 1) + "_KMeans.png"
                    Plot(data, result2.labels_)

                if i == iterationTime - 1 and answer == 1:  # 得到改善数据集的聚类结果
                    ddataPCA = ddata.copy()
                    photoPath = "./data-process/s3/color_" + str(threshold) + "_" + str(k) + "_" + str(T) + "_" + str(
                        i + 1) + "_shrink.png"
                    result1 = sc.AgglomerativeClustering(n_clusters=cluster_num).fit(ddataPCA)
                    result2 = sc.KMeans(n_clusters=cluster_num).fit(ddataPCA)
                    jieguo1 = metrics.adjusted_mutual_info_score(labels, result1.labels_, average_method='max')
                    jieguo2 = metrics.adjusted_mutual_info_score(labels, result2.labels_, average_method='max')
                    print(jieguo1, jieguo2)
                    if dim > 2:
                        ddataPCA = pca.fit_transform(ddataPCA)
                    Plot(ddataPCA, labels)  # 收缩后的图
                    if dim > 2:
                        ddataPCA = pca.fit_transform(data)
                    ##收缩后被用于聚类的图
                    photoPath = "./data-process/s3/1-cluster_after_shrink_" + str(threshold) + "_" + str(k) + "_" + str(
                        T) + "_" + str(i + 1) + ".png"
                    Plot(data_without_nml, result1.labels_)
                    photoPath = "./data-process/s3/2-cluster_after_shrink_" + str(threshold) + "_" + str(k) + "_" + str(
                        T) + "_" + str(i + 1) + ".png"
                    Plot(data_without_nml, result2.labels_)
                ########################################################
            print(iterTime, maxScore1, maxScore2)
