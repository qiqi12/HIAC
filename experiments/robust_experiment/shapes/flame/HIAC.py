# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 20:04:58 2022

@author: 佡儁
"""

import numpy as np
import scipy.spatial.distance as dis
import matplotlib.pyplot as plt
import sklearn.cluster as sc
from matplotlib import ticker
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import load_digits
from sklearn import decomposition as dec
import matplotlib.patches as pat
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
    ax = plt.gca()
    sortNumMax = np.max(labels)
    sortNumMin = np.min(labels)
    color = ['#CE49BF', '#22577E', '#4D96FF', '#125B50', '#FFD93D', '#FF6363',  '#4700D8', '#F900DF', '#95CD41',
             '#FF5F00', '#40DFEF', '#8E3200', '#001E6C', '#C36A2D', '#B91646']  # 正常数据集
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
        fontSize = 20
        colorNum = i - sortNumMin  # 正常数据集
        formNum = 0  # 无需形状代表类别的数据集
        plt.scatter(Together[:, 0], Together[:, 1], fontSize, color[colorNum], lineform[formNum])
    point1 = [0.18, 17.72]
    point11 = [4.02, 23.33]
    point2 = [9.54, 14.82]
    point22 = [14.46, 24.57]
    rect1 = pat.Rectangle(xy=(point1[0], point1[1]), width=(point11[0] - point1[0]), height=(point11[1] - point1[1]),
                          linewidth=3.5, fill=False, edgecolor='r')
    rect2 = pat.Rectangle(xy=(point2[0], point2[1]), width=(point22[0] - point2[0]), height=(point22[1] - point2[1]),
                          linewidth=3.5, fill=False, edgecolor='r')
    #ax.add_patch(rect1)
    #ax.add_patch(rect2)
    #plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(0.002))

    plt.xticks(size=20)
    plt.yticks(size=20)

    plt.title(sign, fontstyle='italic', size=20)
    plt.savefig(photoPath, dpi=300, bbox_inches='tight')
    plt.show()


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
    plt.savefig(photoPath, dpi=200, bbox_inches='tight')  ####改改
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
    filePath = "./Flame.txt"  # 改改改
    labelPath = "./data-sets/noise-datasets/Asymmetric-noise-label.txt"
    with_label = 1
    iterationTime = 64  # 改改改
    k = 25  # 改改改
    Tset = [2.5]  # 改改改
    threshold = 0.7936  # 改改改
    isIter = 1  # 改改改
    draw_clustering = 1  # 是否对原始数据集进行聚类并保存可视化图  改改改
    pca = dec.PCA(n_components=2)

    #####################当不是用sklearn自带数据集时，用以下程序提取数据#################
    data = np.loadtxt(filePath, dtype=np.float32)
    dim = data.shape[1] - 1  # 数据维度（不含label）  改改改
    if with_label == 1:
        labels = data[:, -1]
        labels = np.array(labels, dtype=np.int32())
        data = data[:, :-1]
    else:
        labels = np.loadtxt(labelPath, dtype=np.int32)
    dim = data.shape[1]
    cluster_num = max(labels) - min(labels) + 1

    disIndex = np.zeros([data.shape[0], data.shape[0]], dtype=np.int32)

    ###########对原始数据集聚类##############
    if draw_clustering == 1:
        result_Agg = sc.AgglomerativeClustering(n_clusters=cluster_num).fit(data)
        result_KMeans = sc.KMeans(n_clusters=cluster_num).fit(data)
        sign = '(C)'  # 每幅图上方的标号
        photoPath = "./original-Agg.png"
        Plot(data, result_Agg.labels_)
        sign = '(A)'  # 每幅图上方的标号
        photoPath = "./original-KMeans.png"
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
    data_nml = data.copy()
    photoPath = "./normal-original.png"
    sign = '(B)'  # 每幅图上方的标号
    Plot(data, labels)

    DecisionPath = "./decision_" + str(k) + "_" + str(threshold) + ".png"
    distanceTGP = TGP(data, k, DecisionPath, threshold)  # data after prune,we can determine the threshold

    #########################迭代收缩##################################
    if isIter:
        for T in Tset:
            ddata = data.copy()
            for i in range(iterationTime):
                dataAfterStrink = shrink(ddata, k, T, threshold, distanceTGP, i)
                ddata = dataAfterStrink
            if i == iterationTime - 1:  # 得到改善数据集的聚类结果，通过与i比较大小，可以得到每次迭代后的聚类结果
                ddataPCA = ddata.copy()
                result1 = sc.AgglomerativeClustering(n_clusters=cluster_num).fit(ddataPCA)
                result2 = sc.KMeans(n_clusters=cluster_num).fit(ddataPCA)
                jieguo1 = metrics.adjusted_mutual_info_score(labels, result1.labels_, average_method='max')
                jieguo2 = metrics.adjusted_mutual_info_score(labels, result2.labels_, average_method='max')

                sign = '(C)'
                photoPath = "./Agg-after-hiac.png"
                Plot(data_without_nml, result1.labels_)

                sign = '(A)'
                photoPath = "./Kmeans-after-hiac.png"
                Plot(data_without_nml, result2.labels_)
            photoPath = "after-hiac.png"
            sign = '(G)'
            Plot(ddata, labels)

