# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 16:17:49 2022

@author: 佡儁
"""
'''
此程序可以先指定一系列K值，然后根据这些K值选择对应的阈值，然后开始迭代寻找每个k和阈值下的最佳迭代次数和准确度，最后从所有结果中取最优
程序中，K-set与threshold-set的元素是一一对应的，所以如果同一K需要测试几个阈值，则应重复几次
'''
import numpy as np
import scipy.spatial.distance as dis
import matplotlib.pyplot as plt
import matplotlib
import sklearn.cluster as sc
from sklearn import metrics
from sklearn.datasets import load_wine
from sklearn.datasets import load_digits
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
import xlwt
from DPC import *

global num
num = 1

def list2xls(Para,xlsname):  #文本转换成xls的函数，filename 表示一个要被转换的txt文本，xlsname 表示转换后的文件名
    x = 0                #在excel开始写的位置（y）
    y = 0                #在excel开始写的位置（x）
    xls=xlwt.Workbook()
    sheet = xls.add_sheet('sheet1',cell_overwrite_ok=True) #生成excel的方法，声明excel
    
    for i in range(len(Para)):
        for j in range(len(Para[i])):
            item = Para[i][j]
            if j==0 or j==2:
                item = round(item)
            elif j==1:
                item = round(item,1)
            elif j==4:
                item = round(item,3)
            else:
                item = round(item, 4)
            sheet.write(x,y,item)
            y += 1 #另起一列
        x += 1 #另起一行
        y = 0  #初始成第一列
    xls.save(xlsname+'.xls') #保存


def Plot(data,labels):
    global num
    global photoPath
    plt.figure(num)
    num += 1
    sortNumMax = np.max(labels)
    sortNumMin = np.min(labels)
    #print(sortNumMin,sortNumMax)
    #print(data)
    color = ['r','b','m','tomato','darkkhaki','g','c','plum','hotpink','tan','aqua','darkorange','moccasin','cadetblue']
    lineform = ['*','o','+','x']
    for i in range(sortNumMin,sortNumMax+1):
        Together = []
        flag = 0
        for j in range(data.shape[0]):
            if labels[j] == i:
                flag += 1
                Together.append(data[j])
        Together = np.array(Together)
        Together = Together.reshape(-1,data.shape[1])#  在不区分类时，整理矩阵形状
        #print(Together[:20])
        colorNum = np.mod(i+1,14)
        formNum = np.mod(i+1,4)
        plt.scatter(Together[:,0], Together[:,1], 15, color[colorNum], lineform[formNum])
    plt.xlabel('attribute 1', fontsize=20)
    plt.ylabel('attribute 2', fontsize=20)
    plt.savefig(photoPath, dpi=150, bbox_inches='tight')  ####改改
    
def TGP(data,k,photoPath,threshold):
    global num
    pointNum = data.shape[0]
    probality = np.zeros([int(pointNum/10),2])
    distance = dis.pdist(data)
    distance_matrix = -dis.squareform(distance)+np.max(distance)
    distance_sort = -np.sort(-distance_matrix, axis=1)

    pointWeight=np.mean(distance_sort[:,1:k+1],axis=1)
    dc=(max(pointWeight)-min(pointWeight))*10/pointNum
    for i in range(probality.shape[0]):
        location = min(pointWeight) + dc * i
        probality[i,0] = location
    for i in range(pointNum):
        j = (int)((pointWeight[i] - min(pointWeight))/dc)
        if j < int(pointNum/10):
            probality[j,1] += 1
    probality[:,1] = probality[:,1]/pointNum
    plt.figure(num)
    num += 1
    plt.plot(probality[:, 0], probality[:, 1])
    for i in threshold:
        plt.axvline(x=i, c="r", ls="--", lw=1.5)
    plt.title('C (compact-large)', fontstyle='italic', size=20)
    plt.savefig(photoPath, dpi=150, bbox_inches='tight')  ####改改
    if chooseThreshold == 1:
        plt.show()
    return distance_sort

def shrink(data,knn,T,threshold,distanceTGP,iterationTime):
    global disIndex
    bata = data.copy()
    pointNum = data.shape[0]
    distance = dis.pdist(data)
    distance_matrix = dis.squareform(distance)
    distance_sort = np.sort(distance_matrix, axis=1)
    area = np.mean(distance_sort[:,round(pointNum*0.015)])
    density = np.zeros(pointNum)
    if iterationTime == 0:
        disIndex = np.argsort(distance_matrix, axis=1)#记录第一次迭代时knn
    #计算密度
    for i in range(pointNum):
        num = 1
        while(distance_sort[i,num] < area):
            num += 1
        density[i] = num
    densityThreshold = np.mean(density)
    G = np.mean(distance_sort[:,1])

    #print("G  ",G)
    
    for i in range(pointNum):
        if density[i] < densityThreshold:
            displacement = np.zeros(data.shape[1],dtype=np.float32)
            for j in range(knn+1):#第j个邻居
                if (data[disIndex[i][j]] == data[i]).all():
                    continue
                else:
                    if iterationTime == 0:#第一次迭代，从knn中选择有引力的邻居，无引力的标记为-1
                                        #后续迭代中，用此次迭代的邻居信息
                        if distanceTGP[i, j] < threshold:
                            disIndex[i][j] = -1#消除引力
                    if disIndex[i][j] != -1:
                        ff = (data[disIndex[i][j]] - data[i])
                        fff = (distance_sort[i, 1] / (distance_matrix[i, disIndex[i][j]] * distance_matrix[i, disIndex[i][j]]))
                        displacement += G * ff * fff
            bata[i] = data[i] + displacement * T

        #if i%1000 == 0:
        #   print(i/1000,"  ",bata[i])```````
    return bata

if __name__ == "__main__":
    ######################初始化####################################
    #matplotlib.use('AGG')
    
    global disIndex
    global photoPath
    savePath = "./"
    filePath = "./data_banknote_authentication(done).txt"#改改改
    xlsPath_Kmeans = "./Kmeans_compare_k"
    xlsPath_Agg = "./Agg_compare_k"
    xlsPath_Dpc = "./Dpc_compare_k"

    iterationTime = 20#改改改
    isIter = 1
    chooseThreshold = 0
    Kset = [6, 6, 8, 8, 10, 10, 12, 12, 14, 14, 16, 16, 18, 18, 20, 22, 24, 24]#bank
    ThresholdSet = [1.4193, 1.4345, 1.4167, 1.3913, 1.3804, 1.4180, 1.3730, 1.3960, 1.3785, 1.4012, 1.3899, 1.3567, 1.3681, 1.3949, 1.3604, 1.3644, 1.3717, 1.3565]#bank
    Tset = [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.]#bank

    
    ############################data#################################
    data = np.loadtxt(filePath, dtype=np.float64)
    target = data[:,-1]
    data = data[:,:-1]
    target = np.array(target, dtype=np.int32())
    cluster_num = max(target) - min(target) + 1
    print(max(target), min(target))
    
    #########################归一化###################################
    for j in range(data.shape[1]):
        max_ = max(data[:,j])
        min_ = min(data[:,j])
        if max_ == min_:
            continue
        for i in range(data.shape[0]):
            data[i][j] = (data[i][j] - min_)/(max_ - min_)
    
    np.savetxt("./bank-normal.txt", data)#改改改
    #print(data[:10])
    
    ############################迭代收缩#############################
    if isIter:
        resultParaList_Kmeans = []#存放每次迭代后的结果和参数
        resultParaList_Agg = []#存放每次迭代后的结果和参数
        resultParaList_Dpc = []
        resultPara_everyK_Kmeans = []#记录每个K值下的最佳性能下的参数
        resultPara_everyK_Agg = []
        resultPara_everyK_Dpc = []

        #不在执行的过程中寻找最大值，执行完后统一处理
        for j in range(len(Kset)):
            ########################shrink###################################
            k = Kset[j]
            disIndex = np.zeros([data.shape[0], data.shape[0]], dtype=np.int32)
            threshold = ThresholdSet[j]#得到k值对应的阈值
            photoPath = "./bank-decision.png"
            distanceTGP = TGP(data, k, photoPath, [threshold])#data after prune,we can determine the threshold
            #print(distanceTGP)
            maxScore_Kmeans = -1.
            maxScore_Agg = -1.
            maxScore_Dpc = -1.
            best_para_everyK_Kmeans = np.zeros(5)
            best_para_everyK_Agg = np.zeros(5)
            best_para_everyK_Dpc = np.zeros(5)

            if chooseThreshold == 0:
                print("################## ", k, " #######################")
                for T in Tset:
                    ddata = data.copy()#选取每个T值时都应重头开始迭代
                    for i in range(iterationTime):
                        dataAfterStrink = shrink(ddata, k, T, threshold, distanceTGP, i)
                        ddata = dataAfterStrink
                        result_Kmeans = sc.KMeans(n_clusters=cluster_num).fit(ddata)
                        jieguo_Kmeans = metrics.adjusted_mutual_info_score(target, result_Kmeans.labels_, average_method='max')
                        result_Agg = sc.AgglomerativeClustering(n_clusters=cluster_num).fit(ddata)
                        jieguo_Agg = metrics.adjusted_mutual_info_score(target, result_Agg.labels_, average_method='max')
                        result_Dpc = DPC(ddata, cluster_num)
                        jieguo_Dpc = metrics.adjusted_mutual_info_score(target, result_Dpc, average_method='max')
                        ##########如果需要记录每个k下每个T下的结果值，需要下面三行##################
                        resultParaList_Kmeans.append([1, k, T, i+1, threshold, jieguo_Kmeans])#保存参数
                        resultParaList_Agg.append([2, k, T, i+1, threshold, jieguo_Agg])
                        resultParaList_Dpc.append([3, k, T, i+1, threshold, jieguo_Dpc])

                        if jieguo_Kmeans > maxScore_Kmeans:
                            maxScore_Kmeans = jieguo_Kmeans
                            best_para_everyK_Kmeans = k, T, i+1, threshold, maxScore_Kmeans
                        if jieguo_Agg > maxScore_Agg:
                            maxScore_Agg = jieguo_Agg
                            best_para_everyK_Agg = k, T, i+1, threshold, maxScore_Agg
                        if jieguo_Dpc > maxScore_Dpc:
                            maxScore_Dpc = jieguo_Dpc
                            best_para_everyK_Dpc = k, T, i+1, threshold, maxScore_Dpc


                    print("K:  ", k, " T: ", T, "    maxKmeans: ", maxScore_Kmeans, "    maxAgg: ", maxScore_Agg, "    maxDpc: ", maxScore_Dpc)
                resultPara_everyK_Kmeans.append(best_para_everyK_Kmeans)
                resultPara_everyK_Agg.append(best_para_everyK_Agg)
                resultPara_everyK_Dpc.append(best_para_everyK_Dpc)

        list2xls(resultParaList_Kmeans, xlsPath_Kmeans)
        list2xls(resultParaList_Agg, xlsPath_Agg)
        list2xls(resultParaList_Dpc, xlsPath_Dpc)
        list2xls(resultPara_everyK_Kmeans, xlsPath_Kmeans)
        list2xls(resultPara_everyK_Agg, xlsPath_Agg)
        list2xls(resultPara_everyK_Dpc, xlsPath_Dpc)