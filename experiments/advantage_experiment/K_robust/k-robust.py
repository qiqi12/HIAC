import numpy as np
import scipy.spatial.distance as dis
from sklearn import decomposition as dec
import matplotlib.pyplot as plt
import matplotlib.patches as pat
global num
num = 0

def TGP(data, k, threshold):
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
    #plt.figure(num)
    #num += 1
    #plt.plot(probality[:, 0], probality[:, 1])
    #plt.axvline(x=threshold, c="r", ls="--", lw=1.5)
    #plt.title('Decision graph', fontstyle='italic', size=20)
    #plt.xlabel('wight', fontsize=20)
    #plt.ylabel('probability', fontsize=20)
    #plt.show()
    return distance_sort

def prune(data, knn, threshold, distanceTGP):
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
             # 后续迭代中，用此次迭代的邻居信息
                if distanceTGP[i, j] < threshold:
                    disIndex[i][j] = -1  # 消除引力
    return disIndex

def shrink(data, knn, T, disIndex):
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

    # print("G  ",G)

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



color = ['#125B50', '#4D96FF', '#FFD93D', '#FF6363', '#CE49BF', '#22577E', '#4700D8', '#F900DF', '#95CD41',
             '#FF5F00', '#40DFEF', '#8E3200', '#001E6C', '#C36A2D', '#B91646']  # 正常数据集
lineform = ['o']  # 噪声数据集、无需颜色代表类别的数据集

data1 = np.loadtxt('flame.txt', dtype=np.float32)
data2 = np.loadtxt('Aggregation.txt', dtype=np.float32)
label1 = data1[:, -1]
label2 = data2[:, -1]
data1 = data1[:, :-1]
data2 = data2[:, :-1]

for j in range(data1.shape[1]):
    max_ = max(data1[:, j])
    min_ = min(data1[:, j])
    if max_ == min_:
        continue
    for i in range(data1.shape[0]):
        data1[i][j] = (data1[i][j] - min_) / (max_ - min_)

for j in range(data2.shape[1]):
    max_ = max(data2[:, j])
    min_ = min(data2[:, j])
    if max_ == min_:
        continue
    for i in range(data2.shape[0]):
        data2[i][j] = (data2[i][j] - min_) / (max_ - min_)

size = 20
size1 = 32


plt.figure(1)
plt.figure(figsize=(31, 10))
plt.subplots_adjust(wspace=0.1, hspace =0.1)

ax = plt.subplot(251)
for i in range(1, 3):
    Together = []
    flag = 0
    for j in range(data1.shape[0]):
        if label1[j] == i:
            flag += 1
            Together.append(data1[j])
    Together = np.array(Together)
    Together = Together.reshape(-1, data1.shape[1])  # 在不区分类时，整理矩阵形状
    colorNum = i - 1  # 正常数据集
    formNum = 0  # 无需形状代表类别的数据集
    plt.scatter(Together[:, 0], Together[:, 1], size, color[colorNum], lineform[formNum])
plt.xticks([])
plt.yticks([])
ax.set_ylabel('Flame',fontdict={'style':'italic','weight':'light','size':size1})
ax.set_title('Original', fontstyle='italic', size=size1)

ax = plt.subplot(256)
for i in [1, 2, 4, 5, 6, 7, 3]:
    Together = []
    flag = 0
    for j in range(data2.shape[0]):
        if label2[j] == i:
            flag += 1
            Together.append(data2[j])
    Together = np.array(Together)
    Together = Together.reshape(-1, data2.shape[1])  # 在不区分类时，整理矩阵形状
    colorNum = i - 1  # 正常数据集
    formNum = 0  # 无需形状代表类别的数据集
    plt.scatter(Together[:, 0], Together[:, 1], size, color[colorNum], lineform[formNum])
plt.xticks([])
plt.yticks([])
ax.set_ylabel('Aggregation',fontdict={'style':'italic','weight':'light','size':size1})



distanceTGP1 = TGP(data1, 5, 0)
distanceTGP2 = TGP(data2, 5, 0)

disIndex1 = prune(data1, 25, 0.7936, distanceTGP1)
disIndex2 = prune(data1, 35, 0.7674, distanceTGP1)
disIndex3 = prune(data1, 45, 0.7456, distanceTGP1)
disIndex4 = prune(data1, 55, 0.7264, distanceTGP1)
disIndex5 = prune(data2, 25, 1.2008, distanceTGP2)
disIndex6 = prune(data2, 35, 1.1822, distanceTGP2)
disIndex7 = prune(data2, 45, 1.18369, distanceTGP2)
disIndex8 = prune(data2, 55, 1.17318, distanceTGP2)

bata2 = data2.copy()
bata1 = data1.copy()
for i in range(64):
    b = shrink(bata1, 25, 2.5, disIndex1)
    bata1 = b
ax=plt.subplot(252)
for i in range(1, 3):
    Together = []
    flag = 0
    for j in range(data1.shape[0]):
        if label1[j] == i:
            flag += 1
            Together.append(bata1[j])
    Together = np.array(Together)
    Together = Together.reshape(-1, data1.shape[1])  # 在不区分类时，整理矩阵形状
    colorNum = i - 1  # 正常数据集
    formNum = 0  # 无需形状代表类别的数据集
    plt.scatter(Together[:, 0], Together[:, 1], size, color[colorNum], lineform[formNum])
plt.xticks([])
plt.yticks([])
ax.set_title('K=25', fontstyle='italic', size=size1)

bata1 = data1.copy()
for i in range(51):
    b = shrink(bata1, 35, 1.7, disIndex2)
    bata1 = b
ax=plt.subplot(253)
for i in range(1, 3):
    Together = []
    flag = 0
    for j in range(data1.shape[0]):
        if label1[j] == i:
            flag += 1
            Together.append(bata1[j])
    Together = np.array(Together)
    Together = Together.reshape(-1, data1.shape[1])  # 在不区分类时，整理矩阵形状
    colorNum = i - 1  # 正常数据集
    formNum = 0  # 无需形状代表类别的数据集
    plt.scatter(Together[:, 0], Together[:, 1], size, color[colorNum], lineform[formNum])
plt.xticks([])
plt.yticks([])
ax.set_title('K=35', fontstyle='italic', size=size1)

bata1 = data1.copy()
for i in range(55):
    b = shrink(bata1, 45, 1.35, disIndex3)
    bata1 = b
ax=plt.subplot(254)
for i in range(1, 3):
    Together = []
    flag = 0
    for j in range(data1.shape[0]):
        if label1[j] == i:
            flag += 1
            Together.append(bata1[j])
    Together = np.array(Together)
    Together = Together.reshape(-1, data1.shape[1])  # 在不区分类时，整理矩阵形状
    colorNum = i - 1  # 正常数据集
    formNum = 0  # 无需形状代表类别的数据集
    plt.scatter(Together[:, 0], Together[:, 1], size, color[colorNum], lineform[formNum])
plt.xticks([])
plt.yticks([])
ax.set_title('K=45', fontstyle='italic', size=size1)

bata1 = data1.copy()
for i in range(60):
    b = shrink(bata1, 55, 1.1, disIndex4)
    bata1 = b
ax=plt.subplot(255)
for i in range(1, 3):
    Together = []
    flag = 0
    for j in range(data1.shape[0]):
        if label1[j] == i:
            flag += 1
            Together.append(bata1[j])
    Together = np.array(Together)
    Together = Together.reshape(-1, data1.shape[1])  # 在不区分类时，整理矩阵形状
    colorNum = i - 1  # 正常数据集
    formNum = 0  # 无需形状代表类别的数据集
    plt.scatter(Together[:, 0], Together[:, 1], size, color[colorNum], lineform[formNum])
plt.xticks([])
plt.yticks([])
ax.set_title('K=55', fontstyle='italic', size=size1)


bata2 = data2.copy()
for i in range(35):
    b = shrink(bata2, 25, 2.5, disIndex5)
    bata2 = b
ax=plt.subplot(257)
for i in [1, 2, 4, 5, 6, 7, 3]:
    Together = []
    flag = 0
    for j in range(data2.shape[0]):
        if label2[j] == i:
            flag += 1
            Together.append(bata2[j])
    Together = np.array(Together)
    Together = Together.reshape(-1, data2.shape[1])  # 在不区分类时，整理矩阵形状
    colorNum = i - 1  # 正常数据集
    formNum = 0  # 无需形状代表类别的数据集
    plt.scatter(Together[:, 0], Together[:, 1], size, color[colorNum], lineform[formNum])
plt.xticks([])
plt.yticks([])

bata2 = data2.copy()
for i in range(29):
    b = shrink(bata2, 35, 1.8, disIndex6)
    bata2 = b
ax=plt.subplot(258)
for i in [1, 2, 4, 5, 6, 7, 3]:
    Together = []
    flag = 0
    for j in range(data2.shape[0]):
        if label2[j] == i:
            flag += 1
            Together.append(bata2[j])
    Together = np.array(Together)
    Together = Together.reshape(-1, data2.shape[1])  # 在不区分类时，整理矩阵形状
    colorNum = i - 1  # 正常数据集
    formNum = 0  # 无需形状代表类别的数据集
    plt.scatter(Together[:, 0], Together[:, 1], size, color[colorNum], lineform[formNum])
plt.xticks([])
plt.yticks([])

bata2 = data2.copy()
for i in range(65):
    b = shrink(bata2, 45, 1.5, disIndex7)
    bata2 = b
ax=plt.subplot(259)
for i in [1, 2, 4, 5, 6, 7, 3]:
    Together = []
    flag = 0
    for j in range(data2.shape[0]):
        if label2[j] == i:
            flag += 1
            Together.append(bata2[j])
    Together = np.array(Together)
    Together = Together.reshape(-1, data2.shape[1])  # 在不区分类时，整理矩阵形状
    colorNum = i - 1  # 正常数据集
    formNum = 0  # 无需形状代表类别的数据集
    plt.scatter(Together[:, 0], Together[:, 1], size, color[colorNum], lineform[formNum])
plt.xticks([])
plt.yticks([])

bata2 = data2.copy()
for i in range(65):
    b = shrink(bata2, 55, 1.3, disIndex8)
    bata2 = b
ax=plt.subplot(2,5,10)
for i in [1, 2, 4, 5, 6, 7, 3]:
    Together = []
    flag = 0
    for j in range(data2.shape[0]):
        if label2[j] == i:
            flag += 1
            Together.append(bata2[j])
    Together = np.array(Together)
    Together = Together.reshape(-1, data2.shape[1])  # 在不区分类时，整理矩阵形状
    colorNum = i - 1  # 正常数据集
    formNum = 0  # 无需形状代表类别的数据集
    plt.scatter(Together[:, 0], Together[:, 1], size, color[colorNum], lineform[formNum])
plt.xticks([])
plt.yticks([])

plt.savefig('k_robust.png',dpi=300,bbox_inches='tight')
plt.show()







