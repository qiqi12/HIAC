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
                if (data[disIndex[i][j]] == data[i]).all():
                    continue
                else:
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



data1 = np.loadtxt('dim032.txt', dtype=np.float32)
data2 = np.loadtxt('dim064.txt', dtype=np.float32)
data3 = np.loadtxt('dim128.txt', dtype=np.float32)
data4 = np.loadtxt('dim256.txt', dtype=np.float32)
data5 = np.loadtxt('dim512.txt', dtype=np.float32)
data6 = np.loadtxt('dim1024.txt', dtype=np.float32)

pca = dec.PCA(n_components=2, svd_solver='randomized')

size = 0.5
size1 = 8
co ='#606470'

plt.figure(1)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
ax = plt.subplot(641)
plt.xticks([])
plt.yticks([])
ax.set_ylabel('dim32', fontdict={'style':'italic', 'weight': 'light', 'size': size1})
ax.set_title('original distribution', fontstyle='italic', size=size1)
jiang_original1 = pca.fit_transform(data1)
rect = pat.Rectangle(xy=(-218, -129), width=66, height=75, linewidth=1, fill=False, edgecolor='r')
ax.add_patch(rect)
plt.scatter(jiang_original1[:, 0], jiang_original1[:, 1], s=size, c=co)

ax = plt.subplot(645)
plt.xticks([])
plt.yticks([])
ax.set_ylabel('dim64', fontdict={'style': 'italic', 'weight': 'light', 'size': size1})
jiang_original2 = pca.fit_transform(data2)
rect = pat.Rectangle(xy=(42, -260), width=65, height=89, linewidth=1, fill=False, edgecolor='r')
ax.add_patch(rect)
plt.scatter(jiang_original2[:, 0], jiang_original2[:, 1], s=size, c=co)

ax = plt.subplot(649)
plt.xticks([])
plt.yticks([])
ax.set_ylabel('dim128', fontdict={'style': 'italic', 'weight': 'light', 'size': size1})
jiang_original3 = pca.fit_transform(data3)
plt.scatter(jiang_original3[:, 0], jiang_original3[:, 1], s=size, c=co)

ax = plt.subplot(6, 4, 13)
plt.xticks([])
plt.yticks([])
ax.set_ylabel('dim256', fontdict={'style': 'italic', 'weight': 'light', 'size': size1})
jiang_original4 = pca.fit_transform(data4)
plt.scatter(jiang_original4[:, 0], jiang_original4[:, 1], s=size, c=co)

ax = plt.subplot(6, 4, 17)
plt.xticks([])
plt.yticks([])
ax.set_ylabel('dim512', fontdict={'style': 'italic', 'weight': 'light', 'size': size1})
jiang_original5 = pca.fit_transform(data5)
plt.scatter(jiang_original5[:, 0], jiang_original5[:, 1], s=size, c=co)

ax = plt.subplot(6, 4, 21)
plt.xticks([])
plt.yticks([])
ax.set_ylabel('dim1024', fontdict={'style': 'italic', 'weight': 'light', 'size': size1})
jiang_original6 = pca.fit_transform(data6)
rect1 = pat.Rectangle(xy=(-463, 3), width=130, height=270, linewidth=1, fill=False, edgecolor='r')
rect2 = pat.Rectangle(xy=(359, -60), width=145, height=270, linewidth=1, fill=False, edgecolor='r')
#ax.add_patch(rect1)
#ax.add_patch(rect2)
plt.scatter(jiang_original6[:, 0], jiang_original6[:, 1], s=size, c=co)

distanceTGP32 = TGP(data1, 5, 564.5)
distanceTGP64 = TGP(data2, 5, 678)
distanceTGP128 = TGP(data3, 5, 873.4)
distanceTGP256 = TGP(data4, 5, 1309)
distanceTGP512 = TGP(data5, 5, 1791.4)
distanceTGP1024 = TGP(data6, 5, 2550)

disIndex1 = prune(data1, 5, 564.5, distanceTGP32)
disIndex2 = prune(data2, 5, 678, distanceTGP64)
disIndex3 = prune(data3, 5, 873.4, distanceTGP128)
disIndex4 = prune(data4, 5, 1309, distanceTGP256)
disIndex5 = prune(data5, 5, 1791.4, distanceTGP512)
disIndex6 = prune(data6, 5, 2550, distanceTGP1024)

for i in range(3):
    b1 = shrink(data1, 5, 0.5, disIndex1)
    data1 = b1
    b2 = shrink(data2, 5, 0.5, disIndex2)
    data2 = b2
    b3 = shrink(data3, 5, 0.5, disIndex3)
    data3 = b3
    b4 = shrink(data4, 5, 0.5, disIndex4)
    data4 = b4
    b5 = shrink(data5, 5, 0.5, disIndex5)
    data5 = b5
    b6 = shrink(data6, 5, 0.5, disIndex6)
    data6 = b6

    jiang1 = pca.fit_transform(data1)
    if i == 0:
        ax = plt.subplot(6, 4, 2 + i)
        plt.xticks([])
        plt.yticks([])
        plt.scatter(jiang1[:, 0], jiang1[:, 1], s=size, c=co)
        ax.set_title('first time-segment', fontstyle='italic', size=size1)
    if i == 1:
        ax = plt.subplot(6, 4, 2 + i)
        plt.xticks([])
        plt.yticks([])
        plt.scatter(jiang1[:, 0], jiang1[:, 1], s=size, c=co)
        ax.set_title('second time-segment', fontstyle='italic', size=size1)
    if i == 2:
        ax = plt.subplot(6, 4, 2 + i)
        plt.xticks([])
        plt.yticks([])
        plt.scatter(jiang1[:, 0], jiang1[:, 1], s=size, c=co)
        ax.set_title('final distribution', fontstyle='italic', size=size1)
    jiang2 = pca.fit_transform(data2)
    plt.subplot(6, 4, 6 + i)
    plt.xticks([])
    plt.yticks([])
    plt.scatter(jiang2[:, 0], jiang2[:, 1], s=size, c=co)
    jiang3 = pca.fit_transform(data3)
    plt.subplot(6, 4, 10 + i)
    plt.xticks([])
    plt.yticks([])
    plt.scatter(jiang3[:, 0], jiang3[:, 1], s=size, c=co)
    jiang4 = pca.fit_transform(data4)
    plt.subplot(6, 4, 14 + i)
    plt.xticks([])
    plt.yticks([])
    plt.scatter(jiang4[:, 0], jiang4[:, 1], s=size, c=co)
    jiang5 = pca.fit_transform(data5)
    plt.subplot(6, 4, 18 + i)
    plt.xticks([])
    plt.yticks([])
    plt.scatter(jiang5[:, 0], jiang5[:, 1], s=size, c=co)
    jiang6 = pca.fit_transform(data6)
    plt.subplot(6, 4, 22 + i)
    plt.xticks([])
    plt.yticks([])
    plt.scatter(jiang6[:, 0], jiang6[:, 1], s=size, c=co)
plt.savefig('dim.png', dpi=400, bbox_inches='tight')
plt.show()







