'''
only for DBSCAN and Birch
'''

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas
import sklearn.metrics as sm
import sklearn.utils
from sklearn import cluster
from HIAC import *
import scipy.spatial.distance as dis
from sklearn import decomposition as dec

def Myplot(nmi_or, nmi_hiac, title, save_path):
    '''
    给出两个nmi集合，画出他们的曲线并保存，由于是使用同一方法在多种组合的参数下实验，所以传入的是nmi集合
    Parameters
    ----------
    nmi_or：原始数据集用于聚类后的nmi集合
    nmi_hiac：经HIAC优化后的数据集用于聚类后的nmi集合
    title：图片名称
    save_path：图片保存地址

    Returns
    -------

    '''
    nmi_list = [0 for i in range(10)]
    nmi_list_hiac = [0 for i in range(10)]
    nmi_or = [nmi_or[i] * 10. for i in range(len(nmi_or))]
    nmi_hiac = [nmi_hiac[i] * 10. for i in range(len(nmi_hiac))]
    for i in range(len(nmi_or)):
        if nmi_or[i] == 10.:
            nmi_or[i] = 9
        if nmi_hiac[i] == 10.:
            nmi_hiac[i] = 9
        nmi_list[int(nmi_or[i])] += 1
        nmi_list_hiac[int(nmi_hiac[i])] += 1

    nmi_list = [nmi_list[i] / len(nmi_or) for i in range(len(nmi_list))]
    nmi_list_hiac = [nmi_list_hiac[i] / len(nmi_hiac) for i in range(len(nmi_list_hiac))]
    fig = plt.figure(figsize=(7, 5))
    x = np.array([0.1 * i for i in range(1, len(nmi_list) + 1)], dtype=np.float32)
    nmi_list = np.array(nmi_list, dtype=np.float32)
    nmi_list_hiac = np.array(nmi_list_hiac, dtype=np.float32)
    width = 0.04

    plt.bar(x - width / 2, nmi_list, width, color='#4D96FF', label='Before HIAC optimization')
    plt.bar(x + width / 2, nmi_list_hiac, width, color='#FF6363', label='After HIAC optimization')
    plt.xlabel("accuracy", fontsize=24)
    plt.ylabel("probability", fontsize=24)
    plt.xticks(x, size=20)
    plt.yticks(size=20)
    plt.title(title, fontsize=24, fontstyle='italic')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__  == "__main__":
    is_train = False #使用hiac优化，并计算在各参数组合上的nmi值
    is_train = False #画出使用同一聚类算法对优化前后数据集进行聚类的nmi对比图
    dir_path = "./data-sets/shapes"  #本实验只针对形状数据集，可以一次性跑遍所有需要的形状数据集，但先在字典中设置好参数
    sava_dir = "./experiments/compare_experiment" #保存的NMI表格（csv格式），本实验只测试了birch和DBSCAN优化前后的性能
    file_list = os.listdir(dir_path)
    plot_dir = "./experiments/compare_experiment/Birch" #不用更改，这是birch在各数据集上聚类结果的nmi表格位置
    #plot_dir = "./experiments/compare_experiment/DBSCAN" #不用更改，这是DBSCAN在各数据集上聚类结果的nmi表格位置
    plot_list = os.listdir(plot_dir)
    plot_save = plot_dir

    title_dict = {
        "Aggregation": "(F)",
        "flame": "(G)",
        "love": "(H)",
        "ls3_with_target": "(I)",
        "t710k(nonoise)": "(J)"
    }

    if not is_train:
        for item in plot_list:
            #if item != "Aggregation":  #可以通过此形式来指定只处理这一个数据集，item!=数据集名称
            #   continue
            file = os.path.join(plot_dir, item, item + "_th_after_hiac.csv")
            df = pandas.read_csv(file)
            nmi_or = df["nmi"].values.tolist()
            nmi_hiac = df["nmi_hiac"].values.tolist()
            sava_path = os.path.join(plot_save, item, item + "_nmi_bar_single.png")
            if item in title_dict.keys():
                item = title_dict[item]
            Myplot(nmi_or, nmi_hiac, item+" dataset", sava_path)

    cluster_num = {
        'Aggregation': 7,
        'flame': 2,
        "dim32": 16,
        "dim64": 16,
        "dim128": 16,
        "dim256": 16,
        "dim512": 16,
        "dim1024": 16,
        "Twomoons":2,
        "t710k(nonoise)": 9,
        "ls3_with_target": 6,
        "t48k(nonoise)": 6,
        "t88k(nonoise)": 8,
        "Compound_1": 5,
        "d6_with_target": 4,
        "love": 3
    }
    dic_para = {
        "Aggregation": [1.2008, 35, 1.5, 29],
        "flame": [0.7264, 55, 1.1, 60],
        "dim32": [564.5, 5, 0.5, 4],
        "dim64": [678, 5, 0.5, 4],
        "dim128": [873.4, 5, 0.5, 4],
        "dim256": [1309, 5, 0.5, 4],
        "dim512": [1791.4, 5, 0.5, 4],
        "dim1024": [2550, 5, 0.5, 4],
        "Twomoons": [0.98338, 30, 1., 4],
        "t710k(nonoise)": [1.26686, 50, 0.5, 4],
        #"ls3_with_target": [1.2540, 40, 0.4, 2],
        "ls3_with_target": [1.2540, 40, 1.2, 20],
        "t48k(nonoise)": [1.2572, 50, 0.5, 1],
        "t88k(nonoise)": [1.31587, 100, 0.6, 20],
        "Compound_1": [0.96799, 30, 1.4, 20],
        "d6_with_target": [1.2583, 30, 0.8, 20],
        "love": [0, 10, 3.6, 14]
    }
    bf_seed = {
        "Aggregation": 5,
        "flame": 1,
        "dim32": 100,
        "dim64": 100,
        "dim128": 100,
        "dim256": 100,
        "dim512": 100,
        "dim1024": 100,
        "Twomoons": 100,
        "t710k(nonoise)": 100,
        "ls3_with_target": 100,
        "t48k(nonoise)": 100,
        "t88k(nonoise)": 100,
        "Compound_1": 2,
        "d6_with_target": 10,
        "love": 10
    }

    if is_train: #train前先配置参数
        for item in file_list:
            #if item != "love.txt":
            #    continue
            if len(item.split("-")) > 1:  #如果label是单独的文件，其名称应命名为  数据集名-label.txt，此处跳过是因为将label和数据放在了一个文件夹下，此处跳过label
                continue
            name, _ = item.split(".")
            label_path = os.path.join(dir_path, name + "-label.txt")
            file_path = os.path.join(dir_path, item)
            data = np.loadtxt(file_path).astype(np.float32)
            label = data[:, -1] #此处注意数据集label在第几列，如果数据和label在一个文件中
            data = data[:, :-1]
            #label = np.loadtxt(label_path).astype(np.int32)
            data, label = sklearn.utils.shuffle(data, label)
            sava_dir_birch = os.path.join(sava_dir, "birch", name)
            sava_dir_dbscan = os.path.join(sava_dir, "DBSCAN", name)
            #######归一化#######
            for j in range(data.shape[1]):
                max_ = max(data[:, j])
                min_ = min(data[:, j])
                if max_ == min_:
                    continue
                for i in range(data.shape[0]):
                    data[i][j] = (data[i][j] - min_) / (max_ - min_)
            #######归一化#######

            #######call hiac##########
            distanceTGP = TGP(data, dic_para[name][1], "", dic_para[name][0])
            data2hiac = data.copy()
            for t in range(dic_para[name][3]):
                bata = shrink(iterationTime=t, data=data2hiac, threshold=dic_para[name][0], knn=dic_para[name][1], distanceTGP=distanceTGP, T=dic_para[name][2])
                data2hiac = bata
            #######call hiac##########

            distance = dis.pdist(data)
            distance_matrix = dis.squareform(distance)
            distance_sort = np.sort(distance_matrix)
            threshold = np.mean(distance_sort[:, round(0.02 * data.shape[0]) - 1])

            distance2hiac = dis.pdist(data2hiac)
            distance_matrix2hiac = dis.squareform(distance2hiac)
            distance_sort2hiac = np.sort(distance_matrix2hiac)
            threshold2hiac = np.mean(distance_sort[:, round(0.02 * data.shape[0]) - 1])

            if name == 'Compound_1':
                branch_factor_set = [bf_seed[name] * i for i in range(2, 12)]
                neighbour_num = [i * 1 for i in range(1, 11)]
            else:
                branch_factor_set = [bf_seed[name] * i for i in range(1, 11)]
                neighbour_num = [i for i in range(1, 11)]
            threshold_set = [threshold2hiac * (i/4) for i in range(1, 11)]
            threshold_set_hiac = [threshold2hiac * (i / 4) for i in range(1, 11)]

            if not os.path.exists(sava_dir_birch):
                os.makedirs(sava_dir_birch)
            save_path_birch = os.path.join(sava_dir_birch, name + "_th_after_hiac.csv")
            if not os.path.exists(sava_dir_dbscan):
                os.makedirs(sava_dir_dbscan)
            save_path_dbscan = os.path.join(sava_dir_dbscan, name + "_th_after_hiac.csv")
            para_res_birch = []
            para_res_dbscan = []

            print("*****************************process dataset ", item)
            for th, th_hiac in zip(threshold_set, threshold_set_hiac):
                for bf, number in zip(branch_factor_set, neighbour_num):
                    res = cluster.Birch(threshold=th, branching_factor=bf, n_clusters=cluster_num[name], compute_labels=True).fit(data)
                    res_hiac = cluster.Birch(threshold=th_hiac, branching_factor=bf, n_clusters=cluster_num[name], compute_labels=True).fit(data2hiac)
                    res_dbscan = cluster.DBSCAN(eps=th, min_samples=number).fit(data)
                    res_hiac_dbscan = cluster.DBSCAN(eps=th_hiac, min_samples=number).fit(data2hiac)

                    nmi = sm.adjusted_mutual_info_score(label, res.labels_, average_method='max')
                    nmi_hiac = sm.adjusted_mutual_info_score(label, res_hiac.labels_, average_method='max')
                    nmi_dbscan = sm.adjusted_mutual_info_score(label, res_dbscan.labels_, average_method='max')
                    nmi_hiac_dbscan = sm.adjusted_mutual_info_score(label, res_hiac_dbscan.labels_, average_method='max')

                    para_res_birch.append([th, bf, nmi, nmi_hiac])
                    para_res_dbscan.append([th, number, nmi_dbscan, nmi_hiac_dbscan])
                    #print("th: {}, bf: {}, nmi: {}, hiac nmi: {}".format(th, bf, nmi, nmi_hiac))
                    #print(res.labels_)
            print("******************************finish dataset ", item)
            df = pandas.DataFrame(para_res_birch, columns=['eps', 'min_samples', 'nmi', 'nmi_hiac'])
            #df.to_csv(save_path_birch, index=False)
            new_arr = df.loc[:, ['nmi', 'nmi_hiac']].values
            new_arr = new_arr >= 0.7
            print("valid parameter composition for original BIRCH: {}\nvalid parameter composition for HIAC BIRCH: {}".format(np.sum(new_arr[:, 0]), np.sum(new_arr[:, 1])))
            df_dbscan = pandas.DataFrame(para_res_dbscan, columns=['eps', 'min_samples', 'nmi', 'nmi_hiac'])
            df_dbscan.to_csv(save_path_dbscan, index=False)
            new_arr = df_dbscan.loc[:, ['nmi', 'nmi_hiac']].values
            new_arr = new_arr >= 0.7
            print("valid parameter composition for original DBSCAN: {}\nvalid parameter composition for HIAC DBSCAN: {}".format(np.sum(new_arr[:, 0]), np.sum(new_arr[:, 1])))
