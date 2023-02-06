# HIAC: How to improve the accuracy of clustering algorithms
## Description
Clustering is an important data analysis technique. However, due to the diversity of datasets, each clustering algorithm is unable to produce satisfactory results on some particular datasets. Here, we propose a clustering optimization method called HIAC ([paper link](https://www.sciencedirect.com/science/article/abs/pii/S0020025523000919)). By introducing gravitation, HIAC forces objects in the dataset to move towards similar objects, making the ameliorated dataset more friendly to clustering algorithms (i.e., clustering algorithms can produce more accurate results on the ameliorated dataset). HIAC is independent of clustering principle, so it can optimize different clustering algorithms. In contrast to other similar methods, HIAC is the first to adopt the selective-addition mechanism, i.e., only adding gravitation between valid-neighbors, to avoid dissimilar objects approaching each other. In order to identify valid-neighbors from neighbors, HIAC introduces a decision graph, from which the naked eye can observe a clear division threshold. Additionally, the decision graph can assist HIAC in reducing the negative effects that improper parameter values have on optimization. We conducted numerous experiments to test HIAC. Experiment results show that HIAC can effectively ameliorate high-dimensional datasets, Gaussian datasets, shape datasets, the datasets with outliers, and overlapping datasets. HIAC greatly improves the accuracy of clustering algorithms, and its improvement rates are far higher than that of similar methods. The average improvement rate is as high as 253.6% (except maximum and minimum). Moreover, its runtime is significantly shorter than that of most similar methods. More importantly, with different parameter values, the advantages of HIAC over similar methods are always maintained. 
## Code version
* Python
## Instruction of use
You only need to edit HIAC.py, select appropriate parameters(i.e. d,T,k,threshold) and run it.
```
data = np.loadtxt("data-sets/S-set/s1.txt")# s1.txt can be relpacecd by any other data set under the 'data-sets' folder. If the txt file contains label, you need to separate the data and label. Maybe normalization is necessary for the data.
distanceTGP = TGP(data, k, photo_path, threshold)# You can decide the sava path of the decision-graph by setting the parameter photo_path. If you want to save the decision-graph with the threshold clearly marked, you can replace parameter threshold(default None) with the selected threshold.
neighbor_index = prune(data, k, threshold, distanceTGP)# call prune to clip the invalid-neighbors
for i in range(d):# shrink objects to ameliorate dataset
  bata = shrink(data, k, T, neighbor_index)
  data = bata
```
## Note
1. The code (HIAC.py) can be run directly, and we have enumerated the appropriate parameters for each dataset in file **parameter-config.xls**.
2. All datasets that we used for experiments are saved in the **data-sets** folder and are classified. We used 8 real-datasets in our comparison experiments, four of which (i.e. **Banknote authentication、Seeds、Teaching assistant evaluation、Wireless indoor location**) are given in folder **./data-sets/real-datasets** and the other four (i.e. **Breast cancer、Digit、Iris、Wine**) can be loaded from skearn. The code to load these datasets is as follows:
```
from sklearn.datasets import load_iris# iris dataset
from sklearn.datasets import load_wine# wine dataset
from sklearn.datasets import load_digits# digit dataset
from sklearn.datasets import load_breast_cancer# breast cancer dataset

iris = load_iris()
data = iris.data# 数据
label = iris.target# 标签
```
3. The file DPC.py is used for comparison experiments. Actually, three clustering algorithms are used for comparison experiments, the other two algorithms (i.e. **Kmeans、Agg**) can be loaded from sklearn. The code to load these algorithms is as follows:
```
import sklearn.cluster as sc
from DPC import *
from sklearn import metrics

res = sc.KMeans(n_clusters=cluster_num).fit(data)
label_kmeans = res.labels_

res = sc.AgglomerativeClustering(n_clusters=cluster_num).fit(data)
label_agg = res.labels_

label_dpc = DPC(data, cluster_num)
nmi = metrics.adjusted_mutual_info_score(true-label, label_dpc, average_method='max')
```
