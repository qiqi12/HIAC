# HIAC: How to improve the accuracy of clustering algorithms
## Description
Clustering is an important data analysis technique. However, due to the diversity of datasets, each clustering algorithm is unable to produce satisfactory results on some particular datasets. In this paper, we propose a clustering optimization method called HIAC ([paper link](https://www.sciencedirect.com/science/article/abs/pii/S0020025523000919)). By introducing gravitation, HIAC forces objects in the dataset to move towards similar objects, making the ameliorated dataset more friendly to clustering algorithms (i.e., clustering algorithms can produce more accurate results on the ameliorated dataset). HIAC is independent of clustering principle, so it can optimize different clustering algorithms. In contrast to other similar methods, HIAC is the first to adopt the selective-addition mechanism, i.e., only adding gravitation between valid-neighbors, to avoid dissimilar objects approaching each other. In order to identify valid-neighbors from neighbors, HIAC introduces a decision graph, from which the naked eye can observe a clear division threshold. Additionally, the decision graph can assist HIAC in reducing the negative effects that improper parameter values have on optimization. We conducted numerous experiments to test HIAC. Experiment results show that HIAC can effectively ameliorate high-dimensional datasets, Gaussian datasets, shape datasets, the datasets with outliers, and overlapping datasets. HIAC greatly improves the accuracy of clustering algorithms, and its improvement rates are far higher than that of similar methods. The average improvement rate is as high as 253.6% (except maximum and minimum). Moreover, its runtime is significantly shorter than that of most similar methods. More importantly, with different parameter values, the advantages of HIAC over similar methods are always maintained. 
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
1. All the experiments we have done are saved in the **experiment** folder. They are robustness experiment (folder **robust_experiment**), contrast experiment (folder **compare_experiment**) and advantage analysis experiment (folder **advantage_experiment**) .The code (HIAC.py) in it can be run directly, and we have configured the appropriate parameters of HIAC.
2. All datasets are saved in the **data-sets** folder and are classified.
3. The file DPC.py is used in some python files. The compute_acc_probability.py calculate the accuracy of 100 sets of parameter pairs and plot the graph which describe the probabilities with different levels of accuracy. Before and after optimization, the probabilities of DBSCAN and BIRCH with different levels of accuracy. 
