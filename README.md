# HIAC: How to improve the accuracy of clustering algorithms
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
