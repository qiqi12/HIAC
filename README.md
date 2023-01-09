# HIAC: How to improve the accuracy of clustering algorithms
## Code version
· Python
## Instruction of use
You only need to edit HIAC.py, select appropriate parameters(i.e. d,T,k,threshold) and run it.
'''
  data = np.loadtxt("data-sets/S-set/s1.txt")# s1.txt can be relpacecd by any other data set under the 'data-sets' folder. If the txt file contains label, you need to separate the data and label. Maybe normalization is necessary for the data.
  distanceTGP = TGP(data, k, photo_path, threshold)# You can decide the sava path of the decision-graph by setting the parameter photo_path. If you want to save the decision-graph with the threshold clearly marked, you can replace parameter threshold(default None) with the selected threshold.
  neighbor_index = prune(data, k, threshold, distanceTGP)# call prune to clip the invalid-neighbor
  for i in range(d):# shrink objects to ameliorate dataset
    bata = shrink(data, k, T, neighbor_index)
    data = bata
'''
