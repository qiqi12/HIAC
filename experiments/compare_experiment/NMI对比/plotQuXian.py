import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

font_size = 20
title = "(H)"
ExcelPath = "wine-compare.xls" #三个算法在该数据集上的准确度表格
dataName = "wine" #数据集名称
df = pd.read_excel(ExcelPath, sheet_name='Sheet1')
Acc_descent = np.zeros([4, 3, 10], dtype=np.float32) #对应四个数据集上的KMeans Agg 和 DPC的准确度
nrows = df.shape[0]
ncols = df.columns.size
print(nrows, ncols)
for i in range(3):
    for row in range(10):
        flag = row + i * 11
        Acc_descent[0, i, row] = df.iloc[flag, 5]
        Acc_descent[1, i, row] = df.iloc[flag, 10]
        Acc_descent[2, i, row] = df.iloc[flag, 12]
        Acc_descent[3, i, row] = df.iloc[flag, 15]
print(Acc_descent)

x = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10']
size1 = 22.5

#################################
plt.figure(1)
plt.title(title, fontstyle='italic', size=20)
line1, = plt.plot(x, Acc_descent[3, 0], 'co-', markersize=8, linewidth=3)
line2, = plt.plot(x, Acc_descent[2, 0], 'yo-', markersize=8, linewidth=3)
line3, = plt.plot(x, Acc_descent[1, 0], 'ro-', markersize=8, linewidth=3)
line4, = plt.plot(x, Acc_descent[0, 0], 'bo-', markersize=8, linewidth=3)
plt.ylabel("NMI", fontstyle='italic', size=20)
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
#plt.legend(handles=[line1, line2, line3, line4], labels=["Herd", "SBCA", "HIBOG", "HIAC"], loc="lower left", fontsize=size1)
plt.savefig("./NMI对比图/" + dataName + "-kmeans.png", dpi=300, bbox_inches='tight')
plt.show()

##############################
plt.figure(2)
plt.title(title, fontstyle='italic', size=20)
plt.plot(x, Acc_descent[3, 1], 'co-', markersize=8, linewidth=3)
plt.plot(x, Acc_descent[2, 1], 'yo-', markersize=8, linewidth=3)
plt.plot(x, Acc_descent[1, 1], 'ro-', markersize=8, linewidth=3)
plt.plot(x, Acc_descent[0, 1], 'bo-', markersize=8, linewidth=3)
plt.ylabel("NMI", fontstyle='italic', size=20)
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
#plt.legend(handles=[line1, line2, line3, line4], labels=["Herd", "SBCA", "HIBOG", "HIAC"], loc="lower left", fontsize=size1)
plt.savefig("./NMI对比图/" + dataName + "-agg.png", dpi=300, bbox_inches='tight')
plt.show()

##############################
plt.figure(3)
plt.title(title, fontstyle='italic', size=20)
plt.plot(x, Acc_descent[3, 2], 'co-', markersize=8, linewidth=3)
plt.plot(x, Acc_descent[2, 2], 'yo-', markersize=8, linewidth=3)
plt.plot(x, Acc_descent[1, 2], 'ro-', markersize=8, linewidth=3)
plt.plot(x, Acc_descent[0, 2], 'bo-', markersize=8, linewidth=3)
plt.ylabel("NMI", fontstyle='italic', size=20)
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
#plt.legend(handles=[line1, line2, line3, line4], labels=["Herd", "SBCA", "HIBOG", "HIAC"], loc="lower left", fontsize=size1)
plt.savefig("./NMI对比图/" + dataName + "-dpc.png", dpi=300, bbox_inches='tight')
plt.show()