import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
# encoding = utf-8
from matplotlib.font_manager import FontProperties

font_size = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=20)
fontProperties = font_size


#加载数据集
data_x_label = np.array([600, 755, 1470, 2314, 5488, 14000,17070, 115008])
data_x = np.array([1,2,3,4,5,6,7,8])
# data_y_label = np.array([0.1, 5, 10, 50, 500,1000,5000,20000,50000])
# data_y = np.array([0.01, 0.5, 1, 2, 5, 6, 10, 13, 16])

#data_y_label = np.array([0.05, 0.1, 10, 50, 500,1000,5000,20000,50000])
#data_y = np.array([0.01, 0.5, 1, 2, 5, 6, 10, 13, 16])
data_y_label = np.array([0.05, 0.1, 10, 50, 500,1000,5000,20000,50000])
data_y = np.array([0.01, 0.5, 1, 2, 5, 6, 10, 13, 16])

data_hiac_truth = np.array([0.119, 0.363, 0.027, 0.321, 12.822, 3.763, 0.670, 8.605])
data_hiac = np.array([0.501, 0.513, 0.0054, 0.511, 1.071, 0.685, 0.529, 0.930])

data_hibog_truth =np.array([0.02, 0.023, 0.043, 0.077, 2.07, 3.72, 0.44, 5.79])
#data_hibog =np.array([0.002, 0.0023, 0.0043, 0.0077, 0.207, 0.3372, 0.044, 0.579])
data_hibog =np.array([0.004, 0.0046, 0.0086, 0.2756, 0.599, 0.683, 0.517, 0.787])

data_y_label = np.array([0.05, 0.1, 10, 50, 500,1000,5000,20000,50000])
data_y = np.array([0.01, 0.5, 1, 2, 5, 6, 10, 13, 16])
data_SCBA_truth = np.array([0.08, 0.10, 0.55, 0.81, 15.76, 53.68, 18.40, 378.19])
#data_SCBA = np.array([0.008, 0.01, 0.055, 0.081, 1.144, 2.025, 1.21, 4.188])
data_SCBA = np.array([0.304, 0.5, 0.523, 0.536, 1.144, 2.025, 1.21, 4.188])

data_newm_truh = np.array([56.76, 118.69, 483.43, 1078.10, 6210.75, 43831.61])
data_newm = np.array([2.045, 2.4579, 4.8895, 6.0781, 10.24215, 15.383161])

data_y_label = np.array([0.05, 0.1, 10, 50, 500,1000,5000,20000,50000])
data_y = np.array([0.01, 0.5, 1, 2, 5, 6, 10, 13, 16])
data_herb_truth = np.array([8.11, 5.52, 12.66, 10.31, 3987.01, 15197.34, 602.68, 32307])
#data_herb = np.array([0.811, 0.552, 1.0665, 1.00775, 8.98701, 12.0394, 5.20536, 14.2307])
data_herb = np.array([0.905, 0.774, 1.0665, 1.00775, 8.98701, 12.0394, 5.20536, 14.2307])
cValue = ['#071E3D','#278EA5','#21E6C1','#FFBA5A','#FF7657','#C56868','#6b76ff']
fig = plt.figure(figsize=(7, 5))
# plt.plot(data_x, data_hibog,linewidth= 2.5, c='#fe5f55',label='HIBOG')
# plt.plot(data_x[:6], data_newm,linewidth= 2.5, c='#7b88ff',label='Newtonian')
# plt.plot(data_x[:7], data_herb,linewidth= 2.5, c=cValue[3],label='Herd')

plt.plot(data_x[:6], data_newm,linewidth= 2.5, c='#ffa323',label='Newtonian',marker='o')
plt.plot(data_x, data_herb,linewidth= 2.5, c='#71a0a5',label='Herd',marker='o')
plt.plot(data_x, data_SCBA,linewidth= 2.5, c='#df7599',label='SBCA',marker='o')
plt.plot(data_x, data_hibog,linewidth= 2.5, c='#ff5f00',label='HIBOG',marker='o')
plt.plot(data_x, data_hiac,linewidth= 2.5, c='#3e64ff',label='HIAC',marker='o')
plt.xticks(data_x, data_x_label, size=10)
plt.yticks(data_y, data_y_label, size=10)
plt.title("Running time", fontstyle='italic', size=20)
plt.xlabel('dim*num', fontstyle='italic', size=20)
plt.ylabel('second', fontstyle='italic', size=20)
plt.legend(loc="upper left", fontsize=15)
plt.savefig('./time.png', dpi=300, bbox_inches='tight')
plt.show()


