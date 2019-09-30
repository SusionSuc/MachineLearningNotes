import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap


'''
批量梯度下降在计算大的数据集时代价非常昂贵，因为每向全局最小值走一步都需要重新评估整个训练集。
可以使用随机梯度下降法来代替。
'''

