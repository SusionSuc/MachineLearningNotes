import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# 初始化数据

df_wine = pd.read_csv('./wine.data', header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

df_wine.head()

# Splitting the data into 70% training and 30% test subsets.

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

# PCA是一种无监督的线性变换技术, 它广泛应用于特征提取和降维。
# PCA的核心思想是 : 寻找高维数据中存在最大方差的方向，并将其投影到维数等于或小于原始数据的新子空间。

# python 实现 PCA


from sklearn.preprocessing import StandardScaler

# 1. 标准化
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)

print("X_train_std row    number : ", X_train_std.shape[0])
print("X_train_std column number : ", X_train_std.shape[1])


# 2. 计算协方差矩阵

cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

print("cov_mat row    number ", cov_mat.shape[0])
print("cov_mat column    number ", cov_mat.shape[1])
print("eigen_vals", eigen_vals)
print("eigen_vecs", eigen_vecs)
