import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

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
# print("eigen_vals", eigen_vals)
# print("eigen_vecs", eigen_vecs)

# print(eigen_vecs.T)

# 3. 把特征向量按特征值降序排列

eign_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]

# print("eign_pairs", eign_pairs)

# 4. 收集前两个最大特征值的特征向量

print(eign_pairs[0][1])  # 第一个特征值的所用的特征向量
print((eign_pairs[0][1], eign_pairs[1][1]))  # 两个array 拼成了一个元组
print(eign_pairs[0][1][:, np.newaxis])  # 转置，并增加一个维度

# 5. 创建投影矩阵
# hstack 水平堆叠序列中的数组（列方向）
w = np.hstack((eign_pairs[0][1][:, np.newaxis],
               eign_pairs[1][1][:, np.newaxis]))

print('Matrix W:\n', w)

# 6. 利用投影矩阵将样本 x ( 1 * 13维的列向量) 转换到PCA子空间, 并画出新的样本分布图

X_train_std[0].dot(w)

X_train_pca = X_train_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
#
# for l, c, m in zip(np.unique(y_train), colors, markers):
#     plt.scatter(X_train_pca[y_train == l, 0],
#                 X_train_pca[y_train == l, 1],
#                 c=c, label=l, marker=m)
#
# plt.xlabel('PC 1')
# plt.ylabel('PC 2')
# plt.legend(loc='lower left')
# plt.tight_layout()
# # plt.savefig('images/05_03.png', dpi=300)
# plt.show()

# 使用 scikit learn 中的 PCA

from sklearn.decomposition import PCA

# pca = PCA(n_components=2)
# X_train_pca = pca.fit_transform(X_train_std)
# X_test_pca = pca.transform(X_test_std)
#
# plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1])
# plt.xlabel('PC 1')
# plt.ylabel('PC 2')
# plt.show()

from base.plotClassifier import plot_decision_regions

# 使用 PCA 转换过的数据做逻辑回归
from sklearn.linear_model import LogisticRegression

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

lr = LogisticRegression()
lr = lr.fit(X_train_pca, y_train)

plot_decision_regions(X_train_pca, y_train, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('images/05_04.png', dpi=300)
# plt.show()


# 保留所有主成分
pca = PCA()
X_train_pca = pca.fit_transform(X_train_std)

print("解释方差比:", pca.explained_variance_ratio_)