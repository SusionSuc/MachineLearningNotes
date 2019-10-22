# 逆顺序选择(SBS) 可以从原始特征中选择子集，它可以应对分类器性能最小的衰减来降低初始特征子空间的维数。

from sklearn.metrics import accuracy_score
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from itertools import combinations
import numpy as np


class SBS:
    def __init__(self, estimator, k_features, scoring=accuracy_score, test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size,
                                                            random_state=self.random_state)
        dim = X_train.shape[1]  # 几列， 几个feature
        print("dim :", dim)
        self.indices = tuple(range(dim))
        self.subsets = [self.indices]

        print(" self.indices :", self.indices)
        print(" self.subsets :", self.subsets)

        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices)

        print(" init score:", score)

        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []
            # http://funhacks.net/2017/02/13/itertools/#combinations
            for p in combinations(self.indices, r=dim - 1):
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices = subsets[best]
            self.subsets.append(self.indices)
            dim -= 1

            print("best features ：", subsets[best])
            self.scores_.append(scores[best])

        print("每次迭代后的最佳 accuracy:", self.scores_)
        self.k_score_ = self.scores_[-1]

        print("  self.k_score_ ",   self.k_score_ )

        return self

    def transform(self, X):
        return X[:, self.indices]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score

from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

df_wine = pd.read_csv('./wine.data', header=None)  # 读取葡萄酒数据集
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=0,
                                                    stratify=y)
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)

knn = KNeighborsClassifier(n_neighbors=5)

# selecting features
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)

# plotting performance of feature subsets
k_feat = [len(k) for k in sbs.subsets]

plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
# plt.savefig('images/04_08.png', dpi=300)
plt.show()
