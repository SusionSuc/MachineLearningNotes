import pandas as pd
from io import StringIO
from sklearn.preprocessing import Imputer
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# 将数据集分为 训练集 和 测试集

df_wine = pd.read_csv('./wine.data', header=None)  # 读取葡萄酒数据集

print(df_wine)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']

print('Class labels', np.unique(df_wine['Class label']))  # 三种酒
print(df_wine.head())

from sklearn.model_selection import train_test_split

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

# 把特征保持在同一尺度上 ： 归一化 与 标准化

# 归一化通常是指把特征的比例调整到[0,1]区间

ex = np.array([0, 1, 2, 3, 4, 5])

print('normalized:', (ex - ex.min()) / (ex.max() - ex.min()))

from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()

X_train_norm = mms.fit_transform(X_train)

print(X_train_norm)

X_test_norm = mms.transform(X_test)

print(X_test)

# 标准化对于许多线性模型都十分有必要

print('standardized:', (ex - ex.mean()) / ex.std())

from sklearn.preprocessing import StandardScaler

stdsc = StandardScaler()

X_train_std = stdsc.fit_transform(X_train)

X_test_std = stdsc.transform(X_test)


# 选择有意义的特征

# 正则化

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(penalty='l1', C=1.0)
lr.fit(X_train_std, y_train)

print('Training accuracy:', lr.score(X_train_std, y_train))
print('Test accuracy:', lr.score(X_test_std, y_test))

print(lr.intercept_)

print(lr.coef_)

