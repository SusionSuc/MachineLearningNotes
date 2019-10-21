import pandas as pd
from io import StringIO
from sklearn.preprocessing import Imputer
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

df = pd.read_csv(StringIO(csv_data))

# print(df)

# print("null column : ")
# print(df.isnull().sum())

# print(df.values)

# 删除缺失的数据

deletedByColumn = df.dropna(axis=0)

# print(deletedByColumn)

deletedByRow = df.dropna(axis=1)

# print(deletedByRow)

deletedAllNan = df.dropna(how='all')  # 全部都为 Nan时做删除

# print(deletedAllNan)

# 补全缺失的数据

imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
# print(imputed_data)

# 名词特征和序数特征

df = pd.DataFrame([['green', 'M', 10.1, 'class2'],
                   ['red', 'L', 13.5, 'class1'],
                   ['blue', 'XL', 15.3, 'class2']])

df.columns = ['color', 'size', 'price', 'classlabel']

## 映射序数特征

size_mapping = {'XL': 3, 'L': 2, 'M': 1}
df['size'] = df['size'].map(size_mapping)

print(df)

inv_size_mapping = {v: k for k, v in size_mapping.items()}

# print(inv_size_mapping)

df['size'] = df['size'].map(inv_size_mapping)

print(df)

# 为分类标签编码

print(np.unique(df['classlabel']))

class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}

print(class_mapping)

df['classlabel'] = df['classlabel'].map(class_mapping)

print(df)

inv_class_mapping = {v: k for k, v in class_mapping.items()}

df['classlabel'] = df['classlabel'].map(inv_class_mapping)

# LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)

print(y)

print(class_le.inverse_transform(y))

X = df[['color', 'size', 'price']].values

color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])

print(X)

# 独热编码
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(categorical_features=[0],sparse=False)

# print(ohe.fit_transform(X))

print(pd.get_dummies(df[['price', 'color', 'size']]))


