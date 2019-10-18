import pandas as pd
from io import StringIO
from sklearn.preprocessing import Imputer
from sklearn.impute import  SimpleImputer

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

print(deletedAllNan)

# 补全缺失的数据

imr = Imputer(missing_values='NaN', strategy='mean',axis=0)
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
print(imputed_data)
