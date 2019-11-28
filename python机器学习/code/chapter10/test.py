import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('housing.data', header=None, sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
              'NOX', 'RM', 'AGE', 'DIS', 'RAD',
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head()

cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']


# 散点图矩阵可以展示数据集内部特征之间的关系

# sns.pairplot(df[cols], size=2.5)
# plt.tight_layout()
# plt.savefig('images/10_03.png', dpi=300)
# plt.show()


# 绘制基于关联矩阵的热力图

# cm = np.corrcoef(df[cols].values.T)
# # sns.set(font_scale=1.5)
# hm = sns.heatmap(cm,
#                  cbar=True,
#                  annot=True,
#                  square=True,
#                  fmt='.2f',
#                  annot_kws={'size': 15},
#                  yticklabels=cols,
#                  xticklabels=cols)
#
# plt.tight_layout()
# # plt.savefig('images/10_04.png', dpi=300)
# plt.show()


# 普通最小二乘法线性回归模型的实现

class LinearRegressionGD(object):

    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return self.net_input(X)


X = df[['RM']].values
y = df['MEDV'].values

sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()

lr = LinearRegressionGD()
lr.fit(X_std, y_std)

# plt.plot(range(1, lr.n_iter + 1), lr.cost_)
# plt.ylabel('SSE')
# plt.xlabel('Epoch')
# # plt.tight_layout()
# # plt.savefig('images/10_05.png', dpi=300)
# plt.show()


# sklearn 中的线性回归模型

from sklearn.linear_model import LinearRegression

slr = LinearRegression()
slr.fit(X, y)


def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)
    return


# lin_regplot(X_std, y_std, lr)
# plt.xlabel('Average number of rooms [RM] (standardized)')
# plt.ylabel('Price in $1000s [MEDV] (standardized)')
#
# # plt.savefig('images/10_06.png', dpi=300)
# plt.show()

# 利用RANSAN拟合稳健的回归模型

from sklearn.linear_model import RANSACRegressor

ransac = RANSACRegressor(LinearRegression(),
                         max_trials=100,
                         min_samples=50,
                         loss='absolute_loss',
                         residual_threshold=5.0,
                         random_state=0)

ransac.fit(X, y)

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
# plt.scatter(X[inlier_mask], y[inlier_mask],
#             c='steelblue', edgecolor='white',
#             marker='o', label='Inliers')
# plt.scatter(X[outlier_mask], y[outlier_mask],
#             c='limegreen', edgecolor='white',
#             marker='s', label='Outliers')
# plt.plot(line_X, line_y_ransac, color='black', lw=2)
# plt.xlabel('Average number of rooms [RM]')
# plt.ylabel('Price in $1000s [MEDV]')
# plt.legend(loc='upper left')

# plt.savefig('images/10_08.png', dpi=300)
# plt.show()

print('Slope: %.3f' % ransac.estimator_.coef_[0])
print('Intercept: %.3f' % ransac.estimator_.intercept_)

# 评估线性回归模型的性能

# 残差 : 实际值和预测值之间的差异或垂直距离

# 完美的线性回归模型的残差应该为0

from sklearn.model_selection import train_test_split

X = df.iloc[:, :-1].values
y = df['MEDV'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

slr = LinearRegression()

slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

# plt.scatter(y_train_pred, y_train_pred - y_train,
#             c='steelblue', marker='o', edgecolor='white',
#             label='Training data')
# plt.scatter(y_test_pred, y_test_pred - y_test,
#             c='limegreen', marker='s', edgecolor='white',
#             label='Test data')
# plt.xlabel('Predicted values')
# plt.ylabel('Residuals')
# plt.legend(loc='upper left')
# plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
# plt.xlim([-10, 50])
# plt.tight_layout()
#
# # plt.savefig('images/10_09.png', dpi=300)
# plt.show()

# 多项式回归

# 使用sklearn来增加多项式的项

from sklearn.preprocessing import PolynomialFeatures

X = np.array([258.0, 270.0, 294.0,
              320.0, 342.0, 368.0,
              396.0, 446.0, 480.0, 586.0]) \
    [:, np.newaxis]

y = np.array([236.4, 234.4, 252.8,
              298.6, 314.2, 342.2,
              360.8, 368.0, 391.2,
              390.8])

lr = LinearRegression()
pr = LinearRegression()
quadratic = PolynomialFeatures(degree=2)  # 二次
X_quad = quadratic.fit_transform(X)

# fit linear features
lr.fit(X, y)
X_fit = np.arange(250, 600, 10)[:, np.newaxis]
y_lin_fit = lr.predict(X_fit)

# fit quadratic features
pr.fit(X_quad, y)
y_quad_fit = pr.predict(quadratic.fit_transform(X_fit))

# plot results
plt.scatter(X, y, label='training points')
plt.plot(X_fit, y_lin_fit, label='linear fit', linestyle='--')
plt.plot(X_fit, y_quad_fit, label='quadratic fit')
plt.legend(loc='upper left')

plt.tight_layout()
# plt.savefig('images/10_10.png', dpi=300)
plt.show()
