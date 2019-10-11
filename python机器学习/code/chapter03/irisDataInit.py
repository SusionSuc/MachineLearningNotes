import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

iris = datasets.load_iris()

# print(iris)

X = iris.data[:, [2, 3]]
y = iris.target

# print('y :', y)
print('Class labels:', np.unique(y))  # 应该使用整数作为分类标签，这样可以避免很多问题
# print('X :', X)
print('y :', y, 'y length :', y.size)

# Splitting data into 70% training and 30% test data, 将样本分为 训练集 和 测试集
# train_test_split 它在分割数据前会将数据重新洗牌，避免 训练集和测试集数据不均
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# bincount 函数可以对阵列中每个值进行统计
print('Labels counts in y:', np.bincount(y))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))

# Standardizing the features 对数据进行特征缩放
sc = StandardScaler()
sc.fit(X_train)  # 对数据每个特征维度参数 \mu (样本均值) 和 \sigma (标准偏差) 进行估算
X_train_std = sc.transform(X_train)  # 利用估算好的  \mu (样本均值) 和 \sigma (标准偏差) 对数据进行标准化
X_test_std = sc.transform(X_test)  # 使用相同的 \mu (样本均值) 和 \sigma (标准偏差) 对测试数据进行标准化，保证训练集合测试集的数值具有可比性


# print('X_train : ', X_train)
# print('X_train_std : ', X_train_std)


# 绘制鸢尾花分类图， && 带决策边界
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # 创建网格
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    # ravel 函数可以展平数组元素
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')

    # highlight test samples
    if test_idx:
        # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100,
                    label='test set')
