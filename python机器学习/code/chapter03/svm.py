from base.irisDataInit import X_train_std, y_train, y_test, X_test_std
from base.irisDataInit import plot_decision_regions
from base.irisDataInit import X_combined_std
from base.irisDataInit import y_combined
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

# 感知器算法的目标是把分类误差减少到最小。而支持向量机算法优化的目标是寻找最大化的边界
# 边界定义为分离超平面(决策边界)与其最近的训练样本之间的距离，即所谓的支持向量机。_

svm = SVC(kernel='linear', C=1.0, random_state=1)

svm.fit(X_train_std, y_train)

# plot_decision_regions(X_combined_std, y_combined,
#                       classifier=svm, test_idx=range(105, 150))
#
# plt.xlabel('petal length [standardized]')
# plt.ylabel('petal width [standardized]')
# plt.legend(loc='upper left')
# plt.show()

# 使用核支持向量机求解非线性问题

# 核方法的逻辑是针对线性不可分数据,建立非线性组合，通过映射函数 \phi 把原始特征投影到一个高维空间，特征在该空间变的线性可分。

# # Solving non-linear problems using a kernel SVM

np.random.seed(1)
X_xor = np.random.randn(200, 2)
print("X_xor :", X_xor)

y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
print("y_xor :", y_xor)

y_xor = np.where(y_xor, 1, -1)

print("y_xor :", y_xor)

# plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], c='b', marker='x', label='1')

# plt.scatter(X_xor[y_xor == -1, 0], X_xor[y_xor == -1, 1], c='r', marker='s', label='-1')

# plt.xlim([-3, 3])
# plt.ylim([-3, 3])
# plt.legend(loc='best')
# plt.tight_layout()
# plt.savefig('images/03_12.png', dpi=300)
# plt.show()

# 术语"核"可以理解为两个样本之间的相似函数

# 参数 \gamma 的变化会导致决策边界的紧缩和波动。 并在控制过拟合问题上起着重要的作用
svm = SVC(kernel='rbf', random_state=1, gamma=10.10, C=10.0)
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor, classifier=svm)
plt.legend(loc='upper left')
plt.show()
