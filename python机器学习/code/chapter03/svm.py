from base.irisDataInit import X_train_std, y_train, y_test, X_test_std
from base.irisDataInit import plot_decision_regions
from base.irisDataInit import X_combined_std
from base.irisDataInit import y_combined
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# 感知器算法的目标是把分类误差减少到最小。而支持向量机算法优化的目标是寻找最大化的边界
# 边界定义为分离超平面(决策边界)与其最近的训练样本之间的距离，即所谓的支持向量机。_

svm = SVC(kernel='linear', C=1.0, random_state=1)

svm.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined,
                      classifier=svm, test_idx=range(105, 150))

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

# 使用核支持向量机求解非线性问题

# 核方法的逻辑是针对线性不可分数据,建立非线性组合，通过映射函数 \phi 把原始特征投影到一个高维空间，特征在该空间变的线性可分。

