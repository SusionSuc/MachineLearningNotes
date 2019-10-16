from base.irisDataInit import X_train_std, y_train, y_test, X_test
from base.irisDataInit import plot_decision_regions
from base.irisDataInit import X_combined_std
from base.irisDataInit import y_combined
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# K-近邻(KNN)是一个监督学习算法，它适合数据集只有很少特征时使用。它的核心思想比较简单:
# 1. 选择k个数和一个距离度量
# 2. 找到要分类样本的 k-近邻
# 3. 以多数投票机制确定分类标签

# 由于没有训练步骤，所以不能丢弃训练样本。因此，如果要处理大型数据集，存储空间将面临挑战。

knn = KNeighborsClassifier(n_neighbors=5,
                           p=2,
                           metric='minkowski')
knn.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined,
                      classifier=knn, test_idx=range(105, 150))

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/03_24.png', dpi=300)
plt.show()

