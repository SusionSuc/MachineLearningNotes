from sklearn.linear_model import Perceptron
from base.irisDataInit import X_train_std, y_train, y_test, X_test_std
from base.irisDataInit import plot_decision_regions
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# scikit-learn Perception

ppn = Perceptron(max_iter=40, eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)  # 训练模型

y_pred = ppn.predict(X_test_std)

print('Misclassified samples: %d' % (y_test != y_pred).sum())

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))  # 分类准确度

print('Accuracy: %.2f' % ppn.score(X_test_std, y_test))  # scikit-learn 分类器评分方法

# vstack 用来 竖直堆叠序列中的数组（行方向）  https://www.runoob.com/numpy/numpy-array-manipulation.html
X_combined_std = np.vstack((X_train_std, X_test_std))

print("X_combined_std", X_combined_std)

y_combined = np.hstack((y_train, y_test))  # hstack  用来 水平堆叠序列中的数组（列方向）

print("y_combined", y_combined)

plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn, test_idx=range(105, 150))

plt.xlabel('petal length [standardized]')

plt.ylabel('petal width [standardized]')

plt.legend(loc='upper left')

plt.tight_layout()

# plt.savefig('images/03_01.png', dpi=300)

plt.show()
