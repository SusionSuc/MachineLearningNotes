from sklearn.tree import DecisionTreeClassifier
from base.irisDataInit import X_train, y_train, y_test, X_test
from base.irisDataInit import plot_decision_regions
import numpy as np
import matplotlib.pyplot as plt

tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
tree.fit(X_train, y_train)


X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined,
                      y_combined,
                      classifier=tree,
                      test_idx=range(105, 150))

plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.show()
