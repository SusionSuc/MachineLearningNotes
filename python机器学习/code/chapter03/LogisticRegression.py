from base.irisDataInit import X_train_std, y_train, y_test, X_test_std
from base.irisDataInit import plot_decision_regions
from base.irisDataInit import X_combined_std
from base.irisDataInit import y_combined
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ### Logistic regression intuition and conditional probabilities


# def sigmoid(z):
#     return 1.0 / (1.0 + np.exp(-z))
#
#
# z = np.arange(-7, 7, 0.1)
# phi_z = sigmoid(z)
#
# plt.plot(z, phi_z)
# plt.axvline(0.0, color='k')
# plt.ylim(-0.1, 1.1)
# plt.xlabel('z')
# plt.ylabel('$\phi (z)$')
#
# # y axis ticks and gridline
# plt.yticks([0.0, 0.5, 1.0])
# ax = plt.gca()
# ax.yaxis.grid(True)
#
# plt.tight_layout()


# plt.show()


# ### Learning the weights of the logistic cost function


# def cost_1(z):
#     return - np.log(sigmoid(z))
#
#
# def cost_0(z):
#     return - np.log(1 - sigmoid(z))
#
#
# z = np.arange(-10, 10, 0.1)
# phi_z = sigmoid(z)
#
# c1 = [cost_1(x) for x in z]
# plt.plot(phi_z, c1, label='J(w) if y=1')
#
# c0 = [cost_0(x) for x in z]
# plt.plot(phi_z, c0, linestyle='--', label='J(w) if y=0')
#
# plt.ylim(0.0, 5.1)
# plt.xlim([0, 1])
# plt.xlabel('$\phi$(z)')
# plt.ylabel('J(w)')
# plt.legend(loc='best')
# plt.tight_layout()
# plt.savefig('images/03_04.png', dpi=300)
# plt.show()


# ### Training a logistic regression model with scikit-learn

lr = LogisticRegression(C=100.0, random_state=1, solver='liblinear')
lr.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined,
                      classifier=lr, test_idx=range(105, 150))

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('images/03_06.png', dpi=300)
# plt.show()

res1 = lr.predict_proba(X_test_std[:3, :])  # 预测测试中前3类的概率, 每一列为属于各个种类的概率

print("res1 : ", res1)

sum = lr.predict_proba(X_test_std[:3, :]).sum(axis=1)  # 每一列之和应该为1

print("sum : ", sum)  # [1, 1, 1]

lr.predict_proba(X_test_std[:3, :]).argmax(axis=1)  # 识别每行中最大列值得到预测的分类标签

predictArr = lr.predict(X_test_std[:3, :])  # 直接获得前3个样本的分类标签

print("predictArr : ", predictArr)

reshapeArr = X_test_std[0, :].reshape(1, -1)  # 第一个样本的数据, 一维变二维

print("reshapeArr : ", reshapeArr)

predictOne = lr.predict(reshapeArr) # 预测第一个样本

print("predictOne : ", predictOne)
