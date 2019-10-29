from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.decomposition import KernelPCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# Implementing a kernel principal component analysis in Python

def rbf_kernel_pca(X, gamma, n_components):
    """
      RBF kernel PCA implementation.
      Parameters
      ------------
      X: {NumPy ndarray}, shape = [n_samples, n_features]

      gamma: float
        Tuning parameter of the RBF kernel

      n_components: int
        Number of principal components to return
      Returns
      ------------
       X_pc: {NumPy ndarray}, shape = [n_samples, k_features]
         Projected dataset
      """

    # Calculate pairwise squared Euclidean distances
    # in the MxN dimensional dataset.
    sq_dists = pdist(X, 'sqeuclidean')  # 计算核相似距离

    print('sq_dists', sq_dists)

    # Convert pairwise distances into a square matrix.
    mat_sq_dists = squareform(sq_dists)  # 转换为核矩阵

    print(mat_sq_dists)

    # Compute the symmetric kernel matrix.
    K = exp(-gamma * mat_sq_dists)

    # Center the kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenpairs from the centered kernel matrix
    # scipy.linalg.eigh returns them in ascending order
    eigvals, eigvecs = eigh(K)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]

    # Collect the top k eigenvectors (projected samples)
    X_pc = np.column_stack((eigvecs[:, i] for i in range(n_components)))

    return X_pc


X, y = make_moons(n_samples=100, random_state=123)  # 100 * 2 的数据集 -> 线性不可分

# plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='^', alpha=0.5)
# plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', alpha=0.5)
#
# plt.tight_layout()
# # plt.savefig('images/05_12.png', dpi=300)
# plt.show()

# 使用标准的PCA把数据集投射到主成分上
scikit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X)

# 使用核主成分分析
X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))

ax[0].scatter(X_kpca[y == 0, 0], X_kpca[y == 0, 1], color='red', marker='^', alpha=0.5)
ax[0].scatter(X_kpca[y == 1, 0], X_kpca[y == 1, 1], color='blue', marker='o', alpha=0.5)

ax[1].scatter(X_kpca[y == 0, 0], np.zeros((500, 1)) + 0.02, color='red', marker='^', alpha=0.5)
ax[1].scatter(X_kpca[y == 1, 0], np.zeros((500, 1)) - 0.02, color='blue', marker='o', alpha=0.5)

ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')

plt.tight_layout()
# plt.savefig('images/05_17.png', dpi=300)
plt.show()

# rbf_pca 投影矩阵
def project_x(x_new, X, gamma, alphas, lambdas):
    pair_dist = np.array([np.sum((x_new - row)**2) for row in X])
    k = np.exp(-gamma * pair_dist)
    return k.dot(alphas / lambdas)