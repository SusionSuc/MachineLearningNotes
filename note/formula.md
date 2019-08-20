# 线性回归

## 代价函数 Cost Fuction

$$
J_{(\theta_{0}, \theta_{1})} = \frac{1}{2m}\sum_{i=1}^{m}(h_{\theta}x^{(i)} - y^{(i)})^{2} 
$$

它的目标是: **选择出可以使得建模误差的平方和能够最小的模型参数**


## 批量梯度下降 Batch Gradient Descent

$$
repeat \quad \{ \\
\quad \quad  \quad \quad  \quad \quad  \quad \quad \quad \quad  \quad \quad  \theta_{j} :=   \theta_{j}  - \alpha\frac{\partial}{\partial\theta_{j}}J_{(\theta_{0}, \theta_{1})}
\\
\quad \quad  \quad \quad  \quad \quad  \quad \quad \quad  \quad \quad  \quad  \quad for \quad j = 1 \quad and \quad j = 0
\\
\}
$$

**$\alpha$是学习率（learning rate），它决定了我们沿着能让代价函数下降程度最大的方向向下迈出的步子有多大，在批量梯度下降中，我们每一次都同时让所有的参数减去学习速率乘以代价函数的导数。**

可以尝试:  $\alpha =  0.01, 0.01, 0.1, 0.3, 1, 3, 10$


>特征缩放 Feature Scaling （标准方程法）

$$
x_{n} = \frac{x_{n} - \mu_{n}}{s_{n}}
$$

**其中$\mu_{n}$是平均值，$s_{n}$是标准差**

## 正规方程 Normal Equation

$$
\theta = (X^{T}X)^{-1}X^{T}y
$$

**对于线性模型，在特征变量的数目并不大的情况下，标准方程是一个很好的计算参数的替代方法。具体地说，只要特征变量数量小于一万，我通常使用标准方程法，而不使用梯度下降法。**

# 逻辑回归

逻辑回归算法是分类算法，我们将它作为分类算法使用。它适用于标签取值离散的情况，如：1 0 0 1。

## 假设函数 Hypothesis Function

$$
h_{\theta}(x) = g(\theta^{T}x)
$$

$$
z = \theta^{T}x
$$

$$
g(z) =  \frac{1}{1 + e^{-z}}
$$


