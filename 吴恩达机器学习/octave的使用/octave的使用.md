
## mac 安装 octave

`brew install octave`

## octave 常用的命令

常用的命令可以参考这篇文章 : https://www.jianshu.com/p/02d81dda33ae

## ocatve 实战


### 向量化数据

比如我们要计算下面这个数学表达式:

$$
h_{\theta}(x)= \sum_{j=1}^{n}\theta_{j}x_{j}
$$

上面这个表达式其实可以这样表达:

$$
h_{\theta}(x)= \theta^{T}x
$$

即:

$$
\theta = 
\begin{bmatrix}
\theta_{0}\\
\theta_{1}\\
\theta_{2}\\
\end{bmatrix}
$$

$$
x = 
\begin{bmatrix}
x_{0}\\
x_{1}\\
x_{2}\\
\end{bmatrix}
$$

通过对数据的向量化，在`octave`可以很方便的计算出$h_{\theta}(x)= \sum_{j=1}^{n}\theta_{j}x_{j}$的值。

### 更复杂的例子

比如梯度下降: 

$$
\theta_{j} := \theta_{j} - \alpha\frac{1}{m}\sum_ {i=1}^{m}(h_{\theta}(x^{i}) - y^{i})x_{j}^{i}
$$

假设我们有两个feature : $h_{\theta}(x^{i}) = \theta_{0} + \theta_{1}x^{i}+\theta_{2}x^{i}$

$$
\theta_{0} := \theta_{0} - \alpha\frac{1}{m}\sum_ {i=1}^{m}(h_{\theta}(x^{i}) - y^{i})x_{0}^{i} 
$$

$$
\theta_{1} := \theta_{1} - \alpha\frac{1}{m}\sum_ {i=1}^{m}(h_{\theta}(x^{i}) - y^{i})x_{1}^{i}
$$

$$
\theta_{2} := \theta_{2} - \alpha\frac{1}{m}\sum_ {i=1}^{m}(h_{\theta}(x^{i}) - y^{i})x_{2}^{i}
$$

上面3个公式我们可以在代码中使用for循环来计算出来，不过使用向量的，计算起来会更简单:

- vectorized implementation

把上面3个式子向量化:   $\theta := \theta - \alpha\delta$

$\alpha$为一个实数($\alpha \in R$), $\theta$为一个列向量:

$$
\theta = 
\begin{bmatrix}
    \theta_{0}\\
     \theta_{1}\\
      \theta_{2}
\end{bmatrix}
$$

$\delta$是:

$$
\begin{bmatrix}
\frac{1}{m}\sum_ {i=1}^{m}(h_{\theta}(x^{i}) - y^{i})x_{0}^{i}
\\
\frac{1}{m}\sum_ {i=1}^{m}(h_{\theta}(x^{i}) - y^{i})x_{1}^{i}
\\
\frac{1}{m}\sum_ {i=1}^{m}(h_{\theta}(x^{i}) - y^{i})x_{2}^{i}
\end{bmatrix}
$$
 =
$$
\begin{bmatrix}
\delta_{0}
\\
\delta_{1}
\\
\delta_{2}
\end{bmatrix}
$$



$\sum_ {i=1}^{m}(h_{\theta}(x^{i}) - y^{i})x_{0}^{i}$ 的计算结果也是一个实数。









