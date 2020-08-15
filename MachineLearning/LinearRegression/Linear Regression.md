# 线性回归模型

## 模型

数据集：$(x_i,y_i),i=1,2,...,m$，其中$x_i=[x_i^1,...,x_i^d]^T\in\mathbb{R}^{d\times1}$，$y_i\in\mathbb{R}^{1\times1}$。

- 对每一个数据$(x_i,y_i)$，记$w=[w_1,...,w_d]^T\in\mathbb{R}^{d\times1}$，$b_i\in\mathbb{R}$，满足线性模型：

$$
y_i=w^Tx_i+b_i+\epsilon_i=x_i^Tw+b_i+\epsilon_i
$$

- 对全部数据，记$X=[x_1,x_2,...,x_m]^T\in\mathbb{R}^{m\times d}$，$Y=[y_1,y_2,...,y_m]^T\in\mathbb{R}^{m\times 1}$，$b=[b_1,b_2,...,b_m]^T\in\mathbb{R}^{m\times 1}$，$\epsilon=[\epsilon_1,\epsilon_2,...,\epsilon_m]^T\in\mathbb{R}^{m\times 1}$则

$$
Y=Xw+b+\epsilon
$$

简化模型：对原始数据集作归一化处理，线性模型偏置项一定为0。

- min-max：$x_i^{new}=\frac{x_i-\max(x_i)}{\max{x_i}-\min{x_i}}$。
- z-score：$x_i^{new}=\frac{x_i-\bar{x}}{s}$，其中$\bar{x}$、$s$分别为样本均值和标准差。

## 参数估计

### 最小二乘法

极小化残差平方和：
$$
\min_{w}J(w)=\sum_{i=1}^m(y_i-x_i^Tw)^2=(Y-Xw)^T(Y-Xw)
$$

### 极大似然估计

假设误差$b\sim\mathcal{N}(0,\sigma^2I)$，则在给定$X$、$w$的条件下，$Y\sim\mathcal{N}(Xw,\sigma^2I)$，于是样本的似然函数为：
$$
L(w)=(2\pi)^{-\frac{n}{2}}\sigma^{-n}exp\{-\frac{\sigma^2}{2}(Y-Xw)^T(Y-Xw)\}
$$
对数似然函数：
$$
\log L(w)=-\frac{n}{2}\log(2\pi)-n\log(\sigma)-\frac{\sigma^2}{2}(Y-Xw)^T(Y-Xw)
$$
可以看出极大化对数似然函数等价于极小化最小二乘目标函数：
$$
\max_{w}\log L(w)\Leftrightarrow\min_w J(w)
$$

### 优化问题

参数估计问题转换为以下目标函数（平均损失）的极小化问题：
$$
\begin{aligned}
&分量形式：\min_{w}J(w)=\frac{1}{2n}\sum_{i=1}^m(y_i-x_i^Tw)^2\\
&向量形式：\min_{w}J(w)=\frac{1}{2n}(Y-Xw)^T(Y-Xw)=\frac{1}{2n}\lVert Y-Xw\rVert^2_2
\end{aligned}
$$
目标函数的梯度函数：
$$
\begin{aligned}
&分量形式：\frac{\partial}{\partial w_j}J(w)=-\frac{1}{n}\sum_{i=1}^m(y_i-x_i^Tw)x_i^j\\
&向量形式：\frac{\partial}{\partial w}J(w)=\frac{1}{n}(-X^T)(Y-Xw)=\frac{1}{n}(X^TXw-X^TY)
\end{aligned}
$$

- 倘若$X^TX$可逆，参数有最优解$w=(X^TX)^{-1}X^TY$。

- 梯度下降法求解的参数更新公式为：

$$
\begin{aligned}
&分量形式：w_j^{new}=w_j+\eta\frac{1}{n}\sum_{i=1}^m(y_i-x_i^Tw)x_i^j\\
&向量形式：w^{new}=w+\eta\frac{1}{n}X^T(Y-Xw)
\end{aligned}
$$

# 岭回归

$$
\begin{aligned}
&分量形式：\begin{cases}\min_{w}J(w)=\frac{1}{2n}\sum_{i=1}^m(y_i-x_i^Tw)^2\\
s.t. \quad \sum_{i=1}^dw_i^2<c\end{cases}\\
\\
&向量形式：\begin{cases}\min_{w}J(w)=\frac{1}{2n}\lVert Y-Xw\rVert^2_2\\
s.t. \quad \lVert w\rVert^2_2<c\end{cases}
\end{aligned}
$$

优化问题等价于极小化Lagrange函数：
$$
\begin{aligned}
&分量形式：\min_{w}J(w)=\frac{1}{2n}\sum_{i=1}^m(y_i-x_i^Tw)^2+\lambda\sum_{i=1}^dw_i^2\\
&向量形式：\min_{w}J(w)=\frac{1}{2n}\lVert Y-Xw\rVert^2_2+\lambda\lVert w\rVert^2_2
\end{aligned}
$$

- 岭回归估计：$w=(X^TX+\lambda I)^{-1}X^TY$。
- $\lambda$为正则化参数：如果$\lambda$选取过大，会把所有参数均最小化，造成欠拟合；如果$\lambda$选取过小，会导致对过拟合问题解决不当。

# LASSO

$$
\begin{aligned}
&分量形式：\begin{cases}\min_{w}J(w)=\frac{1}{2n}\sum_{i=1}^m(y_i-x_i^Tw)^2\\
s.t. \quad \sum_{i=1}^d\rvert w_i\lvert<c\end{cases}\\
\\
&向量形式：\begin{cases}\min_{w}J(w)=\frac{1}{2n}\lVert Y-Xw\rVert^2_2\\
s.t. \quad \lVert w\rVert_1<c\end{cases}
\end{aligned}
$$

优化问题等价于极小化Lagrange函数：
$$
\begin{aligned}
&分量形式：\min_{w}J(w)=\frac{1}{2n}\sum_{i=1}^m(y_i-x_i^Tw)^2+\lambda\sum_{i=1}^d\lvert w_i\rvert\\
&向量形式：\min_{w}J(w)=\frac{1}{2n}\lVert Y-Xw\rVert^2_2+\lambda\lVert w\rVert_1
\end{aligned}
$$

- 求解LASSO：坐标下降法，最小角回归

# Notes

1. 线性回归基本假设（经典假设）
   - 解释变量是确定性变量，不是随机变量，且在重复抽样中取固定值；
   - 误差项满足：零均值，同方差，序列不相关；
   - 误差项与解释变量不相关；
   - 误差项服从零均值、同方差、零协方差的正态分布。

2. 为什么假设误差项服从正态分布
   - **中心极限定理**：独立同分布的随机变量$X_1,...,X_n,...$具有有限的期望$E[X_i]=\mu$和方差$Var(X_i)=\sigma^2$，则当$n$很大时，随机变量$Y_n=\frac{\sum_{i=1}^nX_i-n\mu}{\sqrt{n}\sigma}=\frac{\bar{X}-E[\bar{X}]}{Var(\bar{X})}$近似地服从标准正态分布$\mathcal{N}(0,1)$。
   - 一些现象受到许多相互独立的随机因素的影响，如果每个因素所产生的影响都很微小时，总的影响可以看作是服从正态分布的。

3. 最小二乘估计的性质

   - 最小二乘估计是无偏估计；
   - （Gauss-Markov定理）在所有线性无偏估计中，最小二乘估计是唯一具有最小方差的估计；
   - 最小二乘估计的均方误差：$E[(w_{ols}-w)'(w_{ols}-w)]=\sigma^2Tr(X'X)^{-1}=\sigma^2\sum_{i=1}^d1/\lambda_i$。

4. 岭回归与LASSO异同

   - 岭回归估计与LASSO估计均是有偏估计；

   - 岭回归和LASSO回归解决了线性回归出现的过拟合问题以及在通过正规方程求解参数时$X^TX$不可逆的问题；

   - LASSO可以同时实现变量选择和参数压缩，岭回归只能实现参数压缩：

     岭回归是对线性模型参数的L2正则化，L2正则化相当于引入了高斯先验，高斯分布在极值点处是光滑的，也就是高斯先验分布认为参数在极值点0附近取不同值的可能性是接近的；LASSO是对线性模型参数的L1正则化，L1正则化相当于引入了拉普拉斯先验，拉普拉斯分布在极值点0处是尖峰，因此拉普拉斯先验使参数为0的可能性更大。$P(w|x,y)\propto P(y|x,w)P(w)$

   - 在用最小二乘估计有更高的偏差的情况下，岭回归效果最好。