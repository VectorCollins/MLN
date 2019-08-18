# 简明机器学习
**Machine Learning in a Nutshell**

[TOC]

## 1.感知机
**感知机 Perceptron: **可模拟人类感知能力的机器.
函数形式: $f(x)=sign(w^Tx+b)$

实现过程为寻找定义决策边界的超平面作为二元线性分类器, 对样本进行划分.
感知机可组成单隐层前馈神经网络(神经元).
感知机只能解决线性可分问题(如无法表示异或).

**超平面 Hyperplane: **n维欧式空间中的n-1维线性子空间.得名于三维空间中的二维平面概念的高维推广.
又称分离超平面 separating hyperplane

方程表示：$\sum\limits_{i=0}^n\theta_ix_i=0$

向量表示：$w^Tx+b=0$, 其中x为n维向量, $w$为超平面的法向量[*](#超平面法向量的相关说明)
或将$b$并入$w$, 有$w^Tx=0$, 其中x为n+1维向量,$w_0=b$, $x_0=1$

**激活函数 Activation Function: **为神经元的输出提供非线性映射.
感知机使用$sign$函数作为激活函数, 逻辑回归使用$sigmoid$函数作为激活函数.

**损失函数 Loss Function: **一种将一个事件（在一个样本空间中的一个元素）映射到一个表达与其事件相关的经济成本或机会成本的实数上的一种函数.实际过程中多利用真实值和预测值计算损失, 评估模型效果.

感知机使用点到超平面的距离作为损失函数(省去$\frac{1}{\lVert w \rVert}$), 给定一组测试数据$(x_i,y_i)$, 有损失函数$L(w,b)=-\sum\limits_{x_i\in M} y_i(wx_i+b)$, 其中$M$为误分类点集, 寻找目标超平面即为寻找最小化损失函数时的参数$w$和$b$.

**优化器 Optimizer: **更新参数的优化算法.

感知机可使用随机梯度下降(SGD:Stochastic Gradient Descent)或小批量梯度下降(MBGD:Mini-Batch Gradient Descent)

## 2.逻辑斯蒂回归

**Logistic Regression**

二项逻辑斯蒂回归模型假设有线性函数$w^T\cdot x$可对输入$x$进行分类(分类结果 $Y$ 为 $0-1$ 分布)

设输出$Y$为$1$的条件概率为
$P(Y=1|x)=\frac{e^{w^T\cdot x}}{1+e^{w^T\cdot x}}$
则输出$Y$为$0$的条件概率为
$P(Y=0|x)=\frac{1}{1+e^{w^T\cdot x}}$

使用$logit$函数度量事件的几率, 即利用发生概率与不发生概率的比值的对数描述事件几率.
$logit(p)=log\frac{p}{1-p}=log\frac{P(Y=1|x)}{1-P(Y=1|x)}=w^T\cdot x$
即逻辑斯蒂模型中线性函数$w^T\cdot x$代表事件几率

反解上式
$log\frac{p}{1-p}=w^T\cdot x$
$\frac{p}{1-p}=e^{w^T\cdot x}$
$\frac{1}{p}-1=e^{-w^T\cdot x}$
得到事件发生概率 $p=\frac{1}{1+e^{-w^T\cdot x}}$

形如$\frac{1}{1+e^{-x}}$的函数称为Logistic函数或sigmoid函数
这种函数能将任意数值的事件几率$w^T\cdot x$的映射到$(0,1)$之间, 起到二分类作用.

## 3.支持向量机

**Support Vector Machine**

在感知机的基础上对分离超平面增加约束条件求解得到唯一超平面(最大化间隔), 并使用核方法划分线性不可分数据(非线性分类).



---

## **附录**

---

### 从空间到范数

#### 空间

空间是指一种具有特殊性质及一些额外结构的集合. 未指明的空间不是具体的数学对象, 需明确性质和结构后才能进行研究.

#### 向量空间
**Vector Space**

又称矢量空间或线性空间

定义: 设$F$是一个域, $V$是一个非空集合. 若满足以下运算规则和条件, 则称$V$是$P$上的一个向量空间.

1. 向量加法

     对于$V$中任意两个向量$\alpha , \beta$, 有$V$中一个唯一确定的向量与他们对应, 这个向量称为$\alpha$与$\beta$的和, 记作$\alpha + \beta$

2. 标量乘法

     对于域$F$中每一个数$k$和集合$V$中每一个向量$\alpha$, 在$V$中都有唯一确定的向量与他们对应, 这个向量称为$k$与$\alpha$的积, 记作$k \alpha$

3. 向量与标量乘法满足运算律(任取$V$中向量$\alpha, \beta, \gamma$; 任取$F$中数$a,b$)
     ①向量加法交换律: $\alpha + \beta = \beta + \alpha$
     ②向量加法结合律: $(\alpha + \beta) + \gamma = \alpha + (\beta + \gamma)$
     ③向量加法单位元: 集合$V$中存在零向量, 记作$0$. 有性质$0 + \alpha = \alpha$
     ④向量加法负向量: 对于集合$V$中的每一个向量$\alpha$, 在$V$中都有一个向量$\beta$使$\alpha + \beta = 0$. $\beta$称为$\alpha$的负向量, 记为$- \alpha$
     ⑤标量乘法分配律: $a (\alpha + \beta) = a \alpha + b \beta$; $(a + b) \alpha = a \alpha + b \beta$
     ⑥标量乘法结合律: $(a b) \alpha = a (b \alpha)$
     ⑦标量乘法单位元: $1 \alpha = \alpha$. 此处的数$1$称为域$F$的乘法单位元

#### 仿射空间

仿射空间是点和向量的集合, 是对向量空间的推广.

定义: 假设$A$是域$F$上的点集, $V$是域$F$上的向量空间. 若满足以下性质, 则称$A$是关于$V$的**仿射空间**. 
(任取$A$中点$P,Q,R$; 任取$V$中向量$\alpha , \beta$)

1. 对$A$中任意两个有序点$P,Q$, 在向量空间$V$都存在向量$\alpha$与他们对应, 记为$\vec{PQ}$
2. 对$A$中任意三个点$P,Q,R$, 如$P,Q$对应于矢量$\alpha$, $Q,R$对应于矢量$\beta$, 则$P,R$对应于向量$\alpha + \beta$, 即有
   $\vec{PQ} + \vec{QR} = \vec{PR}$

#### 函数空间

函数空间指的是从集合$X$到集合$Y$的给定种类的函数的集合.

函数空间是由**元素**和**规则**定义的

#### 距离

我们使用**距离**描述同一空间中不同元素之间的差异, 定义如下

设$S$是一个非空集合, 若任给一对$S$中的元素$X,Y$, 都能给定一个实数$d(X,Y)$与之对应, 且满足

1. $d(X,Y) \ge 0; \quad d(X,Y) = 0 \Leftrightarrow X=Y;$
2. $d(X,Y) = d(Y,X);$
3. $d(X,Y) \le d(X,Z) + d(Y,Z)$

则称$d(X,Y)$是元素$X,Y$之间的**距离**, 这种可定义元素间距离的集合称为度量空间或距离空间

#### 范数

**范数**比**距离**的限制条件更多, 这种函数能给向量空间中所有的向量赋予非零的正长度或大小.
假设$V$是域$F$上的向量空间, 若函数$p$满足以下性质, 则成$p$是$V$的**半范数**
(任取$V$中向量$\alpha , \beta$; 任取$F$中数$k$)

1. 半正定性 $p(\alpha) \ge 0$
2. 一次齐次性 $p(k \alpha) = |k| p(\alpha)$
3. 三角不等式 $p(\alpha + \beta) \le p(\alpha) + p(\beta)$

**范数**是一个**半范数**加上额外性质:

4. 正定性 $p(\alpha) = 0$, 当且仅当$\alpha$是零向量

拥有范数的空间称作**赋范空间**. 用符号$\lVert X \rVert$表示元素$|X|$的范数.

#### $L_p$范数
得名于数学家勒贝格(Lebesgue)

$L_p$范数定义了一组范数(定义同闵可夫斯基距离)
$L_p = \lVert x \rVert_p = \sqrt[p]{\sum\limits_{i=1}^n x_i^p}$

因$L_p$ 范数在$p \in[0,1)$范围内不满足三角不等式性质, 所以$L_0$范数不是严格意义上的范数.

---

### 超平面法向量的相关说明

1.$w$为超平面法向量：与指定向量$w$垂直且过指定点p向量上的点$x$均满足关系$w^T(x-p)=0$, 其中点$x$的集合即为超平面, 其中$-w^Tp$对应原式中的b.

2.任意点到超平面的距离：过任意点$x_a$可做与原超平面$w^Tx_0+b_0=0$平行的超平面$w^Tx_i+b_i=0$, 则求点$x_i$到原超平面的距离等价于求两个超平面的截距差在法向量上的投影$\frac{|b_0-b_i|}{\lVert w \rVert}$.因$w^Tx_a+b_i=0$, 故点$x_a$到原超平面的距离有$\frac{|b_0-b_i|}{\lVert w \rVert}=\frac{|b_0-(-w^Tx_a)|}{\lVert w \rVert}=\frac{|w^Tx_a+b_0|}{\lVert w \rVert}$.

[感知机 $\Leftarrow$](#1.感知机)

---

### 信息熵含义理解

1.如果我们被告知小概率事件发生了, 代表我们得到更多信息, 所以信息熵是概率的单调(减)函数; 
2.如随机变量$x$和$y$相互独立, 信息熵应满足$h(x+y)=h(x)+h(y)$, 联合概率满足$p(x,y)=p(x)p(y)$, 
信息熵可取概率的$log$函数, 同时需要$0\le p\le1$时$h(x)$非负(呼应单调性), 所以取负号有$h(x)=-log_2p(x) $
3.总信息熵是所有可能信息熵的数学期望

综上, 随机变量不确定性可用以下公式度量 $H(X)=-\sum\limits_xp(x)log_2p(x)$, 信息的作用是消除这种不确定性.

---

### 最大熵原理

1.满足已知条件(约束条件)
2.不做任何未知假设(等概率假设)

信息熵$H(P)=-\sum\limits_XP(X)log(X)$

当一个系统有$n$个概率$p_1,p_2,...,p_n$, 即$n$是$X$的$L_0$范数$|X|$:

$H(P)=-\sum\limits_{i=1}^np_ilogp_i$
约束条件: $\sum\limits_{i=1}^np_i=1$

求解最大熵:

使用拉格朗日乘子法构造
$L(p,\lambda)=-\sum\limits_{i=1}^np_ilog(p_i)+\lambda(\sum\limits_{i=1}^np_i-1)$

对拉格朗日函数求导
$\frac{\partial L(p,\lambda)}{\partial p_i}=-logp_i-1+\lambda=0$
解得 $p_i=e^{\lambda-1}, i=1,2,...,n$

代入约束条件有
$\sum\limits_{i=1}^np_i=ne^{\lambda-1}=1$
得 $e^{\lambda-1}=\frac{1}{n}$, 即等概率 $p_1=p_2,...,=p_n=\frac{1}{n}$ 时(均匀分布)熵有最大值

最大熵为 $logn$
信息熵满足 $0\le H(P)\le logn$

---
### 拉格朗日乘子法
Lagrange Multipliers

有以下原问题
$min \, f(x) \\ s.t. \, g(x) = 0$

借助罚项的思想, 先取令 $f(x)+\lambda g(x)$ 最大的$\lambda$值 $max_\lambda f(x)+\lambda g(x)$ , 保证约束条件满足, 之后再取令 $f(x)+\lambda g(x)$ 最小的$x$值, 所以原问题可以表示为 $min_x max_\lambda f(x)+\lambda g(x)$.

根据拉格朗日对偶性, 可以把问题转化为求解原问题的对偶问题 $max_\lambda min_x L(x,\lambda)$[*](#最优化对偶问题说明)

如果原问题有解, 从几何角度看, 原问题对应于$f(x)$的等高线簇 $f(x)=k$ 与 $g(x)=0$ 相切, 并求其中最小的$k$及对应$x$. 此时在切点处$f$和$g$的法线在一条直线上(反向), 即梯度向量共线, 有数学关系
$\nabla f=-\lambda\nabla g$
联立$g(x)=0$即可求解原问题.

从求解步骤上看, 对于构造函数 $L(x,\lambda)=f(x) + \lambda g(x)$
先求 $\nabla_x L=0$, 得到关于$\lambda$的$x$表达式
再把解代入约束条件(约束条件等价于$\nabla_\lambda L=0$)消去$x$, 解出$\lambda$
回代$\lambda$到第一步的表达式, 解出$x$, 进而代入$f(x)$计算出$min \, f(x)$

这种引入拉格朗日乘子$\lambda$构造拉格朗日函数$L(x,\lambda)$求解带约束条件最优化问题的方法, 称为**拉格朗日乘子法**.

---

### 最优化对偶问题说明

有原问题 

**Problem 1**:
$min \, f(x) \\ s.t. \, g(x) \le 0$

对于

**Problem 2**:
$f(x) \lt \upsilon \\ s.t. \, g(x) \le 0$

**Problem 3**:
拉格朗日函数$L(x,\lambda)=f(x) + \lambda g(x) \lt \upsilon$

如果 **Problem 2** 无解, 则称 $\upsilon$ 是 **Problem 1** 的一个下界;
如果 **Problem 2** 有解, 则对于任意的$\lambda \ge 0$, **Problem 3** 一定有解.

又 **Problem 3** 无解等价于

**Problem 4**:
$\upsilon \le min_x \, L(x,\lambda)$

综上, 根据**原命题与逆否命题的等价性**可知:
如果 **Problem 3** 无解, 则  **Problem 2** 无解, 则 $\upsilon$ 是 **Problem 1** 的一个下界, 即
如果 **Problem 4** 成立, 则 $\upsilon$ 是 **Problem 1** 的一个下界.

取其中最大下界, 即对偶问题 $\upsilon *=max_\lambda min_x L(x,\lambda)$

又因为原问题 **Problem 1** 等价于

**Problem 5**: 
$p*=min_xmax_\lambda L(x,\lambda)$

可见对偶问题就相当于把直接求解原问题转化为求解原问题最大下界的问题.

原问题和对偶问题满足不等式
$\upsilon *=max_\lambda min_x L(x,\lambda) \le min_xmax_\lambda L(x,\lambda) = p*$

当$f(x)$和不等式约束$g(x)$均为凸函数, 等式约束$h(x)$为仿射函数时, **原问题等价于对偶问题**.

[拉格朗日乘子法 $\Leftarrow$](#拉格朗日乘子法)

---

### 最大熵模型

从训练数据中可以获得联合分布$P(X,Y)$的经验分布$\widetilde P(X,Y)$和边缘分布$P(X)$的经验分布$\widetilde P(X)$

定义特征函数$f(x,y)=\begin{equation} \left\{\begin{array}{**lr**} 1, \quad x \, 与 \, y \, 满足某一个事实 \\ 0, \quad 否则 \end{array} \right. \end{equation}$

特征函数$f(x,y)$关于经验分布$\widetilde P(X,Y)$的期望值(训练数据中描述事实的概率和)
$E_{\widetilde P}(f)=\sum\limits_{x,y}\widetilde P(x,y)f(x,y)$

特征函数$f(x,y)$关于模型$P(Y|X)$与经验分布$\widetilde P(X)$的期望值(模型预测结果中为事实的概率和)
$E_{P}(f)=\sum\limits_{x,y}\widetilde P(x)P(y|x)f(x,y)$

如果模型能够学到训练数据中的信息, 那么就可以假设这两个期望值相等
$E_{P}(f)=E_{\widetilde P}(f)$, 即
$\sum\limits_{x,y}\widetilde P(x)P(y|x)f(x,y)=\sum\limits_{x,y}\widetilde P(x,y)f(x,y)$

假设满足所有约束条件的模型集合为
$C \equiv \{ P \in \wp \, | \, E_P(f_i)=E_{\widetilde P}(f_i),\; i=1,2...,n \} $

定义在条件概率分布$P(Y|X)$上的条件熵为
$H(P)=\sum\limits_{x}\widetilde P(x)H(Y|X=x)=-\sum\limits_{x}\widetilde P(x) \sum\limits_{y}\widetilde P(y|x)logP(y|x)=-\sum\limits_{x,y}\widetilde P(x)P(y|x)logP(y|x)$

模型集合$C$中条件熵$H(P)$最大的模型称为最大熵模型.

---

### 求解最大熵模型

最大熵模型的求解等价于以下约束最优化问题
$
\begin{align*}
min \quad &-H(P) = \sum\limits_{x,y} \widetilde P(x)P(y|x)logP(y|x) \\
s.t.\, \quad &E_P(f_i) - E_\widetilde P(f_i) = 0, \, i = 1,2,...,n \\
&\sum\limits_y P(y|x) = 1
\end{align*}
 $

将约束最优化原问题转换为无约束最优化的对偶问题[*](#最优化对偶问题说明), 即使用拉格朗日乘子法求解[*](#拉格朗日乘子法)
构造拉格朗日函数$L(P,w) $

$
\begin{align*}
L(P,w) &\equiv -H(P)+w_0 \big( 1 - \sum\limits_y P(y|x) \big) + \sum\limits_{i=1}^n w_i \big( E_\widetilde P(f_i) - E_P(f_i) \big) \\
&=\sum\limits_{x,y} \widetilde P(x) P(y|x) logP(y|x) + w_0 \big( 1 - \sum\limits_y P(y|x) \big) + \sum\limits_{i=1}^n w_i \big( \sum\limits_{x,y} \widetilde P(x,y)f_i(x,y) - \sum\limits_{x,y} \widetilde P(x)P(y|x)f_i(x,y) \big)
\end{align*}
 $

因为$L(P,w)$是凸函数, 等式约束均为仿射函数, 所以对偶问题$max_P min_w L(P,w)$与原问题$min_w max_P L(P,w)$的解等价.

第一步求解内层$min_P L(P,w)$, 记**对偶函数**$\Psi (w) = min_P L(P,w)=L(P_w,w)$, 记其解$P_w=arg \, min_P L(P,w) = P_w(y|x)$
具体地, 求$L(P,w)$对$P(y|x)$的偏导
$
\begin{align*}
\frac{\partial L(P,w)}{\partial P(y|x)} &= \sum\limits_{x,y} \widetilde P(x)(logP(y|x)+1) - \sum\limits_y w_0 - \sum\limits_{x,y} \big( \widetilde P(x)\sum\limits_{i=1}^n w_i f_i(x,y) \big) \\
&=\sum\limits_{x,y} \widetilde P(x) \big( logP(y|x) + 1 - w_0 - \sum\limits_{i=1}^n w_i f_i(x,y) \big) 
\end{align*}
$

令偏导$\frac{\partial L(P,w)}{\partial P(y|x)}=0$, 在$\widetilde P(x)>0$的情况下, 解得
$P(y|x)=e^{\sum\limits_{i=1}^n w_i f_i(x,y) + w_0 - 1} = \frac{e^{\sum\limits_{i=1}^n w_i f_i(x,y)}}{e^{(1 - w_0)}}$
由于$\sum\limits_y P(y|x)=1$, 得$P_w(y|x)=\frac{e^{\sum\limits_{i=1}^n w_i f_i(x,y)}}{\sum\limits_y e^{\sum\limits_{i=1}^n w_i f_i(x,y)}}=\frac{e^{\sum\limits_{i=1}^n w_i f_i(x,y)}}{Z_w(x)}$
其中记规范化因子$Z_w(x) = \sum\limits_y e^{\sum\limits_{i=1}^n w_i f_i(x,y)}$
第二步求解外层$max_w \Psi(x)$
记其解$w*=arg max_w \Psi(x)$

---

### 最大熵模型的极大似然估计

使用模型条件概率分布$P(X|Y)$和训练数据经验概率分布$\widetilde P(X,Y)$度量模型学习的似然程度, 具体地
取**对数似然函数**$L_\widetilde P(P_w) = log \prod\limits_{x,y} P(y|x)^{\widetilde P(x,y)} = \sum\limits_{x,y} \widetilde P(x,y) logP(y|x)$

如果其中条件概率分布使用最大熵模型推论的取值, 即$P_w(y|x) = \frac{e^{\sum\limits_{i=1}^n w_i f_i(x,y)}}{Z_w(x)}$, 有对数似然函数
$
\begin{align*}
L_\widetilde P(P_w) &= \sum\limits_{x,y} \widetilde P(x,y) logP(y|x) \\
&= \sum\limits_{x,y} \widetilde P(x,y) \sum\limits_{i=1}^n w_i f_i(x,y) - \sum\limits_{x} \widetilde P(x,y) logZ_w(x)
\end{align*}
$

又因为求解最大熵模型过程中有对偶函数
$
\begin{align*}
\Psi (w) &= \sum\limits_{x,y} \widetilde P(x) P(y|x) logP_w(y|x) + \sum\limits_{i=1}^n w_i \big( \sum\limits_{x,y} \widetilde P_w(x,y)f_i(x,y) - \sum\limits_{x,y} \widetilde P(x)P_w(y|x)f_i(x,y) \big) \\
&= \sum\limits_{x,y} \widetilde P(x,y) \sum\limits_{i=1}^n w_i f_i(x,y) + \sum\limits_{x,y} \widetilde P(x) P_w(y|x) \big( log P_w(y|x) - \sum\limits_{i=1}^n w_i f_i(x,y) \big) \\
&= \sum\limits_{x,y} \widetilde P(x,y) \sum\limits_{i=1}^n w_i f_i(x,y) + \sum\limits_{x,y} \widetilde P(x) P_w(y|x) log Z_w(x) \\
&= \sum\limits_{x,y} \widetilde P(x,y) \sum\limits_{i=1}^n w_i f_i(x,y) + \sum\limits_{x} \widetilde P(x) log Z_w(x)
\end{align*}
 $

综上可知 $\Psi(w) = L_\widetilde P(P_w)$
即求解最大熵模型过程中**对偶函数极大化等价于最大熵模型的极大似然估计**

归纳最大熵模型的一般形式
$P_w(y|x) = \frac{e^{\sum\limits_{i=1}^n w_i f_i(x,y)}}{Z_w(x)}$
其中$Z_w(x) = \sum\limits_y e^{\sum\limits_{i=1}^n w_i f_i(x,y)}$

---

### 最大似然估计和最大后验概率估计求解实例

#### 二项分布最大似然估计

最大化似然函数$P(x_0|\theta)$

二项分布似然函数$P(x_0|\theta)=f(x_0,\theta)=\theta^a(1-\theta)^b$, 求导得

$\frac{d[x^a(1-x)^b]}{dx}$

$=ax^{a-1}(1-x)^b-bx^a(1-x)^{b-1}$

$=x^{a-1}(1-x)^{b-1}(a(1-x)-bx)$

$=x^{a-1}(1-x)^{b-1}[a-(a+b)x]$

$根据导数在x\in(0,1)上的取值, 当x=\frac{a}{a+b}时, 似然函数有极大值(\frac{a}{a+b})^a(\frac{b}{a+b})^b$

#### 二项分布最大后验概率估计

最大化后验概率$P(\theta|x_0)=\frac{P(x_0|\theta)P(\theta)}{P(x_0)}$

高斯分布$N(\mu,\sigma^2)$的后验概率$P(\theta|x_0)=\frac{P(x_0|\theta)P(\theta)}{P(x_0)}\propto x^a(1-x)^be^{-\frac{(x-\mu)^2}{2\sigma^2}}$, 求导有

$\frac{d[x^a(1-x)^be^{-\frac{(x-\mu)^2}{2\sigma^2}}]}{dx}$

$=x^{a-1}(1-x)^{b-1}[a-(a+b)x]e^{-\frac{(x-\mu)^2}{2\sigma^2}}+x^a(1-x)^b(-\frac{1}{\sigma^2})(x-\mu)e^{-\frac{(x-\mu)^2}{2\sigma^2}}$

$=x^{a-1}(1-x)^{b-1}e^{-\frac{(x-\mu)^2}{2\sigma^2}}[a-(a+b)x-\frac{1}{\sigma^2}x(1-x)(x-\mu)]$

根据导数在$x\in(0,1)$上的取值, 零点$x_0$介于$\frac{a}{a+b}$,$\mu$之间, $x=x_0$时后验概率有极大值

---

Any problem, contact [ves](https://github.com/VectorCollins) :)