<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

## 目录
### 1 监督学习关键概念回顾 
#### 1.1 监督学习中原理
#### 1.2 监督学习中的目标函数
#### 1.3 模型的目标函数和正则项举例
### 2. 回归树与集成学习
#### 2.1 回归树原理
#### 2.2 算法描述
#### 2.3 单变量回归树
#### 2.4 回归树拟合情况
#### 2.5 回归树集成
### 3 XGBoost梯度提升树  
#### 3.1 最小二乘损失函数推导
#### 3.2 泰勒展开损失函数
#### 3.3 目标函数求解
##### 3.3.1 回归树求解
##### 3.3.2 正则项求解
##### 3.3.3 合并简化
##### 3.3.4 目标函数再次简化
##### 3.3.5 打分计算案例
#### 3.4 分裂算法
##### 3.4.1 贪婪算法
##### 3.4.2 近似算法
#### 3.5 剪枝与正则化
#### 3.6 其他问题
##### 3.6.1 XGBoost如何处理分类变量？
##### 3.6.2 如何构建一棵提升树，使得每个实例都有重要的权重？
##### 3.6.3 回到时间序列的问题，如果想学习阶跃函数。除了从上到下的分割方法，还有其他学习时间分割的方法吗?
##### 3.6.4 XGBoost如何处理缺失值？
### 4 XGBoost与GBDT的联系和区别有哪些？
### 5 XGBoost的参数

</p>
</p>
</p>



## 1 监督学习关键概念回顾 
### 1.1 监督学习中原理
![](https://github.com/flysaint/Datawhale-/blob/master/XGBOOST/XGBOOST%E8%AE%BA%E6%96%87_Elements_in_Supervised_Learning.png)

**说明：**——怎么通过样本得到预测值 $\hat{y}_i$
1) 线性模型的预测值$\hat{y}_i$  = $\sum_j w_j x_{ij}$。即预测值 $\hat{y}_i$ 是由 每个样本$x_i$的和权值$w$乘积的累加。

2) 线性模型中，预测值 $\hat{y}_i$ 是一个预测分数。

3) 在逻辑回归中，1/(1+exp(-$\hat{y}_i$ ))是一个概率。

### 1.2 监督学习中的目标函数
![](https://github.com/flysaint/Datawhale-/blob/master/XGBOOST/XGBOOST%E8%AE%BA%E6%96%87_Elements_continued_Objective_Function.png)
**说明：**
1. 目标函数Obj($\Theta$)，由损失函数$L$($\Theta$)和正则项$\Omega$($\Theta$)是正则项构成。
$L$($\Theta$) 衡量模型和训练数据的拟合程度。
$\Omega$($\Theta$) 衡量模型的复杂度，用于防止过拟合。
2. 常用的有损失函数有：
Square loss(最小二乘损失): ($y_i$ - $\hat{y}_i$)$^2$。
Logsitic loss: (在逻辑回归基础上,通过最大似然法得到)
$y_i ln(1+e^{-\hat{y}_i}) + (1-y_i) ln(1+e^{\hat{y}_i})$


注意：为什么在线性模型中可以直接使用最小二乘解？因为在残差符合正态分布的情况下，最小二乘解和极大似然解重叠。而极大似然解是最有实际意义的。

3. 常用的正则项。
L1范式。权值向量$w$中各元素的绝对值之和。
L2范式。权值向量$w$中各元素平方和的平均值。
### 1.3 模型的目标函数和正则项举例
![](https://github.com/flysaint/Datawhale-/blob/master/XGBOOST/%E6%A8%A1%E5%9E%8B%E7%9B%AE%E6%A0%87%E5%87%BD%E6%95%B0%E4%B8%8E%E6%AD%A3%E5%88%99%E9%A1%B9%E4%B8%BE%E4%BE%8B.png)
说明：对常见的模型进行举例说明。


## 2. 回归树与集成学习

回归树是决策树的一种。决策树是将空间用超平面进行划分的一种方法，每次分割时，将当前的空间一分为二， 使得每个叶子节点都在空间中的一个不相交的区域。在进行决策的时候，会根据输入样本每一维feature的值，一步一步往下，最后使得样本落入N个区域中的一个（假设有N个叶子节点），如下图所示。
![](https://github.com/flysaint/Datawhale-/blob/master/XGBOOST/%E5%9B%9E%E5%BD%92%E6%A0%91%E4%B8%8E%E9%9B%86%E6%88%90%E6%A8%A1%E5%9E%8B.png)

三种比较常见的**分类决策树**分支划分方式包括：ID3, C4.5, CART
![](https://github.com/flysaint/Datawhale-/blob/master/XGBOOST/%E4%B8%89%E7%A7%8D%E5%B8%B8%E8%A7%81%E7%9A%84%E5%86%B3%E7%AD%96%E6%A0%91.png)

### 2.1 回归树原理
回归树如何选择划分点？如何决定叶节点的输出值？

在回归树中，采用的是启发式的方法。假如有n个特征，每个特征有$s_i$(i∈(1,n))个取值。我们遍历所有特征，尝试该特征所有取值，对空间进行划分，直到取到特征j的取值s，这个s使得损失函数最小，这样就得到了一个划分点。描述该过程的公式如下：
$$ \min_{js}[\min_{c_1}Loss(y_i,c_1) + \min_{c_2}Loss(y_i,c_2)] $$


### 2.2 算法描述
![](https://github.com/flysaint/Datawhale-/blob/master/XGBOOST/cart%E6%A0%91_%E7%AE%97%E6%B3%95%E6%8F%8F%E8%BF%B0%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95.png)

### 2.3 单变量回归树
![](https://github.com/flysaint/Datawhale-/blob/master/XGBOOST/%E5%8D%95%E5%8F%98%E9%87%8F%E5%9B%9E%E5%BD%92%E6%A0%91.png)
如上图所示，假设回归树模型是通过时间划分的，则当 

t < 2010/03/20时，回归树 = 0.2 对应最左边的直线。

t >= 2011/03/01时，回归树 = 1.0，对应最右边的直线。

t < 2011/03/01 且 t>= 2010/03/20时,对应中间的直线。

### 2.4 回归树拟合情况

![](https://github.com/flysaint/Datawhale-/blob/master/XGBOOST/%E5%9B%9E%E5%BD%92%E6%A0%91%E6%8B%9F%E5%90%88%E6%83%85%E5%86%B5.png)

1. 过拟合。分裂太多，叶子节点太多，惩罚项极高，如图2。
2. 欠拟合。错误的分裂。损失函数极高，如图3.
3. 良好拟合。损失函数和惩罚项均衡，如图4.

### 2.5 回归树集成
单变量回归树，我们已经看过了，那么多变量回归树该如何组合呢？
如下图所示，假设现在我们有两课回归树，tree1使用age和性别作为分裂特征，tree2使用 Use Computer Daily作为分裂特征。
此时 单个样本点的残差就等于两个树残差之和，如图中的小男孩和老人。

![](https://github.com/flysaint/Datawhale-/blob/master/XGBOOST/%E5%9B%9E%E5%BD%92%E6%A0%91%E9%9B%86%E6%88%90.jpg)


公式如下图，$\hat{y}_i = \sum_{k=1}^Kf_kx_{ij}$, $f_k\in F$ 。每个样本点的残差是所有回归树残差之和。

![](https://github.com/flysaint/Datawhale-/blob/master/XGBOOST/%E5%9B%9E%E5%BD%92%E6%A0%91%E9%9B%86%E6%88%902.png)

## 3 XGBoost梯度提升树  
什么是提升树算法？一句话是，每t轮的预测结果 = 第 t-1 预测结果 + t轮新的函数值。如下图step4。

step1 表示 初始化时（理解未第0轮），预测值 = 0。

step2 表示第1轮的预测提升树的预测值。

step3 表示第2轮的预测提升树的预测值。

step4 表示第t轮的预测提升树的预测值。

![](https://github.com/flysaint/Datawhale-/blob/master/XGBOOST/XGBoost%E9%A2%84%E6%B5%8B%E5%80%BC_00.png)

### 3.1 最小二乘损失函数推导
![](https://github.com/flysaint/Datawhale-/blob/master/XGBOOST/3.1%E6%9C%80%E5%B0%8F%E4%BA%8C%E4%B9%98%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%E6%8E%A8%E5%AF%BC.png)
说明：

step1 是提升树在 第 t轮的预测值。

step2 将目标函数展开未 损失函数和正则项

step3 是 将第t轮损失函数的预测值 $\hat{y}_i$ 拆分为第t-1轮的预测值$\hat{y}_i^{(t-1)}$和第t轮的函数值$f_t(x_i)$。将第t轮的累积正则项 $\sum$$_{i=1}^t\Omega(f_i)$拆分为 第t的正则项$\Omega(f_i)$ + 常数constant(前t-1轮累加的正则项，已计算得出，可看作常数)。</p>

Step4 假设损失函数是square loss(最小二乘损失)。</p>

Step5 将损失函数展开，并且将 $(y_i$ - $\hat{y}_i^{(t-1)})^2$作为常数并入 constant，因为 在第t轮，$\hat{y}_i^{(t-1)}$ 和 $y_i$都是已知值。


### 3.2 泰勒展开损失函数
![](https://github.com/flysaint/Datawhale-/blob/master/XGBOOST/3.2%E6%B3%B0%E5%8B%92%E5%B1%95%E5%BC%80%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.png)
说明：

step1。第t轮的目标函数，这里的t-1已经展开。

step2。就是泰勒二阶展开式的公式。

step3。在第t轮，$g_i$和$h_i$都可以看作常数。因为 $y_i$和$\hat{y}_i^{(t-1)}$都是已知值。

step4。将step1，使用step2中泰勒二阶展开后，再使用step3带入后的损失函数的结果。

step5。将$g_i$和$h_i$进一步做简化方便，用于后续进一步简化目标函数。

step6。最终的目标函数结果。

### 3.3 目标函数求解
如3.2 中讨论的，目标函数由回归树和正则项构成，下面我们来分别求解回归树和正则项。
#### 3.3.1 回归树求解
![](https://github.com/flysaint/Datawhale-/blob/master/XGBOOST/3.3.1%E5%9B%9E%E5%BD%92%E6%A0%91%E6%B1%82%E8%A7%A3.jpg)
说明：
$w \in R^T$。$w是T维向量，$T$代表叶子节点的数量。$R^T$代表样本在各叶子节点上的拟合值。

$q:R^d-> $ {1,2,...,$T$}。$d$代表样本的维度。$R^d$代表样本，因此$q$ 用来 就是将样本映射到叶子节点上（这就是决策树的作用）。

$w_{q(x)}$ 就是样本x通过$q$映射到对应的回归树，再获取回归树相应叶子节点上的拟合值（权重）。

#### 3.3.2 正则项求解
![](https://github.com/flysaint/Datawhale-/blob/master/XGBOOST/3.3.2%E6%AD%A3%E5%88%99%E9%A1%B9%E6%B1%82%E8%A7%A3.jpg)
如上图所示，正则项如下：
$\Omega(f_t)$ = $\gamma T + \frac 12 \lambda
\sum$$_{j=1}^Tw_j^2$

其中 $\gamma$和 $\lambda $都是超参数，需要不断调参获得最优。
$T$代表叶子节点的数量，最大不超过样本数量N。
$w_j^2$代表L2正则项，用于防止模型过拟合。

#### 3.3.3 合并简化
![](https://github.com/flysaint/Datawhale-/blob/master/XGBOOST/3.3.3%E5%90%88%E5%B9%B6%E7%AE%80%E5%8C%96.png)

**step1** 表示样本i，被映射到第j号叶子。

**step2** 直接使用 3.2小节 中的目标函数的泰勒二级展开式结果。

**step3** 使用3.3.1 和3.3.2 中简化的结果。

step4 将step3中按 样本残差累加，转变为按照叶子的权值进行累加。可以想象一个 样本为行，叶子节点为列的拟合矩阵。step3是按照行进行累加，step4是按照列进行累加。

比如：
$\sum_{i=1}^n g_iw_{q(x_i)}$ 是将每n样本点对应的拟合值的累加。
$\sum_{j=1}^T(\sum_{i \in I_j}g_i)w_j$ 是上式n个样本点的拟合值累加，转换到$T$个叶子节点拟合值累加。

#### 3.3.4 目标函数再次简化
![](https://github.com/flysaint/Datawhale-/blob/master/XGBOOST/3.3.4%E7%9B%AE%E6%A0%87%E5%87%BD%E6%95%B0%E5%86%8D%E6%AC%A1%E7%AE%80%E5%8C%96.png)
说明：

**step1**: 目标函数 $argmin_x Gx + \frac 12 Hx^2$，求导可得，在 $ x= - \frac GH,H>0时，$取得最小值 $ -\frac12 \frac {G^2}H $。

**step2,3**：将 $G_j和H_j$带入后简化。

**step4**: 将step1和step2,3中的结论带入。将 step3中的$w_j^* ,G_j,(H_j+ \lambda),分别看作step1中的x,G,H$,则此时目标函数 $Obj = -\frac12 \frac {G^2}H $ = $ -\frac 12 \sum_{j=1}^T \frac {G_j^2}{H_j+\lambda} + \gamma T$
当树的结构固定时，我们就可以求得参数，让目标函数取得极小值。

#### 3.3.5 打分计算案例
##### 样本说明
![](https://github.com/flysaint/Datawhale-/blob/master/XGBOOST/3.3.5%E6%89%93%E5%88%86%E6%A1%88%E4%BE%8B.jpg)

如上图，$I_1,I_2,I_3分别代表分配到各个节点上的样本点$
此时目标函数 $Obj =  - \sum_{j=1} \frac {G_j^2}{H_j+\lambda} + 3\gamma$，越小说明树的结构越好。

### 3.4 分裂算法
#### 3.4.1 贪心算法
每一次尝试去对已有的叶子加入一个分割

![](https://github.com/flysaint/Datawhale-/blob/master/XGBOOST/%E8%B4%AA%E5%BF%83%E7%AE%97%E6%B3%95.png)
对于每次扩展，我们还是要枚举所有可能的分割方案，如何高效地枚举所有的分割呢？我假设我们要枚举所有x < a 这样的条件，对于某个特定的分割a我们要计算a左边和右边的导数和。
![](https://github.com/flysaint/Datawhale-/blob/master/XGBOOST/%E8%B4%AA%E5%BF%83%E7%AE%97%E6%B3%95_2.png)
我们可以发现对于所有的a，我们只要做一遍从左到右的扫描就可以枚举出所有分割的梯度和GL和GR。然后用上面的公式计算每个分割方案的分数就可以了。

观察这个目标函数，大家会发现第二个值得注意的事情就是引入分割不一定会使得情况变好，因为我们有一个引入新叶子的惩罚项。优化这个目标对应了树的剪枝， 当引入的分割带来的增益小于一个阀值的时候，我们可以剪掉这个分割。
下面是论文中的算法：
![](https://github.com/flysaint/Datawhale-/blob/master/XGBOOST/%E8%B4%AA%E5%BF%83%E7%AE%97%E6%B3%95_3.png)

#### 3.4.2 近似算法
主要针对数据太大，不能直接进行计算
![](https://github.com/flysaint/Datawhale-/blob/master/XGBOOST/%E8%BF%91%E4%BC%BC%E7%AE%97%E6%B3%95.png)

### 3.5 剪枝与正则化
![](https://github.com/flysaint/Datawhale-/blob/master/XGBOOST/%E5%89%AA%E6%9E%9D%E4%B8%8E%E6%AD%A3%E5%88%99%E5%8C%96.png)
什么时候停止分裂，进行剪枝呢？这里有两种剪枝策略。

1 前剪枝。当最佳分裂节点，出现负的gain时，就进行剪枝。缺点是，继续分裂可能得到更好的gain，这样剪枝会错过更好gain。

2.后剪枝。先让子树生长到最大的深度，从下往上，把所有Gain为负的叶子剪掉，直到节点的gain为正数为止。

### 3.6 其他问题
#### 3.6.1 XGBoost如何处理分类变量？
XGBoost只能处理连续变量，因此如果出现分类变量，如性别等，要进行 one-hot编码。
#### 3.6.2 如何构建一棵提升树，使得每个实例都有重要的权重？
定义目标函数，计算 $g_i,h_i$，放回原来没有计算权重的树学习算法中。
重新将模型和目标分开思考：理论怎么可以帮助机器学习工具。
#### 3.6.3 回到时间序列的问题，如果想学习阶跃函数。除了从上到下的分割方法，还有其他学习时间分割的方法吗?
![](https://github.com/flysaint/Datawhale-/blob/master/XGBOOST/3.6.3%E6%97%B6%E9%97%B4%E5%BA%8F%E5%88%97%E9%97%AE%E9%A2%98.png)
关键还是在于计算分裂的目标函数得分。有三种方案：

1）自顶向下贪婪。

2）自底向上贪婪。将每个点当作单独分组，用贪婪算法合并邻近的分组。

3）动态规划。可以找到最优的解决方案。

#### 3.6.4 XGBoost如何处理缺失值？
XGBoost会把缺失值当作稀疏矩阵来对待，进行分裂时，只计算非缺失的数值，然后把缺失值放到左右子树再次计算Gain，选择Gain较大的方式分裂。
如果出现训练集没有缺失值，而测试集出现缺失值，则缺失值会默认分配到右子树进行计算。具体算法如下：

![](https://github.com/flysaint/Datawhale-/blob/master/XGBOOST/%E7%BC%BA%E5%A4%B1%E5%80%BC%E5%A4%84%E7%90%86.png)

## 4 XGBoost与GBDT的联系和区别有哪些？

（1）GBDT是机器学习算法，XGBoost是该算法的工程实现。

（2）原始的GBDT算法基于经验损失函数的负梯度来构造新的决策树，只是在决策树构建完成后再进行剪枝。而XGBoost在决策树构建阶段就加入了正则项，有利于防止过拟合，从而提高模型的泛化能力。

（3）GBDT在模型训练时只使用了代价函数的一阶导数信息，XGBoost对代价函数进行二阶泰勒展开，可以同时使用一阶和二阶导数。

（4）传统的GBDT采用CART作为基分类器，XGBoost支持多种类型的基分类器，比如线性分类器。

（5）传统的GBDT在每轮迭代时使用全部的数据，XGBoost则采用了与随机森林相似的策略，支持对数据进行采样。

（6）传统的GBDT没有设计对缺失值进行处理，XGBoost能够自动学习出缺失值的处理策略。




## 5、XGBoost的参数

XGBoost的作者把所有的参数分成了三类： 
1、通用参数：宏观函数控制。 

2、Booster参数：控制每一步的booster(tree/regression)。 

3、学习目标参数：控制训练目标的表现。 
在这里我会类比GBM来讲解，所以作为一种基础知识，强烈推荐先阅读[这篇文章](http://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/)。

### 通用参数

这些参数用来控制XGBoost的宏观功能。

**1、booster[默认gbtree]**

*   选择每次迭代的模型，有两种选择： 
    gbtree：基于树的模型 
    gbliner：线性模型

**2、silent[默认0]**

*   当这个参数值为1时，静默模式开启，不会输出任何信息。
*   一般这个参数就保持默认的0，因为这样能帮我们更好地理解模型。

**3、nthread[默认值为最大可能的线程数]**

*   这个参数用来进行多线程控制，应当输入系统的核数。
*   如果你希望使用CPU全部的核，那就不要输入这个参数，算法会自动检测它。 
    还有两个参数，XGBoost会自动设置，目前你不用管它。接下来咱们一起看booster参数。

### booster参数

尽管有两种booster可供选择，我这里只介绍**tree booster**，因为它的表现远远胜过**linear booster**，所以linear booster很少用到。

**1、eta[默认0.3]**

*   和GBM中的 learning rate 参数类似。
*   通过减少每一步的权重，可以提高模型的鲁棒性。
*   典型值为0.01-0.2。

**2、min_child_weight[默认1]**

*   决定最小叶子节点样本权重和。
*   和GBM的 min_child_leaf 参数类似，但不完全一样。XGBoost的这个参数是最小_样本权重的和_，而GBM参数是最小_样本总数_。
*   这个参数用于避免过拟合。当它的值较大时，可以避免模型学习到局部的特殊样本。
*   但是如果这个值过高，会导致欠拟合。这个参数需要使用CV来调整。

**3、max_depth[默认6]**

*   和GBM中的参数相同，这个值为树的最大深度。
*   这个值也是用来避免过拟合的。max_depth越大，模型会学到更具体更局部的样本。
*   需要使用CV函数来进行调优。
*   典型值：3-10

**4、max_leaf_nodes**

*   树上最大的节点或叶子的数量。
*   可以替代max_depth的作用。因为如果生成的是二叉树，一个深度为n的树最多生成$n_2$个叶子。
*   如果定义了这个参数，GBM会忽略max_depth参数。

**5、gamma[默认0]**

*   在节点分裂时，只有分裂后损失函数的值下降了，才会分裂这个节点。Gamma指定了节点分裂所需的最小损失函数下降值。
*   这个参数的值越大，算法越保守。这个参数的值和损失函数息息相关，所以是需要调整的。

**6、max_delta_step[默认0]**

*   这参数限制每棵树权重改变的最大步长。如果这个参数的值为0，那就意味着没有约束。如果它被赋予了某个正值，那么它会让这个算法更加保守。
*   通常，这个参数不需要设置。但是当各类别的样本十分不平衡时，它对逻辑回归是很有帮助的。
*   这个参数一般用不到，但是你可以挖掘出来它更多的用处。

**7、subsample[默认1]**

*   和GBM中的subsample参数一模一样。这个参数控制对于每棵树，随机采样的比例。
*   减小这个参数的值，算法会更加保守，避免过拟合。但是，如果这个值设置得过小，它可能会导致欠拟合。
*   典型值：0.5-1

**8、colsample_bytree[默认1]**

*   和GBM里面的max_features参数类似。用来控制每棵随机采样的列数的占比(每一列是一个特征)。
*   典型值：0.5-1

**9、colsample_bylevel[默认1]**

*   用来控制树的每一级的每一次分裂，对列数的采样的占比。
*   我个人一般不太用这个参数，因为subsample参数和colsample_bytree参数可以起到相同的作用。但是如果感兴趣，可以挖掘这个参数更多的用处。

**10、lambda[默认1]**

*   权重的L2正则化项。(和Ridge regression类似)。
*   这个参数是用来控制XGBoost的正则化部分的。虽然大部分数据科学家很少用到这个参数，但是这个参数在减少过拟合上还是可以挖掘出更多用处的。

**11、alpha[默认1]**

*   权重的L1正则化项。(和Lasso regression类似)。
*   可以应用在很高维度的情况下，使得算法的速度更快。

**12、scale_pos_weight[默认1]**

*   在各类别样本十分不平衡时，把这个参数设定为一个正值，可以使算法更快收敛。

### 学习目标参数

这个参数用来控制理想的优化目标和每一步结果的度量方法。

**1、objective[默认reg:linear]**

*   这个参数定义需要被最小化的损失函数。最常用的值有： 

    *   binary:logistic 二分类的逻辑回归，返回预测的概率(不是类别)。
    *   multi:softmax 使用softmax的多分类器，返回预测的类别(不是概率)。 

        *   在这种情况下，你还需要多设一个参数：num_class(类别数目)。
    *   multi:softprob 和multi:softmax参数一样，但是返回的是每个数据属于各个类别的概率。

**2、eval_metric[默认值取决于objective参数的取值]**

*   对于有效数据的度量方法。
*   对于回归问题，默认值是rmse，对于分类问题，默认值是error。
*   典型值有： 

    *   rmse 均方根误差($\sqrt{\frac{\sum_{i=1}^N \epsilon^2}{N}}$)
    *   mae 平均绝对误差($\frac{\sum_{i=1}^N|\epsilon|}{N}$)
    *   logloss 负对数似然函数值
    *   error 二分类错误率(阈值为0.5)
    *   merror 多分类错误率
    *   mlogloss 多分类logloss损失函数
    *   auc 曲线下面积

**3、seed(默认0)**

*   随机数的种子
*   设置它可以复现随机数据的结果，也可以用于调整参数



## 参考资料

[Regression Tree 回归树](https://blog.csdn.net/weixin_40604987/article/details/79296427)

[xgboost原理](https://blog.csdn.net/a819825294/article/details/51206410)

[怎么理解决策树、xgboost能处理缺失值？而有的模型(svm)对缺失值比较敏感呢?](https://www.zhihu.com/question/58230411)

[XGBoost参数调优完全指南（附Python代码）](https://blog.csdn.net/u010657489/article/details/51952785)

《百面机器学习》-葫芦娃

《统计学习方法》-李航

《机器学习》-周志华



