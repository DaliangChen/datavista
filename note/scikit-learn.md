<link rel="stylesheet" href="../style.css">

# 1. Supervised learning

## 1.1. Linear Models

目标值预期是特征的线性组合。用数学符号表示，如果 $\hat{y}$
是预测值。

$$\hat{y}(w, x) = w_0 + w_1 x_1 + ... + w_p x_p$$

我们指定向量 $w = (w_1,..., w_p)$
作为 $coef_$ 和 $w_0$作为 $intercept_$

### 1.1.1. Ordinary Least Squares

📌 主要思想

我们假设因变量 $y$ 与输入变量 $x$ 之间存在线性关系：

$$
y = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_p x_p + \varepsilon
$$

核心目标：最小化残差平方和（RSS）

我们希望选出一个最优的 $w$，使得：

$$
\text{RSS}(w) = \sum_{i=1}^n (y_i - X_i w)^2
$$

即：让模型预测值 $Xw$ 尽可能接近真实值 $y$。

📌 注意事项

| 类别     | 注意点                         | 说明                                             |
| -------- | ------------------------------ | ------------------------------------------------ |
| 数据要求 | 线性关系假设                   | 目标变量 y 与输入 X 之间要“近似线性”             |
| 数据问题 | 异常值影响大                   | 因为用的是平方误差，异常值权重大，容易“拉歪”模型 |
| 特征关系 | 特征共线性（多重共线性）       | 特征之间高度相关会导致解不稳定（或不可逆）       |
| 维度问题 | 特征维度高于样本数             | 会导致 $X^TX$ 不可逆，模型无法求解闭式解         |
| 误差假设 | 残差独立、方差齐性             | 残差（误差项）应满足 i.i.d，误差方差一致         |
| 输入规模 | 特征未标准化可能影响数值稳定性 | 特征值差异大时，系数难以比较，计算精度也受影响   |
| 输出解释 | 系数只能解释“边际线性影响”     | 多数真实关系是非线性的，不宜过度解读系数         |

- 线性关系假设
  - 如果真实关系是非线性的，OLS 就无法很好拟合。
  - 解决方法：可以对特征做多项式扩展、log 转换，或者直接用非线性模型（如树模型）。
- 对异常值特别敏感
  - 因为误差是平方的，一个大的偏差会对损失函数影响很大。
  - 解决方法：
    - 使用 鲁棒回归（如 Huber 回归）
    - 对数据做异常值检测或 Winsorize
    - 改用 L1 损失（绝对值误差）模型
- 特征共线性（Multicollinearity）
  - 当两个或多个特征高度相关时，$X^TX$ 接近不可逆，OLS 解会非常不稳定，系数变动大。
  - 解决方法：
    - 用 Ridge 回归（L2 正则化）来缓解共线性；
    - 使用 PCA 等降维方法；
    - 去掉冗余特征（如变量选择）
- 特征维度大于样本数
  - 如果 p > n，OLS 无法唯一解出参数（无解或多解）。
  - 解决方法：使用 Ridge（能处理 p > n 的问题），或降维。
- 误差项的统计假设
  - OLS 经典假设包括：
    - 误差项独立同分布（i.i.d）
    - 误差期望为 0
    - 方差一致（Homoskedasticity）
  - 如果这些假设不成立，估计量依然无偏，但统计性质（如置信区间）就不可靠了。
  - 解决方法：用 White 标准误、加权最小二乘（WLS）等。
- 特征未标准化影响系数解释
  - 如果一个特征值在 0-1 之间，另一个在 0-10000，OLS 的系数就不具有比较性。
  - 解决方法：使用 `StandardScaler` 标准化特征。
- 模型容易过拟合（尤其是高维）
  - OLS 没有正则项，对小样本/多特征问题特别容易过拟合。
  - 解决方法：
    - 使用 Ridge 或 Lasso 回归；
    - 做交叉验证选择模型；
    - 降维或特征选择。
- 系数解释有限
  - OLS 的系数是对单一变量的“边际影响”解释，前提是其他变量不变。
    但如果变量之间相关性高，这种解释是不可靠的。

### 1.1.2. Ridge regression and classification

📌 主要思想

1. Ridge 回归的基本形式：

Ridge 回归是在普通最小二乘回归（OLS）的基础上添加了 L2 正则化项，目的是防止过拟合。

其目标函数如下：

$$
\min_w \|y - Xw\|_2^2 + \alpha \|w\|_2^2
$$

1. 正则化的作用：

- 在数据具有共线性（特征之间高度相关）时，普通最小二乘估计会变得不稳定。
- 引入正则化项可以对权重 $w$ 进行收缩，从而减少模型复杂度、提升稳定性。
- 随着 $\alpha$ 增大，模型系数会被进一步压缩。

5. 使用建议：

- 对于特征之间存在强相关性的回归问题，Ridge 回归通常优于普通线性回归。
- 如果你想要在正则化的同时进行特征选择（即让某些系数为零），应使用 Lasso 回归（L1 正则）。

📌 注意事项

一、特征需要标准化（归一化）

- 原因：
  岭回归对特征的尺度敏感，因为正则化项 $\|w\|_2^2$ 直接对权重施加约束。
- 做法：
  使用 `StandardScaler` 对所有特征进行标准化（均值为 0，方差为 1）。

二、正则化系数 $\alpha$ 需要调参

- 小 α： 正则化弱，模型复杂，可能过拟合
- 大 α： 正则化强，模型过于简单，可能欠拟合
- 推荐做法：

  - 使用交叉验证（如 `GridSearchCV` 或 `RidgeCV`）寻找最优 α
  - 典型调参范围：如 $\alpha \in [10^{-3}, 10^3]$

三、不适合做特征选择

- 岭回归不会将系数压缩为 0，因此不能自动剔除无关特征。
- 如果目标是寻找最重要的特征，应考虑：

  - Lasso 回归（L1 正则）
  - 或 ElasticNet（L1 + L2 结合）

四、结果解释性弱于普通线性回归

- 因为 Ridge 会缩小所有系数，解释变量和目标的直接线性关系被“压制”了。
- 如果你关心回归系数的实际意义（如医学/社会科学建模），可能不如线性回归直观。

五、目标变量不需要标准化，但特征必须

- $y$ 不需要标准化，岭回归对 $y$ 没有特别要求
- 但特征（$X$）必须标准化，否则正则化作用会失衡

六、适用于线性问题，不适合非线性建模

- 岭回归只能拟合线性或近似线性关系
- 如果数据存在显著的非线性关系，可以考虑：

  - 加入多项式特征（`PolynomialFeatures`）
  - 使用核方法（如 `KernelRidge`）

七、不要在稀疏数据上盲目使用 Ridge

- 如果你的输入数据是稀疏矩阵（如 TF-IDF 文本数据），标准化会破坏稀疏性
- 建议用不需要标准化的算法（如朴素贝叶斯），或者用特化的正则方法（如 Lasso）

八、与岭分类器（RidgeClassifier）的区别

- `Ridge` 是回归模型
- `RidgeClassifier` 是一个将分类转化为回归求解的模型，但其训练方式略有不同，不使用概率输出（不像 `LogisticRegression`）

### 1.1.3. Lasso

📌 主要思想

Lasso 回归的目标是在最小化预测误差的同时，引入对模型参数的稀疏约束，从而达到特征选择的效果。

Lasso 回归通过在普通最小二乘（OLS）的损失函数中加入一个 L1 范数（L1-norm） 的惩罚项，约束回归系数的大小：

$$
\min_{\beta} \left\{ \frac{1}{2n} \sum_{i=1}^n \left( y_i - X_i \cdot \beta \right)^2 + \alpha \sum_{j=1}^{p} |\beta_j| \right\}
$$

核心思想总结：

| 核心点       | 说明                                                                   |
| ------------ | ---------------------------------------------------------------------- |
| L1 正则化    | 通过绝对值惩罚来使部分特征的系数变为 0，起到“特征选择”的作用           |
| 稀疏解       | 与 Ridge 回归（L2 正则）不同，Lasso 能产生稀疏解 —— 只保留最有用的特征 |
| 降维能力     | 自动去除不重要的特征，尤其适用于高维数据（p ≫ n）的情况                |
| 模型解释性强 | 保留的特征少，模型更易于解释                                           |

适用场景

- 特征很多，但期望只保留一部分显著特征；
- 高维数据（如基因数据、文本分类等）；
- 希望构建简单、可解释的模型。

📌 主要特点

| 特点类别       | 内容                                                             |     |     |
| -------------- | ---------------------------------------------------------------- | --- | --- |
| 正则类型       | 使用 L1 正则化对模型参数施加约束                                 |
| 特征选择能力   | 能将部分回归系数压缩为 0，即自动去除不重要的特征（稀疏解）       |     |     |
| 模型可解释性强 | 因为只保留重要变量，模型更简单、更易解释                         |     |     |
| 防止过拟合     | 正则项有助于降低模型复杂度，避免在小样本下过拟合                 |     |     |
| 超参数控制     | 通过正则强度 $\alpha$ 控制模型复杂度与变量稀疏性之间的权衡       |     |     |
| 适用场景       | 高维数据（例如 p ≫ n），如文本分类、基因数据分析等               |     |     |
| 与 Ridge 区别  | Lasso 用 L1 正则（可稀疏）；Ridge 用 L2 正则（不会产生稀疏系数） |     |     |
| 多重共线性处理 | 对高度相关的变量，Lasso 往往只保留其中一个，不能很好地共享权重   |     |     |

📌 注意事项

特征标准化（非常重要）

- 原因：Lasso 依赖正则化惩罚（$\alpha \sum |\beta_j|$），而特征的尺度会影响惩罚力度。
- 做法：在使用 Lasso 前必须对输入特征进行 标准化（StandardScaler）或归一化（MinMaxScaler）。

正则系数 $\alpha$ 的选择

- 太小：模型接近普通线性回归，容易过拟合；
- 太大：模型过度稀疏，重要特征可能被压缩为 0，导致欠拟合；
- 建议：使用交叉验证（如 `LassoCV`）自动选择最优 $\alpha$。

对共线性特征敏感（变量互相相关）

- 如果多个特征高度相关，Lasso 可能只保留其中一个，其余设为 0；
- 解决办法：

  - 可使用 ElasticNet（结合 L1 和 L2）更稳健地处理共线性；
  - 或提前使用降维方法如 PCA；

稀疏性不是万能的

- 稀疏解适合解释性要求高的任务；
- 如果目标是最大化预测精度，而不是解释变量，Lasso 可能不是最佳选择；
- 建议根据任务目标选择合适的模型。

小样本 + 高维度时要谨慎

- 虽然 Lasso 适用于高维，但如果样本数量极少（n ≪ p），结果可能不稳定；
- 特别是在重要特征被错误压为 0 的风险下，建议使用更鲁棒的方式（如 bootstrap 验证特征稳定性）；

输出系数容易震荡（不稳定）

- Lasso 回归的输出结果对数据扰动较敏感；
- 对于“边缘重要”的变量，其是否被选中可能随着样本稍微变动而不同；
- 可使用模型稳定性分析（stability selection）辅助判断特征是否可信。

目标变量不要标准化

- 特征需要标准化
- 目标变量 y 通常不要标准化，否则预测值解释会变复杂；
- 除非你专门对 y 有分布约束或预测范围要求。

### 1.1.4. Multi-task Lasso

📌 主要思想

在具有多个相关输出变量的线性回归问题中，同时进行所有任务的稀疏特征选择，以利用任务间的共同结构，提高预测准确性与模型简洁性。

数学表达（与普通 Lasso 的区别）

对于多个输出变量（比如 $Y \in \mathbb{R}^{n \times T}$，有 T 个任务），MultiTaskLasso 优化以下目标：

$$
\min_{W} \left\{ \frac{1}{2n} \| Y - XW \|_F^2 + \alpha \sum_{j=1}^{p} \| W_{j, :} \|_2 \right\}
$$

核心机制

| 特点                       | 说明                                                                            |
| -------------------------- | ------------------------------------------------------------------------------- |
| L2/L1 组合正则化           | 对每一列特征在所有任务上的系数使用 L2 范数，但所有特征加和仍是稀疏的（L1 结构） |
| 行稀疏性（Group sparsity） | 若某特征对所有任务都不重要，其对应的整行 $W_{j,:}$ 会被压缩为 0                 |
| 共享特征选择               | 所有任务共享同一组被选择的特征，适合多个输出高度相关的情况                      |
| 任务间信息共享             | 相比独立训练多个 Lasso，更能利用多个输出间的关联信息提高泛化能力                |

与普通 Lasso 的区别

| 对比项   | Lasso                        | MultiTaskLasso                     |
| -------- | ---------------------------- | ---------------------------------- |
| 适用任务 | 单任务（一个输出）           | 多任务（多个输出）                 |
| 特征选择 | 每个任务单独决定是否使用特征 | 所有任务共享相同特征选择           |
| 稀疏结构 | 每个任务稀疏独立             | 行稀疏（某特征同时为所有任务置 0） |
| 正则化   | L1（单个参数）               | L2/L1 混合（组稀疏）               |

📌 主要特点

| 特点类别       | 说明                                                                                |
| -------------- | ----------------------------------------------------------------------------------- |
| 多任务处理     | 同时拟合多个相关的回归任务，输出多个连续变量的预测结果。                            |
| 共享特征选择   | 通过组稀疏正则化，所有任务共享相同的特征子集，实现统一的特征选择。                  |
| 组稀疏结构     | 对每个特征对应的所有任务系数一起正则化（L2 范数），若该组系数全为零则该特征被丢弃。 |
| 正则化类型     | 使用混合的 L1/L2 正则化，结合了 Lasso 的稀疏性和多任务的结构信息。                  |
| 提高泛化能力   | 通过利用任务间的相关性，提升模型在多个任务上的预测准确性和稳定性。                  |
| 避免过拟合     | 正则化帮助减少模型复杂度，避免在多任务学习中的过拟合问题。                          |
| 适用于高维数据 | 可以处理高维特征数据，自动筛选出重要特征，减小模型规模。                            |
| 模型解释性强   | 因为共享特征选择，便于理解多个任务间的共同影响因素。                                |

📌 注意事项

特征预处理必不可少

- 必须对输入特征进行标准化（均值为 0，方差为 1），否则正则化惩罚会因特征尺度不同而失效。
- 多任务数据通常维度较高，标准化对模型稳定性和收敛速度都非常关键。

正则化参数 $\alpha$ 调节

- $\alpha$ 控制模型的稀疏性和拟合程度：

  - $\alpha$ 太大，会导致过度稀疏，重要特征被忽略；
  - $\alpha$ 太小，模型复杂，可能过拟合。

- 推荐使用交叉验证（例如 `MultiTaskLassoCV`）自动调优。

任务相关性假设

- MultiTaskLasso 假设多个任务之间共享相同重要特征，即任务间存在相关性。
- 若任务差异很大，强制共享特征可能反而降低性能。
- 此时应考虑单独建模或使用更灵活的多任务模型。

高度相关特征的处理

- 虽然 MultiTaskLasso 通过组稀疏选择特征，但对高度共线性特征仍然敏感。
- 可结合 ElasticNet 多任务版或先做降维、特征筛选。

样本数量要求

- 多任务模型需要足够的样本数支持多输出拟合，否则可能导致欠拟合或不稳定。
- 样本数远少于特征数时，结果可能不稳定，需要谨慎。

解释和使用场景

- 适合多个输出变量相关且共享影响因素的场景。
- 不适合任务完全独立或特征影响机制差异极大的情况。

目标变量标准化

- 目标变量一般不需要标准化，但若不同任务量纲相差很大，适当归一化有助于平衡训练。

### 1.1.5. Elastic-Net

📌 主要思想

1. 目标函数

Elastic Net 的优化目标是最小化下面的损失函数：

$$
\min_{\beta} \left\{ \frac{1}{2n} \| y - X\beta \|_2^2 + \alpha \left( \rho \|\beta\|_1 + \frac{1 - \rho}{2} \|\beta\|_2^2 \right) \right\}
$$

2. 核心思想

- 结合 L1 和 L2 正则化的优点：L1 带来模型稀疏性（变量选择），L2 带来模型稳定性和处理多重共线性；
- 解决 Lasso 在高度相关特征中的不稳定选择问题：Elastic Net 允许相关特征共同进入模型，避免了 Lasso 只选择其中一个的缺陷；
- 适合高维数据：在 $p \gg n$ 的场景下，Elastic Net 依然表现优异。

3. 优势总结

| 优势                 | 说明                                       |
| -------------------- | ------------------------------------------ |
| 变量选择与正则化结合 | 既能自动选择重要变量，又能控制模型复杂度   |
| 处理共线性           | 对高度相关特征更稳定，不随数据波动随机选择 |
| 适合高维数据         | 适合特征数大于样本数的情况                 |
| 灵活调节参数         | $\rho$ 控制 L1/L2 比例，模型可定制         |

📌 主要特点

| 特点类别           | 说明                                                                  |
| ------------------ | --------------------------------------------------------------------- |
| 结合 L1 和 L2 正则 | 同时使用 L1（稀疏）和 L2（稳定）正则，融合两者优势。                  |
| 自动特征选择       | L1 正则项使部分系数变为零，实现特征筛选。                             |
| 处理共线性         | L2 正则项减缓高度相关特征间的随机选择问题，使相关变量可以被一起保留。 |
| 适合高维数据       | 对于特征数远大于样本数的场景，表现优异。                              |
| 参数灵活调节       | 通过参数 $\rho$ 灵活控制 L1 和 L2 比例，适应不同数据特性。            |
| 模型稳定性强       | 相比纯 Lasso，模型在数据波动时更稳定，避免过度稀疏。                  |
| 广泛应用           | 在基因表达、金融建模、文本挖掘等领域表现出色。                        |

📌 注意事项

特征预处理

- 必须对输入特征进行标准化（StandardScaler），否则正则项会因特征尺度不同导致惩罚不均。
- 标准化有助于算法收敛和参数解释。

参数调节

- Elastic Net 有两个重要超参数：

  - $\alpha$：整体正则化强度；
  - $\rho$（或 l1_ratio）：控制 L1 和 L2 正则化的权重比例。

- 需要使用交叉验证（如 `ElasticNetCV`）来自动寻找最佳组合，避免欠拟合或过拟合。

对共线性的优势

- Elastic Net 比纯 Lasso 更适合高度相关特征的场景，但在极端多重共线性时仍需谨慎。
- 可以结合降维或特征选择方法提高表现。

稀疏性与稳定性的权衡

- $\rho$ 趋近于 1，模型更稀疏，但稳定性可能下降；
- $\rho$ 趋近于 0，模型更平滑稳定，但稀疏性弱；
- 根据实际需求调整。

计算资源

- Elastic Net 计算相对 Ridge 和 Lasso 略复杂，尤其在大规模数据时，注意计算成本。

目标变量处理

- 一般不需要对目标变量进行标准化，除非不同任务量纲差异大。

解释性

- 由于部分系数不为零，模型有一定的可解释性，但注意系数受参数影响较大，需谨慎解读。

### 1.1.6. Multi-task Elastic-Net

📌 主要思想

在多个相关回归任务中，同时使用 L1（稀疏）和 L2（稳定）正则化，并通过共享特征选择（行稀疏），提升模型性能和泛化能力。

数学表达式

$$
\min_{W} \left\{ \frac{1}{2n} \| Y - XW \|_F^2 + \alpha \left( \rho \sum_{j=1}^p \|W_{j,:}\|_2 + \frac{1 - \rho}{2} \|W\|_F^2 \right) \right\}
$$

| 特点             | 描述                                                           |
| ---------------- | -------------------------------------------------------------- |
| 多任务回归       | 同时预测多个输出（多个 Y），而不是多个独立回归                 |
| 任务共享特征选择 | 所有任务使用同一组重要特征（行稀疏结构）                       |
| L1 + L2 正则结合 | 提供 Lasso 的稀疏性 + Ridge 的稳定性，特别适合相关特征多的情形 |
| 稳定 + 稀疏      | 相比 MultiTaskLasso，更稳健、更抗共线性                        |
| 适合高维多任务   | 在样本数小、特征多、多任务的实际问题中表现优秀                 |

适用场景

- 多输出变量高度相关；
- 希望多个任务共享相同的特征子集（解释性好）；
- 特征数多、样本少（如基因数据、传感器信号等）；
- 存在特征共线性，需要模型稳健；

📌 主要特点

| 类别                   | 说明                                                                                                         |
| ---------------------- | ------------------------------------------------------------------------------------------------------------ |
| 多任务建模             | 可以同时预测多个相关输出变量（回归任务），提升任务间协同建模能力。                                           |
| 共享特征选择（行稀疏） | 对所有任务使用相同的特征子集，提升模型简洁性和可解释性。即，如果某个特征对所有任务都不重要，它会整体被剔除。 |
| 融合 L1 与 L2 正则化   | 同时具备 Lasso 的特征选择能力（L1）和 Ridge 的抗共线性能力（L2），通过 `l1_ratio` 控制平衡。                 |
| 模型更稳定             | 相比 MultiTaskLasso，对特征共线性更鲁棒，输出系数更平滑、不容易震荡。                                        |
| 适合高维多输出问题     | 尤其适合样本数少、特征数多、多输出变量同时预测的复杂问题。                                                   |
| 可调节性强             | 可通过 `alpha` 和 `l1_ratio` 控制整体正则强度和 L1/L2 权重，实现更灵活的模型调节。                           |

📌 注意事项

特征标准化是必须的

- Elastic Net 依赖 L1/L2 正则项，如果特征量纲不一致，会导致正则惩罚失衡。
- 必须对输入特征进行标准化（如 `StandardScaler`）：

超参数调节要谨慎

- 两个关键超参数：
  - `alpha`: 控制正则强度（整体惩罚力度）；
  - `l1_ratio`: 控制 L1（稀疏）与 L2（平滑）之间的权重，常设为 0.1\~0.9。
- 推荐使用交叉验证（如 `MultiTaskElasticNetCV`）自动选择最优组合。

任务应具有一定的相关性

- MultiTaskElasticNet 假设多个任务（多个输出变量）共享特征子集；
- 如果任务之间完全独立，强制共享特征可能反而影响性能；
- 可先做相关性分析（例如输出变量之间的皮尔逊相关系数）再决定使用与否。

注意输出维度匹配

- `Y` 必须是二维矩阵（即 shape 为 `[n_samples, n_outputs]`）；
- 如果只预测一个输出，不建议使用 MultiTask 版本，使用普通 `ElasticNet` 即可。

模型可解释性要结合业务理解

- 虽然模型能自动选特征，但最终解释哪些特征被选择，应结合任务含义、领域知识判断；
- 特别是在生物医学、金融等对“因果解释”要求高的场景中。

样本不足时易不稳定

- 在样本数较少的情况下，多任务模型仍可能发生过拟合；
- 可使用正则较强的初始模型（较大 `alpha`），或先进行特征降维、过滤。

多任务模型训练较慢

- 相比单任务 Lasso/Ridge，MultiTaskElasticNet 的计算复杂度更高，特别是在高维数据下；
- 可提前用稀疏性强的模型进行预筛选特征，减轻计算负担。

### 1.1.7. Least Angle Regression

📌 主要思想

LARS 是一种类似于前向逐步回归的算法，在每一步中只引入最相关的特征，但它采用了一种更温和且线性地前进的策略，从而更高效地逼近 Lasso 路径。

类比传统前向回归：

- 传统前向回归每一步固定选择与残差最相关的变量并完全加入；
- LARS 不会一下子用满这个变量，而是逐渐沿最相关变量方向前进；
- 当另一个变量与当前残差变得一样相关时，LARS 开始沿两个变量的方向同时前进。

工作流程简述：

1. 初始化：所有系数为 0；
2. 在每一步：

   - 选择与当前残差最相关的变量（即相关系数最大）；
   - 沿着该变量的方向逐步增加系数；
   - 一旦另一个变量与残差的相关性追平，就将它加入活动集合；
   - 沿着这些变量的“等角方向”（least angle）继续移动；

3. 重复，直到所有变量都被加入，或满足停止条件。

特点概括

| 特点                       | 描述                                                        |
| -------------------------- | ----------------------------------------------------------- |
| 高效逼近 Lasso 路径        | LARS 通过逐步调整系数方向，模拟出与 Lasso 相似的正则路径    |
| 速度快                     | 特别适合高维数据（p ≫ n）场景下，相比 Lasso 更快            |
| 系数路径可解释性强         | LARS 返回的是一个路径解（所有变量加入的顺序、系数变化过程） |
| 稀疏性                     | 每一步最多增加一个变量，天然产生稀疏模型                    |
| 精确控制变量进入模型的顺序 | 有助于理解变量的重要性及模型结构                            |

适用场景

- 高维小样本数据（如基因数据分析）；
- 想快速估算完整的 Lasso 解路径；
- 希望理解变量进入模型的顺序和过程；
- 模型追求稀疏和可解释性，而不是极致预测精度。

📌 注意事项

适合高维稀疏场景

- LARS 设计初衷就是用于特征数远大于样本数（p ≫ n）且只有少数特征有用的情况。
- 如果特征数量不多，或者变量并不稀疏，LARS 的优势会减少，普通线性回归或 Ridge 会更合适。

对特征共线性敏感

- 与 Lasso 类似，LARS 对高度相关的特征会在路径中随机选择其一，结果可能不稳定。
- 可使用 LassoLars（LARS 的 Lasso 变体）来自动进行正则化以缓解这个问题。

输入特征需标准化

- 因为 LARS 是基于变量间相关性前进的，如果特征没有统一量纲，会影响变量进入模型的顺序。
- 推荐使用 `StandardScaler` 对所有输入特征进行标准化：

只适用于线性模型

- LARS 是线性回归方法，不能直接用于非线性关系建模；
- 如果数据存在显著非线性关系，需考虑核方法、非线性回归或特征工程。

计算稳定性差于坐标下降法

- 尽管 LARS 路径构建高效，但在浮点精度、噪声数据上，数值稳定性可能略逊于 Lasso 坐标下降（如 `LassoCV`）。
- 尤其是在大规模稠密数据上，坐标下降有更强的收敛保证。

不直接提供正则强度调节

- 与 Lasso 的 $\alpha$ 不同，标准 LARS 没有一个可控的正则化强度超参数；
- 不过可以通过“早停”或者选择特定路径点来间接控制模型复杂度。

模型选择依赖路径截断点

- 使用 LARS 时，往往需根据交叉验证或 AIC/BIC 选择最佳截断点（变量数量），否则容易过拟合；
- `LarsCV` 提供交叉验证支持，自动选择截断点。

### 1.1.8. LARS Lasso

📌 主要思想  
📌 主要特点
📌 注意事项

### 1.1.9. Orthogonal Matching Pursuit (OMP)

📌 主要思想  
📌 主要特点
📌 注意事项

### 1.1.10. Bayesian Regression

📌 主要思想  
📌 主要特点
📌 注意事项

### 1.1.11. Logistic regression

### 1.1.12. Generalized Linear Models

### 1.1.13. Stochastic Gradient Descent

### 1.1.14. Perceptron

### 1.1.15. Passive Aggressive Algorithms

### 1.1.16. Robustness regression: outliers and modeling errors

### 1.1.17. Quantile Regression

### 1.1.18. Polynomial regression: extending linear models with basis functions

## 1.2. Linear and Quadratic Discriminant Analysis

## 1.3. Kernel ridge regression

## 1.4. Support Vector Machines

### 1.4.1. Classification

### 1.4.2. Regression

### 1.4.3. Density estimation, novelty detection

### 1.4.4. Complexity

### 1.4.5. Tips on Practical Use

### 1.4.6. Kernel functions

### 1.4.7. Mathematical formulation

### 1.4.8. Implementation details

## 1.5. Stochastic Gradient Descent

### 1.5.1. Classification

### 1.5.2. Regression

### 1.5.3. Online One-Class SVM

### 1.5.4. Stochastic Gradient Descent for sparse data

### 1.5.5. Complexity

### 1.5.6. Stopping criterion

### 1.5.7. Tips on Practical Use

### 1.5.8. Mathematical formulation

### 1.5.9. Implementation details

## 1.6. Nearest Neighbors

### 1.6.1. Unsupervised Nearest Neighbors

### 1.6.2. Nearest Neighbors Classification

### 1.6.3. Nearest Neighbors Regression

### 1.6.4. Nearest Neighbor Algorithms

### 1.6.5. Nearest Centroid Classifier

### 1.6.6. Nearest Neighbors Transformer

### 1.6.7. Neighborhood Components Analysis

## 1.7. Gaussian Processes

### 1.7.1. Gaussian Process Regression (GPR)

### 1.7.2. Gaussian Process Classification (GPC)

### 1.7.3. GPC examples

### 1.7.4. Kernels for Gaussian Processes

## 1.8. Cross decomposition

### 1.8.1. PLSCanonical

### 1.8.2. PLSSVD

### 1.8.3. PLSRegression

### 1.8.4. Canonical Correlation Analysis

## 1.9. Naive Bayes

### 1.9.1. Gaussian Naive Bayes

### 1.9.2. Multinomial Naive Bayes

### 1.9.3. Complement Naive Bayes

### 1.9.4. Bernoulli Naive Bayes

### 1.9.5. Categorical Naive Bayes

### 1.9.6. Out-of-core naive Bayes model fitting

## 1.10. Decision Trees

### 1.10.1. Classification

### 1.10.2. Regression

### 1.10.3. Multi-output problems

### 1.10.4. Complexity

### 1.10.5. Tips on practical use

### 1.10.6. Tree algorithms: ID3, C4.5, C5.0 and CART

### 1.10.7. Mathematical formulation

### 1.10.8. Missing Values Support

### 1.10.9. Minimal Cost-Complexity Pruning

## 1.11. Ensembles: Gradient boosting, random forests, bagging, voting, stacking

### 1.11.1. Gradient-boosted trees

### 1.11.2. Random forests and other randomized tree ensembles

### 1.11.3. Bagging meta-estimator

### 1.11.4. Voting Classifier

### 1.11.5. Voting Regressor

### 1.11.6. Stacked generalization

### 1.11.7. AdaBoost

## 1.12. Multiclass and multioutput algorithms

### 1.12.1. Multiclass classification

### 1.12.2. Multilabel classification

### 1.12.3. Multiclass-multioutput classification

### 1.12.4. Multioutput regression

## 1.13. Feature selection

### 1.13.1. Removing features with low variance

### 1.13.2. Univariate feature selection

### 1.13.3. Recursive feature elimination

### 1.13.4. Feature selection using SelectFromModel

### 1.13.5. Sequential Feature Selection

### 1.13.6. Feature selection as part of a pipeline

## 1.14. Semi-supervised learning

### 1.14.1. Self Training

### 1.14.2. Label Propagation

## 1.15. Isotonic regression

## 1.16. Probability calibration

### 1.16.1. Calibration curves

### 1.16.2. Calibrating a classifier

### 1.16.3. Usage

## 1.17. Neural network models (supervised)

### 1.17.1. Multi-layer Perceptron

### 1.17.2. Classification

### 1.17.3. Regression

### 1.17.4. Regularization

### 1.17.5. Algorithms

### 1.17.6. Complexity

### 1.17.7. Tips on Practical Use

### 1.17.8. More control with warm_start

# 2. Unsupervised learning

## 2.1. Gaussian mixture models

### 2.1.1. Gaussian Mixture

### 2.1.2. Variational Bayesian Gaussian Mixture

## 2.2. Manifold learning

### 2.2.1. Introduction

### 2.2.2. Isomap

### 2.2.3. Locally Linear Embedding

### 2.2.4. Modified Locally Linear Embedding

### 2.2.5. Hessian Eigenmapping

### 2.2.6. Spectral Embedding

### 2.2.7. Local Tangent Space Alignment

### 2.2.8. Multi-dimensional Scaling (MDS)

### 2.2.9. t-distributed Stochastic Neighbor Embedding (t-SNE)

### 2.2.10. Tips on practical use

## 2.3. Clustering

### 2.3.1. Overview of clustering methods

### 2.3.2. K-means

### 2.3.3. Affinity Propagation

### 2.3.4. Mean Shift

### 2.3.5. Spectral clustering

### 2.3.6. Hierarchical clustering

### 2.3.7. DBSCAN

### 2.3.8. HDBSCAN

### 2.3.9. OPTICS

### 2.3.10. BIRCH

### 2.3.11. Clustering performance evaluation

## 2.4. Biclustering

### 2.4.1. Spectral Co-Clustering

### 2.4.2. Spectral Biclustering

### 2.4.3. Biclustering evaluation

## 2.5. Decomposing signals in components (matrix factorization problems)

### 2.5.1. Principal component analysis (PCA)

### 2.5.2. Kernel Principal Component Analysis (kPCA)

### 2.5.3. Truncated singular value decomposition and latent semantic analysis

### 2.5.4. Dictionary Learning

### 2.5.5. Factor Analysis

### 2.5.6. Independent component analysis (ICA)

### 2.5.7. Non-negative matrix factorization (NMF or NNMF)

### 2.5.8. Latent Dirichlet Allocation (LDA)

## 2.6. Covariance estimation

### 2.6.1. Empirical covariance

### 2.6.2. Shrunk Covariance

### 2.6.3. Sparse inverse covariance

### 2.6.4. Robust Covariance Estimation

## 2.7. Novelty and Outlier Detection

### 2.7.1. Overview of outlier detection methods

### 2.7.2. Novelty Detection

### 2.7.3. Outlier Detection

### 2.7.4. Novelty detection with Local Outlier Factor

## 2.8. Density Estimation

### 2.8.1. Density Estimation: Histograms

### 2.8.2. Kernel Density Estimation

## 2.9. Neural network models (unsupervised)

### 2.9.1. Restricted Boltzmann machines

# 3. Model selection and evaluation

## 3.1. Cross-validation: evaluating estimator performance

### 3.1.1. Computing cross-validated metrics

### 3.1.2. Cross validation iterators

### 3.1.3. A note on shuffling

### 3.1.4. Cross validation and model selection

### 3.1.5. Permutation test score

## 3.2. Tuning the hyper-parameters of an estimator

### 3.2.1. Exhaustive Grid Search

### 3.2.2. Randomized Parameter Optimization

### 3.2.3. Searching for optimal parameters with successive halving

### 3.2.4. Tips for parameter search

### 3.2.5. Alternatives to brute force parameter search

## 3.3. Tuning the decision threshold for class prediction

### 3.3.1. Post-tuning the decision threshold

## 3.4. Metrics and scoring: quantifying the quality of predictions

### 3.4.1. Which scoring function should I use?

### 3.4.2. Scoring API overview

### 3.4.3. The `scoring` parameter: defining model evaluation rules

### 3.4.4. Classification metrics

### 3.4.5. Multilabel ranking metrics

### 3.4.6. Regression metrics

### 3.4.7. Clustering metrics

### 3.4.8. Dummy estimators

## 3.5. Validation curves: plotting scores to evaluate models

### 3.5.1. Validation curve

### 3.5.2. Learning curve

# 4. Metadata Routing

## 4.1. Usage Examples

### 4.1.1. Weighted scoring and fitting

### 4.1.2. Weighted scoring and unweighted fitting

### 4.1.3. Unweighted feature selection

### 4.1.4. Different scoring and fitting weights

## 4.2. API Interface

## 4.3. Metadata Routing Support Status

# 5. Inspection

## 5.1. Partial Dependence and Individual Conditional Expectation plots

### 5.1.1. Partial dependence plots

### 5.1.2. Individual conditional expectation (ICE) plot

### 5.1.3. Mathematical Definition

### 5.1.4. Computation methods

## 5.2. Permutation feature importance

### 5.2.1. Outline of the permutation importance algorithm

### 5.2.2. Relation to impurity-based importance in trees

### 5.2.3. Misleading values on strongly correlated features

# 6. Visualizations

## 6.1. Available Plotting Utilities

### 6.1.1. Display Objects

# 7. Dataset transformations

## 7.1. Pipelines and composite estimators

### 7.1.1. Pipeline: chaining estimators

### 7.1.2. Transforming target in regression

### 7.1.3. FeatureUnion: composite feature spaces

### 7.1.4. ColumnTransformer for heterogeneous data

### 7.1.5. Visualizing Composite Estimators

## 7.2. Feature extraction

### 7.2.1. Loading features from dicts

### 7.2.2. Feature hashing

### 7.2.3. Text feature extraction

### 7.2.4. Image feature extraction

## 7.3. Preprocessing data

### 7.3.1. Standardization, or mean removal and variance scaling

### 7.3.2. Non-linear transformation

### 7.3.3. Normalization

### 7.3.4. Encoding categorical features

### 7.3.5. Discretization

### 7.3.6. Imputation of missing values

### 7.3.7. Generating polynomial features

### 7.3.8. Custom transformers

## 7.4. Imputation of missing values

### 7.4.1. Univariate vs. Multivariate Imputation

### 7.4.2. Univariate feature imputation

### 7.4.3. Multivariate feature imputation

### 7.4.4. Nearest neighbors imputation

### 7.4.5. Keeping the number of features constant

### 7.4.6. Marking imputed values

### 7.4.7. Estimators that handle NaN values

## 7.5. Unsupervised dimensionality reduction

### 7.5.1. PCA: principal component analysis

### 7.5.2. Random projections

### 7.5.3. Feature agglomeration

## 7.6. Random Projection

### 7.6.1. The Johnson-Lindenstrauss lemma

### 7.6.2. Gaussian random projection

### 7.6.3. Sparse random projection

### 7.6.4. Inverse Transform

## 7.7. Kernel Approximation

### 7.7.1. Nystroem Method for Kernel Approximation

### 7.7.2. Radial Basis Function Kernel

### 7.7.3. Additive Chi Squared Kernel

### 7.7.4. Skewed Chi Squared Kernel

### 7.7.5. Polynomial Kernel Approximation via Tensor Sketch

### 7.7.6. Mathematical Details

## 7.8. Pairwise metrics, Affinities and Kernels

### 7.8.1. Cosine similarity

### 7.8.2. Linear kernel

### 7.8.3. Polynomial kernel

### 7.8.4. Sigmoid kernel

### 7.8.5. RBF kernel

### 7.8.6. Laplacian kernel

### 7.8.7. Chi-squared kernel

## 7.9. Transforming the prediction target (`y`)

### 7.9.1. Label binarization

### 7.9.2. Label encoding

# 8. Dataset loading utilities

## 8.1. Toy datasets

### 8.1.1. Iris plants dataset

### 8.1.2. Diabetes dataset

### 8.1.3. Optical recognition of handwritten digits dataset

### 8.1.4. Linnerrud dataset

### 8.1.5. Wine recognition dataset

### 8.1.6. Breast cancer Wisconsin (diagnostic) dataset

## 8.2. Real world datasets

### 8.2.1. The Olivetti faces dataset

### 8.2.2. The 20 newsgroups text dataset

### 8.2.3. The Labeled Faces in the Wild face recognition dataset

### 8.2.4. Forest covertypes

### 8.2.5. RCV1 dataset

### 8.2.6. Kddcup 99 dataset

### 8.2.7. California Housing dataset

### 8.2.8. Species distribution dataset

## 8.3. Generated datasets

### 8.3.1. Generators for classification and clustering

### 8.3.2. Generators for regression

### 8.3.3. Generators for manifold learning

### 8.3.4. Generators for decomposition

## 8.4. Loading other datasets

### 8.4.1. Sample images

### 8.4.2. Datasets in svmlight / libsvm format

### 8.4.3. Downloading datasets from the openml.org repository

### 8.4.4. Loading from external datasets

# 9. Computing with scikit-learn

## 9.1. Strategies to scale computationally: bigger data

### 9.1.1. Scaling with instances using out-of-core learning

## 9.2. Computational Performance

### 9.2.1. Prediction Latency

### 9.2.2. Prediction Throughput

### 9.2.3. Tips and Tricks

## 9.3. Parallelism, resource management, and configuration

### 9.3.1. Parallelism

### 9.3.2. Configuration switches

# 10. Model persistence

## 10.1. Workflow Overview

### 10.1.1. Train and Persist the Model

## 10.2. ONNX

## 10.3. `skops.io`

## 10.4. `pickle`, `joblib`, and `cloudpickle`

## 10.5. Security & Maintainability Limitations

### 10.5.1. Replicating the training environment in production

### 10.5.2. Serving the model artifact

## 10.6. Summarizing the key points

# 11. Common pitfalls and recommended practices

## 11.1. Inconsistent preprocessing

## 11.2. Data leakage

### 11.2.1. How to avoid data leakage

### 11.2.2. Data leakage during pre-processing

## 11.3. Controlling randomness

### 11.3.1. Using `None` or `RandomState` instances, and repeated calls to `fit` and `split`

### 11.3.2. Common pitfalls and subtleties

### 11.3.3. General recommendations

# 12. Dispatching

## 12.1. Array API support (experimental)

### 12.1.1. Example usage

### 12.1.2. Support for `Array API`\-compatible inputs

### 12.1.3. Input and output array type handling

### 12.1.4. Common estimator checks

# 13. Choosing the right estimator

# 14. External Resources, Videos and Talks

## 14.1. The scikit-learn MOOC

## 14.2. Videos

## 14.3. New to Scientific Python?

## 14.4. External Tutorials
