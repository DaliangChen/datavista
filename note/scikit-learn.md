<link rel="stylesheet" href="../style.css">

# 1. Supervised learning

## 1.1. Linear Models

目标值预期是特征的线性组合。用数学符号表示，如果 $\hat{y}$
是预测值。

$$\hat{y}(w, x) = w_0 + w_1 x_1 + ... + w_p x_p$$

我们指定向量 $w = (w_1,..., w_p)$
作为 $coef_$ 和 $w_0$作为 $intercept_$

### 1.1.1. Ordinary Least Squares

普通最小二乘（OLS）是一种线性回归方法，通过最小化残差平方和（RSS）来拟合一个线性模型。

其基本形式是：

$$
\hat{y}(w, x) = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_p x_p
$$

其中：

- $\hat{y}$ 是预测值
- $w$ 是回归系数（包括截距项 $w_0$）
- $x$ 是特征向量

OLS 通过最小化以下损失函数来估计参数：

$$
\min_w \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

适用于：

- 特征之间无多重共线性（即特征之间不强相关）
- 数据近似线性
- 噪声为正态分布的情形效果最佳

注意事项

- OLS 不适合处理高维稀疏数据（推荐使用 `Ridge` 或 `Lasso`）
- 多重共线性会导致系数不稳定

### 1.1.2. Ridge regression and classification

Ridge 回归 是一种线性回归的正则化版本，通过在最小二乘损失中加入 ℓ2 正则项来约束模型复杂度，有助于应对共线性、过拟合等问题。

Ridge 回归最小化以下目标函数：

$$
\min_w \|Xw - y\|^2_2 + \alpha \|w\|^2_2
$$

其中：

- $\|Xw - y\|^2_2$：残差平方和（RSS）
- $\|w\|^2_2$：回归系数的平方和（ℓ2 范数）
- $\alpha$：正则化强度（超参数）

ℓ2 正则项限制了系数的大小，使模型不容易过拟合。

与普通最小二乘（OLS）的区别

- OLS：最小化残差平方和（不正则化）
- Ridge：在 OLS 基础上加了 ℓ2 正则化
- Ridge 更适合高维或多重共线性的场景

应用场景

- 特征维度高（如文本数据）
- 特征之间存在多重共线性
- 需要稳定的解（相比 OLS）

### 1.1.3. Lasso

Lasso（Least Absolute Shrinkage and Selection Operator）是一种线性回归方法，通过加入 ℓ1 正则项来实现特征选择和系数稀疏化。
相比 Ridge 回归，Lasso 可以将一些回归系数精确压缩为 0，从而实现 自动特征选择。

Lasso 回归的目标是最小化以下函数：

$$
\min_w \frac{1}{2n_{\text{samples}}} \|Xw - y\|^2_2 + \alpha \|w\|_1
$$

其中：

- $\|Xw - y\|^2_2$：残差平方和（RSS）
- $\|w\|_1$：回归系数的绝对值之和（ℓ1 范数）
- $\alpha$：正则化强度，控制稀疏程度

ℓ1 正则化可以使一些系数为 0，从而有效去除不重要的特征。

Lasso 的优势

- 自动进行特征选择（稀疏解）
- 可解释性强（仅保留重要特征）
- 适合特征数量大于样本数的高维数据（如文本、基因数据）

注意事项

- 如果特征之间高度相关，Lasso 可能在它们之间随意选择其中一个，忽略其他。
- 不建议直接在未经缩放的数据上使用 Lasso（应先进行标准化处理）。

### 1.1.4. Multi-task Lasso

Multi-task Lasso 是 Lasso 的扩展，适用于 多输出回归（multi-output regression）问题。
它假设多个输出（任务）之间共享相同的特征子集，因此可以在多个任务之间 联合进行特征选择，实现更好的泛化能力。

应用场景

- 多个相关输出变量（多任务）
- 高维稀疏特征空间
- 希望对多个任务使用相同的特征子集建模

Multi-task Lasso 最小化以下目标函数：

$$
\frac{1}{2n_{\text{samples}}} \|Y - XW\|^2_{\text{Fro}} + \alpha \|W\|_{2,1}
$$

其中：

- $Y \in \mathbb{R}^{n \times T}$：目标变量矩阵（T 个输出任务）
- $X \in \mathbb{R}^{n \times p}$：输入特征
- $W \in \mathbb{R}^{p \times T}$：模型参数矩阵
- $\|W\|_{2,1} = \sum_{i=1}^{p} \sqrt{\sum_{j=1}^{T} w_{ij}^2}$：对每一行（对应一个特征）的 ℓ2 范数求和，鼓励整个行向量为零 → 联合特征选择

优势

- 联合多个任务进行学习，提高鲁棒性和准确率
- 自动进行联合特征选择（即一个特征要么被所有任务使用，要么被全部丢弃）
- 比独立地为每个任务建模更具有解释力和泛化能力

注意事项

- 所有输出（`Y`）必须是 连续变量（回归任务），不能用于分类
- 输出必须是二维数组（即使只有一个任务）

### 1.1.5. Elastic-Net

Elastic Net 是结合了 Lasso (ℓ1 正则) 和 Ridge (ℓ2 正则) 的线性回归方法。

它同时具备：

- Lasso 的 稀疏性（特征选择）
- Ridge 的 稳定性（尤其在特征高度相关时）

适用于：

- 高维数据
- 特征数大于样本数
- 特征之间高度相关

Elastic Net 最小化以下目标函数：

$$
\frac{1}{2n_{\text{samples}}} \|Xw - y\|^2_2 + \alpha \left( \rho \|w\|_1 + \frac{1 - \rho}{2} \|w\|^2_2 \right)
$$

其中：

- $\alpha$：整体正则化强度（控制收缩程度）
- $\rho \in [0, 1]$：权衡 ℓ1 与 ℓ2 正则化的比例（`l1_ratio` 参数）

  - $\rho = 1$：等价于 Lasso
  - $\rho = 0$：等价于 Ridge

优势

- 能处理高维、特征相关性强的情形
- 稀疏（但比 Lasso 稀疏程度稍弱）
- 稳定性好于 Lasso（在共线性特征中不会随机丢弃）

注意事项

- 推荐对特征进行标准化（如使用 `StandardScaler`）
- Lasso 可能在多个强相关特征中任意选择一个，而 Elastic Net 会平均分配权重
- 当样本数远小于特征数时，Elastic Net 优于 Lasso 和 Ridge

### 1.1.6. Multi-task Elastic-Net

Multi-task Elastic Net 是 Elastic Net 的扩展，用于 多输出回归任务（multi-output regression），即一次性预测多个相关目标变量。
它结合了：

- Lasso（ℓ1） 的特征选择能力
- Ridge（ℓ2） 的鲁棒性
- 并通过 行级稀疏性（ℓ2,1 范数） 实现 跨多个任务的一致性特征选择

Multi-task Elastic Net 最小化以下目标函数：

$$
\frac{1}{2n_{\text{samples}}} \|Y - XW\|^2_{\text{Fro}} + \alpha \left( \rho \|W\|_{2,1} + \frac{1 - \rho}{2} \|W\|^2_{\text{Fro}} \right)
$$

其中：

- $Y \in \mathbb{R}^{n \times T}$：目标矩阵（T 个输出任务）
- $X \in \mathbb{R}^{n \times p}$：特征矩阵
- $W \in \mathbb{R}^{p \times T}$：系数矩阵
- $\|W\|_{2,1} = \sum_{i=1}^p \sqrt{\sum_{j=1}^T w_{ij}^2}$：对系数矩阵每一行的 ℓ2 范数求和
- $\|W\|^2_{\text{Fro}}$：Frobenius 范数，相当于所有元素平方和

特点

- 在所有任务上共同选择一组特征
- 更稳定的特征选择效果（尤其在任务相关性强时）
- 相比单任务模型更能提升泛化能力

适用于：

- 多任务、多输出回归问题
- 高维稀疏特征空间
- 任务之间存在相关性，希望共用一组特征

注意事项

- 所有输出必须为 连续值，不能用于分类
- 特征建议标准化（如使用 `StandardScaler`）
- 输出必须为二维数组，即使只有一个任务

### 1.1.7. Least Angle Regression

Least Angle Regression（LARS） 是一种高效的回归算法，尤其适合用于变量选择（feature selection）问题，尤其在 特征数远大于样本数 的情况下（即 `n_features >> n_samples`）。

它是一种迭代的建模方法，特征是逐步纳入模型的方式类似于前向逐步回归（Forward Stepwise Regression），但更新方式更为“温和”和计算高效。

LARS 的主要思想

- 初始模型为空。
- 每次找到与当前残差最相关（最“角度小”）的特征。
- 沿着该方向前进，直到另一个特征与残差达到相同的相关度。
- 然后向这两个方向同时前进。
- 重复，直到所有特征都被纳入模型或满足某种停止条件。

这种“最小角度”策略让 LARS 非常适合高维问题，且生成的系数路径可以用作变量选择的依据。

应用场景

- 变量选择（feature selection）
- 高维小样本问题（如基因表达数据分析）
- 解释性分析：路径跟踪方便查看变量是何时进入模型的

注意事项

- 对于 LARS 及其变种，输入特征建议中心化。
- 不建议用于极大规模数据集（在某些场景下可能内存消耗较大）。
- 对于 Lasso 路径，建议使用 `LassoLars` 而不是普通的 `Lasso`（在样本数少的情况下）。

### 1.1.8. LARS Lasso

LARS Lasso 是一种高效算法，用于计算 Lasso 回归 的 完整解路径（regularization path）。它基于 Least Angle Regression（LARS）算法。

背景

- Lasso（Least Absolute Shrinkage and Selection Operator）是一种带 L1 正则化的线性回归方法，可以同时实现 变量选择 + 正则化。
- 传统求解 Lasso 使用的是 坐标下降法（Coordinate Descent），但当特征维度远大于样本数时，这种方法效率低。
- LARS-Lasso 利用 LARS 算法，可以 一步一步构造整个 Lasso 路径，非常适合小样本高维特征数据。

LARS Lasso 的思想

- 从零系数开始，逐渐增加非零系数的数量。
- 每一步选择与残差最相关的特征，并向这个方向移动。
- 当某个变量的系数趋近于 0 时，它会被“踢出”模型。
- LARS-Lasso 的路径与 LARS 类似，但因为 L1 正则化的存在，路径会出现“折返”（非单调增加）。

应用场景

- 基因数据分析：n_features ≫ n_samples
- 特征选择（feature selection）
- 稀疏建模
- 正则路径可视化

注意事项

- 不适合 n_samples ≫ n_features 的数据集（可能比坐标下降更慢）。
- 特征必须标准化（推荐使用 `StandardScaler`）。
- 对噪声较多的数据不够鲁棒，可能会导致不稳定的变量选择。

### 1.1.9. Orthogonal Matching Pursuit (OMP)

Orthogonal Matching Pursuit (OMP) 是一种贪心算法，用于在高维稀疏信号恢复中进行 稀疏线性回归。

OMP 通过逐步选择与当前残差最相关的特征（字典元素），迭代地更新模型，直到达到预设的稀疏度或误差阈值。

OMP 的基本思想

- 目标是找到一个系数向量，使得预测值尽可能拟合目标，同时系数保持稀疏（即大部分系数为 0）。
- 每次迭代：

  1. 找到与当前残差最相关的特征（字典列）。
  2. 将该特征加入当前活跃集。
  3. 在活跃集上做最小二乘拟合，更新系数。
  4. 计算新的残差。

- 重复直到满足停止条件（达到最大非零系数个数或残差足够小）。

数学形式

给定输入矩阵 $X$ 和响应向量 $y$，求解：

$$
\min_{\beta} \|y - X\beta\|_2^2 \quad \text{subject to} \quad \|\beta\|_0 \leq k
$$

其中 $\|\beta\|_0$ 是非零系数的个数，限制稀疏度。

应用场景

- 稀疏信号恢复，如压缩感知（Compressed Sensing）
- 高维稀疏回归
- 特征选择
- 字典学习

优缺点对比

| 优点                   | 缺点                     |
| ---------------------- | ------------------------ |
| 计算速度快，简单易实现 | 可能对噪声敏感           |
| 模型系数稀疏，便于解释 | 贪心算法，不保证全局最优 |
| 逐步选择变量，易于理解 | 迭代次数需合理设置       |

### 1.1.10. Bayesian Regression

贝叶斯回归是一种基于贝叶斯统计理论的线性回归方法，它通过引入先验概率对模型参数进行正则化，从而解决过拟合问题并给出参数的不确定性估计。

主要思想

- 假设模型参数服从某种先验分布（一般为高斯分布）。
- 观测数据根据模型和噪声产生，噪声也服从高斯分布。
- 利用贝叶斯定理结合先验与似然，得到参数的后验分布。
- 模型不仅预测目标值，还能提供预测的置信区间。

数学模型

- 线性模型：

  $$
  y = Xw + \epsilon, \quad \epsilon \sim \mathscr{N}(0, \sigma^2 I)
  $$

- 参数先验：

  $$
  p(w | \alpha) = \mathscr{N}(0, \alpha^{-1} I)
  $$

- 超参数 $\alpha$ 控制先验的强度，类似正则化项。
- 通过观测数据估计参数后验：

  $$
  p(w | X, y, \alpha, \sigma^2)
  $$

优缺点及应用场景

| 优点                                 | 缺点                             |
| ------------------------------------ | -------------------------------- |
| 自动调节正则化强度，减少调参麻烦     | 计算复杂度较高，适合中小型数据集 |
| 给出预测的置信区间，提供不确定性估计 | 对异常值敏感                     |
| ARD 可以自动进行特征选择             | 对非线性关系建模有限             |
| 能够处理共线性问题                   |                                  |

应用场景

- 需要不确定性估计的回归问题
- 高维小样本问题
- 自动特征选择
- 需要贝叶斯解释的机器学习任务

### 1.1.11. Logistic regression

逻辑回归是一种经典的广义线性模型，主要用于二分类及多分类问题。它通过对线性组合输入变量进行 sigmoid（或 softmax）变换，将输出映射为概率值。

主要思想

- 预测目标 $y \in \{0,1\}$（或多类别）对应某一类别的概率。
- 通过逻辑函数（logistic function）拟合类别概率：

  $$
  P(y=1|X) = \frac{1}{1 + e^{-Xw}}
  $$

- 目标是最大化似然函数，或等价地最小化对数损失（log loss）。

数学模型

- 二分类：

  $$
  p(y=1|x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}
  $$

- 多分类（多项式逻辑回归）：

  $$
  p(y=k|x) = \frac{e^{w_k^T x + b_k}}{\sum_{j} e^{w_j^T x + b_j}}
  $$

应用场景

- 二分类任务（如垃圾邮件分类、疾病预测）
- 多分类任务（如数字识别、文本分类）
- 解释性强的模型需求
- 线性可分或近似线性可分的数据集

注意事项

- 特征最好做标准化（如 `StandardScaler`），尤其使用 `lbfgs`、`saga` 等 solver。
- 对于高度不平衡数据，可使用 `class_weight='balanced'`。
- `l1` 正则化有助于特征选择，但需要特定 solver。
- 多分类时，默认采用 One-vs-Rest 策略，若需更精确可选 `multinomial`。

### 1.1.12. Generalized Linear Models

广义线性模型（GLM）是线性模型的推广，能够处理非正态分布的响应变量（如二项分布、泊松分布等），通过链接函数（link function）将线性预测值映射到响应变量的期望值。

主要思想

- 响应变量 $y$ 的条件分布属于指数族（Exponential family），如正态分布、二项分布、泊松分布等。
- 预测变量 $X$ 的线性组合 $\eta = Xw$ 通过链接函数 $g(\cdot)$ 与响应变量的期望相关联：

  $$
  g(\mathbb{E}[y|X]) = \eta = Xw
  $$

- 链接函数根据数据类型选择：

  - 线性回归：恒等函数 $g(\mu) = \mu$
  - 逻辑回归：logit 函数 $g(\mu) = \log \frac{\mu}{1-\mu}$
  - 泊松回归：log 函数 $g(\mu) = \log \mu$

应用场景

- 计数数据建模（如事故次数、事件发生频率）
- 保险精算、医疗事件分析
- 非负响应变量的回归问题
- 广义线性回归，涉及非正态分布

注意事项

- 目标变量必须符合对应的分布假设，如计数数据适合泊松回归。
- 特征通常需要进行适当预处理。
- TweedieRegressor 的 `power` 参数决定了分布类型，需要根据任务合理设置。
- 训练过程中使用的优化算法是坐标下降或其他凸优化方法。

### 1.1.13. Stochastic Gradient Descent

随机梯度下降（SGD）是一种高效的优化算法，适用于大规模和高维度的线性模型训练。它通过对训练样本逐个（或小批量）计算梯度，迭代更新模型参数。

SGD 可用于多种线性模型的训练，包括线性回归、逻辑回归、支持向量机（SVM）等。

主要思想

- 相比传统批量梯度下降（计算所有样本梯度），SGD 每次只用一个样本更新参数，计算更快。
- 由于引入随机性，收敛路径不稳定，但通常能更快逼近最优。
- 支持多种损失函数和正则化方式，灵活构建不同线性模型。

优缺点及应用场景

| 优点                       | 缺点                           |
| -------------------------- | ------------------------------ |
| 适合大规模数据，计算效率高 | 需要调节学习率和正则化等超参数 |
| 支持多种损失函数，灵活性强 | 收敛不稳定，可能震荡           |
| 可以在线学习（增量训练）   | 依赖初始学习率和迭代次数       |
| 支持稀疏数据和稀疏模型     | 对噪声较敏感                   |

注意事项

- 需对输入特征做归一化或标准化处理，否则影响收敛速度。
- 学习率调节策略（`learning_rate`）对训练效果影响大，推荐从 `'optimal'` 或 `'adaptive'` 试起。
- 对于大规模和稀疏数据非常适合。
- 可以结合早停机制防止过拟合。

### 1.1.14. Perceptron

### 1.1.15. Passive Aggressive Algorithms

### 1.1.16. Robustness regression: outliers and modeling errors

### 1.1.17. Quantile Regression

### 1.1.18. Polynomial regression: extending linear models with basis functions

## 1.2. Linear and Quadratic Discriminant Analysis

### 1.2.1. Dimensionality reduction using Linear Discriminant Analysis

### 1.2.2. Mathematical formulation of the LDA and QDA classifiers

### 1.2.3. Mathematical formulation of LDA dimensionality reduction

### 1.2.4. Shrinkage and Covariance Estimator

### 1.2.5. Estimation algorithms

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

# 2\. Unsupervised learning

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

# 3\. Model selection and evaluation

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

# 4\. Metadata Routing

## 4.1. Usage Examples

### 4.1.1. Weighted scoring and fitting

### 4.1.2. Weighted scoring and unweighted fitting

### 4.1.3. Unweighted feature selection

### 4.1.4. Different scoring and fitting weights

## 4.2. API Interface

## 4.3. Metadata Routing Support Status

# 5\. Inspection

## 5.1. Partial Dependence and Individual Conditional Expectation plots

### 5.1.1. Partial dependence plots

### 5.1.2. Individual conditional expectation (ICE) plot

### 5.1.3. Mathematical Definition

### 5.1.4. Computation methods

## 5.2. Permutation feature importance

### 5.2.1. Outline of the permutation importance algorithm

### 5.2.2. Relation to impurity-based importance in trees

### 5.2.3. Misleading values on strongly correlated features

# 6\. Visualizations

## 6.1. Available Plotting Utilities

### 6.1.1. Display Objects

# 7\. Dataset transformations

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

# 8\. Dataset loading utilities

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

# 9\. Computing with scikit-learn

## 9.1. Strategies to scale computationally: bigger data

### 9.1.1. Scaling with instances using out-of-core learning

## 9.2. Computational Performance

### 9.2.1. Prediction Latency

### 9.2.2. Prediction Throughput

### 9.2.3. Tips and Tricks

## 9.3. Parallelism, resource management, and configuration

### 9.3.1. Parallelism

### 9.3.2. Configuration switches

# 10\. Model persistence

## 10.1. Workflow Overview

### 10.1.1. Train and Persist the Model

## 10.2. ONNX

## 10.3. `skops.io`

## 10.4. `pickle`, `joblib`, and `cloudpickle`

## 10.5. Security & Maintainability Limitations

### 10.5.1. Replicating the training environment in production

### 10.5.2. Serving the model artifact

## 10.6. Summarizing the key points

# 11\. Common pitfalls and recommended practices

## 11.1. Inconsistent preprocessing

## 11.2. Data leakage

### 11.2.1. How to avoid data leakage

### 11.2.2. Data leakage during pre-processing

## 11.3. Controlling randomness

### 11.3.1. Using `None` or `RandomState` instances, and repeated calls to `fit` and `split`

### 11.3.2. Common pitfalls and subtleties

### 11.3.3. General recommendations

# 12\. Dispatching

## 12.1. Array API support (experimental)

### 12.1.1. Example usage

### 12.1.2. Support for `Array API`\-compatible inputs

### 12.1.3. Input and output array type handling

### 12.1.4. Common estimator checks

# 13\. Choosing the right estimator

# 14\. External Resources, Videos and Talks

## 14.1. The scikit-learn MOOC

## 14.2. Videos

## 14.3. New to Scientific Python?

## 14.4. External Tutorials
