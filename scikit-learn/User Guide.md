<link rel="stylesheet" href="../style.css">

# 1. 监督学习

## 1.1. 线性模型

线性模型模块提供了一系列算法，其预测形式为：

$$
\hat{y}(w, x) = w_0 + \sum_{j=1}^{p} w_j x_j
$$

对应的模型属性包括系数 `coef_`（对应 $w$）和截距 `intercept_`（对应 $w_0$）。

### 1.1.1. 普通最小二乘法

https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares

最小化残差平方和（Residual Sum of Squares, RSS），即：

$$
\min_{w} \|X w - y\|_2^2
$$

几个常见的注意点：

1. 特征共线性（Multicollinearity）
   - 如果 `X` 中某些特征高度相关，$X^\top X$ 会接近奇异矩阵，导致系数估计不稳定（方差大、符号乱变）。
   - 解决方法：
     - 删除高度相关的特征
     - 使用 Ridge（岭回归）替代 OLS
     - 做 PCA 降维
2. 输入特征的缩放
   - OLS 本身对缩放不敏感（因为系数会自动调整），但如果后续要比较系数的大小来判断特征重要性，就需要统一量纲。
   - 如果只关心预测而不是解释，可以不缩放。
3. 异常值（Outliers）影响大
   - OLS 使用平方误差，异常值会被过度放大，导致模型严重偏斜。
   - 解决方法：
     - 清理数据中的异常值
     - 用 HuberRegressor、RANSACRegressor 等鲁棒模型替代
4. 样本量与特征数关系
   - 当特征数 $p$ 接近或超过样本数 $n$ 时：
     - OLS 可能完全拟合训练集（过拟合）
     - 系数估计极不稳定
   - 解决方法：
     - 增加样本
     - 用正则化（Ridge / Lasso）
     - 先做特征选择
5. 截距与特征中心化
   - `LinearRegression` 默认会拟合截距 `fit_intercept=True`，如果你自己已经在数据中加了常数列，需要 `fit_intercept=False`。
   - 在数值计算时，先对特征做中心化（减去均值）能减少数值误差。
6. 假设条件与解释性
   - 线性关系成立
   - 误差项独立同分布
   - 方差齐性（homoscedasticity）
   - 无多重共线性, 如果假设严重不满足，系数解释可能无意义，预测性能也会下降。

### 1.1.2. 岭回归和分类

https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification

模型训练的目标变成了最小化以下损失：

$$
\min_{w} \|Xw - y\|_2^2 + \alpha \|w\|_2^2
$$

当 $\alpha = 0$ 时，Ridge 恢复为普通的 OLS（Ordinary Least Squares）。但不建议使用 `Ridge(alpha=0)`，而应使用 `LinearRegression`。

注意事项：Ridge 和 OLS 相比要多考虑几个正则化相关的因素。

1. 正则化系数 α 的选择
   - 影响
     - α 太小 → 接近普通最小二乘，可能过拟合，系数波动大。
     - α 太大 → 系数过于收缩，模型欠拟合，预测精度下降。
   - 做法
     - 用交叉验证自动选择：`RidgeCV`、`RidgeClassifierCV`
     - 在不同数量级（比如 0.1、1、10、100）测试 α，找出最优值
2. 特征缩放（Scaling）是必须的
   - Ridge 惩罚项 $\alpha \|w\|_2^2$ 会受到特征量纲影响。
   - 特征数值差异大时，大尺度特征的系数会被过度收缩，小尺度特征的系数可能几乎不受惩罚。
   - 建议：在 fit 之前做 `StandardScaler` 或 `MinMaxScaler`。
3. 多重共线性问题
   - Ridge 比 OLS 更能处理共线性，因为 L2 正则化让矩阵 $X^T X + \alpha I$ 总是可逆。
   - 但如果特征数量很多且冗余，仍可能需要特征选择或降维（PCA）来提高可解释性。
4. 系数的解释性降低
   - 由于 L2 正则化，系数被缩小，大小关系依然可比较，但数值不再直接反映“特征变化 1 单位对预测值的影响”。
   - 如果你的目标是特征选择，可以考虑 Lasso 或 Elastic Net，因为 Ridge 不会让系数变成零。
5. 截距处理
   - 默认 `fit_intercept=True`，并且在缩放时通常会去掉均值（center）。
   - 如果数据已经中心化且包含截距列，需要设置 `fit_intercept=False`，否则会重复引入截距。
6. 分类场景注意事项
   - RidgeClassifier 的决策边界和 LogisticRegression（L2 正则）不完全相同：
     - RidgeClassifier 是回归 + 取符号，适合高维稠密数据
     - LogisticRegression 输出的是概率，更适合需要概率预测的任务
   - 在类别不均衡时，RidgeClassifier 没有 `class_weight='balanced'` 的自动权重，需要你手动调整 `sample_weight`。
7. 多输出任务
   - `Ridge` 原生支持多输出回归（y 为二维数组时），会对每个输出独立拟合。
   - 如果多个输出相关性强，可以考虑多任务模型（MultiTaskElasticNet）。

### 1.1.3. 套索

https://scikit-learn.org/stable/modules/linear_model.html#lasso



### [1.1.4. 多任务套索](https://scikit-learn.org/stable/modules/linear_model.html#multi-task-lasso)

### [1.1.5. 弹性网络](https://scikit-learn.org/stable/modules/linear_model.html#elastic-net)

### [1.1.6. 多任务弹性网络](https://scikit-learn.org/stable/modules/linear_model.html#multi-task-elastic-net)

### [1.1.7. 最小角回归](https://scikit-learn.org/stable/modules/linear_model.html#least-angle-regression)

### [1.1.8. LARS 套索](https://scikit-learn.org/stable/modules/linear_model.html#lars-lasso)

### [1.1.9. 正交匹配追踪（OMP）](https://scikit-learn.org/stable/modules/linear_model.html#orthogonal-matching-pursuit-omp)

### [1.1.10. 贝叶斯回归](https://scikit-learn.org/stable/modules/linear_model.html#bayesian-regression)

### [1.1.11. 逻辑回归](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)

### [1.1.12. 广义线性模型](https://scikit-learn.org/stable/modules/linear_model.html#generalized-linear-models)

### [1.1.13. 随机梯度下降 - SGD](https://scikit-learn.org/stable/modules/linear_model.html#stochastic-gradient-descent-sgd)

### [1.1.14. 感知器](https://scikit-learn.org/stable/modules/linear_model.html#perceptron)

### [1.1.15. 被动攻击算法](https://scikit-learn.org/stable/modules/linear_model.html#passive-aggressive-algorithms)

### [1.1.16. 稳健性回归：异常值和建模误差](https://scikit-learn.org/stable/modules/linear_model.html#robustness-regression-outliers-and-modeling-errors)

### [1.1.17. 分位数回归](https://scikit-learn.org/stable/modules/linear_model.html#quantile-regression)

### [1.1.18. 多项式回归：利用基函数扩展线性模型](https://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression-extending-linear-models-with-basis-functions)

## [1.2 线性和二次判别分析](https://scikit-learn.org/stable/modules/lda_qda.html)

### [1.2.1. 使用线性判别分析进行降维](https://scikit-learn.org/stable/modules/lda_qda.html#dimensionality-reduction-using-linear-discriminant-analysis)

### [1.2.2. LDA 和 QDA 分类器的数学公式](https://scikit-learn.org/stable/modules/lda_qda.html#mathematical-formulation-of-the-lda-and-qda-classifiers)

### [1.2.3. LDA 降维的数学公式](https://scikit-learn.org/stable/modules/lda_qda.html#mathematical-formulation-of-lda-dimensionality-reduction)

### [1.2.4. 收缩和协方差估计器](https://scikit-learn.org/stable/modules/lda_qda.html#shrinkage-and-covariance-estimator)

### [1.2.5. 估计算法](https://scikit-learn.org/stable/modules/lda_qda.html#estimation-algorithms)

## [1.3. 核岭回归](https://scikit-learn.org/stable/modules/kernel_ridge.html)

## [1.4. 支持向量机](https://scikit-learn.org/stable/modules/svm.html)

### [1.4.1. 分类](https://scikit-learn.org/stable/modules/svm.html#classification)

### [1.4.2. 回归](https://scikit-learn.org/stable/modules/svm.html#regression)

### [1.4.3. 密度估计，新颖性检测](https://scikit-learn.org/stable/modules/svm.html#density-estimation-novelty-detection)

### [1.4.4. 复杂性](https://scikit-learn.org/stable/modules/svm.html#complexity)

### [1.4.5. 实际使用技巧](https://scikit-learn.org/stable/modules/svm.html#tips-on-practical-use)

### [1.4.6. 核函数](https://scikit-learn.org/stable/modules/svm.html#kernel-functions)

### [1.4.7. 数学公式](https://scikit-learn.org/stable/modules/svm.html#mathematical-formulation)

### [1.4.8. 实施细节](https://scikit-learn.org/stable/modules/svm.html#implementation-details)

## [1.5. 随机梯度下降](https://scikit-learn.org/stable/modules/sgd.html)

### [1.5.1. 分类](https://scikit-learn.org/stable/modules/sgd.html#classification)

### [1.5.2. 回归](https://scikit-learn.org/stable/modules/sgd.html#regression)

### [1.5.3. 在线单类 SVM](https://scikit-learn.org/stable/modules/sgd.html#online-one-class-svm)

### [1.5.4. 稀疏数据的随机梯度下降](https://scikit-learn.org/stable/modules/sgd.html#stochastic-gradient-descent-for-sparse-data)

### [1.5.5. 复杂性](https://scikit-learn.org/stable/modules/sgd.html#complexity)

### [1.5.6. 停止标准](https://scikit-learn.org/stable/modules/sgd.html#stopping-criterion)

### [1.5.7. 实际使用技巧](https://scikit-learn.org/stable/modules/sgd.html#tips-on-practical-use)

### [1.5.8. 数学公式](https://scikit-learn.org/stable/modules/sgd.html#mathematical-formulation)

### [1.5.9. 实施细节](https://scikit-learn.org/stable/modules/sgd.html#implementation-details)

## [1.6. 最近邻](https://scikit-learn.org/stable/modules/neighbors.html)

### [1.6.1. 无监督最近邻](https://scikit-learn.org/stable/modules/neighbors.html#unsupervised-nearest-neighbors)

### [1.6.2. 最近邻分类](https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-classification)

### [1.6.3. 最近邻回归](https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-regression)

### [1.6.4. 最近邻算法](https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbor-algorithms)

### [1.6.5. 最近质心分类器](https://scikit-learn.org/stable/modules/neighbors.html#nearest-centroid-classifier)

### [1.6.6. 最近邻变换器](https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-transformer)

### [1.6.7. 邻域成分分析](https://scikit-learn.org/stable/modules/neighbors.html#neighborhood-components-analysis)

## [1.7. 高斯过程](https://scikit-learn.org/stable/modules/gaussian_process.html)

### [1.7.1. 高斯过程回归（GPR）](https://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process-regression-gpr)

### [1.7.2. 高斯过程分类（GPC）](https://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process-classification-gpc)

### [1.7.3. GPC 示例](https://scikit-learn.org/stable/modules/gaussian_process.html#gpc-examples)

### [1.7.4 高斯过程的核](https://scikit-learn.org/stable/modules/gaussian_process.html#kernels-for-gaussian-processes)

## [1.8. 交叉分解](https://scikit-learn.org/stable/modules/cross_decomposition.html)

### [1.8.1. PLSCanaonical](https://scikit-learn.org/stable/modules/cross_decomposition.html#plscanonical)

### [1.8.2. PLSSVD](https://scikit-learn.org/stable/modules/cross_decomposition.html#plssvd)

### [1.8.3. 偏最小二乘回归](https://scikit-learn.org/stable/modules/cross_decomposition.html#plsregression)

### [1.8.4. 典型相关分析](https://scikit-learn.org/stable/modules/cross_decomposition.html#canonical-correlation-analysis)

## [1.9. 朴素贝叶斯](https://scikit-learn.org/stable/modules/naive_bayes.html)

### [1.9.1. 高斯朴素贝叶斯](https://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes)

### [1.9.2. 多项式朴素贝叶斯](https://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes)

### [1.9.3. 补充朴素贝叶斯](https://scikit-learn.org/stable/modules/naive_bayes.html#complement-naive-bayes)

### [1.9.4. 伯努利朴素贝叶斯](https://scikit-learn.org/stable/modules/naive_bayes.html#bernoulli-naive-bayes)

### [1.9.5. 分类朴素贝叶斯](https://scikit-learn.org/stable/modules/naive_bayes.html#categorical-naive-bayes)

### [1.9.6. 核外朴素贝叶斯模型拟合](https://scikit-learn.org/stable/modules/naive_bayes.html#out-of-core-naive-bayes-model-fitting)

## [1.10. 决策树](https://scikit-learn.org/stable/modules/tree.html)

### [1.10.1. 分类](https://scikit-learn.org/stable/modules/tree.html#classification)

### [1.10.2. 回归](https://scikit-learn.org/stable/modules/tree.html#regression)

### [1.10.3. 多输出问题](https://scikit-learn.org/stable/modules/tree.html#multi-output-problems)

### [1.10.4. 复杂性](https://scikit-learn.org/stable/modules/tree.html#complexity)

### [1.10.5. 实际使用技巧](https://scikit-learn.org/stable/modules/tree.html#tips-on-practical-use)

### [1.10.6. 树算法：ID3、C4.5、C5.0 和 CART](https://scikit-learn.org/stable/modules/tree.html#tree-algorithms-id3-c4-5-c5-0-and-cart)

### [1.10.7. 数学公式](https://scikit-learn.org/stable/modules/tree.html#mathematical-formulation)

### [1.10.8. 缺失值支持](https://scikit-learn.org/stable/modules/tree.html#missing-values-support)

### [1.10.9. 最小成本复杂度修剪](https://scikit-learn.org/stable/modules/tree.html#minimal-cost-complexity-pruning)

## [1.11. 集成：梯度提升、随机森林、bagging、投票、stacking](https://scikit-learn.org/stable/modules/ensemble.html)

### [1.11.1. 梯度提升树](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosted-trees)

### [1.11.2. 随机森林和其他随机树集合](https://scikit-learn.org/stable/modules/ensemble.html#random-forests-and-other-randomized-tree-ensembles)

### [1.11.3. Bagging 元估计器](https://scikit-learn.org/stable/modules/ensemble.html#bagging-meta-estimator)

### [1.11.4. 投票分类器](https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier)

### [1.11.5. 投票回归器](https://scikit-learn.org/stable/modules/ensemble.html#voting-regressor)

### [1.11.6. 堆叠泛化](https://scikit-learn.org/stable/modules/ensemble.html#stacked-generalization)

### [1.11.7. AdaBoost](https://scikit-learn.org/stable/modules/ensemble.html#adaboost)

## [1.12. 多类和多输出算法](https://scikit-learn.org/stable/modules/multiclass.html)

### [1.12.1. 多类分类](https://scikit-learn.org/stable/modules/multiclass.html#multiclass-classification)

### [1.12.2. 多标签分类](https://scikit-learn.org/stable/modules/multiclass.html#multilabel-classification)

### [1.12.3. 多类多输出分类](https://scikit-learn.org/stable/modules/multiclass.html#multiclass-multioutput-classification)

### [1.12.4. 多输出回归](https://scikit-learn.org/stable/modules/multiclass.html#multioutput-regression)

## [1.13. 特征选择](https://scikit-learn.org/stable/modules/feature_selection.html)

### [1.13.1 删除方差较小的特征](https://scikit-learn.org/stable/modules/feature_selection.html#removing-features-with-low-variance)

### [1.13.2. 单变量特征选择](https://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection)

### [1.13.3. 递归特征消除](https://scikit-learn.org/stable/modules/feature_selection.html#recursive-feature-elimination)

### [1.13.4. 使用 SelectFromModel 进行特征选择](https://scikit-learn.org/stable/modules/feature_selection.html#feature-selection-using-selectfrommodel)

### [1.13.5. 顺序特征选择](https://scikit-learn.org/stable/modules/feature_selection.html#sequential-feature-selection)

### [1.13.6. 特征选择作为流程的一部分](https://scikit-learn.org/stable/modules/feature_selection.html#feature-selection-as-part-of-a-pipeline)

## [1.14. 半监督学习](https://scikit-learn.org/stable/modules/semi_supervised.html)

### [1.14.1. 自我训练](https://scikit-learn.org/stable/modules/semi_supervised.html#self-training)

### [1.14.2. 标签传播](https://scikit-learn.org/stable/modules/semi_supervised.html#label-propagation)

## [1.15. 等渗回归](https://scikit-learn.org/stable/modules/isotonic.html)

## [1.16. 概率校准](https://scikit-learn.org/stable/modules/calibration.html)

### [1.16.1. 校准曲线](https://scikit-learn.org/stable/modules/calibration.html#calibration-curves)

### [1.16.2. 校准分类器](https://scikit-learn.org/stable/modules/calibration.html#calibrating-a-classifier)

### [1.16.3. 使用](https://scikit-learn.org/stable/modules/calibration.html#usage)

## [1.17. 神经网络模型（监督）](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)

### [1.17.1. 多层感知器](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#multi-layer-perceptron)

### [1.17.2. 分类](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification)

### [1.17.3. 回归](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#regression)

### [1.17.4. 正则化](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#regularization)

### [1.17.5. 算法](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#algorithms)

### [1.17.6. 复杂性](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#complexity)

### [1.17.7. 实际使用技巧](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#tips-on-practical-use)

### [1.17.8. 使用 warm_start 进行更多控制](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#more-control-with-warm-start)

# [2.无监督学习](https://scikit-learn.org/stable/unsupervised_learning.html)

## [2.1. 高斯混合模型](https://scikit-learn.org/stable/modules/mixture.html)

### [2.1.1. 高斯混合](https://scikit-learn.org/stable/modules/mixture.html#gaussian-mixture)

### [2.1.2. 变分贝叶斯高斯混合](https://scikit-learn.org/stable/modules/mixture.html#variational-bayesian-gaussian-mixture)

## [2.2. 流形学习](https://scikit-learn.org/stable/modules/manifold.html)

### [2.2.1. 简介](https://scikit-learn.org/stable/modules/manifold.html#introduction)

### [2.2.2. 等值线图](https://scikit-learn.org/stable/modules/manifold.html#isomap)

### [2.2.3. 局部线性嵌入](https://scikit-learn.org/stable/modules/manifold.html#locally-linear-embedding)

### [2.2.4 改进的局部线性嵌入](https://scikit-learn.org/stable/modules/manifold.html#modified-locally-linear-embedding)

### [2.2.5. Hessian 特征映射](https://scikit-learn.org/stable/modules/manifold.html#hessian-eigenmapping)

### [2.2.6. 谱嵌入](https://scikit-learn.org/stable/modules/manifold.html#spectral-embedding)

### [2.2.7. 局部切线空间对齐](https://scikit-learn.org/stable/modules/manifold.html#local-tangent-space-alignment)

### [2.2.8. 多维尺度分析（MDS）](https://scikit-learn.org/stable/modules/manifold.html#multi-dimensional-scaling-mds)

### [2.2.9. t 分布随机邻域嵌入（t-SNE）](https://scikit-learn.org/stable/modules/manifold.html#t-distributed-stochastic-neighbor-embedding-t-sne)

### [2.2.10. 实际使用技巧](https://scikit-learn.org/stable/modules/manifold.html#tips-on-practical-use)

## [2.3. 聚类](https://scikit-learn.org/stable/modules/clustering.html)

### [2.3.1 聚类方法概述](https://scikit-learn.org/stable/modules/clustering.html#overview-of-clustering-methods)

### [2.3.2. K 均值](https://scikit-learn.org/stable/modules/clustering.html#k-means)

### [2.3.3. 亲和传播](https://scikit-learn.org/stable/modules/clustering.html#affinity-propagation)

### [2.3.4. 均值漂移](https://scikit-learn.org/stable/modules/clustering.html#mean-shift)

### [2.3.5. 谱聚类](https://scikit-learn.org/stable/modules/clustering.html#spectral-clustering)

### [2.3.6. 层次聚类](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering)

### [2.3.7. DBSCAN](https://scikit-learn.org/stable/modules/clustering.html#dbscan)

### [2.3.8. HDBSCAN](https://scikit-learn.org/stable/modules/clustering.html#hdbscan)

### [2.3.9. 光学](https://scikit-learn.org/stable/modules/clustering.html#optics)

### [2.3.10. 桦木](https://scikit-learn.org/stable/modules/clustering.html#birch)

### [2.3.11. 聚类性能评估](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)

## [2.4. 双聚类](https://scikit-learn.org/stable/modules/biclustering.html)

### [2.4.1. 谱协同聚类](https://scikit-learn.org/stable/modules/biclustering.html#spectral-co-clustering)

### [2.4.2. 谱双聚类](https://scikit-learn.org/stable/modules/biclustering.html#spectral-biclustering)

### [2.4.3. 双聚类评估](https://scikit-learn.org/stable/modules/biclustering.html#biclustering-evaluation)

## [2.5. 信号分解（矩阵分解问题）](https://scikit-learn.org/stable/modules/decomposition.html)

### [2.5.1. 主成分分析（PCA）](https://scikit-learn.org/stable/modules/decomposition.html#principal-component-analysis-pca)

### [2.5.2. 核主成分分析（kPCA）](https://scikit-learn.org/stable/modules/decomposition.html#kernel-principal-component-analysis-kpca)

### [2.5.3. 截断奇异值分解与潜在语义分析](https://scikit-learn.org/stable/modules/decomposition.html#truncated-singular-value-decomposition-and-latent-semantic-analysis)

### [2.5.4. 词典学习](https://scikit-learn.org/stable/modules/decomposition.html#dictionary-learning)

### [2.5.5. 因子分析](https://scikit-learn.org/stable/modules/decomposition.html#factor-analysis)

### [2.5.6. 独立成分分析（ICA）](https://scikit-learn.org/stable/modules/decomposition.html#independent-component-analysis-ica)

### [2.5.7. 非负矩阵分解（NMF 或 NNMF）](https://scikit-learn.org/stable/modules/decomposition.html#non-negative-matrix-factorization-nmf-or-nnmf)

### [2.5.8. 潜在狄利克雷分配（LDA）](https://scikit-learn.org/stable/modules/decomposition.html#latent-dirichlet-allocation-lda)

## [2.6. 协方差估计](https://scikit-learn.org/stable/modules/covariance.html)

### [2.6.1. 经验协方差](https://scikit-learn.org/stable/modules/covariance.html#empirical-covariance)

### [2.6.2. 收缩协方差](https://scikit-learn.org/stable/modules/covariance.html#shrunk-covariance)

### [2.6.3. 稀疏逆协方差](https://scikit-learn.org/stable/modules/covariance.html#sparse-inverse-covariance)

### [2.6.4. 稳健协方差估计](https://scikit-learn.org/stable/modules/covariance.html#robust-covariance-estimation)

## [2.7. 新颖性和异常值检测](https://scikit-learn.org/stable/modules/outlier_detection.html)

### [2.7.1. 异常值检测方法概述](https://scikit-learn.org/stable/modules/outlier_detection.html#overview-of-outlier-detection-methods)

### [2.7.2. 新颖性检测](https://scikit-learn.org/stable/modules/outlier_detection.html#novelty-detection)

### [2.7.3. 异常值检测](https://scikit-learn.org/stable/modules/outlier_detection.html#id1)

### [2.7.4. 利用局部异常因子进行新颖性检测](https://scikit-learn.org/stable/modules/outlier_detection.html#novelty-detection-with-local-outlier-factor)

## [2.8. 密度估计](https://scikit-learn.org/stable/modules/density.html)

### [2.8.1. 密度估计：直方图](https://scikit-learn.org/stable/modules/density.html#density-estimation-histograms)

### [2.8.2. 核密度估计](https://scikit-learn.org/stable/modules/density.html#kernel-density-estimation)

## [2.9. 神经网络模型（无监督）](https://scikit-learn.org/stable/modules/neural_networks_unsupervised.html)

### [2.9.1. 受限玻尔兹曼机](https://scikit-learn.org/stable/modules/neural_networks_unsupervised.html#restricted-boltzmann-machines)

# [3\. 模型选择与评估](https://scikit-learn.org/stable/model_selection.html)

## [3.1. 交叉验证：评估估计器性能](https://scikit-learn.org/stable/modules/cross_validation.html)

### [3.1.1. 计算交叉验证指标](https://scikit-learn.org/stable/modules/cross_validation.html#computing-cross-validated-metrics)

### [3.1.2. 交叉验证迭代器](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators)

### [3.1.3. 关于改组](https://scikit-learn.org/stable/modules/cross_validation.html#a-note-on-shuffling)

### [3.1.4 交叉验证和模型选择](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-and-model-selection)

### [3.1.5. 排列检验分数](https://scikit-learn.org/stable/modules/cross_validation.html#permutation-test-score)

## [3.2. 调整估计器的超参数](https://scikit-learn.org/stable/modules/grid_search.html)

### [3.2.1. 穷举网格搜索](https://scikit-learn.org/stable/modules/grid_search.html#exhaustive-grid-search)

### [3.2.2. 随机参数优化](https://scikit-learn.org/stable/modules/grid_search.html#randomized-parameter-optimization)

### [3.2.3. 通过逐次减半寻找最优参数](https://scikit-learn.org/stable/modules/grid_search.html#searching-for-optimal-parameters-with-successive-halving)

### [3.2.4. 参数搜索提示](https://scikit-learn.org/stable/modules/grid_search.html#tips-for-parameter-search)

### [3.2.5. 强力参数搜索的替代方案](https://scikit-learn.org/stable/modules/grid_search.html#alternatives-to-brute-force-parameter-search)

## [3.3. 调整类别预测的决策阈值](https://scikit-learn.org/stable/modules/classification_threshold.html)

### [3.3.1. 调整决策阈值](https://scikit-learn.org/stable/modules/classification_threshold.html#post-tuning-the-decision-threshold)

## [3.4. 指标和评分：量化预测质量](https://scikit-learn.org/stable/modules/model_evaluation.html)

### [3.4.1. 我应该使用哪个评分函数？](https://scikit-learn.org/stable/modules/model_evaluation.html#which-scoring-function-should-i-use)

### [3.4.2. 评分 API 概述](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-api-overview)

### [3.4.3.`scoring`参数：定义模型评估规则](https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules)

### [3.4.4. 分类指标](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)

### [3.4.5. 多标签排名指标](https://scikit-learn.org/stable/modules/model_evaluation.html#multilabel-ranking-metrics)

### [3.4.6. 回归指标](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)

### [3.4.7. 聚类指标](https://scikit-learn.org/stable/modules/model_evaluation.html#clustering-metrics)

### [3.4.8. 虚拟估计量](https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators)

## [3.5. 验证曲线：绘制分数来评估模型](https://scikit-learn.org/stable/modules/learning_curve.html)

### [3.5.1. 验证曲线](https://scikit-learn.org/stable/modules/learning_curve.html#validation-curve)

### [3.5.2. 学习曲线](https://scikit-learn.org/stable/modules/learning_curve.html#learning-curve)

# [4\. 元数据路由](https://scikit-learn.org/stable/metadata_routing.html)

## [4.1. 使用示例](https://scikit-learn.org/stable/metadata_routing.html#usage-examples)

### [4.1.1. 加权评分和拟合](https://scikit-learn.org/stable/metadata_routing.html#weighted-scoring-and-fitting)

### [4.1.2. 加权评分和非加权拟合](https://scikit-learn.org/stable/metadata_routing.html#weighted-scoring-and-unweighted-fitting)

### [4.1.3. 无加权特征选择](https://scikit-learn.org/stable/metadata_routing.html#unweighted-feature-selection)

### [4.1.4 不同的评分和拟合权重](https://scikit-learn.org/stable/metadata_routing.html#different-scoring-and-fitting-weights)

## [4.2. API 接口](https://scikit-learn.org/stable/metadata_routing.html#api-interface)

## [4.3. 元数据路由支持状态](https://scikit-learn.org/stable/metadata_routing.html#metadata-routing-support-status)

# [5.检查](https://scikit-learn.org/stable/inspection.html)

## [5.1. 部分依赖和个体条件期望图](https://scikit-learn.org/stable/modules/partial_dependence.html)

### [5.1.1. 部分依赖图](https://scikit-learn.org/stable/modules/partial_dependence.html#partial-dependence-plots)

### [5.1.2. 个体条件期望（ICE）图](https://scikit-learn.org/stable/modules/partial_dependence.html#individual-conditional-expectation-ice-plot)

### [5.1.3. 数学定义](https://scikit-learn.org/stable/modules/partial_dependence.html#mathematical-definition)

### [5.1.4 计算方法](https://scikit-learn.org/stable/modules/partial_dependence.html#computation-methods)

## [5.2. 排列特征重要性](https://scikit-learn.org/stable/modules/permutation_importance.html)

### [5.2.1. 排列重要性算法概述](https://scikit-learn.org/stable/modules/permutation_importance.html#outline-of-the-permutation-importance-algorithm)

### [5.2.2. 与树中基于杂质的重要性的关系](https://scikit-learn.org/stable/modules/permutation_importance.html#relation-to-impurity-based-importance-in-trees)

### [5.2.3. 强相关特征的误导性值](https://scikit-learn.org/stable/modules/permutation_importance.html#misleading-values-on-strongly-correlated-features)

# [6.可视化](https://scikit-learn.org/stable/visualizations.html)

## [6.1. 可用的绘图实用程序](https://scikit-learn.org/stable/visualizations.html#available-plotting-utilities)

### [6.1.1. 显示对象](https://scikit-learn.org/stable/visualizations.html#display-objects)

# [7.数据集转换](https://scikit-learn.org/stable/data_transforms.html)

## [7.1. 管道和复合估计器](https://scikit-learn.org/stable/modules/compose.html)

### [7.1.1. 管道：链接估算器](https://scikit-learn.org/stable/modules/compose.html#pipeline-chaining-estimators)

### [7.1.2. 回归中的目标变换](https://scikit-learn.org/stable/modules/compose.html#transforming-target-in-regression)

### [7.1.3. FeatureUnion：复合特征空间](https://scikit-learn.org/stable/modules/compose.html#featureunion-composite-feature-spaces)

### [7.1.4. 用于异构数据的 ColumnTransformer](https://scikit-learn.org/stable/modules/compose.html#columntransformer-for-heterogeneous-data)

### [7.1.5 可视化复合估计量](https://scikit-learn.org/stable/modules/compose.html#visualizing-composite-estimators)

## [7.2. 特征提取](https://scikit-learn.org/stable/modules/feature_extraction.html)

### [7.2.1. 从字典加载特性](https://scikit-learn.org/stable/modules/feature_extraction.html#loading-features-from-dicts)

### [7.2.2. 特征哈希](https://scikit-learn.org/stable/modules/feature_extraction.html#feature-hashing)

### [7.2.3. 文本特征提取](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)

### [7.2.4. 图像特征提取](https://scikit-learn.org/stable/modules/feature_extraction.html#image-feature-extraction)

## [7.3. 预处理数据](https://scikit-learn.org/stable/modules/preprocessing.html)

### [7.3.1. 标准化，或均值去除和方差缩放](https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling)

### [7.3.2 非线性变换](https://scikit-learn.org/stable/modules/preprocessing.html#non-linear-transformation)

### [7.3.3. 规范化](https://scikit-learn.org/stable/modules/preprocessing.html#normalization)

### [7.3.4. 编码分类特征](https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features)

### [7.3.5. 离散化](https://scikit-learn.org/stable/modules/preprocessing.html#discretization)

### [7.3.6. 缺失值的填补](https://scikit-learn.org/stable/modules/preprocessing.html#imputation-of-missing-values)

### [7.3.7. 生成多项式特征](https://scikit-learn.org/stable/modules/preprocessing.html#generating-polynomial-features)

### [7.3.8. 自定义变压器](https://scikit-learn.org/stable/modules/preprocessing.html#custom-transformers)

## [7.4. 缺失值的填补](https://scikit-learn.org/stable/modules/impute.html)

### [7.4.1. 单变量与多变量插补](https://scikit-learn.org/stable/modules/impute.html#univariate-vs-multivariate-imputation)

### [7.4.2. 单变量特征插补](https://scikit-learn.org/stable/modules/impute.html#univariate-feature-imputation)

### [7.4.3. 多元特征插补](https://scikit-learn.org/stable/modules/impute.html#multivariate-feature-imputation)

### [7.4.4. 最近邻插补](https://scikit-learn.org/stable/modules/impute.html#nearest-neighbors-imputation)

### [7.4.5. 保持特征数量不变](https://scikit-learn.org/stable/modules/impute.html#keeping-the-number-of-features-constant)

### [7.4.6. 标记估算值](https://scikit-learn.org/stable/modules/impute.html#marking-imputed-values)

### [7.4.7. 处理 NaN 值的估算器](https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values)

## [7.5. 无监督降维](https://scikit-learn.org/stable/modules/unsupervised_reduction.html)

### [7.5.1. PCA：主成分分析](https://scikit-learn.org/stable/modules/unsupervised_reduction.html#pca-principal-component-analysis)

### [7.5.2. 随机投影](https://scikit-learn.org/stable/modules/unsupervised_reduction.html#random-projections)

### [7.5.3. 特征聚集](https://scikit-learn.org/stable/modules/unsupervised_reduction.html#feature-agglomeration)

## [7.6. 随机投影](https://scikit-learn.org/stable/modules/random_projection.html)

### [7.6.1. Johnson-Lindenstrauss 引理](https://scikit-learn.org/stable/modules/random_projection.html#the-johnson-lindenstrauss-lemma)

### [7.6.2. 高斯随机投影](https://scikit-learn.org/stable/modules/random_projection.html#gaussian-random-projection)

### [7.6.3. 稀疏随机投影](https://scikit-learn.org/stable/modules/random_projection.html#sparse-random-projection)

### [7.6.4. 逆变换](https://scikit-learn.org/stable/modules/random_projection.html#inverse-transform)

## [7.7. 核近似](https://scikit-learn.org/stable/modules/kernel_approximation.html)

### [7.7.1. 核近似的 Nystroem 方法](https://scikit-learn.org/stable/modules/kernel_approximation.html#nystroem-method-for-kernel-approximation)

### [7.7.2. 径向基函数核](https://scikit-learn.org/stable/modules/kernel_approximation.html#radial-basis-function-kernel)

### [7.7.3. 加性卡方核](https://scikit-learn.org/stable/modules/kernel_approximation.html#additive-chi-squared-kernel)

### [7.7.4. 偏斜卡方核](https://scikit-learn.org/stable/modules/kernel_approximation.html#skewed-chi-squared-kernel)

### [7.7.5. 通过 Tensor Sketch 进行多项式核近似](https://scikit-learn.org/stable/modules/kernel_approximation.html#polynomial-kernel-approximation-via-tensor-sketch)

### [7.7.6. 数学细节](https://scikit-learn.org/stable/modules/kernel_approximation.html#mathematical-details)

## [7.8. 成对度量、亲和力和核](https://scikit-learn.org/stable/modules/metrics.html)

### [7.8.1. 余弦相似度](https://scikit-learn.org/stable/modules/metrics.html#cosine-similarity)

### [7.8.2. 线性核](https://scikit-learn.org/stable/modules/metrics.html#linear-kernel)

### [7.8.3. 多项式核](https://scikit-learn.org/stable/modules/metrics.html#polynomial-kernel)

### [7.8.4. S 形核](https://scikit-learn.org/stable/modules/metrics.html#sigmoid-kernel)

### [7.8.5. RBF 核](https://scikit-learn.org/stable/modules/metrics.html#rbf-kernel)

### [7.8.6. 拉普拉斯核](https://scikit-learn.org/stable/modules/metrics.html#laplacian-kernel)

### [7.8.7. 卡方核](https://scikit-learn.org/stable/modules/metrics.html#chi-squared-kernel)

## [7.9. 变换预测目标（`y`）](https://scikit-learn.org/stable/modules/preprocessing_targets.html)

### [7.9.1. 标签二值化](https://scikit-learn.org/stable/modules/preprocessing_targets.html#label-binarization)

### [7.9.2. 标签编码](https://scikit-learn.org/stable/modules/preprocessing_targets.html#label-encoding)

# [8\. 数据集加载实用程序](https://scikit-learn.org/stable/datasets.html)

## [8.1. 玩具数据集](https://scikit-learn.org/stable/datasets/toy_dataset.html)

### [8.1.1. 鸢尾花植物数据集](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-plants-dataset)

### [8.1.2. 糖尿病数据集](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)

### [8.1.3. 手写数字光学识别数据集](https://scikit-learn.org/stable/datasets/toy_dataset.html#optical-recognition-of-handwritten-digits-dataset)

### [8.1.4. Linnerrud 数据集](https://scikit-learn.org/stable/datasets/toy_dataset.html#linnerrud-dataset)

### [8.1.5. 葡萄酒识别数据集](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-recognition-dataset)

### [8.1.6. 乳腺癌威斯康星州（诊断）数据集](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset)

## [8.2. 真实世界数据集](https://scikit-learn.org/stable/datasets/real_world.html)

### [8.2.1. Olivetti 人脸数据集](https://scikit-learn.org/stable/datasets/real_world.html#the-olivetti-faces-dataset)

### [8.2.2. 20 个新闻组文本数据集](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset)

### [8.2.3. Labeled Faces in the Wild 人脸识别数据集](https://scikit-learn.org/stable/datasets/real_world.html#the-labeled-faces-in-the-wild-face-recognition-dataset)

### [8.2.4. 森林覆盖类型](https://scikit-learn.org/stable/datasets/real_world.html#forest-covertypes)

### [8.2.5. RCV1 数据集](https://scikit-learn.org/stable/datasets/real_world.html#rcv1-dataset)

### [8.2.6. Kddcup 99 数据集](https://scikit-learn.org/stable/datasets/real_world.html#kddcup-99-dataset)

### [8.2.7. 加州住房数据集](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)

### [8.2.8. 物种分布数据集](https://scikit-learn.org/stable/datasets/real_world.html#species-distribution-dataset)

## [8.3. 生成的数据集](https://scikit-learn.org/stable/datasets/sample_generators.html)

### [8.3.1. 分类和聚类生成器](https://scikit-learn.org/stable/datasets/sample_generators.html#generators-for-classification-and-clustering)

### [8.3.2. 回归生成器](https://scikit-learn.org/stable/datasets/sample_generators.html#generators-for-regression)

### [8.3.3. 流形学习生成器](https://scikit-learn.org/stable/datasets/sample_generators.html#generators-for-manifold-learning)

### [8.3.4. 分解生成器](https://scikit-learn.org/stable/datasets/sample_generators.html#generators-for-decomposition)

## [8.4 加载其他数据集](https://scikit-learn.org/stable/datasets/loading_other_datasets.html)

### [8.4.1. 示例图像](https://scikit-learn.org/stable/datasets/loading_other_datasets.html#sample-images)

### [8.4.2. svmlight / libsvm 格式的数据集](https://scikit-learn.org/stable/datasets/loading_other_datasets.html#datasets-in-svmlight-libsvm-format)

### [8.4.3 从 openml.org 存储库下载数据集](https://scikit-learn.org/stable/datasets/loading_other_datasets.html#downloading-datasets-from-the-openml-org-repository)

### [8.4.4. 从外部数据集加载](https://scikit-learn.org/stable/datasets/loading_other_datasets.html#loading-from-external-datasets)

# [9\. 使用 scikit-learn 进行计算](https://scikit-learn.org/stable/computing.html)

## [9.1. 计算扩展策略：更大的数据](https://scikit-learn.org/stable/computing/scaling_strategies.html)

### [9.1.1. 使用核外学习进行实例扩展](https://scikit-learn.org/stable/computing/scaling_strategies.html#scaling-with-instances-using-out-of-core-learning)

## [9.2. 计算性能](https://scikit-learn.org/stable/computing/computational_performance.html)

### [9.2.1. 预测延迟](https://scikit-learn.org/stable/computing/computational_performance.html#prediction-latency)

### [9.2.2. 预测吞吐量](https://scikit-learn.org/stable/computing/computational_performance.html#prediction-throughput)

### [9.2.3. 技巧和窍门](https://scikit-learn.org/stable/computing/computational_performance.html#tips-and-tricks)

## [9.3. 并行性、资源管理和配置](https://scikit-learn.org/stable/computing/parallelism.html)

### [9.3.1. 并行性](https://scikit-learn.org/stable/computing/parallelism.html#parallelism)

### [9.3.2. 配置开关](https://scikit-learn.org/stable/computing/parallelism.html#configuration-switches)

# [10\. 模型持久化](https://scikit-learn.org/stable/model_persistence.html)

## [10.1. 工作流程概述](https://scikit-learn.org/stable/model_persistence.html#workflow-overview)

### [10.1.1. 训练并持久化模型](https://scikit-learn.org/stable/model_persistence.html#train-and-persist-the-model)

## [10.2. ONNX](https://scikit-learn.org/stable/model_persistence.html#onnx)

## [10.3.`skops.io`](https://scikit-learn.org/stable/model_persistence.html#skops-io)

## [10.4. `pickle`，，`joblib`和`cloudpickle`](https://scikit-learn.org/stable/model_persistence.html#pickle-joblib-and-cloudpickle)

## [10.5. 安全性和可维护性限制](https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations)

### [10.5.1. 在生产中复制训练环境](https://scikit-learn.org/stable/model_persistence.html#replicating-the-training-environment-in-production)

### [10.5.2. 提供模型工件](https://scikit-learn.org/stable/model_persistence.html#serving-the-model-artifact)

## [10.6. 总结要点](https://scikit-learn.org/stable/model_persistence.html#summarizing-the-key-points)

# [11\. 常见陷阱和建议做法](https://scikit-learn.org/stable/common_pitfalls.html)

## [11.1. 不一致的预处理](https://scikit-learn.org/stable/common_pitfalls.html#inconsistent-preprocessing)

## [11.2. 数据泄露](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)

### [11.2.1. 如何避免数据泄露](https://scikit-learn.org/stable/common_pitfalls.html#how-to-avoid-data-leakage)

### [11.2.2. 预处理过程中的数据泄漏](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage-during-pre-processing)

## [11.3. 控制随机性](https://scikit-learn.org/stable/common_pitfalls.html#controlling-randomness)

### [11.3.1. 使用`None`或实例，以及对和的`RandomState`重复调用` fit``split `](https://scikit-learn.org/stable/common_pitfalls.html#using-none-or-randomstate-instances-and-repeated-calls-to-fit-and-split)

### [11.3.2. 常见的陷阱和细微之处](https://scikit-learn.org/stable/common_pitfalls.html#common-pitfalls-and-subtleties)

### [11.3.3. 一般建议](https://scikit-learn.org/stable/common_pitfalls.html#general-recommendations)

# [12.调度](https://scikit-learn.org/stable/dispatching.html)

## [12.1. 数组 API 支持（实验性）](https://scikit-learn.org/stable/modules/array_api.html)

### [12.1.1. 示例用法](https://scikit-learn.org/stable/modules/array_api.html#example-usage)

### [12.1.2. 支持兼容输入`Array API`](https://scikit-learn.org/stable/modules/array_api.html#support-for-array-api-compatible-inputs)

### [12.1.3. 输入和输出数组类型处理](https://scikit-learn.org/stable/modules/array_api.html#input-and-output-array-type-handling)

### [12.1.4. 常见的估计器检查](https://scikit-learn.org/stable/modules/array_api.html#common-estimator-checks)

# [13\. 选择正确的估算器](https://scikit-learn.org/stable/machine_learning_map.html)

# [14\. 外部资源、视频和讲座](https://scikit-learn.org/stable/presentations.html)

## [14.1. scikit-learn MOOC](https://scikit-learn.org/stable/presentations.html#the-scikit-learn-mooc)

## [14.2. 视频](https://scikit-learn.org/stable/presentations.html#videos)

## [14.3. 刚接触科学计算 Python？](https://scikit-learn.org/stable/presentations.html#new-to-scientific-python)

## [14.4. 外部教程](https://scikit-learn.org/stable/presentations.html#external-tutorials)
