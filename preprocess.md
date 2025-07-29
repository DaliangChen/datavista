当然！这个页面主要介绍的是：



🎯 Calibration Curves（校准曲线）

也叫：Reliability Diagrams（置信度可靠性图）



✅ 它是用来做什么的？

> 校准曲线用来衡量模型输出的概率是否可靠。

我们通过画出：
模型预测的概率值 vs 实际正类的发生频率
来可视化判断一个模型是否“校准好”。



📊 一、图像长什么样？

校准曲线图的基本结构如下：

* 横轴（X）：模型预测的概率（如 0.1, 0.2, ..., 0.9）
* 纵轴（Y）：这些概率对应的样本，真实属于正类的比例

如果模型完全校准完美，那么：

* 每个点都应该在对角线上（即 y = x）；
* 比如，模型预测概率为 0.7 的样本里，真的有 70% 是正类；



❌ 校准不好的模型长什么样？

* 曲线偏上：模型低估了概率（太保守）
* 曲线偏下：模型高估了概率（太自信）

这就是“过置信”或“欠置信”的体现。



🛠️ 二、怎么画 Calibration Curve？（代码讲解）

```python
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

y_test: 真实标签
probas: 模型预测的概率（比如 calibrated_model.predict_proba(X)[:, 1]）

fraction_of_positives, mean_predicted_value = calibration_curve(y_test, probas, n_bins=10)

绘图
plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Your Model")
plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.title("Calibration curve (reliability diagram)")
plt.legend()
plt.grid()
plt.show()
```



🧮 参数说明：

* `y_test`：测试集真实标签
* `probas`：模型预测概率（注意，不是 decision function）
* `n_bins=10`：把预测概率划成 10 个区间（比如 0~~0.1，0.1~~0.2 ...）



📌 三、补充指标（和校准曲线一起使用）：

| 指标名             | 用途                          |
| --------------- | --------------------------- |
| Brier Score | 衡量概率预测的“均方误差”，越小越好（越接近真实概率） |
| Log Loss    | 衡量概率与实际标签的对数损失，也常用于概率质量评估   |

这些指标能量化地补充你从校准曲线中看到的图像信息。



🎯 四、典型用途

* 对比不同模型的“置信度质量”：

  * 随机森林 vs 校准后的随机森林
  * LightGBM vs Platt scaling 后的 LightGBM
* 评估一个模型是否适合用概率输出直接作为决策输入
* 检查一个分类器是否“过度自信”或“过度保守”



🧠 五、总结一句话：

> 校准曲线是判断模型预测概率是否可信的直观方法。

你可以很清楚地看到：
“模型说它有 80% 的把握，结果真的有 80% 成功吗？”



需要我用一个实际例子画出校准前后对比图，帮助你更直观理解吗？
