好的，下面是对 `pandas.pivot_table()` 的**详细解析**，包括用法、参数、示例、对比和进阶技巧。

---

**🧩 `pivot_table()` 是什么？**

`pivot_table()` 是 `pandas` 中用于\*\*透视分析（pivoting）和汇总聚合（aggregation）\*\*的核心函数。它比 `pivot()` 更强大，允许数据中有重复项，并支持各种聚合函数（如平均、求和、计数等）。

---

**📘 基本语法**

```python
pd.pivot_table(
    data,
    values=None,
    index=None,
    columns=None,
    aggfunc='mean',
    fill_value=None,
    margins=False,
    margins_name='All',
    dropna=True
)
```

---

**🔑 参数详解**

| 参数             | 类型         | 说明                                                  |
| -------------- | ---------- | --------------------------------------------------- |
| `data`         | DataFrame  | 原始数据                                                |
| `values`       | str 或 list | 要聚合的字段（可多个）                                         |
| `index`        | str 或 list | 行索引（用于分组）                                           |
| `columns`      | str 或 list | 列索引（变为新列）                                           |
| `aggfunc`      | 函数/列表/字典   | 聚合函数，默认 `'mean'`（如：`sum`, `count`, `max`, `np.std`） |
| `fill_value`   | 标量         | 替换缺失值                                               |
| `margins`      | bool       | 是否添加总计行/列（默认为 False）                                |
| `margins_name` | str        | 总计的名称，默认 `'All'`                                    |
| `dropna`       | bool       | 是否丢弃含 NA 的列组合（默认 True）                              |

---

**🧪 示例：简单聚合**

```python
import pandas as pd

df = pd.DataFrame({
    'region': ['East', 'East', 'West', 'West', 'East', 'West'],
    'product': ['A', 'B', 'A', 'B', 'A', 'A'],
    'sales': [100, 150, 80, 200, 130, 90]
})

print(df)
```

输出：

```
  region product  sales
0   East       A    100
1   East       B    150
2   West       A     80
3   West       B    200
4   East       A    130
5   West       A     90
```

---

**📌 1. 基本 pivot\_table**

```python
pd.pivot_table(df, index='region', columns='product', values='sales')
```

输出：

```
product      A      B
region
East     115.0  150.0
West      85.0  200.0
```

解释：

* `A` 产品在 East 的平均销售 = (100 + 130)/2 = 115
* 默认 `aggfunc='mean'`，可以改为其他函数

---

**📌 2. 使用 `aggfunc='sum'`**

```python
pd.pivot_table(df, index='region', columns='product', values='sales', aggfunc='sum')
```

输出：

```
product     A      B
region
East     230    150
West     170    200
```

---

**📌 3. 多层索引（多维分析）**

```python
pd.pivot_table(df, index=['region', 'product'], values='sales', aggfunc='sum')
```

输出：

```
                 sales
region product
East   A           230
       B           150
West   A           170
       B           200
```

---

**📌 4. 多值字段聚合**

```python
df['quantity'] = [1, 2, 1, 3, 1, 1]

pd.pivot_table(
    df,
    index='region',
    columns='product',
    values=['sales', 'quantity'],
    aggfunc='sum'
)
```

输出：

```
            sales           quantity
product         A      B         A    B
region
East          230    150         2    2
West          170    200         2    3
```

---

**📌 5. 填充缺失值**

```python
pd.pivot_table(df, index='region', columns='product', values='sales', fill_value=0)
```

→ 用 0 填补缺失组合。

---

**📌 6. 添加合计（margins）**

```python
pd.pivot_table(df, index='region', columns='product', values='sales', aggfunc='sum', margins=True)
```

输出：

```
product      A      B    All
region
East       230    150    380
West       170    200    370
All        400    350    750
```

---

**📌 7. 多聚合函数**

```python
import numpy as np

pd.pivot_table(df, index='region', columns='product', values='sales',
               aggfunc=[np.sum, np.mean])
```

输出是多层列的 DataFrame，显示总和与平均值：

```
             sum           mean
product        A      B      A      B
region
East         230    150  115.0  150.0
West         170    200   85.0  200.0
```

---

**⚠️ 与 `groupby()` 的区别？**

| 特性        | `pivot_table` | `groupby` + `agg` |
| --------- | ------------- | ----------------- |
| 输出是表格（透视） | ✅             | ❌（通常是扁平结果）        |
| 聚合函数支持    | ✅ `aggfunc`   | ✅ `agg`           |
| 多列聚合      | ✅ 多值、多聚合      | ✅                 |
| 多索引支持     | ✅             | ✅                 |
| 用于可视化表格   | 非常适合          | 不太直观              |

---

**✅ 总结**

| 功能      | 用法                                    |
| ------- | ------------------------------------- |
| 行列转换+汇总 | `pivot_table(index, columns, values)` |
| 支持聚合函数  | `aggfunc='sum'`、`count`、`mean`、自定义函数  |
| 多字段聚合   | `values=['sales', 'quantity']`        |
| 多级索引/多列 | 支持多层次的数据分析                            |
| 缺失值处理   | `fill_value=0`                        |
| 总计统计    | `margins=True`                        |

---

如果你提供具体的数据结构或业务场景（例如销售、库存、考试成绩），我可以帮你构建定制的 `pivot_table()` 示例。需要试一个吗？
