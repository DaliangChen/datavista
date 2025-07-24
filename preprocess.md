
输出结果：

```
city      Beijing  Shanghai
date
2024-01         5         8
2024-02         7        10
```

解释：

* `index="date"`：按月份排列行；
* `columns="city"`：将 `city` 中的唯一值（Beijing、Shanghai）转成列；
* `values="temperature"`：填入表格中的数值。

---

**🧨 注意事项：不能有重复项！**

如果数据中出现 `index + columns` 的组合重复，`pivot()` 会报错：

```python
ValueError: Index contains duplicate entries, cannot reshape
```

解决方法：使用 `pivot_table()` 替代（它支持聚合）。

---

**🔁 pivot vs pivot\_table**

| 方法            | 是否支持重复项 | 是否支持聚合 | 默认聚合函数     |
| ------------- | ------- | ------ | ---------- |
| `pivot`       | ❌ 不支持   | ❌ 不支持  | 无          |
| `pivot_table` | ✅ 支持    | ✅ 支持   | `mean`（可改） |

例如：

```python
df.pivot_table(index="date", columns="city", values="temperature", aggfunc="mean")
```

---

**🔄 反向操作：`melt`**

如果你使用了 `pivot` 让数据变宽，想恢复成长格式，可以用 `melt`：

```python
df_pivot.reset_index().melt(id_vars="date", value_name="temperature")
```

---

**🧠 总结**

| 功能            | 描述                               |
| ------------- | -------------------------------- |
| `pivot`       | 按 index/columns 重排表格；必须无重复；只用于重构 |
| `pivot_table` | 同上，但允许重复并能聚合                     |
| `melt`        | 把宽表“还原”为长表                       |

---

如果你有具体的应用场景，比如财务表、销售报表、考勤数据等，我可以给出对应的 `pivot` 实战案例。需要吗？
