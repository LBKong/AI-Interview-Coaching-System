# Expert score CSV format

每位专家使用一个 CSV 文件，例如 `expert_1.csv`。表头必须是：

```csv
sample_id,accuracy,specificity,actionability,coverage,overall_usefulness
```

`sample_id` 必须与 `scores/` 中对应 judge JSON 的**完整文件名 stem 完全一致**：只去掉
末尾的 `.json`，不得缩短或改写 session ID，也不得删除 `_q{idx}`。

例如：

```text
scores/session_1783895535160_q0.json
→ sample_id = session_1783895535160_q0
```

五个评分维度都必须填写 1–5 的整数。空白、越界、重复 `sample_id` 或无法与 judge
score 对齐的行不会被静默插补；校准程序会校验并报告不完整样本。
