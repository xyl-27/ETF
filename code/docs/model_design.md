# 模型设计文档

## 1. 整体框架

所有模型遵循统一的 **Stock Ranking** 范式：

```
输入: [batch_size, num_stocks, seq_len, num_features]
  ↓
展平股票维度: [batch_size * num_stocks, seq_len, num_features]
  ↓
Input Projection (Linear)
  ↓
时间序列编码器 (各模型不同)
  ↓
时间维度聚合 → [batch_size * num_stocks, d_model]
  ↓
重塑: [batch_size, num_stocks, d_model]
  ↓
CrossStockAttention (可选)
  ↓
Ranking MLP (2~3层 MLP + LayerNorm + ReLU + Dropout)
  ↓
Score Head (Linear → 1)
  ↓
输出: [batch_size, num_stocks]  (每只股票的排序分数)
```

**核心设计原则：**
- 所有模型共享统一的输入/输出接口
- 时间序列编码器是可替换的模块
- CrossStockAttention 和 MMoE 为可选增强模块
- 损失函数统一为 **WeightedRankingLoss** (Listwise + Pairwise)

---

## 2. 数据管道

### 2.1 特征工程

| 特征集 | 特征数 | 说明 |
|--------|--------|------|
| `39` | 39 | 基础量价因子 (JQ_FACTORS) |
| `158plus39` | 197 | 158 扩展因子 + 39 基础 |
| `97` | 97 | 聚宽因子 |
| `39plus97` | 136 | 39 基础 + 97 聚宽 |
| `158plus97` | 255 | 158 扩展 + 97 聚宽 |
| `158plus39plus97` | 294 | 全量因子 |

特征包括：动量、波动率、相关性、成交量、价格位置、换手率、乖离率等。

### 2.2 数据预处理

- **标准化**: `StandardScaler` 按截面（同一日期）标准化
- **序列构建**: 滑动窗口 `seq_len=60` 天
- **标签**: 未来 5 日开盘收益率 (`fwd_5d_open_return`)
- **数据集划分**:
  - 训练集: `end_date` 前若干完整月
  - 验证集: `val_start_date` ~ `val_end_date`（支持固定日期集）
  - 滑动验证集: 按月切片滚动验证

### 2.3 DataLoader 输出格式

```python
{
    "sequences":  [batch, max_stocks, seq_len, num_features]  # 填充到 max_stocks
    "targets":    [batch, max_stocks]    # 未来收益率
    "relevance":  [batch, max_stocks]    # 排序标签（排名倒数）
    "masks":      [batch, max_stocks]    # 有效股票掩码
    "hs300_rets": [batch, 1]             # 同期沪深300收益
}
```

---

## 3. 公共组件

### 3.1 PositionalEncoding (`positional_encoding.py`)

标准正弦余弦位置编码，来自 Transformer (Vaswani et al. 2017)。注册为 buffer，非训练参数。

### 3.2 FeatureAttention (`attention.py:FeatureAttention`)

时间步级注意力聚合：

```
Linear(d_model → d_model/2) → Tanh → Linear(d_model/2 → 1) → Softmax(dim=1)
```

对每个时间步学习权重，加权求和聚合序列维度。

### 3.3 CrossStockAttention (`attention.py:CrossStockAttention`)

股票间交互注意力：

```
MultiheadAttention(stock_features, stock_features, stock_features)
  → residual + LayerNorm
```

使用标准多头自注意力建模股票之间的相关性，带残差连接。

### 3.4 MMoE (`mmoe.py`)

单门控专家混合：

```
Gate: Linear → Softmax(num_experts)
Experts: [MLP(input_dim → input_dim/2 → output_dim)] × num_experts
Output: Σ gate_i × expert_i(x)
```

可选模块，在 LSTM/GRU/TCN/iTransformer 中通过 `num_experts` 参数控制。

### 3.5 Inception_Block_V1 (`layers.py`)

多核 2D 卷积：

```
Conv2d(kernel=1), Conv2d(kernel=3), ..., Conv2d(kernel=2*num_kernels-1)
  → stack → mean
```

用于 TimesNet 的时序 2D 变换。

---

## 4. 模型详解

### 4.1 Transformer (`transformer.py`)

| 组件 | 配置 |
|------|------|
| 输入映射 | `Linear(num_features → d_model)` |
| 位置编码 | 正弦余弦 |
| 时序编码器 | `TransformerEncoderLayer × num_layers` |
| 时序聚合 | FeatureAttention |
| 股票交互 | CrossStockAttention |
| Ranking MLP | `Linear(d → d) → LayerNorm → ReLU → Dropout → Linear(d → d/2)` |
| Score Head | `Linear(d/2 → d/4) → ReLU → Dropout → Linear(d/4 → 1)` |

**特点**: 标准的 Transformer 架构，self-attention 捕捉时间依赖关系。

### 4.2 LSTM (`lstm.py`)

| 组件 | 配置 |
|------|------|
| 输入映射 | `Linear(num_features → d_model)` |
| 时序编码器 | `LSTM(d_model, d_model, num_layers, batch_first)` |
| 时序聚合 | FeatureAttention |
| 可选增强 | MMoE (num_experts>0) |
| 股票交互 | CrossStockAttention |
| Ranking MLP | 同 Transformer |
| Score Head | 同 Transformer |

**特点**: 双向/单向 LSTM 捕捉长期时间依赖，带 MMoE 选项。

### 4.3 GRU (`gru.py`)

结构与 LSTM 完全相同，仅将 LSTM 替换为 GRU。

### 4.4 TCN (`tcn.py`)

| 组件 | 配置 |
|------|------|
| 输入映射 | `Linear(num_features → d_model)` |
| 时序编码器 | `TemporalBlock × num_layers` (膨胀卷积) |
| 时序聚合 | FeatureAttention |
| 可选增强 | MMoE |
| 股票交互 | CrossStockAttention |
| Ranking MLP | 同 Transformer |

**TemporalBlock**: 2 层 Conv1d + BatchNorm + ReLU + Dropout + 残差连接，膨胀率 `2^i` 指数增长。因果卷积（padding 裁剪）。

### 4.5 iTransformer (`itransformer.py`)

| 组件 | 配置 |
|------|------|
| 输入映射 | `Linear(seq_len → d_model)` 转置后映射 |
| 时序编码器 | `TransformerEncoderLayer × num_layers` |
| 池化 | `mean(dim=1)` 在特征维度池化 |
| 可选增强 | MMoE |
| 股票交互 | CrossStockAttention |
| Ranking MLP | 同 Transformer |

**特点**: 与传统 Transformer 不同，iTransformer 对**特征维度**做 attention（`Linear(seq_len, d_model)`），将时间序列在特征维度编码。这允许它捕捉跨特征的相关性。

### 4.6 TimesNet (`timesnet.py`)

| 组件 | 配置 |
|------|------|
| 输入映射 | `Linear(num_features → d_model)` |
| 时序编码器 | `TimesBlock × num_layers` |
| 时序聚合 | `mean(dim=1)` |
| 股票交互 | 无 |
| Ranking MLP | `Linear(d → d/2) → LayerNorm → ReLU → Dropout → Linear(d/2 → d/4)` |
| Score Head | `Linear(d/4 → 1)` |

**TimesBlock**:
```
FFT → 发现主导周期 → 按周期 reshape 为 2D
  → Inception_Block_V1 (2D Conv) × 2
  → reshape 回 1D → softmax 加权聚合各周期结果
```

**特点**: 不使用 CrossStockAttention 和 MMoE，通过 FFT 发现时序内在周期，2D 卷积捕捉周期内和周期间模式。

### 4.7 DLinear (`dlinear.py`)

| 组件 | 配置 |
|------|------|
| 分解 | `series_decomp` (移动平均分离趋势+季节) |
| 输入映射 | `Linear(num_features → d_model)` |
| 线性层 | `Linear(seq_len → seq_len)` × 2 (季节+趋势) |
| 时序聚合 | `mean(dim=1)` |
| 股票交互 | 无 |
| Ranking MLP | `Linear(d → d/2) → LayerNorm → ReLU → Dropout → Linear(d/2 → d/4)` |
| Score Head | `Linear(d/4 → 1)` |

**特点**: 极简结构，序列分解后两个线性层分别处理季节和趋势分量。初始化权重为 `1/seq_len`。

### 4.8 NLinear (`nlinear.py`)

| 组件 | 配置 |
|------|------|
| 输入映射 | `Linear(num_features → d_model)` |
| 归一化 | 减去最后一个时间步的值 |
| 线性层 | `Linear(seq_len → seq_len)` |
| 反归一化 | 加回最后一个时间步 |
| 时序聚合 | `mean(dim=1)` |
| 股票交互 | 无 |
| Ranking MLP | 同 DLinear |
| Score Head | 同 DLinear |

**特点**: N-BEATS 风格，先减后加最后一个值做归一化/反归一化。最简结构。

### 4.9 PatchTST (`patchtst.py`)

| 组件 | 配置 |
|------|------|
| 输入映射 | `Linear(num_features → d_model)` |
| 分片 | `patch_len=8, stride=4` → `num_patches=14` (seq_len=60) |
| 片映射 | `Linear(patch_len × d_model → d_model)` |
| 位置编码 | 可学习位置编码 `randn(1, num_patches, d_model)` |
| 时序编码器 | `TransformerEncoderLayer × num_layers` |
| 池化 | `mean(dim=1)` |
| 股票交互 | 无 |
| Ranking MLP | 同 DLinear |
| Score Head | 同 DLinear |

**特点**: 将时间序列分割为 patches，每个 patch 展开后映射到 d_model，再用 Transformer 编码 patches 之间的关系。

### 4.10 Mamba (`mamba_simple.py`)

| 组件 | 配置 |
|------|------|
| 输入映射 | `Linear(num_features → d_model)` |
| 门控 | SiLU(GateProj(x)) |
| 卷积 | `Conv1d(d_model, d_model, kernel=4, padding=3, groups=d_model)` |
| 状态空间 | `SSMBlock × num_layers` |
| 时序聚合 | `mean(dim=1)` |
| 股票交互 | 无 |
| Ranking MLP | 同 DLinear |
| Score Head | 同 DLinear |

**SSMBlock**:
```
x → x_proj → dt_proj → softplus(δ)
  → B_param, C_param
  → y_t = h_t · C_t + D · x_t
  → h_{t+1} = exp(δA) · h_t + δB · x_t
```

**特点**: 基于状态空间模型的选择性扫描，线性复杂度 O(L)，带门控卷积。不需要 CrossStockAttention。

---

## 5. 训练方法

### 5.1 损失函数: `WeightedRankingLoss`

```
Loss = listwise_CE + pairwise_weight × pairwise_loss
```

- **Listwise损失**: 预测分数和目标分数的 softmax 分布之间的交叉熵
- **Pairwise损失**: sigmoid 排序损失，鼓励预测排序与真实排序一致
- **Top-K加权**: 真实收益 Top-K 的样本权重乘以 `weight_factor` (默认 2.0)
- **温度**: softmax 温度参数 `temperature=1.0`

### 5.2 优化器

| 参数 | 值 |
|------|-----|
| 优化器 | AdamW |
| 学习率 | 搜索范围 `1e-5` ~ `1e-3` (因模型而异) |
| 权重衰减 | `1e-5` |
| 调度器 | `LinearLR` (start=1.0, end=0.2, total_iters=num_epochs) |
| 梯度裁剪 | `max_grad_norm=5.0` |

### 5.3 训练参数

| 参数 | 默认值 |
|------|--------|
| batch_size | 4 |
| num_epochs | 30 |
| top_k | 3 |
| 验证频率 | 每 epoch |

### 5.4 模型保存策略

| 文件 | 选择标准 |
|------|----------|
| `best_model.pth` | 验证集 `final_score` 最高 |
| `best_model_sliding.pth` | 滑动验证集 `final_score` 最高 |
| `best_model_optuna.pth` | Optuna 搜索目标最优 (Sharpe) |

---

## 6. 评估指标

在验证集上按周计算（`train.py:calculate_ranking_metrics`）：

| 指标 | 说明 |
|------|------|
| `final_score` | 综合评分 (Top-K 命中 + NDCG + 超额收益的加权组合) |
| `ndcg` | 归一化折损累计增益 (Normalized Discounted Cumulative Gain) |
| `topk_hit_rate` | Top-K 命中率 |
| `excess_return` | 相对沪深300的超额收益 |
| `rank_ic` | 排序信息系数 (Spearman/Pearson) |
| `ks` | Kolmogorov-Smirnov 统计量 |
| `win_rate` | 日胜率 |
| `rebalance_win_rate` | 调仓日胜率 |

---

## 7. 超参数搜索

### 7.1 Optuna 贝叶斯搜索 (`train_search_v2.py`)

| 参数 | 搜索范围 |
|------|----------|
| `learning_rate` | `1e-5` ~ `1e-3` (log-uniform) |
| `d_model` | `[32, 64, 128]` (因模型而异) |
| `num_layers` | `1` ~ `4` (因模型而异) |
| `dropout` | `0.1` ~ `0.2` |
| `nhead` | `[4]` 或 `[8]` (仅 Transformer 类) |
| `num_experts` | `[None, 3]` (仅 MoE 类) |

搜索空间通过 `config.py:get_search_space()` 定义，按模型类型返回 Optuna trial 建议函数。

### 7.2 搜索流程

```
for each model_type:
  create Optuna study (study_name=f"etf_{model_type}_{top_k}")
  for trial in 1..80:
    params = search_space_fn(trial)
    train model with params
    evaluate on validation set
    report final_score to Optuna
  save search_results.json (含所有 trial 的 exp_idx, params, score)
```

### 7.3 模型选择

训练完成后，`find_best_model()` 函数选择 `search_results.json` 中 score 最高的实验目录。选择结果记录在 `output/model_selection.yaml`：

```yaml
models:
  - dir: "./model/bayes_lstm_74_3_.../exp_5"
    file: "best_model_sliding.pth"
    enabled: true
  - dir: "./model/bayes_transformer_74_3_.../exp_2"
    file: "best_model.pth"
    enabled: true
master: "juejin"      # 日报主序列
average: true         # 是否启用平均模型
voting: true          # 是否启用投票模型
```

---

## 8. 推理与集成

### 8.1 预测流程 (`predict.py` / `daily_eval.py`)

```
加载 model_selection.yaml
  → 遍历 enabled 模型
  → 每模型加载 best_model.pth + config.json + scaler.pkl
  → 对每个交易日生成预测分数
  → 保存 predictions.json
```

**predictions.json 结构:**
```json
{
  "lstm_exp_5": {
    "2026-04-01": [{"rank": 1, "stock_id": "510050", "score": 0.95}, ...],
    "2026-04-02": [...],
    ...
  },
  "transformer_exp_2": { ... },
  "average": { ... },       // 多模型分数平均
  "voting": { ... },        // 多模型投票
  "_meta": {
    "start_date": "2026-04-01",
    "backtest_dates": ["2026-04-01", ...]
  }
}
```

### 8.2 集成策略

| 策略 | 说明 |
|------|------|
| **单模型** | 使用单个最佳模型的预测分数 |
| **Average** | 多个模型的分数等权平均 |
| **Voting** | 每个模型的 Top-K 投票，按出现频次排序 |
| **Juejin** | 掘金实盘/仿真交易，通过 API 执行 |

### 8.3 回测集成

`run_backtest_sequence()` 对每个序列（各单模型 + average + voting + juejin）分别运行 `BacktestEngine.run()`，收集 `today_pnl`、`trades`、`pre_rebalance_positions`，合并返回用于日报生成。

---

## 9. 模型能力对比

| 模型 | 时序建模 | 股票交互 | 多专家 | 周期发现 | 参数量级 | 速度 |
|------|----------|----------|--------|----------|----------|------|
| Transformer | Self-Attention | ✅ | ❌ | ❌ | 中 | 中 |
| LSTM | LSTM | ✅ | ✅ 可选 | ❌ | 大 | 慢 |
| GRU | GRU | ✅ | ✅ 可选 | ❌ | 大 | 中 |
| TCN | 膨胀卷积 | ✅ | ✅ 可选 | ❌ | 中 | 快 |
| iTransformer | 特征维 Attention | ✅ | ✅ 可选 | ❌ | 中 | 中 |
| TimesNet | 2D Inception | ❌ | ❌ | ✅ FFT | 大 | 中 |
| DLinear | 线性分解 | ❌ | ❌ | ❌ | 极小 | 极快 |
| NLinear | 线性+归一化 | ❌ | ❌ | ❌ | 极小 | 极快 |
| PatchTST | Patch+Transformer | ❌ | ❌ | ❌ | 中 | 中 |
| Mamba | 状态空间模型 | ❌ | ❌ | ❌ | 中 | 快 |

---

## 10. 回测与交易

### 10.1 BacktestEngine

| 参数 | 默认值 |
|------|--------|
| 初始资金 | ¥100,000 |
| 持仓比例 | 95% |
| 调仓频率 | 每 5 个交易日 |
| 选股数量 | Top-K (默认 3) |
| 手续费 | 万分之三 (0.03%) |
| 滑点 | 千分之一 (0.1%) |
| 交易模式 | `open` (开盘价) / `close` (收盘价) |

### 10.2 买卖逻辑

- **买入**: 排名 Top-K 的目标持仓，等权分配资金
- **卖出**: 不在目标持仓的现有持仓全部卖出
- **涨跌停/停牌**: 涨停买不进 / 跌停卖不出时跳过，保留现有持仓
- **调仓快照**: 记录调仓前后持仓对比

### 10.3 风险指标

| 指标 | 说明 |
|------|------|
| 累计收益率 | 总收益率 |
| 年化收益率 | 年化 |
| 年化波动率 | 收益波动 |
| 最大回撤 | 峰值到谷底最大跌幅 |
| 夏普比率 | 风险调整后收益 |
| 卡玛比率 | 收益/最大回撤 |
| 日胜率 | 正收益天数占比 |
| 调仓胜率 | 调仓日正收益占比 |
| VaR (95%) | 在险价值 |
| CVaR (95%) | 条件在险价值 |
| Ulcer 指数 | 回撤深度与持续时间 |
| 获利因子 | 总盈利/总亏损 |
| 平均恢复天数 | 回撤后恢复所需平均天数 |
