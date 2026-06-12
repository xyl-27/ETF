# ETF 量化选股系统

基于深度学习的 ETF 量化选股与每日测评系统。支持多模型并行回测、自适应调仓、Optuna 贝叶斯超参搜索、自动邮件报告。

## 系统架构

```
                          ┌─────────────┐
                          │  get_etf_data │──→ etf_74.csv
                          └─────────────┘
                                ↓
               ┌────────────────┼────────────────┐
               ↓                ↓                 ↓
         train_search_v2    daily_eval.py    reproduce_backtest.ipynb
         (贝叶斯超参搜索)    (5种模式)        (批量回测 + 模型选择)
               │                │                 │
               ↓                ↓                 ↓
          model/bayes_*     output/*.json    notebooks/选模型Cell
          exp_*/模型文件     + 图表/邮件       (头部15%综合评分)
                                ↑
                          config.yaml
                      (单一配置源, ML+DL混用)
```

## 工作流

### 1. 超参搜索

对每个模型类型执行 Optuna 贝叶斯搜索：

```bash
# 搜索全部模型
python code/src/train_search_v2.py

# 搜索单个模型
python code/src/train_search_v2.py --model-type patchtst

# 如果搜索空间已变更，用 --fresh 清除旧study
python code/src/train_search_v2.py --model-type itransformer --fresh
```

输出到 `model/bayes_{model_type}_{N}_{topk}_{date}/exp_*/`。

### 2. 批量回测 + 模型选择

在 `notebooks/reproduce_backtest.ipynb` 中完成：

- **Cell 1-8**: 遍历所有实验 (DL + ML)，生成预测缓存 + 回测
- **Cell 12**: 对 Top 模型测试 6 种权重策略
- **Cell 14**: 多维度策略对比分析（收益/回撤/胜率/稳定性/偏度）
- **Cell 16**: **选模型** — 分模型类别取头部 15%，综合评分推荐

#### 选模型评分方法

```
对每组 (model_type, strategy):
  1. 按 return 排序，取 top 15% 为"头部组"
  2. 评分公式：
     head_score  = return² / dd × (1 + avg_return/100)
  3. 头部组内取 head_score 最高的具体 exp
  4. 策略总评 = 各组 composite_score 的中位数
```

#### 6 种权重策略

| 策略 | 说明 | 参数 |
|------|------|------|
| equal | 等权分配 | — |
| softmax | Softmax 概率权重 | temperature |
| rank_linear | 线性排名权重 | — |
| risk_parity | 风险平价（波动率倒数） | vol_window |
| score_risk | 评分修正风险平价（score/vol²） | vol_window |
| score_risk_v1 | 评分修正风险平价（score/vol） | vol_window |
| kelly | Kelly 最优增长 | — |
| liquidity | 流动性优先 | — |

### 3. 每日测评

5 种运行模式：

```bash
# Mode 0: 全流程（更新数据→预测→回测→发日报）
python code/src/daily_eval.py

# Mode 1: 仅更新数据
python code/src/daily_eval.py --update-only

# Mode 2: 仅生成预测信号
python code/src/daily_eval.py --predictions-only

# Mode 3: 从已有预测做回测+日报
python code/src/daily_eval.py --from-predictions

# Mode 4: 从Juejin状态生成日报
python code/src/daily_eval.py --from-juejin
```

### 4. 配置

`config.yaml` 是单一配置源，支持 ML + DL 混用：

```yaml
models:
  - dir: juejin/live/bayes_patchtst_74_3_2026-01-01_2026-03-31/exp_27
    file: best_model_optuna.pth
    type: dl
    enabled: true
  - dir: model/bayes_lightgbm_74_3_2026-01-01_2026-03-31/exp_12
    file: model.pkl
    type: lightgbm
    enabled: true

average: true
voting: true
master: "first"

weight_strategy: "risk_parity"   # equal | softmax | rank_linear | risk_parity | score_risk | score_risk_v1 | kelly | liquidity
top_k: 3
trade_mode: "open"
```

### 5. 数据获取

```bash
# JoinQuant (Playwright)
python code/src/get_etf_data.py

# 或使用掘金API
python juejin/download_etf_data.py
```

## 支持的模型类型

### 深度学习 (10种)

| 模型 | 说明 |
|------|------|
| PatchTST | Patch 时序 Transformer（当前最优） |
| iTransformer | 改进 Transformer，关注序列维度变换 |
| Transformer | 标准 Transformer 编码器 |
| TCN | 时序卷积网络 |
| LSTM | 长短期记忆网络 |
| GRU | 门控循环单元 |
| DLinear | 线性分解模型 |
| NLinear | 归一化线性模型 |
| TimesNet | 时序二维变换 |
| Mamba | 状态空间模型 |

### 机器学习 (3种)

| 模型 | 类型 |
|------|------|
| XGBoost | 梯度提升树排名器 |
| LightGBM | 高效梯度提升树排名器 |
| CatBoost | 类别特征增强梯度提升器 |

所有模型通过 `models/factory.py` 统一创建。

## 报告输出

每个交易日生成：

| 文件 | 说明 |
|------|------|
| `output/latest_report.html` | HTML 报告 |
| `output/latest_report.json` | 结构化报告数据 |
| `output/history_report/{date}.html` | 历史调仓日报告 |
| `output/equity_curves.png` | 多模型收益曲线对比 |
| `output/portfolio.json` | 当前持仓快照 |
| `output/predictions.json` | 模型预测信号 |
| `output/backtest_state.json` | 完整回测状态（6序列） |

### 报告指标

- 策略总收益 / 年化收益 / 超额收益
- 沪深300 基准收益
- 日胜率 / 夏普比率 / 卡玛比率 / 索提诺比率
- 最大回撤区间与持续时间
- 近 5 天 / 近 1 月窗口表现
- 今日 P&L（总值 + 各持仓明细）

## 技术栈

- **Python 3.10+** + PyTorch 2.6
- **Optuna** 贝叶斯超参搜索（TPESampler + SQLite 持久化）
- **Pandas / NumPy** 数据处理
- **Matplotlib / Seaborn** 图表绘制（notebook 分析）
- **SMTP** 邮件报告发送（QQ邮箱）
- **Jupyter** 批量回测分析与模型选择

## 环境要求

- Python >= 3.10, < 3.13
- CUDA 12.8（Linux）或 CPU（Windows）
- 依赖见 `pyproject.toml`

```bash
uv sync
```

## 项目结构

```
├── code/src/                 # 核心代码
│   ├── backtest.py           # 回测引擎
│   ├── config.py             # 模型参数与搜索空间
│   ├── daily_eval.py         # 每日测评入口（5种模式）
│   ├── ml_backtester.py      # ML 模型回测封装
│   ├── models/               # 10种 DL 模型实现
│   ├── predict.py            # 模型预测
│   ├── select_models.py      # 模型选择（旧方式）
│   ├── send_report.py        # 邮件报告 + HTML 构建
│   ├── train.py              # DL 模型训练
│   ├── train_search_v2.py    # Optuna 超参搜索
│   ├── train_ml.py           # ML 模型训练
│   ├── train_ml_search.py    # ML 超参搜索
│   ├── utils.py              # 数据处理（158+因子）
│   └── config.yaml → 根目录  # 单一配置源
├── notebooks/
│   └── reproduce_backtest.ipynb  # 批量回测 + 策略对比 + 选模型
├── juejin/                   # 掘金量化集成
├── etf_data/                 # ETF 日线数据
├── model/                    # 训练好的模型权重
├── output/                   # 报告与回测状态
└── config.yaml               # 主配置文件
```

## TODO

- 交易记录的仓位字段 
