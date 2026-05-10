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
         train_search_v2    select_models    daily_eval.ps1
         (贝叶斯超参搜索)    (模型筛选)       (每日测评)
               │                │                 │
               ↓                ↓                 ↓
          model/search_*   model_selection    output/*.json
                .yaml            + 图表/邮件
```

## 工作流

### 1. 超参搜索 (`search.ps1`)
对每个模型类型 (itransformer, gru, tcn, lstm 等) 执行 Optuna 贝叶斯搜索或网格搜索，输出到 `model/search_{model_type}_{N}_{topk}/exp_*/`。

```powershell
.\search.ps1
```

### 2. 模型选择 (`select_models.py`)
从所有搜索目录中按验证集分数挑选最佳实验，写入 `output/model_selection.yaml`。

```powershell
python code/src/select_models.py --top-n 5
python code/src/select_models.py --manual exp_54 exp_64 exp_6
```

### 3. 每日测评 (`daily_eval.ps1`)
加载 `model_selection.yaml` 中启用的模型，对全部 74 只 ETF 进行回测，计算 equity curve + 各项指标，生成 HTML 报告并自动发送邮件。

```powershell
# 初始化（清理旧数据，指定起始日）
.\daily_eval.ps1 Init

# 更新数据并运行
.\daily_eval.ps1 Update

# 仅运行（不更新数据）
.\daily_eval.ps1 Run
```

### 4. 数据获取 (`get_etf_data.py`)
从 joinquant 下载 74 只 ETF 的日线数据，自动合并到 `etf_74.csv`。

```powershell
python code/src/get_etf_data.py
```

## 多模型模式

`output/model_selection.yaml` 控制回测的模型组合：

```yaml
average: true          # 启用平均模型（各模型分数等权平均）
voting: true           # 启用投票模型（按top-k出现频率排名）
master: first          # 报告默认展示第一个模型序列

models:
  - dir: model/search_itransformer_74_3/exp_54
    file: best_model.pth
    enabled: true
  - dir: model/search_itransformer_74_3/exp_64
    file: best_model.pth
    enabled: true
```

- **单模型**: 每个 exp 独立回测，显示各自的持仓和交易
- **Average**: 取所有 exp 模型得分的均值，按均值排序选股
- **Voting**: 统计各 ETF 被 exp 模型选入 top-k 的频率，按频率排序选股

## 报告输出

每个交易日自动生成：

| 文件 | 说明 |
|------|------|
| `output/latest_report.json` | 结构化报告数据（API 消费） |
| `output/latest_report.html` | HTML 报告 |
| `output/history_report/{date}.html` | 历史调仓日报告 |
| `output/equity_curves.png` | 多模型收益曲线对比图 |
| `output/portfolio.json` | 当前持仓快照 |
| `output/model_selection.yaml` | 模型选择配置 |

### 报告指标

- 策略总收益 / 年化收益 / 超额收益
- 沪深300 基准收益
- 日胜率 / 夏普比率 / 卡玛比率 / 索提诺比率
- 最大回撤区间与持续时间
- 近 5 天 / 近 1 月窗口表现
- 今日 P&L（总值 + 各持仓明细）
- 调仓盈亏（持仓成本 vs 市价）

### 交易表优势信号

每笔买入交易附带 **优势** 列，量化选股置信度：
- **单模型 / Average**: `(score_i − score_cutoff) / score_std` — z-score 标准化
- **Voting**: `votes_i − votes_{k+1}` — 票数原始差值

## 支持的模型类型

| 模型 | 说明 |
|------|------|
| iTransformer | 改进版 Transformer，关注序列维度变换 |
| Transformer | 标准 Transformer 编码器 |
| TCN | 时序卷积网络 |
| LSTM | 长短期记忆网络 |
| GRU | 门控循环单元 |
| DLinear | 线性分解模型 |
| TimesNet | 时序二维变换 |
| MMoE | 多门控专家混合 |

所有模型通过 `models/factory.py` 统一创建，支持 `model.py:create_model()` 接口。

## 技术栈

- **Python 3.10+** + PyTorch 2.6
- **Optuna** 贝叶斯超参搜索（TPESampler + SQLite 持久化）
- **Pandas / NumPy** 数据处理
- **Matplotlib** 图表绘制
- **SMTP** 邮件报告发送（QQ邮箱）
- **PowerShell** 自动化脚本（任务计划）

## 环境要求

- Python >= 3.10, < 3.13
- CUDA 12.8（Linux）或 CPU（Windows）
- 依赖见 `pyproject.toml`

### 安装

```powershell
uv sync
.\init.ps1
```

### 定时任务

```powershell
.\setup_daily_task.ps1
```

## 项目结构

```
├── code/src/                 # 核心代码
│   ├── backtest.py           # 回测引擎
│   ├── config.py             # 模型参数与搜索空间
│   ├── daily_eval.py         # 每日测评入口
│   ├── model.py              # 模型创建接口
│   ├── models/               # 各模型实现
│   ├── predict.py            # 预测
│   ├── select_models.py      # 模型选择
│   ├── send_report.py        # 邮件报告 + HTML 构建
│   ├── train.py              # 模型训练 + RankingDataset
│   ├── train_search_v2.py    # Optuna 超参搜索
│   └── utils.py              # 数据处理工具
├── etf_data/                 # ETF 数据
│   ├── etf_74.csv            # 主数据文件
│   └── etf_list_before_2022_74.csv  # ETF 列表
├── model/                    # 训练好的模型
│   └── search_*/exp_*/       # 实验目录
├── output/                   # 输出报告
│   ├── latest_report.json
│   ├── latest_report.html
│   ├── history_report/
│   ├── equity_curves.png
│   └── model_selection.yaml
├── notebooks/                # Jupyter 分析
├── *.ps1                     # PowerShell 脚本
└── pyproject.toml            # 项目配置
```


# TODO

日报中的模型预热

模型失效预警

<!-- 日报的交易曲线的HTML可视化 -->

围绕NDCG建模

<!-- 今日调仓的优化需要更直观 -->

<!-- 历史日报除了调仓日也要生成,调仓日在命名中为 日期(调仓日) -->

<!-- 冗余计算改用储存到交易记录而非重新计算 -->