# Workflow

```mermaid
flowchart TD
    A["`python daily_eval.py`"] --> B{`--clear`?}
    B -->|yes| Z[删除 output/*<br>保留 model_selection.yaml]
    Z --> C{模式选择}
    B -->|no| C

    C -->|无参数| D0[Mode 0 全流程]
    D0 --> D0a[update_etf_data]
    D0a --> D0b[generate_predictions]
    D0b --> D0c[run_backtest_sequence<br>6序列: 3模型+average+voting+juejin]
    D0c --> D0d[保存 backtest_state.json]
    D0d --> D0e[发送日报邮件]

    C -->|--update-only| D1[Mode 1 仅更新数据]
    D1 --> D1a[update_etf_data]
    D1a --> D1b[保存 etf_74.csv]

    C -->|--predictions-only| D2[Mode 2 仅预测]
    D2 --> D2a[generate_predictions_only]
    D2a --> D2b[输出 predictions.json<br>model_selection.yaml]

    C -->|--from-predictions| D3[Mode 3 从预测回测]
    D3 --> D3a[读 predictions.json]
    D3a --> D3b[run_backtest_sequence<br>6序列]
    D3b --> D3c[保存 backtest_state.json]
    D3c --> D3d[发送日报邮件]

    C -->|--from-juejin| D4[Mode 4 从状态恢复]
    D4 --> D4a[读 backtest_state.json<br>合并 juejin_state.json]
    D4a --> D4b[_resolve_report_key<br>选主序列]
    D4b --> D4c[读 etf_74.csv<br>构造持仓+日报]
    D4c --> D4d[发送日报邮件]

    subgraph run_backtest_sequence 内部
        RBS[遍历6序列] --> RBS1[BacktestEngine.run]
        RBS1 --> RBS2[收集 today_pnl<br>trades<br>pre_rebalance_positions]
        RBS2 --> RBS3[合并返回]
    end
```

## Daily Report Generation (5 modes)

Any mode supports `--clear` to delete old output files first (preserves `model_selection.yaml`):
```bash
python code/src/daily_eval.py --clear --from-juejin
```

### Mode 0: Full pipeline (模型推理 + 回测 + 日报)
```bash
python code/src/daily_eval.py
```
流程: `update_etf_data()` → `generate_predictions()` → `run_backtest_sequence()`(6序列: 3模型+average+voting+juejin占位) → 保存 `backtest_state.json` → `send_report()`

### Mode 1: Update data only (仅更新数据)
```bash
python code/src/daily_eval.py --update-only
```
只下载更新 ETF 日线数据到 `etf_74.csv`。

### Mode 2: Predictions only (仅模型推理，保存预测信号)
```bash
python code/src/daily_eval.py --predictions-only
```
Saved to `output/predictions.json` + `model_selection.yaml`. Skips backtest/report.

### Mode 3: Report from saved predictions (用已保存的预测信号)
```bash
python code/src/daily_eval.py --from-predictions
```
读 `predictions.json` → `run_backtest_sequence()`(6序列) → 保存 `backtest_state.json` → 生成并发送日报。

### Mode 4: Report from juejin state (从回测状态生成日报)
```bash
python code/src/daily_eval.py --from-juejin
```
读 `backtest_state.json` + 合并 `juejin_state.json` → `_resolve_report_key()` 选主序列 → 读 `etf_74.csv` → 构造持仓(close+调仓日→旧持仓) → 生成并发送日报。

### Typical three-step workflow
```bash
# Step 1: update data
python code/src/daily_eval.py --update-only

# Step 2: generate predictions (slow, needs model)
python code/src/daily_eval.py --predictions-only --no-update

# Step 3: generate report from saved predictions (fast)
python code/src/daily_eval.py --from-predictions --no-update
```

### Juejin-based workflow
```bash
# Step 1: update data
python code/src/daily_eval.py --update-only

# Step 2: generate predictions + local backtest
python code/src/daily_eval.py --from-predictions --no-update

# Step 3: run Juejin backtest
python juejin/main.py

# Step 4: generate report from juejin state
python code/src/daily_eval.py --from-juejin
```

## config.yaml 配置

单一配置源，支持 ML + DL 混合模型。详见根目录 `config.yaml`。

关键字段：
- `models`: 模型列表，每个支持 `type: dl|xgb|lightgbm|catboost`
- `weight_strategy`: 加权策略（equal/softmax/rank_linear/risk_parity/score_risk）
- `master`: 主序列（first/juejin/average/voting/模型key）

## 模型选择 (reproduce_backtest.ipynb Cell 16)

批量回测后，在 notebook 中执行模型选择：

```
对每个 (model_type, strategy) 组合:
  1. 按 return 取 top 15% 为头部组
  2. 头部组内按 score = return² / dd × (1 + avg_return/100) 排序
  3. 取每组第1名作为该组合的最佳实验
  4. 输出:
     - 策略总评表（各策略在所有模型类别上的中位数得分）
     - 模型类别 Top 3 推荐表
     - 全推荐排名表（按 ranking 排序）
```

## Juejin Strategy

### juejin/main.py
- Reads `output/predictions.json` (raw model scores)
- Picks Top-K by score each rebalance day
- Executes trades via Juejin API
- Saves results as `"juejin"` sequence in `output/juejin_state.json`

### Sequence priority (report_key resolution)
1. `model_selection.yaml` → `master: juejin` (explicit)
2. Fallback: `juejin` → `average` → `voting` → first

## `run_backtest_sequence()` 内部
遍历 6 个序列（3 单模型 + average + voting + juejin）→ 每序列 `BacktestEngine.run()` → 收集 `today_pnl`、`trades`、`pre_rebalance_positions` → 合并返回。

## Critical: predictions.json must cover ALL trading days

When `trade_mode="open"`, `BacktestEngine` calls `predictions_func(pred_date)` where `pred_date` = **previous trading day**. If `predictions.json` only has rebalance-day entries, the lookup returns `None` and the rebalance is skipped → 0 trades.

**Always regenerate predictions.json via `--predictions-only` after any data or model change.**
```bash
python code/src/daily_eval.py --predictions-only --no-update
```

## Optuna 搜索空间变更

如果 `config.py` 中的搜索空间（`get_search_space()`）发生过变更，已存在的 Optuna study 会拒绝新 trial。
使用 `--fresh` 重置：
```bash
python code/src/train_search_v2.py --model-type itransformer --fresh
```

## Test Commands
```bash
python -m py_compile code/src/daily_eval.py
python -m py_compile code/src/send_report.py
python -m py_compile juejin/main.py
```

## Compare --live-only 参数详解

```bash
# 默认模式: 扫描 juejin/live 所有实验，测试全部 3 个 .pth 文件，使用 weight_strategy="equal"
python code/src/model_manager.py compare --live-only

# --use-config 模式: 仅测试 config.yaml 中启用的 (model_dir, model_file)，使用 config 的 weight_strategy
python code/src/model_manager.py compare --live-only --use-config
```

参数:
- `--use-config`: 从 config.yaml 读取模型列表和 `weight_strategy` + `strategy_params`；仅测试启用模型且仅测试配置指定的 `.pth` 文件
- `--repro-val-start` (默认 `2025-01-01`)
- `--repro-val-end` (默认 `2025-06-30`)

### Return 对比根因

`compare --live-only` 当前模型收益(~13%) vs 日报收益(~30%+)的差异根因:
1. **硬编码 `weight_strategy="equal"`** (主因): `_backtest_one_model()` 中 hardcoded，而 config 使用 `risk_parity`。修复后 → ~22.5%。
2. **全部 3 个 `.pth` 取平均**: 稀释了最优文件。`--use-config` 仅测配置指定的文件。
3. **残余差距**: `feature_num`/scaler 等数据加载参数差异，及日报可能使用融合预测(average/voting/master first)。
