# Workflow

## Daily Report Generation (3+1 modes)

Any mode supports `--clear` to delete old output files first (preserves `model_selection.yaml`):
```bash
python code/src/daily_eval.py --clear --from-state
```

## Daily Report Generation (3+1 modes)

### Mode 0: Full pipeline (模型推理 + 回测 + 日报)
```bash
python code/src/daily_eval.py
```

### Mode 1: Update data only (仅更新数据)
```bash
python code/src/daily_eval.py --update-only
```

### Mode 2: Predictions only (仅模型推理，保存预测信号)
```bash
python code/src/daily_eval.py --predictions-only
```
Saved to `output/predictions.json`. Skips backtest/report.

### Mode 3: Report from saved predictions (用已保存的预测信号)
```bash
python code/src/daily_eval.py --from-predictions
```
Loads `output/predictions.json`, runs fast backtest (no model inference), generates report.

### Mode 4: Report from backtest state (从回测状态生成日报)
```bash
python code/src/daily_eval.py --from-state
```
Reads `output/backtest_state.json`, generates report. 用于掘金回测完成后生成日报。

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

# Step 2: generate predictions + local backtest (6 sequences: 3 models + average + voting + juejin placeholder)
python code/src/daily_eval.py --from-predictions --no-update

# Step 3: run Juejin backtest (reads predictions.json, merges "juejin" sequence into state)
python juejin/main.py

# Step 4: generate report from backtest state (master: juejin → 掘金作主序列)
python code/src/daily_eval.py --from-state
```

## Juejin Strategy

### juejin/main.py
- Reads `output/predictions.json` (raw model scores)
- Picks Top-K by score each rebalance day
- Executes trades via Juejin API
- Saves results as `"juejin"` sequence in `output/juejin_state.json` (独立文件)
- Also saves `output/juejin_result.json` for reference

### After Juejin backtest finishes
```bash
# Generate report with juejin as master sequence:
python code/src/daily_eval.py --from-state
```

### Sequence priority (report_key resolution)
1. `model_selection.yaml` → `master: juejin` (explicit)
2. Fallback: `juejin` → `average` → `voting` → first

## Test Commands
```bash
python -m py_compile code/src/daily_eval.py
python -m py_compile code/src/send_report.py
python -m py_compile juejin/main.py
```
