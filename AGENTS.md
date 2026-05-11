# Workflow

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

# Step 2: generate predictions (slow, needs model)
python code/src/daily_eval.py --predictions-only --no-update

# Step 3: run Juejin backtest (reads predictions.json, saves backtest_state.json)
python juejin/main.py

# Step 4: generate report from backtest state (no model, no backtest)
python code/src/daily_eval.py --from-state
```

## Juejin Strategy

### juejin/main.py
- Reads `output/predictions.json` (raw model scores)
- Picks Top-K by score each rebalance day
- Executes trades via Juejin API
- Saves results to `output/backtest_state.json` and `output/juejin_result.json`

### After Juejin backtest finishes
```bash
# Generate report from juejin's backtest results:
python code/src/daily_eval.py --from-state
```

### Sync specific fields manually
```bash
python juejin/sync_report.py --from-juejin output/juejin_result.json --send-email
python juejin/sync_report.py --override-cash 50000 --override-total-value 135000 --send-email
```

### Dry run (preview without saving)
```python
from juejin.sync_report import apply_juejin_result, load_report
import copy

report = load_report()
result = apply_juejin_result("juejin_result.json", copy.deepcopy(report), dry_run=True)
```

## Test Commands
```bash
python -m py_compile code/src/daily_eval.py
python -m py_compile code/src/send_report.py
python -m py_compile juejin/main.py
python -m py_compile juejin/sync_report.py
```
