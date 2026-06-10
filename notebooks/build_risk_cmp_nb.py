import json, os

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "cells": []
}

_NB_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_NB_DIR)

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [l + "\n" if not l.endswith("\n") else l for l in src.split("\n")]}

def md(src):
    return {"cell_type": "markdown", "metadata": {},
            "source": [l + "\n" if not l.endswith("\n") else l for l in src.split("\n")]}

nb["cells"].append(md("""# 止损策略对比回测

对比 `BacktestEngine` 在不同止损策略下的表现。
使用已保存的 predictions.json（推荐模型 `tcn_43 + gru_14 + dlinear_22 + tcn_54` 的融合 average 序列）。"""))

nb["cells"].append(code(f"""import os, sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.chdir({repr(_NB_DIR)})
sys.path.insert(0, '../code/src')

from backtest import BacktestEngine
from risk_strategies import (
    NoRiskControl, MarketBreadthStrategy, VolatilityTargetStrategy,
    TrendFilterStrategy, DrawdownStopStrategy
)

np.random.seed(42)"""))

nb["cells"].append(code("""# ── 加载 ETF 数据 ──
data_file = '../etf_data/etf_74.csv'
raw = pd.read_csv(data_file, dtype={'股票代码': str})
raw['股票代码'] = raw['股票代码'].astype(str).str.zfill(6)
raw['日期'] = pd.to_datetime(raw['日期'])
print(f'ETF: {raw["股票代码"].nunique()} 只, {raw["日期"].nunique()} 天')

# ── 加载 predictions ──
with open('../output/predictions.json') as f:
    all_preds = json.load(f)

meta = all_preds.pop('_meta', {})
backtest_dates_str = meta.get('backtest_dates', [])
print(f'预测数据: {len(all_preds)} 个模型/序列, {len(backtest_dates_str)} 天')

# ── 构造 predictions_func: 使用 average 序列 ──
avg_preds = all_preds.get('average')
if not avg_preds:
    # fallback: 取第一个模型
    avg_preds = list(all_preds.values())[0]
    print('⚠️ 未找到 average 序列, 使用第一个模型替代')

def predictions_func(date):
    d = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
    # 预测日期用前一日
    entries = avg_preds.get(d)
    if entries is None:
        return None
    # 确保按 score 降序
    sorted_entries = sorted(entries, key=lambda x: x['score'], reverse=True)
    return sorted_entries

# 测试
test_date = pd.Timestamp('2026-04-01')
test_pred = predictions_func(test_date)
print(f'测试预测 ({test_date}): {len(test_pred) if test_pred else 0} 只')"""))

nb["cells"].append(code("""# ── 配置 ──
INITIAL_CAPITAL = 100000
TOP_K = 3
REBALANCE_DAYS = 5
POSITION_PCT = 0.95
COMMISSION = 0.0003
SLIPPAGE = 0.001
TRADE_MODE = "open"

start_date = meta.get('start_date', '2026-04-01')
end_date = backtest_dates_str[-1] if backtest_dates_str else '2026-06-10'

# ── 准备回测日期 ──
all_dates = sorted(raw['日期'].unique())
start_ts = pd.Timestamp(start_date)
end_ts = pd.Timestamp(end_date)
bt_dates = [d for d in all_dates if start_ts <= d < end_ts]
print(f'回测区间: {start_date} ~ {end_date}  ({len(bt_dates)} 天)')"""))

nb["cells"].append(code("""# ── 运行回测 ──
def run_one(name, risk_manager_config):
    engine = BacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        commission=COMMISSION,
        slippage=SLIPPAGE,
        top_k=TOP_K,
        position_pct=POSITION_PCT,
        risk_manager_config=risk_manager_config,
    )
    engine.run(
        dates=bt_dates,
        price_data=raw,
        predictions_func=predictions_func,
        rebalance_days=REBALANCE_DAYS,
        first_rebalance_date=start_ts,
        trade_mode=TRADE_MODE,
    )
    ec = engine.equity_curve
    total_value = ec[-1]['total_value'] if ec else INITIAL_CAPITAL
    ret = (total_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    values = [e['total_value'] for e in ec]
    peak = max(np.maximum.accumulate(values))
    dd = (peak - values[-1]) / peak * 100 if peak > 0 else 0
    n_trades = len(engine.trades)
    print(f'  {name:<30s} 收益={ret:>6.2f}%  回撤={dd:>5.2f}%  交易={n_trades}')
    return {'name': name, 'equity_curve': ec, 'return': ret, 'max_dd': dd, 'trades': n_trades, 'engine': engine}

configs = [
    ('无风控 (none)',             {'enabled': False}),
    ('市场广度 (market_breadth)',  {'enabled': True, 'strategy': 'market_breadth', 'params': {'lookback_days': 20, 'high_threshold': 0.30, 'low_threshold': 0.10}}),
    ('波动率目标 (volatility)',    {'enabled': True, 'strategy': 'volatility_target', 'params': {'lookback_days': 20, 'n_std': 1.0, 'max_std': 2.0}}),
    ('趋势过滤 (trend_filter)',    {'enabled': True, 'strategy': 'trend_filter', 'params': {'fast': 20, 'slow': 60, 'entry_threshold': 1.0, 'exit_threshold': 0.98}}),
    ('回撤止损 (drawdown_stop)',   {'enabled': True, 'strategy': 'drawdown_stop', 'params': {'dd_low': 0.05, 'dd_high': 0.10}}),
]

print(f'回测 5 种策略...')
results = []
for name, cfg in configs:
    r = run_one(name, cfg)
    results.append(r)

print()
print('=' * 60)
print('策略对比汇总')
print('=' * 60)
print(f'{"策略":<30s} {"收益率":>8s} {"最大回撤":>8s} {"交易次数":>8s}')
print('-' * 60)
for r in sorted(results, key=lambda x: x['return'], reverse=True):
    print(f'{r["name"]:<30s} {r["return"]:>7.2f}%  {r["max_dd"]:>7.2f}%  {r["trades"]:>8d}')"""))

nb["cells"].append(code("""# ── 累计收益曲线对比 ──
fig, ax = plt.subplots(figsize=(14, 6))
colors = ['#333333', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for r, c in zip(results, colors):
    ec = r['equity_curve']
    if not ec:
        continue
    values = np.array([e['total_value'] for e in ec])
    ret = (values - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    days = np.arange(len(ret))
    ax.plot(days, ret, lw=1.5, label=f"{r['name']} ({r['return']:+.2f}%)", color=c)

ax.axhline(0, color='black', ls='--', lw=0.5)
ax.set_xlabel('交易日')
ax.set_ylabel('累计收益率 (%)')
ax.set_title('止损策略对比: 累计收益率', fontsize=13, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()"""))

nb["cells"].append(code("""# ── 风险指标仪表板 ──
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

names = [r['name'][:20] for r in results]
returns = [r['return'] for r in results]
dds = [r['max_dd'] for r in results]
trades = [r['trades'] for r in results]

# 收益率柱状图
axes[0].barh(names, returns, color=['#333333','#1f77b4','#ff7f0e','#2ca02c','#d62728'])
axes[0].set_title('累计收益率 (%)')
axes[0].axvline(0, color='black', lw=0.5)

# 最大回撤
axes[1].barh(names, dds, color=['#333333','#1f77b4','#ff7f0e','#2ca02c','#d62728'])
axes[1].set_title('最大回撤 (%)')
axes[1].axvline(0, color='black', lw=0.5)

# 交易次数
axes[2].barh(names, trades, color=['#333333','#1f77b4','#ff7f0e','#2ca02c','#d62728'])
axes[2].set_title('交易次数')

for ax in axes:
    ax.grid(True, alpha=0.2, axis='x')

plt.tight_layout()
plt.show()"""))

nb["cells"].append(code("""# ── 市场广度策略: 仓位乘数可视化 ──
breadth_cfg = {'enabled': True, 'strategy': 'market_breadth', 'params': {'lookback_days': 20, 'high_threshold': 0.30, 'low_threshold': 0.10}}
breadth_engine = BacktestEngine(
    initial_capital=INITIAL_CAPITAL, commission=COMMISSION, slippage=SLIPPAGE,
    top_k=TOP_K, position_pct=POSITION_PCT,
    risk_manager_config=breadth_cfg, log=False,
)
breadth_engine.run(
    dates=bt_dates, price_data=raw, predictions_func=predictions_func,
    rebalance_days=REBALANCE_DAYS, first_rebalance_date=start_ts, trade_mode=TRADE_MODE,
)

# 从风控日志重建仓位乘数序列
strategy = MarketBreadthStrategy(breadth_cfg['params'])
multipliers = []
multi_dates = []
for d in bt_dates:
    m = strategy.get_multiplier(d, raw, {}, [])
    multipliers.append(m)
    multi_dates.append(d)

fig, ax1 = plt.subplots(figsize=(14, 5))

# 净值曲线
ec = breadth_engine.equity_curve
values = np.array([e['total_value'] for e in ec])
rets = (values - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
ax1.plot(np.arange(len(rets)), rets, 'b-', lw=1.5, label='累计收益 (市场广度)')
ax1.set_ylabel('累计收益率 (%)', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# 仓位乘数
ax2 = ax1.twinx()
ax2.fill_between(np.arange(len(multipliers)), multipliers, alpha=0.3, color='orange')
ax2.plot(np.arange(len(multipliers)), multipliers, 'orange', lw=1, label='仓位乘数')
ax2.set_ylabel('仓位乘数 (0~1)', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')
ax2.set_ylim(-0.05, 1.05)

# 调仓日标记
rebalance_indices = list(range(0, len(bt_dates), REBALANCE_DAYS))
for idx in rebalance_indices:
    if idx < len(bt_dates):
        ax1.axvline(idx, color='gray', lw=0.5, ls=':', alpha=0.3)

ax1.set_xlabel('交易日')
ax1.set_title('市场广度策略: 仓位乘数 vs 累计收益 (竖线=调仓日)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.2)
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9), fontsize=9)
plt.tight_layout()
plt.show()"""))

with open("notebooks/risk_cmp.ipynb", "w") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
    f.write("\n")
print(f"Created: {len(nb['cells'])} cells")
