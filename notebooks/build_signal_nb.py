import json, os

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "cells": []
}

_NB_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_NB_DIR)
_TSCV_DIR = os.path.join(_ROOT, 'model/TSCV')
_ETF_PATH = os.path.join(_ROOT, 'etf_data/etf_74.csv')
_OUT_PATH = os.path.join(_ROOT, 'output/signal_vs_random_summary.csv')

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [l + "\n" if not l.endswith("\n") else l for l in src.split("\n")]}

def md(src):
    return {"cell_type": "markdown", "metadata": {},
            "source": [l + "\n" if not l.endswith("\n") else l for l in src.split("\n")]}

# ── 0 ──
nb["cells"].append(md("""# 模型信号 vs 随机排序（多折交叉验证）

仅关注 Top-3 多头等权持有，不看空头端。
使用 TSCV 4 折全部验证期数据，评估跨市场时期的模型稳定性。
比较真实实验 vs 10000 次随机选 3 只的差异。"""))

# ── 1 ──
nb["cells"].append(code(f"""import os, sys, json, glob
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

os.chdir({repr(_NB_DIR)})

TSCV_DIR = '../model/TSCV'
_OUT_DIR = '../output'
np.random.seed(42)"""))

# ── 2 ──
nb["cells"].append(code("""etf = pd.read_csv('../etf_data/etf_74.csv')
etf_sorted = etf.sort_values(['股票代码', '日期']).copy()
etf_sorted['_next_close'] = etf_sorted.groupby('股票代码')['收盘'].shift(-1)
etf_sorted['_next_open'] = etf_sorted.groupby('股票代码')['开盘'].shift(-1)
etf_sorted['fwd_ret'] = (etf_sorted['_next_close'] - etf_sorted['_next_open']) / etf_sorted['_next_open']
etf_sorted['日期'] = pd.to_datetime(etf_sorted['日期'])
fwd_lookup = etf_sorted[['股票代码', '日期', 'fwd_ret']].dropna()

all_dates = sorted(pd.to_datetime(etf['日期'].unique()))
date_to_idx = {d: i for i, d in enumerate(all_dates)}
idx_to_date = {i: d for d, i in date_to_idx.items()}
print(f'ETF: {etf["股票代码"].nunique()} 只, {len(all_dates)} 天')"""))

# ── 3 ──
nb["cells"].append(code("""# 遍历所有实验, 多折合并, 逐日计算 Top-3 收益
real_records = []  # 每个元素: {model_type, exp_idx, daily_returns, daily_top3,
                   #             fold_returns, fold_top3s, fold_boundaries}
etf_list = sorted(etf['股票代码'].unique())
stock_to_idx = {s: i for i, s in enumerate(etf_list)}
FOLD_ORDER = ['fold_0', 'fold_1', 'fold_2', 'fold_3']

def _fold_rets(vp_dict):
    \"\"\"从一组预测计算逐日 Top-3 收益.\"\"\"
    rets, top3 = [], []
    for d_str, entries in sorted(vp_dict.items()):
        dt = pd.to_datetime(d_str)
        idx = date_to_idx.get(dt)
        if idx is None: continue
        next_dt = idx_to_date.get(idx + 1)
        if next_dt is None: continue
        fwd_data = fwd_lookup[fwd_lookup['日期'] == next_dt]
        if len(fwd_data) < 10: continue
        scores = sorted(entries, key=lambda x: x['score'], reverse=True)[:3]
        top3_ids = [e['stock_id'] for e in scores]
        _rets = []
        for sid in top3_ids:
            match = fwd_data[fwd_data['股票代码'] == sid]
            _rets.append(match['fwd_ret'].iloc[0] if len(match) > 0 else 0)
        rets.append(np.mean(_rets))
        top3.append(top3_ids)
    return np.array(rets), top3

for mt_dir in sorted(glob.glob(f'{TSCV_DIR}/*/')):
    full_name = os.path.basename(os.path.normpath(mt_dir))
    parts = full_name.split('_')
    mt_short = parts[1] if len(parts) >= 2 else full_name
    for exp_dir in sorted(glob.glob(f'{mt_dir}exp_*/')):
        exp_idx = int(os.path.basename(os.path.normpath(exp_dir)).split('_')[1])

        # Load all folds
        fold_vps = {}  # fold_name -> vp_dict
        for fn in FOLD_ORDER:
            fp = os.path.join(exp_dir, fn, 'val_predictions.json')
            if os.path.exists(fp):
                fold_vps[fn] = json.load(open(fp))

        if not fold_vps:
            continue

        # Merged predictions (folds are non-overlapping by TSCV design)
        merged = {}
        for fn in FOLD_ORDER:
            if fn in fold_vps:
                merged.update(fold_vps[fn])

        daily_returns, daily_top3 = _fold_rets(merged)
        if len(daily_returns) < 10:
            continue

        # Per-fold (also tracks valid-day boundaries for plotting)
        fold_returns = {}
        fold_top3s = {}
        boundaries = []
        total_valid = 0
        for fn in FOLD_ORDER:
            if fn in fold_vps:
                fr, ft = _fold_rets(fold_vps[fn])
                if len(fr) >= 5:
                    boundaries.append((fn, total_valid))
                    total_valid += len(fr)
                    fold_returns[fn] = fr
                    fold_top3s[fn] = ft

        # Load tscv_results.json for cross-fold Sharpe comparison
        sr = {'cv_score': None, 'fold_sharpe': None, 'mean_sharpe': None, 'std_sharpe': None}
        sr_path = os.path.join(exp_dir, 'tscv_results.json')
        if os.path.exists(sr_path):
            sr_data = json.load(open(sr_path))
            if sr_data.get('success'):
                sr = {k: sr_data.get(k) for k in sr}

        real_records.append({
            'model_type': mt_short, 'exp_idx': exp_idx,
            'daily_returns': daily_returns,
            'daily_top3': daily_top3,
            'fold_returns': fold_returns,
            'fold_top3s': fold_top3s,
            'fold_boundaries': boundaries,
            'search_results': sr,
        })

print(f'加载了 {len(real_records)} 个实验')
if real_records:
    print(f'合并总天数: {len(real_records[0][\"daily_returns\"])}')
    fc = [len(r['fold_returns']) for r in real_records]
    print(f'有效折数: min={min(fc)}, max={max(fc)}, avg={np.mean(fc):.1f}')"""))

# ── 4 ──
nb["cells"].append(code("""def compute_features(daily_returns, daily_top3):
    returns = np.array(daily_returns)
    n = len(returns)
    if n < 2:
        return {k: 0 for k in ['mean_ret', 'win_rate', 'max_dd', 'ret_dd_ratio', 'turnover', 'cum_ret']}
    mean_ret = float(np.mean(returns))
    win_rate = float(np.mean(returns > 0))
    cum = np.cumsum(returns)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0
    ret_dd_ratio = mean_ret / (max_dd + 1e-10)
    cum_ret = float(cum[-1])
    if n < 2:
        turnover = 0
    else:
        overlaps = []
        for i in range(1, n):
            overlap = len(set(daily_top3[i]) & set(daily_top3[i-1]))
            overlaps.append(1 - overlap / 3)
        turnover = float(np.mean(overlaps)) if overlaps else 0
    return {'mean_ret': mean_ret, 'win_rate': win_rate, 'max_dd': max_dd,
            'ret_dd_ratio': ret_dd_ratio, 'turnover': turnover, 'cum_ret': cum_ret}

real_features = []
for rec in real_records:
    feat = compute_features(rec['daily_returns'], rec['daily_top3'])
    feat['model_type'] = rec['model_type']
    feat['exp_idx'] = rec['exp_idx']
    feat['label'] = 'real'
    real_features.append(feat)

real_df = pd.DataFrame(real_features)
print(f'真实实验: {len(real_df)} 个')
display(real_df[['model_type','exp_idx','mean_ret','win_rate','max_dd','ret_dd_ratio','turnover','cum_ret']].head())"""))

# ── 5: Generate 10000 random ──
nb["cells"].append(code("""# 构建 daily_fwd_returns 矩阵: (n_days, n_stocks)
n_days = len(real_records[0]['daily_returns'])
fwd_matrix = np.zeros((n_days, len(etf_list)))
for d_idx in range(n_days):
    dt = idx_to_date[d_idx]
    fwd_data = fwd_lookup[fwd_lookup['日期'] == dt]
    for _, row in fwd_data.iterrows():
        sid_idx = stock_to_idx.get(row['股票代码'])
        if sid_idx is not None:
            fwd_matrix[d_idx, sid_idx] = row['fwd_ret']

print(f'收益矩阵: ({n_days} 天 x {len(etf_list)} 只 ETF)')

# 生成 10000 次随机 Top-3
n_random = 10000
rand_daily_returns = np.zeros((n_random, n_days))
rand_daily_top3_ids = []

for i in range(n_random):
    day_rets = []
    day_top3 = []
    for j in range(n_days):
        picks = np.random.choice(len(etf_list), 3, replace=False)
        day_top3.append([etf_list[p] for p in picks])
        day_rets.append(fwd_matrix[j, picks].mean())
    rand_daily_returns[i] = day_rets
    rand_daily_top3_ids.append(day_top3)

print(f'生成了 {n_random} 次随机 Top-3 序列')"""))

# ── 6 ──
nb["cells"].append(code("""rand_features = []
for i in range(n_random):
    feat = compute_features(rand_daily_returns[i], rand_daily_top3_ids[i])
    feat['label'] = 'random'
    rand_features.append(feat)

rand_df = pd.DataFrame(rand_features)
print(f'随机实验: {len(rand_df)} 次')

all_df = pd.concat([real_df, rand_df], ignore_index=True)
print(f'总计: {len(all_df)} 个样本')"""))

# ── 7: Cross-fold stability ──
nb["cells"].append(code("""# ═══════════════════════════════════════════
# 跨折稳定性分析
# ═══════════════════════════════════════════
FOLD_NAMES = ['fold_0', 'fold_1', 'fold_2', 'fold_3']
FOLD_LABELS = ['2024H2', '2025H1', '2025H2', '2026H1']

rand_cum_vals = np.cumsum(rand_daily_returns, axis=1)[:, -1]

stability_rows = []
for rec in real_records:
    mt, ei = rec['model_type'], rec['exp_idx']
    row = {'model_type': mt, 'exp_idx': ei}
    fold_mean_rets = []
    fold_cum_rets = []
    for fn in FOLD_NAMES:
        if fn in rec['fold_returns']:
            feat = compute_features(rec['fold_returns'][fn], rec['fold_top3s'][fn])
            for m in ['mean_ret', 'win_rate', 'cum_ret', 'ret_dd_ratio', 'max_dd']:
                row[f'{fn}_{m}'] = feat[m]
            fold_mean_rets.append(feat['mean_ret'])
            fold_cum_rets.append(feat['cum_ret'])
        else:
            for m in ['mean_ret', 'win_rate', 'cum_ret', 'ret_dd_ratio', 'max_dd']:
                row[f'{fn}_{m}'] = None

    row['n_folds'] = len(fold_mean_rets)
    if len(fold_mean_rets) >= 2:
        row['cv_mean_ret'] = float(np.mean(fold_mean_rets))
        row['cv_std_ret'] = float(np.std(fold_mean_rets))
        row['cv_sharpe'] = row['cv_mean_ret'] / (row['cv_std_ret'] + 1e-10)
    else:
        row['cv_mean_ret'] = float(fold_mean_rets[0]) if fold_mean_rets else 0
        row['cv_std_ret'] = 0
        row['cv_sharpe'] = 0
    row['n_folds_positive'] = sum(1 for v in fold_cum_rets if v > 0)

    # search_results.json comparison (backtest-based Sharpe)
    sr = rec.get('search_results', {})
    row['sr_cv_score'] = sr.get('cv_score')
    row['sr_mean_sharpe'] = sr.get('mean_sharpe')
    sr_sharpes = sr.get('fold_sharpe')
    if sr_sharpes and len(sr_sharpes) >= 2:
        row['sr_std_sharpe'] = float(np.std(sr_sharpes))
    else:
        row['sr_std_sharpe'] = None

    stability_rows.append(row)

stability_df = pd.DataFrame(stability_rows)

# Top-15 by stability
print('═' * 60)
print('Top 15 稳定模型 (cv_sharpe = 跨折日均收益均值/标准差)')
print('═' * 60)
top_s = stability_df.sort_values('cv_sharpe', ascending=False).head(15)
for _, r in top_s.iterrows():
    cum_str = '/'.join(f'{r.get(f"{fn}_cum_ret", 0):.4f}' for fn in FOLD_NAMES)
    sr_cv = r.get('sr_cv_score')
    sr_str = f'  sr_cv={sr_cv:.3f}' if pd.notna(sr_cv) else ''
    print(f'  {r["model_type"]}_{r["exp_idx"]:>3d}  '
          f'cv_sharpe={r["cv_sharpe"]:>8.3f}{sr_str}  '
          f'cv_mean={r["cv_mean_ret"]:>9.5f}  '
          f'cv_std={r["cv_std_ret"]:>9.5f}  '
          f'正折={r["n_folds_positive"]}/4')

# Per-fold percentile vs random
print()
print('每折 cum_ret percentiles vs random (0-100%)')
print('           ' + ''.join(f'{l:>8}' for l in FOLD_LABELS))
print('-' * 60)
for _, r in stability_df.sort_values('cv_sharpe', ascending=False).head(15).iterrows():
    pcts = []
    for fn in FOLD_NAMES:
        v = r.get(f'{fn}_cum_ret', None)
        if pd.notna(v):
            p = (rand_cum_vals < v).mean()
        else:
            p = None
        pcts.append(p)
    line = f'{r["model_type"]}_{r["exp_idx"]:<3d}  '
    for p in pcts:
        if p is not None:
            line += f'{p*100:>7.0f}% '
        else:
            line += '     - '
    print(line)"""))


# ── 8: 6 histograms ──
nb["cells"].append(code("""# 标注实验 (加入最稳定的模型)
highlight = [
    ('dlinear', 17, 'dlinear_17(最稳定)', '#ff7f0e'),
    ('tcn', 43, 'tcn_43(最强)', '#8c564b'),
    ('tcn', 22, 'tcn_22(次稳定)', '#2ca02c'),
    ('tcn', 5, 'tcn_5(当前)', '#d62728'),
    ('gru', 16, 'gru_16(当前)', '#9467bd'),
    ('gru', 10, 'gru_10', '#1f77b4'),
    ('tcn', 28, 'tcn_28', '#e377c2'),
]
hl_map = {(mt, ei): (name, c) for mt, ei, name, c in highlight}

metric_names = [
    ('mean_ret', '日均收益', True),
    ('win_rate', '胜率', True),
    ('max_dd', '最大回撤', False),
    ('ret_dd_ratio', '收益/回撤', True),
    ('turnover', '换手率', False),
    ('cum_ret', '累计收益', True),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('真实实验 vs 10000次随机: Top-3 特征分布', fontsize=14, fontweight='bold')

for ax, (m, label, hb) in zip(axes.flat, metric_names):
    rv = rand_df[m].values
    ax.hist(rv, bins=60, alpha=0.5, color='#aaaaaa', density=True, label='随机')
    for j, v in real_df[m].items():
        mt, ei = real_df.iloc[j]['model_type'], real_df.iloc[j]['exp_idx']
        key = (mt, ei)
        if key in hl_map:
            nm, c = hl_map[key]
            ax.scatter(v, 0.35, c=c, s=80, zorder=5, edgecolors='black', linewidths=0.5)
            ax.annotate(nm, (v, 0.4), fontsize=7, ha='center', color=c, fontweight='bold')
        else:
            ax.scatter(v, 0.25, c='#333333', s=15, alpha=0.6, zorder=3)

    p5, p50, p95 = np.percentile(rv, [5, 50, 95])
    for p, ls in [(p5, '--'), (p50, '-'), (p95, '--')]:
        ax.axvline(p, color='red', ls=ls, lw=0.8, alpha=0.6)
    ax.set_xlabel(label)
    ax.set_ylabel('密度')
    ax.set_title(f'{label}\\n(P50={p50:.4f})', fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()"""))

# ── 9: PCA ──
nb["cells"].append(code("""feature_cols = ['mean_ret', 'win_rate', 'max_dd', 'ret_dd_ratio', 'turnover', 'cum_ret']
X = all_df[feature_cols].copy().replace([np.inf, -np.inf], 0)
X_std = (X - X.mean()) / (X.std() + 1e-10)

pca = PCA(n_components=2)
coords = pca.fit_transform(X_std)

fig, ax = plt.subplots(figsize=(10, 7))
rand_mask = all_df['label'] == 'random'
real_mask = all_df['label'] == 'real'
samp = np.random.choice(np.where(rand_mask)[0], min(2000, n_random), replace=False)
ax.scatter(coords[samp, 0], coords[samp, 1], c='#cccccc', s=5, alpha=0.3, label='随机')

for idx in np.where(real_mask)[0]:
    mt, ei = all_df.iloc[idx]['model_type'], all_df.iloc[idx]['exp_idx']
    key = (mt, ei)
    if key in hl_map:
        nm, c = hl_map[key]
        ax.scatter(coords[idx, 0], coords[idx, 1], c=c, s=100, zorder=5, edgecolors='black', linewidths=0.5)
        ax.annotate(nm, coords[idx], fontsize=8, fontweight='bold', ha='center', va='bottom', color=c)
    else:
        ax.scatter(coords[idx, 0], coords[idx, 1], c='#333333', s=30, alpha=0.7)

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax.set_title(f'PCA: 真实({len(real_df)}) vs 随机({n_random}) 多折合并', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()

print('PCA 载荷:')
for i, col in enumerate(feature_cols):
    print(f'  {col}: PC1={pca.components_[0,i]:+.3f}  PC2={pca.components_[1,i]:+.3f}')"""))

# ── 10: Cumulative curves ──
nb["cells"].append(code("""fig, ax = plt.subplots(figsize=(14, 6))

rand_cum = np.cumsum(rand_daily_returns, axis=1)
p5 = np.percentile(rand_cum, 5, axis=0)
p50 = np.percentile(rand_cum, 50, axis=0)
p95 = np.percentile(rand_cum, 95, axis=0)
x_days = np.arange(n_days)
ax.fill_between(x_days, p5, p95, alpha=0.15, color='gray', label='随机 5%-95%')
ax.plot(x_days, p50, '-', color='gray', lw=1, alpha=0.5, label='随机中位')

# 更新高亮: 加入 dlinear_17(最稳定)
hl_subset = [
    ('gru', 10, 'gru_10', '#2ca02c'),
    ('tcn', 28, 'tcn_28', '#1f77b4'),
    ('dlinear', 17, 'dlinear_17(最稳定)', '#ff7f0e'),
    ('tcn', 5, 'tcn_5(当前)', '#d62728'),
    ('gru', 16, 'gru_16(当前)', '#9467bd'),
    ('tcn', 43, 'tcn_43(最强)', '#8c564b'),
]

# 按 fold 分段的累计曲线
for rec in real_records:
    mt, ei = rec['model_type'], rec['exp_idx']
    match = [h for h in hl_subset if h[0] == mt and h[1] == ei]
    if not match:
        continue
    name, c = match[0][2], match[0][3]

    # 合并累计曲线
    cum = np.cumsum(rec['daily_returns'])
    ax.plot(x_days[:len(cum)], cum, lw=2, label=name, color=c)

    # 在每条曲线上标记 fold 边界
    for fn, start in rec['fold_boundaries']:
        if start > 0 and start < len(cum):
            ax.axvline(start, color=c, lw=0.5, ls=':', alpha=0.4)

# 随机置信区间的 fold 边界 (用第一个实验的边界)
if real_records:
    bnd = real_records[0]['fold_boundaries']
    for fn, start in bnd:
        if start > 0:
            ax.axvline(start, color='black', lw=0.8, ls='--', alpha=0.3)
            # 标注折名
            label_map = {'fold_0': 'fold_0\\n2024H2', 'fold_1': 'fold_1\\n2025H1',
                         'fold_2': 'fold_2\\n2025H2', 'fold_3': 'fold_3\\n2026H1'}
            ax.text(start + 2, ax.get_ylim()[1] * 0.95, label_map.get(fn, fn),
                    fontsize=7, alpha=0.5)

ax.axhline(0, color='black', ls='--', lw=0.5)
ax.set_xlabel('交易日')
ax.set_ylabel('累计收益')
ax.set_title('Top-3 累计收益: 模型 vs 随机置信区间 (竖线=折边界)', fontsize=13, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()"""))

# ── 11: Conclusion table ──
nb["cells"].append(code("""def pct_in_random(real_val, random_vals, hb=True):
    return (random_vals < real_val).mean() if hb else (random_vals > real_val).mean()

def flag(p):
    return '✅' if p >= 0.95 else ('⚠️' if p >= 0.50 else '❌')

# ── Part A: 综合评级（全部合并数据） ──
print('═' * 60)
print('综合评级 (全部折叠合并)')
print('═' * 60)
header = f'{"实验":<16} {"日均收益":>6} {"胜率":>6} {"最大回撤":>6} {"收益/回撤":>6} {"换手率":>6} {"累计收益":>6} {"综合":>6}'
print(header)
print('-' * 80)

# Select: top-5 by cum_ret + top-5 by cv_sharpe + deploy_check
top_cum = real_df.sort_values('cum_ret', ascending=False).head(8)
top_stab = stability_df.sort_values('cv_sharpe', ascending=False).head(8)
deploy_check = [('tcn',5), ('gru',16), ('tcn',28), ('dlinear',17), ('gru',10), ('tcn',43)]
selected, seen = [], set()
for _, r in top_cum.iterrows():
    k = (r['model_type'], r['exp_idx'])
    if k not in seen: selected.append(r); seen.add(k)
for _, r in top_stab.iterrows():
    k = (r['model_type'], r['exp_idx'])
    if k not in seen:
        match = real_df[(real_df['model_type']==k[0]) & (real_df['exp_idx']==k[1])]
        if len(match):
            selected.append(match.iloc[0]); seen.add(k)
for mt, ei in deploy_check:
    k = (mt, ei)
    if k not in seen:
        match = real_df[(real_df['model_type']==k[0]) & (real_df['exp_idx']==k[1])]
        if len(match):
            selected.append(match.iloc[0]); seen.add(k)

rand_vals_m = {m: rand_df[m].values for m, _, _ in metric_names}
rows = []
for r in selected:
    tag = f"{r['model_type']}_{r['exp_idx']}"
    pcts = []
    for m, _, hb in metric_names:
        pcts.append(pct_in_random(r[m], rand_vals_m[m], hb))
    comp = np.mean(pcts)
    # 稳定性标签
    stb_row = stability_df[(stability_df['model_type']==r['model_type']) & (stability_df['exp_idx']==r['exp_idx'])]
    cv_str = ''
    sr_str = ''
    if len(stb_row):
        cv = stb_row.iloc[0]['cv_sharpe']
        cv_str = f' cv={cv:.1f}'
        sr_cv = stb_row.iloc[0].get('sr_cv_score')
        if pd.notna(sr_cv):
            sr_str = f' sr={sr_cv:.2f}'
    parts = [tag + cv_str + sr_str] + [flag(p) for p in pcts] + [flag(comp)]
    print('  '.join(f'{p:<6}' for p in parts))

    row_dict = {
        '实验': tag, 'cv_sharpe': stb_row.iloc[0]['cv_sharpe'] if len(stb_row) else None,
        '日均收益': flag(pcts[0]), '胜率': flag(pcts[1]),
        '最大回撤': flag(pcts[2]), '收益/回撤': flag(pcts[3]),
        '换手率': flag(pcts[4]), '累计收益': flag(pcts[5]),
        '综合': flag(comp),
    }
    # Add numeric search_results comparison + per-fold metrics to CSV
    if len(stb_row):
        row_dict['sr_cv_score'] = stb_row.iloc[0].get('sr_cv_score')
        row_dict['sr_mean_sharpe'] = stb_row.iloc[0].get('sr_mean_sharpe')
        row_dict['sr_std_sharpe'] = stb_row.iloc[0].get('sr_std_sharpe')
        for fn in FOLD_NAMES:
            for m in ['mean_ret', 'cum_ret']:
                v = stb_row.iloc[0].get(f'{fn}_{m}')
                row_dict[f'{fn}_{m}'] = v
    rows.append(row_dict)

print()
summary = pd.DataFrame(rows)
display(summary)
print()
summary.to_csv(os.path.join(_OUT_DIR, 'signal_vs_random_summary.csv'), index=False)
print(f'已保存: {{_OUT_DIR}}/signal_vs_random_summary.csv')

# ── Part B: 跨折稳定性表 ──
print()
print('═' * 60)
print('跨折稳定性排行 (仅展示稳定度 Top 10)')
print('═' * 60)
stb_header = f'{"实验":<16} {"sr_cv":>8} {"cv_sharpe":>10} {"cv_mean":>11} {"cv_std":>10} {"正折":>6} {"f0_prc":>7} {"f1_prc":>7} {"f2_prc":>7} {"f3_prc":>7}'
print(stb_header)
print('-' * 100)
rand_cum_v = rand_df['cum_ret'].values
for _, r in stability_df.sort_values('cv_sharpe', ascending=False).head(10).iterrows():
    tag = f"{r['model_type']}_{r['exp_idx']}"
    sr_cv = r.get('sr_cv_score')
    sr_str = f'{sr_cv:>8.3f}' if pd.notna(sr_cv) else '     nan '
    pcts_str = []
    for fn in ['fold_0', 'fold_1', 'fold_2', 'fold_3']:
        v = r.get(f'{fn}_cum_ret', None)
        p = f'{((rand_cum_v < v).mean()*100):>6.0f}%' if v is not None else '    - '
        pcts_str.append(p)
    print(f'{tag:<16} {sr_str} {r["cv_sharpe"]:>10.3f} {r["cv_mean_ret"]:>11.5f} {r["cv_std_ret"]:>10.5f} '
          f'{r["n_folds_positive"]:>3d}/4 {" ".join(pcts_str)}')"""))

# ── 12 ──
nb["cells"].append(code("""# 总体结论
print('=' * 60)
print('模型信号 vs 随机排序: 整体结论 (4折交叉验证)')
print('=' * 60)

# 合并数据整体 percentile
for m, label, hb in metric_names:
    pcts = [pct_in_random(real_df[m].iloc[i], rand_vals_m[m], hb) for i in range(len(real_df))]
    print(f'{label:<10} 平均百分位: {np.mean(pcts):.1%} 中位: {np.median(pcts):.1%}')

print('-' * 60)
pos_real = (real_df['cum_ret'] > 0).sum()
pos_rand = (rand_df['cum_ret'] > 0).mean()
print(f'累计收益为正: 真实 {pos_real}/{len(real_df)} = {pos_real/len(real_df):.1%}')
print(f'累计收益为正: 随机 {pos_rand:.1%}')

print()
print('═' * 60)
print('跨折稳定性总结')
print('═' * 60)
# 绝对稳定: cv_sharpe > 1.0
stable = stability_df[stability_df['cv_sharpe'] > 1.0]
print(f'cv_sharpe > 1.0 (跨折高度稳定): {len(stable)}/{len(stability_df)} 个')
if len(stable):
    for _, r in stable.sort_values('cv_sharpe', ascending=False).iterrows():
        print(f'  {r["model_type"]}_{r["exp_idx"]:>3d}  cv_sharpe={r["cv_sharpe"]:>.2f}')

# 全折正收益
all_pos = stability_df[stability_df['n_folds_positive'] == 4]
print(f'4 折全部正收益: {len(all_pos)}/{len(stability_df)} 个')
if len(all_pos):
    for _, r in all_pos.sort_values('cv_sharpe', ascending=False).iterrows():
        print(f'  {r["model_type"]}_{r["exp_idx"]:>3d}  cv_sharpe={r["cv_sharpe"]:>.2f}')

# 在全部 4 折中都超过随机中位
def all_folds_above_median(row):
    for fn in FOLD_NAMES:
        v = row.get(f'{fn}_cum_ret', None)
        if v is None or v < np.median(rand_cum_v):
            return False
    return True
all_beat_median = stability_df[stability_df.apply(all_folds_above_median, axis=1)]
print(f'4 折均超随机中位数: {len(all_beat_median)}/{len(stability_df)} 个')
if len(all_beat_median):
    for _, r in all_beat_median.sort_values('cv_sharpe', ascending=False).iterrows():
        print(f'  {r["model_type"]}_{r["exp_idx"]:>3d}  cv_sharpe={r["cv_sharpe"]:>.2f}')"""))

# ── 13: 模型部署推荐 ──
nb["cells"].append(code("""# ═══════════════════════════════════════════
# 模型选取结论：综合 cv_sharpe + sr_cv_score 双指标
# ═══════════════════════════════════════════

def fmt(v):
    return f'{v:.2f}' if pd.notna(v) else 'nan'

print('=' * 70)
print('模型部署推荐 — 综合 cv_sharpe(日频稳定) + sr_cv_score(回测Sharpe)')
print('=' * 70)

# 双指标均有效的模型
valid = stability_df[stability_df['sr_cv_score'].notna()].copy()
# 综合得分: 两个指标的归一化和
max_cv = valid['cv_sharpe'].max()
max_sr = valid['sr_cv_score'].max()
valid['composite'] = valid['cv_sharpe'] / max_cv + valid['sr_cv_score'] / max_sr
valid['rank'] = valid['composite'].rank(ascending=False)

# ── 第一档: 双优模型 (两个指标都高) ──
dual_win = valid[(valid['cv_sharpe'] > 2.0) & (valid['sr_cv_score'] > 2.0)]
print()
print('【第一档】★★★★★ 双指标优秀 (cv_sharpe>2.0 & sr_cv_score>2.0) — 强烈推荐')
print(f'{"实验":<16} {"cv_sharpe":>10} {"sr_cv":>8} {"sr_mean":>8} {"sr_std":>8} {"composite":>10} {"4折正收益?":>10}')
print('-' * 70)
for _, r in dual_win.sort_values('composite', ascending=False).iterrows():
    tag = f"{r['model_type']}_{r['exp_idx']}"
    pos = '✅' if r['n_folds_positive'] == 4 else '⚠️'
    print(f'{tag:<16} {fmt(r["cv_sharpe"]):>10} {fmt(r["sr_cv_score"]):>8} '
          f'{fmt(r["sr_mean_sharpe"]):>8} {fmt(r["sr_std_sharpe"]):>8} '
          f'{fmt(r["composite"]):>10} {pos:>10}')

# ── 第二档: 单指标突出 ──
sr_only = valid[(valid['cv_sharpe'] <= 2.0) & (valid['sr_cv_score'] > 2.0)].sort_values('composite', ascending=False)
print()
print('【第二档】★★★★ 回测 Sharpe 优秀 (sr_cv_score>2.0)，日频稳定性一般 — 可选用')
print(f'{"实验":<16} {"cv_sharpe":>10} {"sr_cv":>8} {"sr_mean":>8} {"sr_std":>8} {"composite":>10} {"4折正收益?":>10}')
print('-' * 70)
for _, r in sr_only.iterrows():
    tag = f"{r['model_type']}_{r['exp_idx']}"
    pos = '✅' if r['n_folds_positive'] == 4 else '⚠️'
    print(f'{tag:<16} {fmt(r["cv_sharpe"]):>10} {fmt(r["sr_cv_score"]):>8} '
          f'{fmt(r["sr_mean_sharpe"]):>8} {fmt(r["sr_std_sharpe"]):>8} '
          f'{fmt(r["composite"]):>10} {pos:>10}')

cv_only = valid[(valid['cv_sharpe'] > 2.0) & (valid['sr_cv_score'] <= 2.0)].sort_values('composite', ascending=False)
if len(cv_only):
    print()
    print('【第二档B】★★★★ 日频稳定 (cv_sharpe>2.0)，回测一般')
    print(f'{"实验":<16} {"cv_sharpe":>10} {"sr_cv":>8} {"sr_mean":>8} {"sr_std":>8} {"composite":>10} {"4折正收益?":>10}')
    print('-' * 70)
    for _, r in cv_only.iterrows():
        tag = f"{r['model_type']}_{r['exp_idx']}"
        pos = '✅' if r['n_folds_positive'] == 4 else '⚠️'
        print(f'{tag:<16} {fmt(r["cv_sharpe"]):>10} {fmt(r["sr_cv_score"]):>8} '
              f'{fmt(r["sr_mean_sharpe"]):>8} {fmt(r["sr_std_sharpe"]):>8} '
              f'{fmt(r["composite"]):>10} {pos:>10}')

# ── 当前实盘 vs 推荐 ──
print()
print('=' * 70)
print('当前实盘模型评估')
print('=' * 70)
deploy_current = [('gru',10), ('gru',16), ('tcn',28), ('dlinear',17)]
print(f'{"实验":<16} {"cv_sharpe":>10} {"sr_cv":>8} {"sr_mean":>8} {"sr_std":>8} {"composite":>10} {"结论":>10}')
print('-' * 70)
for mt, ei in deploy_current:
    match = valid[(valid['model_type']==mt) & (valid['exp_idx']==ei)]
    if len(match):
        r = match.iloc[0]
        tag = f"{r['model_type']}_{r['exp_idx']}"
        dual = r['cv_sharpe'] > 2.0 and r['sr_cv_score'] > 2.0
        verdict = '✅保留' if dual else ('🟡可换' if r['cv_sharpe'] > 1.0 else '🔴建议替换')
        print(f'{tag:<16} {fmt(r["cv_sharpe"]):>10} {fmt(r["sr_cv_score"]):>8} '
              f'{fmt(r["sr_mean_sharpe"]):>8} {fmt(r["sr_std_sharpe"]):>8} '
              f'{fmt(r["composite"]):>10} {verdict:>10}')

# ── 最终推荐 Top-4 ──
print()
print('=' * 70)
print('最终推荐 (Top-4, 保持模型类型多样性)')
print('=' * 70)
top4 = dual_win.sort_values('composite', ascending=False)
used_types = set()
final_picks = []
for _, r in top4.iterrows():
    mt = r['model_type']
    if mt not in used_types or len(final_picks) < 4:
        if len(final_picks) < 4:
            final_picks.append(r)
            used_types.add(mt)

# 若不足4个，从sr_only补
if len(final_picks) < 4:
    for _, r in sr_only.iterrows():
        if len(final_picks) >= 4: break
        mt = r['model_type']
        tag = f"{r['model_type']}_{r['exp_idx']}"
        if tag not in [f"{p['model_type']}_{p['exp_idx']}" for p in final_picks]:
            final_picks.append(r)

print(f'{"推荐":<4} {"实验":<16} {"类型":<8} {"cv_sharpe":>10} {"sr_cv":>8} {"sr_mean":>8} {"sr_std":>8}')
print('-' * 70)
for i, r in enumerate(final_picks):
    tag = f"{r['model_type']}_{r['exp_idx']}"
    print(f'Top-{i+1:<2} {tag:<16} {r["model_type"]:<8} '
          f'{fmt(r["cv_sharpe"]):>10} {fmt(r["sr_cv_score"]):>8} '
          f'{fmt(r["sr_mean_sharpe"]):>8} {fmt(r["sr_std_sharpe"]):>8}')

print()
print(f'建议组合: {" ".join(f"{p["model_type"]}_{p["exp_idx"]}" for p in final_picks)}')
print(f'配置建议: master first → 自动选第1个, 或切换到特定模型')"""))

# ── 14: 推荐模型累计收益曲线 ──
nb["cells"].append(code("""fig, ax = plt.subplots(figsize=(14, 6))

# 随机置信区间 (同 cell 10)
rand_cum = np.cumsum(rand_daily_returns, axis=1)
p5 = np.percentile(rand_cum, 5, axis=0)
p50 = np.percentile(rand_cum, 50, axis=0)
p95 = np.percentile(rand_cum, 95, axis=0)
x_days = np.arange(n_days)
ax.fill_between(x_days, p5, p95, alpha=0.15, color='gray', label='随机 5%-95%')
ax.plot(x_days, p50, '-', color='gray', lw=1, alpha=0.5, label='随机中位')

# 推荐模型配色 (按类型区分)
palette = {'gru': '#2ca02c', 'tcn': '#1f77b4', 'dlinear': '#ff7f0e'}
extra = {'tcn': '#8c564b'}  # 第2个TCN用棕色区分
type_count = {}

# 在 real_records 中查找推荐模型
pick_labels = []
for r in final_picks:
    mt, ei = r['model_type'], r['exp_idx']
    tag = f'{mt}_{ei}'
    type_count[mt] = type_count.get(mt, 0) + 1
    color = palette.get(mt, '#333333')
    # 同类型第2个用备选色
    if type_count[mt] >= 2 and mt in extra:
        color = extra[mt]
    # 在 real_records 中匹配
    for rec in real_records:
        if rec['model_type'] == mt and rec['exp_idx'] == ei:
            cum = np.cumsum(rec['daily_returns'])
            label = f"{tag} (cv={r['cv_sharpe']:.1f} sr={r['sr_cv_score']:.1f})"
            ax.plot(x_days[:len(cum)], cum, lw=2, label=label, color=color)
            pick_labels.append(label)
            # 折边界竖线
            for fn, start in rec['fold_boundaries']:
                if start > 0 and start < len(cum):
                    ax.axvline(start, color=color, lw=0.5, ls=':', alpha=0.4)
            break

# 随机区间背景的折边界 (用第一个有数据的记录的边界)
anchor = None
for rec in real_records:
    if len(rec['fold_boundaries']) >= 2:
        anchor = rec['fold_boundaries']
        break
if anchor:
    fold_labels = {'fold_0': '2024H2', 'fold_1': '2025H1', 'fold_2': '2025H2', 'fold_3': '2026H1'}
    for fn, start in anchor:
        if start > 0:
            ax.axvline(start, color='black', lw=0.8, ls='--', alpha=0.3)
            ax.text(start + 2, ax.get_ylim()[1] * 0.95, fold_labels.get(fn, fn),
                    fontsize=8, alpha=0.5)

ax.axhline(0, color='black', ls='--', lw=0.5)
ax.set_xlabel('交易日')
ax.set_ylabel('累计收益')
ax.set_title('推荐模型累计收益: 模型 vs 随机置信区间 (竖线=折边界)', fontsize=13, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()"""))

# ── 15: 全验证集止损策略对比 ──
nb["cells"].append(code("""# ═══════════════════════════════════════════
# 全验证集: 无止损 vs 市场广度止损 对比
# ═══════════════════════════════════════════

# 用 fwd_matrix 计算每日市场趋势
# fwd_matrix[d] = 各 stock 在 day d+1 的 open-close 收益
# 所以 "day d 的市场情况" ≈ fwd_matrix[d-1] 的统计量
LOOKBACK = 20
HIGH_TH = 0.30   # 正收益ETF占比≥此值 → 正常仓位
LOW_TH = 0.10    # 正收益ETF占比≤此值 → 空仓

def _market_multiplier(day_idx):
    \"\"\"return 0.0~1.0 仓位乘数\"\"\"
    if day_idx < 1:
        return 1.0
    start = max(0, day_idx - 1 - LOOKBACK)
    end = day_idx  # fwd_matrix[end-1] 是 day_idx-1 的收益
    window = fwd_matrix[start:end]  # shape: (N, n_stocks)
    if len(window) < 5:
        return 1.0
    # 计算每只股票在窗口内的累计 fwd_ret
    cum_rets = np.nansum(window, axis=0)
    valid = cum_rets[~np.isnan(cum_rets)]
    if len(valid) < 5:
        return 1.0
    pos_ratio = (valid > 0).mean()
    if pos_ratio >= HIGH_TH:
        return 1.0
    if pos_ratio <= LOW_TH:
        return 0.0
    return (pos_ratio - LOW_TH) / (HIGH_TH - LOW_TH)

# 对每个实验计算有止损版本的 cumulative return
sl_improvements = []
for rec in real_records:
    mt, ei = rec['model_type'], rec['exp_idx']
    tag = f'{mt}_{ei}'
    orig = rec['daily_returns']
    # 止损版本: 每天 apply multiplier
    sl_ret = []
    for d in range(len(orig)):
        mult = _market_multiplier(d)
        sl_ret.append(orig[d] * mult)
    sl_ret = np.array(sl_ret)
    orig_cum = np.cumsum(orig)[-1]
    sl_cum = np.cumsum(sl_ret)[-1]
    imp = sl_cum - orig_cum
    sl_improvements.append((tag, orig_cum, sl_cum, imp))

print('=' * 70)
print(f'全验证集止损对比 ({len(sl_improvements)} 个实验)')
print(f'策略: 正收益ETF占比 < {LOW_TH:.0%} 空仓, > {HIGH_TH:.0%} 正常, 中间线性')
print(f'回看: {LOOKBACK} 天')
print('=' * 70)
print(f'{"":<16} {"原始累计":>10} {"止损累计":>10} {"差值":>10} {"改善?":>6}')
print('-' * 52)

better = [x for x in sl_improvements if x[3] > 0.001]
worse = [x for x in sl_improvements if x[3] < -0.001]
neutral = [x for x in sl_improvements if abs(x[3]) <= 0.001]

print(f'改善: {len(better)} 个 ({len(better)/len(sl_improvements)*100:.0f}%)')
print(f'变差: {len(worse)} 个 ({len(worse)/len(sl_improvements)*100:.0f}%)')
print(f'持平: {len(neutral)} 个 ({len(neutral)/len(sl_improvements)*100:.0f}%)')

if better:
    avg_imp = np.mean([x[3] for x in better])
    print(f'改善平均: +{avg_imp:.4f}')
if worse:
    avg_dec = np.mean([x[3] for x in worse])
    print(f'变差平均: {avg_dec:.4f}')

# 配对 t 检验
from scipy.stats import ttest_rel
orig_vals = np.array([x[1] for x in sl_improvements])
sl_vals = np.array([x[2] for x in sl_improvements])
t_stat, p_val = ttest_rel(orig_vals, sl_vals)
print()
print(f'配对 t 检验: t={t_stat:.3f}, p={p_val:.4f}')
if p_val < 0.05:
    print('结论: 止损策略有统计学显著影响')
else:
    print('结论: 止损策略无统计学显著影响')

# ── 按模型类型分组 ──
print()
print('按模型类型:')
for mt in sorted(set(x[0].split('_')[0] for x in sl_improvements)):
    group = [x for x in sl_improvements if x[0].startswith(mt)]
    g_orig = np.mean([x[1] for x in group])
    g_sl = np.mean([x[2] for x in group])
    g_imp = np.mean([x[3] for x in group])
    print(f'  {mt:<10s} n={len(group):3d}  原始均值={g_orig:.4f}  止损均值={g_sl:.4f}  差值={g_imp:+.4f}')

# ── 频次统计: 止损触发频率 ──
all_mults = []
for d in range(n_days):
    all_mults.append(_market_multiplier(d))
all_mults = np.array(all_mults)
print()
print(f'止损触发统计 ({n_days} 个交易日):')
print(f'  完全空仓 (mult=0): {(all_mults == 0).sum()} 天 ({(all_mults == 0).mean()*100:.1f}%)')
print(f'  部分缩仓 (0<mult<1): {((all_mults > 0) & (all_mults < 1)).sum()} 天')
print(f'  正常仓位 (mult=1): {(all_mults == 1).sum()} 天 ({(all_mults == 1).mean()*100:.1f}%)")"""))

with open("notebooks/signal_vs_random.ipynb", "w") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
    f.write("\n")
print(f"Created: {len(nb['cells'])} cells")
