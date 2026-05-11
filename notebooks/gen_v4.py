"""
Notebook v4: Health Score = P(future return > 0) via logistic regression.
Raw indicators -> rolling features -> predict binary forward outcome.
"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12.0"}
}

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md("""# Model Degradation Experiment v4: Probability-based Health Score

**Goal**: Health score = P(future return > 0 | current indicators). 
High health = high probability of making money in the next N days.

**Why this is better**:
- Natural interpretation (0-100% = probability of profit)
- Can validate with AUC/accuracy/log-loss, not just correlation
- Model is trained to predict profit, not just correlate
""")

md("## 1. Load Data")

code("""import numpy as np, pandas as pd, json, matplotlib.pyplot as plt, itertools
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, linregress
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import roc_auc_score, brier_score_loss
import warnings
warnings.filterwarnings('ignore')

_NB_DIR = Path().resolve()
for _p in [_NB_DIR, _NB_DIR.parent, Path('/mnt/c/Users/xyl/Desktop/ETF')]:
    if (_p / 'output').exists(): ROOT = _p; break
else: ROOT = _NB_DIR

with open(str(ROOT / 'output' / 'backtest_state.json')) as f:
    state = json.load(f)
seqs = state.get('sequences', state)

MODELS = ['search_itransformer_exp_54','search_itransformer_exp_64','search_itransformer_exp_6']
MSHORT = {m: k for m,k in zip(MODELS,['i54','i64','i6'])}
MCOLOR = {'i54':'#E24A33','i64':'#348ABD','i6':'#988ED5'}

print('sklearn loaded')
""")

code("""# Parse all raw data
bareraw = {}; equities = {}; trades = {}
for m in MODELS:
    s = seqs[m]
    met = s.get('metrics', {})
    by_date = {}
    for k_short, k_raw in [('ic','_rank_ic_raw'),('ndcg','_ndcg_raw'),
                            ('mrr','_mrr_raw'),('ksp','_ks_p_raw')]:
        for e in met.get(k_raw, []):
            d = e['date']
            if d not in by_date: by_date[d] = {'date': d}
            by_date[d][k_short] = e['value']
    bareraw[m] = sorted(by_date.values(), key=lambda x: x['date'])

    eq = s.get('equity_curve', [])
    for e in eq:
        d = e['date'].strftime('%Y-%m-%d') if hasattr(e['date'],'strftime') else str(e['date'])[:10]
        e['date'] = d
    equities[m] = sorted(eq, key=lambda x: x['date'])

    tr = []
    for t in s.get('trades', []):
        d = t['date'].strftime('%Y-%m-%d') if hasattr(t['date'],'strftime') else str(t['date'])[:10]
        tr.append({'date': d, 'action': t.get('action'), 'stock': t.get('stock'),
                   'score': t.get('score'), 'price': t.get('price')})
    trades[m] = tr

all_dates = sorted(set(e['date'] for m in MODELS for e in equities[m]))
print(f'Dates: {all_dates[0]} ~ {all_dates[-1]} ({len(all_dates)} days)')

# Cumulative returns-to-date (%)
cum_rets = {}
for m in MODELS:
    eq = equities[m]; init = eq[0]['total_value']
    cum_rets[m] = {e['date']: (e['total_value']/init - 1)*100 for e in eq}

# Daily returns (%)
daily_rets = {}
for m in MODELS:
    eq = equities[m]; rets = {}
    for i in range(1, len(eq)):
        rets[eq[i]['date']] = (eq[i]['total_value']/eq[i-1]['total_value'] - 1)*100
    daily_rets[m] = rets

# Trade PnL
trade_pnl = {}
for m in MODELS:
    pnl = []; prices = {}
    for t in trades[m]:
        if t['action'] == chr(20080)+chr(20837):
            prices[t['stock']] = t['price']
        elif t['action'] == chr(20986)+chr(20986) and t['stock'] in prices and prices[t['stock']] > 0:
            buy_p = prices[t['stock']]
            pnl_pct = (t['price'] - buy_p) / buy_p * 100
            pnl.append({'date': t['date'], 'stock': t['stock'], 'pnl_pct': pnl_pct})
            del prices[t['stock']]
    trade_pnl[m] = pnl

print('Data loaded')
""")

md("## 2. Build Feature Matrix + Target")

code("""# --- Rolling helpers ---
def roll_mean_idx(vals, n):
    out = {}
    for i in range(len(vals)):
        s = max(0, i-n+1)
        out[i] = float(np.mean(vals[s:i+1]))
    return out

def roll_slope_idx(vals, n):
    out = {}
    for i in range(len(vals)):
        s = max(0, i-n+1)
        y = vals[s:i+1]
        if len(y) < 5: continue
        slope,_,_,_,_ = linregress(np.arange(len(y)), y)
        out[i] = slope
    return out

def roll_sharpe_idx(rets, n):
    out = {}
    for i in range(len(rets)):
        s = max(0, i-n+1)
        sub = rets[s:i+1]
        if len(sub) >= 5 and np.std(sub) > 0:
            out[i] = np.mean(sub) / np.std(sub) * np.sqrt(252)
    return out

N_WINDOWS = [3, 5, 10, 15, 20]

def build_features(m, lookahead=5):
    # Build feature matrix X and target y for model m.
    #
    # Features per date:
    # - r_XX_n: rolling mean of XX over n days (XX = ic, ndcg, mrr, ksp)
    # - t_XX_n: rolling slope of XX
    # - r_winrate_n: rolling win rate of daily returns
    # - r_sharpe_n: rolling sharpe
    # - ddepth_n: drawdown depth
    #
    # Target:
    # - y = 1 if cumulative return over next `lookahead` days > 0, else 0
    ms = MSHORT[m]
    dates = [e['date'] for e in equities[m]]
    
    # Raw metric series (indexed by position)
    raw_ic = []; raw_ndcg = []; raw_mrr = []; raw_ksp = []
    raw_ic_dates = []; raw_ndcg_dates = []; raw_mrr_dates = []; raw_ksp_dates = []
    for e in bareraw[m]:
        if e.get('ic') is not None: raw_ic.append(e['ic']); raw_ic_dates.append(e['date'])
        if e.get('ndcg') is not None: raw_ndcg.append(e['ndcg']); raw_ndcg_dates.append(e['date'])
        if e.get('mrr') is not None: raw_mrr.append(e['mrr']); raw_mrr_dates.append(e['date'])
        if e.get('ksp') is not None: raw_ksp.append(e['ksp']); raw_ksp_dates.append(e['date'])
    
    # Daily returns
    dr_vals = [daily_rets[m].get(d, 0) for d in dates]
    eq_vals = [e['total_value'] for e in equities[m]]
    
    features = {}  # {date: {feat_name: value}}
    
    for pos, d in enumerate(dates):
        row = {}
        
        # Rolling mean of metrics (need date-position mapping)
        for name, raw_list in [('ic', raw_ic), ('ndcg', raw_ndcg), ('mrr', raw_mrr), ('ksp', raw_ksp)]:
            # Find position of this date in raw_list
            try:
                ri = raw_ic_dates.index(d) if name == 'ic' else \
                     raw_ndcg_dates.index(d) if name == 'ndcg' else \
                     raw_mrr_dates.index(d) if name == 'mrr' else \
                     raw_ksp_dates.index(d)
            except ValueError:
                continue
            for n in N_WINDOWS:
                s = max(0, ri-n+1)
                sub = raw_list[s:ri+1]
                if sub:
                    row[f'r_{name}_{n}'] = float(np.mean(sub))
                # Slope
                if len(sub) >= 5:
                    slope,_,_,_,_ = linregress(np.arange(len(sub)), sub)
                    row[f't_{name}_{n}'] = slope
        
        # Rolling win rate
        for n in N_WINDOWS:
            s = max(0, pos-n+1)
            sub_dr = dr_vals[s:pos+1]
            if sub_dr:
                row[f'winrate_{n}'] = sum(1 for v in sub_dr if v > 0) / len(sub_dr)
        
        # Rolling Sharpe
        for n in N_WINDOWS:
            s = max(0, pos-n+1)
            sub_dr = dr_vals[s:pos+1]
            if len(sub_dr) >= 5 and np.std(sub_dr) > 0:
                row[f'sharpe_{n}'] = float(np.mean(sub_dr) / np.std(sub_dr) * np.sqrt(252))
        
        # Drawdown
        running_max = max(eq_vals[:pos+1])
        row['ddepth_cur'] = (eq_vals[pos] / running_max - 1) * 100
        
        # Score std (forward filled)
        # (skip for now)
        
        features[d] = row
    
    # Build DataFrame
    X = pd.DataFrame.from_dict(features, orient='index')
    X.index.name = 'date'
    
    # Target: 1 if forward cumulative return > 0
    cum_vals = [cum_rets[m].get(d) for d in dates]
    y_vals = []
    y_dates = []
    for i, d in enumerate(dates):
        if i + lookahead < len(cum_vals) and cum_vals[i+lookahead] is not None and cum_vals[i] is not None:
            fwd = cum_vals[i+lookahead] - cum_vals[i]
            y_vals.append(1 if fwd > 0 else 0)
            y_dates.append(d)
    
    y = pd.Series(y_vals, index=y_dates)
    
    # Only keep X rows where y exists
    X = X.loc[X.index.intersection(y.index)]
    y = y.loc[X.index]
    
    return X, y

print('Building features for all models (lookahead=5)...')
all_X = {}
all_y = {}
for m in MODELS:
    X, y = build_features(m, lookahead=5)
    all_X[MSHORT[m]] = X
    all_y[MSHORT[m]] = y
    print(f'  {MSHORT[m]}: X {X.shape}, y {y.sum()}/{len(y)} positive')
""")

md("""## 3. Train Health Score (Logistic Regression)

Train on all 3 models pooled together (more samples), then evaluate on each model individually.
""")

code("""# Pool data from all models
X_pool = pd.concat([all_X[ms] for ms in ['i54','i64','i6']], axis=0)
y_pool = pd.concat([all_y[ms] for ms in ['i54','i64','i6']], axis=0)

print(f'Pooled: X {X_pool.shape}, y {y_pool.sum()}/{len(y_pool)} positive')

# Drop columns with NaN
X_pool = X_pool.dropna(axis=1)
print(f'After dropping NaN columns: {X_pool.shape[1]} features')

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pool)

# Train logistic regression (with regularization for small sample)
lr = LogisticRegression(
    penalty='l1',
    solver='saga',
    C=1.0,
    max_iter=1000,
    random_state=42
)
lr.fit(X_scaled, y_pool)

# Feature importance
coef_df = pd.DataFrame({
    'feature': X_pool.columns,
    'coef': lr.coef_[0]
}).sort_values('coef', ascending=False)

print(f'\\n=== Logistic Regression Coefficients ===')
print(f'Intercept: {lr.intercept_[0]:.4f}')
print()
print(f'{"Feature":20s} {"Coef":10s}')
print('-' * 30)
for _, row in coef_df.iterrows():
    if abs(row['coef']) > 0.001:
        print(f'{row["feature"]:20s} {row["coef"]:+.4f}')

print(f'\\nNon-zero features: {(lr.coef_[0] != 0).sum()} / {len(lr.coef_[0])}')
""")

md("""## 4. Evaluate Health Score

Health = predicted probability of forward return > 0.
Evaluate: AUC, accuracy, and correlation with forward return magnitude.
""")

code("""# Predict probabilities
y_prob = lr.predict_proba(X_scaled)[:, 1]

# Build df_by_model for forward return computation
df_rows = []
for d in all_dates:
    for m in MODELS:
        ret = cum_rets[m].get(d)
        if ret is None: continue
        df_rows.append({'date': d, 'model': MSHORT[m], 'ret_cum': ret})
df_all = pd.DataFrame(df_rows)
df_by_model = {ms: df_all[df_all['model'] == ms] for ms in ['i54','i64','i6']}

def forward_rets(df_model, la):
    vals = list(df_model['ret_cum'])
    dates = list(df_model['date'])
    fwd = {}
    for i, d in enumerate(dates):
        if i + la < len(vals):
            fwd[d] = vals[i+la] - vals[i]
    return fwd

health_dict = {}
idx = 0
for ms in ['i54','i64','i6']:
    n = len(all_y[ms])
    health_dict[ms] = {d: y_prob[idx + i] * 100 for i, d in enumerate(all_y[ms].index)}
    idx += n

# Evaluate per model
print("=== Per-Model Evaluation ===\\n")
for ms in ['i54','i64','i6']:
    y_true = all_y[ms]
    X_ms = all_X[ms][X_pool.columns].dropna()
    y_prob_ms = lr.predict_proba(scaler.transform(X_ms))[:, 1]
    
    auc = roc_auc_score(y_true, y_prob_ms)
    brier = brier_score_loss(y_true, y_prob_ms)
    acc = ((y_prob_ms > 0.5) == y_true).mean()
    
    # Correlation with forward return magnitude
    sub = df_by_model[ms]
    fwd = forward_rets(sub, 5)
    pairs = [(health_dict[ms][d], fwd[d]) for d in health_dict[ms] if d in fwd]
    if len(pairs) > 5:
        h = np.array([p[0] for p in pairs])
        f = np.array([p[1] for p in pairs])
        pr, pp = pearsonr(h, f)
    else:
        pr, pp = 0, 1
    
    print(f'{ms}: AUC={auc:.3f}  Acc={acc:.3f}  Brier={brier:.3f}  r(health,forward)={pr:+.3f}(p={pp:.3f})')
    print(f'     y_prob range: [{y_prob_ms.min():.1f}%, {y_prob_ms.max():.1f}%], n={len(y_true)}')
    print()

# Forward return correlation in detail
lookaheads_to_check = [1, 3, 5, 10]
print("=== Health vs Forward Return Correlation (All Lookaheads) ===\\n")
for ms in ['i54','i64','i6']:
    sub = df_by_model[ms]
    print(f'--- {ms} ---')
    for la in lookaheads_to_check:
        fwd = forward_rets(sub, la)
        pairs = [(health_dict[ms][d], fwd[d]) for d in health_dict[ms] if d in fwd]
        if len(pairs) < 5: continue
        h = np.array([p[0] for p in pairs])
        f = np.array([p[1] for p in pairs])
        pr, pp = pearsonr(h, f)
        sr, sp = spearmanr(h, f)
        sig = '***' if pp < 0.01 else ('**' if pp < 0.05 else ('*' if pp < 0.1 else ''))
        acc = ((h > 50) == (f > 0)).mean()
        print(f'  la={la:2d}d  r={pr:+.3f}(p={pp:.4f}){sig}  rho={sr:+.3f}  acc={acc:.2f}  n={len(pairs)}')
    print()
""")

md("""## 5. Compare: Logistic Health Score vs Simple Weighted Score""")

code("""# Simple baseline: equal-weighted z-score of top features
print("=== Comparison: Logistic Regression vs Simple Weighted ===\\n")
top_feats = coef_df[coef_df['coef'] != 0]['feature'].head(6).tolist()
print(f'Top features for simple baseline: {top_feats}')
print()

for ms in ['i54','i64','i6']:
    sub = df_by_model[ms]
    X_ms = all_X[ms][X_pool.columns].dropna()
    
    # Logistic health
    y_prob_log = lr.predict_proba(scaler.transform(X_ms))[:, 1] * 100
    
    # Simple weighted baseline
    X_sub = X_ms[top_feats].copy()
    X_sub = (X_sub - X_sub.mean()) / X_sub.std().clip(lower=1e-6)
    simple_health = X_sub.mean(axis=1).clip(lower=0) * 50 + 50  # shift to 0-100 range
    
    # Compare correlations
    print(f'{ms}:')
    for la in [1, 3, 5, 10]:
        fwd = forward_rets(sub, la)
        pairs_log = [(y_prob_log[i], fwd[d]) for i, d in enumerate(X_ms.index) if d in fwd]
        pairs_simple = [(simple_health.iloc[i], fwd[d]) for i, d in enumerate(X_ms.index) if d in fwd]
        
        if len(pairs_log) < 5: continue
        h_log = np.array([p[0] for p in pairs_log])
        h_simple = np.array([p[0] for p in pairs_simple])
        f = np.array([p[1] for p in pairs_simple])
        
        r_log, p_log = pearsonr(h_log, f)
        r_simple, p_simple = pearsonr(h_simple, f)
        
        sig_log = '**' if p_log < 0.05 else ''
        sig_simple = '**' if p_simple < 0.05 else ''
        print(f'  la={la:2d}d  logistic: r={r_log:+.3f}(p={p_log:.3f}){sig_log}  simple: r={r_simple:+.3f}(p={p_simple:.3f}){sig_simple}')
    print()
""")

md("""## 6. Visualize""")

code("""fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ai, ms in enumerate(['i54','i64','i6']):
    ax = axes[ai]
    h = health_dict[ms]
    dates = pd.to_datetime(list(h.keys()))
    vals = list(h.values())
    ax.plot(dates, vals, color=MCOLOR[ms], lw=2.5)
    ax.fill_between(dates, 0, vals, alpha=0.15, color=MCOLOR[ms])
    ax.axhline(80, color='green', ls=':', alpha=0.7)
    ax.axhline(60, color='orange', ls=':', alpha=0.7)
    ax.axhline(40, color='red', ls=':', alpha=0.7)
    last_v = vals[-1]
    flag = 'GREEN' if last_v >= 80 else ('YELLOW' if last_v >= 60 else ('ORANGE' if last_v >= 40 else 'RED'))
    ax.annotate(f'{last_v:.0f} ({flag})', xy=(dates[-1], last_v),
                xytext=(10, 10), textcoords='offset points', fontsize=9, fontweight='bold',
                color=MCOLOR[ms])
    ax.set_title(f'{ms} | P(profit) = {last_v:.0f}%', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.set_ylabel('Health = P(fwd ret > 0) (%)') 
    ax.grid(True, alpha=0.3)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
plt.suptitle('Health = Predicted Probability of Profit', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
""")

md("""## 7. Final Configuration""")

code("""print('=== FINAL CONFIGURATION ===\\n')
print('Method: Logistic Regression (L1-regularized)')
print(f'Features used: {len(X_pool.columns)}')
print(f'Non-zero coefficients: {(lr.coef_[0] != 0).sum()}')
print()
print(f'Top predictors:')
for _, row in coef_df.head(8).iterrows():
    print(f'  +{row["coef"]:+.4f}  {row["feature"]}')
print()
print(f'Worst predictors:')
for _, row in coef_df.tail(8).iterrows():
    print(f'  {row["coef"]:+.4f}  {row["feature"]}')
print()
print(f'Thresholds: GREEN>=80, YELLOW>=60, ORANGE>=40, RED<40')
print()
for ms in ['i54','i64','i6']:
    h = health_dict[ms]
    last_d = list(h.keys())[-1]
    last_v = list(h.values())[-1]
    flag = 'GREEN' if last_v >= 80 else ('YELLOW' if last_v >= 60 else ('ORANGE' if last_v >= 40 else 'RED'))
    print(f'  [{flag:6s}] {ms}: {last_v:.1f}% (as of {last_d})')
""")

nb.cells = cells
out_path = '/mnt/c/Users/xyl/Desktop/ETF/notebooks/model_degradation_experiment_v4.ipynb'
nbf.write(nb, out_path)
print(f'Done: {out_path}')
