"""
v5: Health = predicted future win rate.
Core features: rolling win rate + PnL ratio at multiple windows + MRR/IC/NDCG.
All 5 models (i54, i64, i6, avg, vote). Ridge regression + LOOCV.
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

md("""# Model Degradation v5: Health = Predicted Future Win Rate

**Core features**: rolling PnL ratio, win rate at multiple windows.
**Auxiliary**: MRR, IC, NDCG.
**Method**: Ridge regression, target = forward daily win rate.
**All 5 models**: i54, i64, i6, avg, vote.
""")

md("## 1. Load Data")

code("""import numpy as np, pandas as pd, json, matplotlib.pyplot as plt, itertools
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, linregress
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

_NB_DIR = Path().resolve()
for _p in [_NB_DIR, _NB_DIR.parent, Path('/mnt/c/Users/xyl/Desktop/ETF')]:
    if (_p / 'output').exists(): ROOT = _p; break
else: ROOT = _NB_DIR

with open(str(ROOT / 'output' / 'backtest_state.json')) as f:
    state = json.load(f)
seqs = state.get('sequences', state)

MODELS = ['search_itransformer_exp_54','search_itransformer_exp_64',
          'search_itransformer_exp_6','average','voting']
MSHORT = {'search_itransformer_exp_54':'i54','search_itransformer_exp_64':'i64',
          'search_itransformer_exp_6':'i6','average':'avg','voting':'vote'}
MCOLOR = {'i54':'#E24A33','i64':'#348ABD','i6':'#988ED5','avg':'#2ECC40','vote':'#FF851B'}
ALL_MS = ['i54','i64','i6','avg','vote']
""")

code("""# Parse raw data
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

cum_rets = {}
for m in MODELS:
    eq = equities[m]; init = eq[0]['total_value']
    cum_rets[m] = {e['date']: (e['total_value']/init - 1)*100 for e in eq}

daily_rets = {}
for m in MODELS:
    eq = equities[m]; rets = {}
    for i in range(1, len(eq)):
        rets[eq[i]['date']] = (eq[i]['total_value']/eq[i-1]['total_value'] - 1)*100
    daily_rets[m] = rets

# Trade PnL per trade
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

md("## 2. Feature Engineering")

code("""WINDOWS = [3, 5, 10, 15, 20]

def build_dataset(m, lookahead=5):
    ms = MSHORT[m]
    dates = [e['date'] for e in equities[m]]
    dr_vals = [daily_rets[m].get(d, 0) for d in dates]
    eq_vals = [e['total_value'] for e in equities[m]]
    
    # Daily win flag
    daily_win = [1 if v > 0 else 0 for v in dr_vals]
    
    # Trade PnL per date
    pnl_by_date = {}
    for tp in trade_pnl[m]:
        pnl_by_date.setdefault(tp['date'], []).append(tp['pnl_pct'])
    
    # Raw metrics (may be empty for voting)
    raw = {}
    has_metric = len(bareraw[m]) > 0
    if has_metric:
        for k in ['ic','ndcg','mrr','ksp']:
            raw[k+'_dates'] = [e['date'] for e in bareraw[m] if e.get(k) is not None]
            raw[k+'_vals'] = [e[k] for e in bareraw[m] if e.get(k) is not None]
    
    features = []; targets = []; feat_dates = []
    
    for pos, d in enumerate(dates):
        row = {}
        
        # -- Core feature 1: Rolling daily win rate (different windows) --
        for w in WINDOWS:
            s = max(0, pos-w+1)
            sub = daily_win[s:pos+1]
            if sub:
                row[f'wr_{w}'] = sum(sub) / len(sub)
        
        # -- Core feature 2: Rolling avg PnL of closed trades --
        for w in WINDOWS:
            trade_pnls = []
            for i in range(max(0, pos-w+1), pos+1):
                if dates[i] in pnl_by_date:
                    trade_pnls.extend(pnl_by_date[dates[i]])
            if trade_pnls:
                row[f'avgpnl_{w}'] = float(np.mean(trade_pnls))
                row[f'n_trades_{w}'] = len(trade_pnls)
                # PnL ratio: avg profit / abs(avg loss)
                profits = [v for v in trade_pnls if v > 0]
                losses = [v for v in trade_pnls if v <= 0]
                avg_prof = np.mean(profits) if profits else 0
                avg_loss = abs(np.mean(losses)) if losses else 0
                row[f'plratio_{w}'] = avg_prof / avg_loss if avg_loss > 0 else (5 if avg_prof > 0 else 0)
                # Trade win rate
                row[f'twr_{w}'] = len(profits) / len(trade_pnls) if trade_pnls else 0
        
        # -- Core feature 3: Rolling Sharpe --
        for w in WINDOWS:
            s = max(0, pos-w+1)
            sub = dr_vals[s:pos+1]
            if len(sub) >= 5 and np.std(sub) > 0:
                row[f'sharpe_{w}'] = float(np.mean(sub) / np.std(sub) * np.sqrt(252))
        
        # -- Core feature 4: Drawdown --
        running_max = max(eq_vals[:pos+1])
        row['ddepth'] = (eq_vals[pos] / running_max - 1) * 100
        
        # -- Auxiliary: Rolling metric means (if available) --
        if has_metric:
            for k in ['ic','ndcg','mrr','ksp']:
                kdates = raw[k+'_dates']
                kvals = raw[k+'_vals']
                ri = -1
                for idx, kd in enumerate(kdates):
                    if kd <= d: ri = idx
                    else: break
                if ri >= 0:
                    for w in WINDOWS:
                        s_pos = max(0, ri-w+1)
                        sub = kvals[s_pos:ri+1]
                        if sub:
                            row[f'{k}_m_{w}'] = float(np.mean(sub))
                        if len(sub) >= 5:
                            slope,_,_,_,_ = linregress(np.arange(len(sub)), sub)
                            row[f'{k}_t_{w}'] = slope
        
        # -- Target: future win rate over next `lookahead` days --
        if pos + lookahead < len(daily_win):
            fwd_wr = sum(daily_win[pos:pos+lookahead]) / lookahead
            features.append(row)
            targets.append(fwd_wr)
            feat_dates.append(d)
    
    X = pd.DataFrame(features, index=feat_dates)
    y = pd.Series(targets, index=feat_dates, name='fwd_wr')
    return X, y

print('Building datasets (lookahead=5)...')
all_X = {}; all_y = {}
for m in MODELS:
    X, y = build_dataset(m, lookahead=5)
    all_X[MSHORT[m]] = X
    all_y[MSHORT[m]] = y
    print(f'  {MSHORT[m]:4s}: X {str(X.shape):10s} y mean={y.mean():.3f} std={y.std():.3f}')
""")

md("## 3. Train Ridge Regression")

code("""# Pool data from all models
X_pool = pd.concat([all_X[ms] for ms in ALL_MS], axis=0)
y_pool = pd.concat([all_y[ms] for ms in ALL_MS], axis=0)

# Drop features that are all NaN
X_pool = X_pool.dropna(axis=1, how='all')
# Fill remaining NaN with column mean
X_pool = X_pool.fillna(X_pool.mean())

print(f'Pooled: X {X_pool.shape}, y mean={y_pool.mean():.3f}')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pool)

# Ridge with LOOCV
alphas = np.logspace(-3, 3, 20)
ridge_cv = RidgeCV(alphas=alphas, scoring='r2', cv=LeaveOneOut())
ridge_cv.fit(X_scaled, y_pool)
print(f'Best alpha: {ridge_cv.alpha_:.4f}')
print(f'LOOCV R2: {ridge_cv.best_score_:.4f}')

# LOOCV predictions
y_pred_cv = cross_val_predict(Ridge(alpha=ridge_cv.alpha_), X_scaled, y_pool, cv=LeaveOneOut())
loo_r2 = r2_score(y_pool, y_pred_cv)
loo_rmse = np.sqrt(mean_squared_error(y_pool, y_pred_cv))
print(f'LOOCV R2: {loo_r2:.4f}, RMSE: {loo_rmse:.4f}')

# Feature importance
coef_df = pd.DataFrame({
    'feature': X_pool.columns,
    'coef': ridge_cv.coef_
}).sort_values('coef', ascending=False)
print(f'\\n=== Top 15 Features ===')
for _, row in coef_df.head(15).iterrows():
    print(f'  +{row["coef"]:+.4f}  {row["feature"]}')
print(f'\\n=== Bottom 10 Features ===')
for _, row in coef_df.tail(10).iterrows():
    print(f'  {row["coef"]:+.4f}  {row["feature"]}')
""")

md("## 4. Evaluate Health Score vs Forward Returns")

code("""# Predict health scores
y_pred = ridge_cv.predict(X_scaled)
health_all = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min() + 1e-10) * 100

idx = 0
health_dict = {}
for ms in ALL_MS:
    n = len(all_y[ms])
    health_dict[ms] = {d: health_all[idx + i] for i, d in enumerate(all_y[ms].index)}
    idx += n

# df_by_model for forward returns
df_rows = []
for d in all_dates:
    for m in MODELS:
        ret = cum_rets[m].get(d)
        if ret is None: continue
        df_rows.append({'date': d, 'model': MSHORT[m], 'ret_cum': ret})
df_all = pd.DataFrame(df_rows)
df_by_model = {ms: df_all[df_all['model'] == ms] for ms in ALL_MS}

def forward_rets(df_model, la):
    vals = list(df_model['ret_cum'])
    dates = list(df_model['date'])
    fwd = {}
    for i, d in enumerate(dates):
        if i + la < len(vals):
            fwd[d] = vals[i+la] - vals[i]
    return fwd

lookaheads = [1, 3, 5, 10]
print("=== Health vs Forward Return Correlation ===\\n")
for ms in ALL_MS:
    sub = df_by_model[ms]
    print(f'--- {ms} ---')
    for la in lookaheads:
        fwd = forward_rets(sub, la)
        pairs = [(health_dict[ms][d], fwd[d]) for d in health_dict[ms] if d in fwd]
        if len(pairs) < 5: continue
        h = np.array([p[0] for p in pairs])
        f = np.array([p[1] for p in pairs])
        if np.std(h)==0 or np.std(f)==0: continue
        pr, pp = pearsonr(h, f)
        sr, sp = spearmanr(h, f)
        sig = '***' if pp < 0.01 else ('**' if pp < 0.05 else ('*' if pp < 0.1 else ''))
        print(f'  la={la:2d}d  r={pr:+.3f}(p={pp:.4f}){sig}  rho={sr:+.3f}  n={len(pairs)}')
    print()
""")

md("## 5. Visualize")

code("""fig, axes = plt.subplots(2, 3, figsize=(18, 8))
axes = axes.flatten()
for ai, ms in enumerate(ALL_MS):
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
    ax.set_title(f'{ms}: Health={last_v:.0f}', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
plt.suptitle('Health Score = Predicted Future Win Rate (5 models)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
""")

md("## 6. Final Config")

code("""print('=== FINAL CONFIGURATION ===')
print(f'Model: Ridge (alpha={ridge_cv.alpha_:.4f})')
print(f'Target: forward win rate (lookahead=5d)')
print(f'Features: {X_pool.shape[1]}')
print(f'Non-zero coefficients: {(abs(ridge_cv.coef_) > 1e-6).sum()}')
print(f'LOOCV R2: {loo_r2:.3f}')
print(f'Samples: {X_pool.shape[0]} (pooled from {len(ALL_MS)} models)')
print(f'Thresholds: GREEN>=80, YELLOW>=60, ORANGE>=40, RED<40')
print()
print(f'{"Model":6s} {"Health":8s} {"Flag":8s}  As of')
print('-' * 40)
for ms in ALL_MS:
    h = health_dict[ms]
    last_d = list(h.keys())[-1]
    last_v = list(h.values())[-1]
    flag = 'GREEN' if last_v >= 80 else ('YELLOW' if last_v >= 60 else ('ORANGE' if last_v >= 40 else 'RED'))
    print(f'  {ms:4s}   {last_v:6.1f}   {flag:6s}  {last_d}')
""")

nb.cells = cells
out_path = '/mnt/c/Users/xyl/Desktop/ETF/notebooks/model_degradation_experiment_v5.ipynb'
nbf.write(nb, out_path)
print(f'Done: {out_path}')
