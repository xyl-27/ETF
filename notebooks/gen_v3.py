"""
Notebook v3: search-based optimization for model health score.
Goal: find (indicator, window, weight) such that weighted sum has max correlation with forward returns.

This is a 3-level search:
  Level 1: which indicators?     (e.g. win_rate, avg_profit_loss, MRR, IC, NDCG, etc.)
  Level 2: what window size n?    (e.g. 3, 5, 10, 15, 20 days)
  Level 3: what weight w?         (found via grid or random search)
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

md("""# Model Degradation Experiment v3: Weighted Health Score Search

**Problem**: We want `health = sum(w_i * indicator_i)` where each indicator uses a rolling window n_i.
Weights and windows are unknown. We want to maximize correlation between health and future returns.

**Approach**: Treat this as a search problem:
- Candidate indicators: win_rate, profit_loss_ratio, MRR, IC, NDCG, Sharpe, etc.
- For each indicator, try multiple window sizes (3, 5, 10, 15, 20)
- Use random search or grid search to find best weights
- Objective: maximize avg Pearson r across models and lookaheads
""")

md("## 1. Load Data")

code("""import numpy as np, pandas as pd, json, matplotlib.pyplot as plt, itertools, random
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, linregress
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

# Cumulative returns-to-date
cum_rets = {}
for m in MODELS:
    eq = equities[m]; init = eq[0]['total_value']
    cum_rets[m] = {e['date']: (e['total_value']/init - 1)*100 for e in eq}

# Daily returns
daily_rets = {}
for m in MODELS:
    eq = equities[m]; rets = {}
    for i in range(1, len(eq)):
        rets[eq[i]['date']] = (eq[i]['total_value']/eq[i-1]['total_value'] - 1)*100
    daily_rets[m] = rets

# Trades with profits
trade_pnl = {}  # {model: [{date, stock, action, score, pnl_pct}]}
for m in MODELS:
    pnl = []
    prices = {}
    for t in trades[m]:
        if t['action'] == '买入':
            prices[t['stock']] = t['price']
        elif t['action'] == '卖出' and t['stock'] in prices and prices[t['stock']] > 0:
            buy_p = prices[t['stock']]
            pnl_pct = (t['price'] - buy_p) / buy_p * 100
            pnl.append({'date': t['date'], 'stock': t['stock'],
                        'score': t.get('score'), 'pnl_pct': pnl_pct})
            del prices[t['stock']]
    trade_pnl[m] = pnl

# Buy scores per rebalance
buy_scores = {}
for m in MODELS:
    bs = {}
    for t in trades[m]:
        if t.get('action') == chr(20080)+chr(20837) and t.get('score') is not None:
            bs.setdefault(t['date'], []).append(t['score'])
    buy_scores[m] = bs

print('Data loaded')
""")

md("""## 2. Compute Raw Indicator Time Series

For each indicator, compute raw values per date (no rolling yet, just the per-period raw values).
""")

code("""def compute_raw_indicators(m):
    '''Return dict: {indicator_name: [(date, value), ...]}'''
    ms = MSHORT[m]
    result = {}

    # -- 1. Win rate: fraction of winning trades that closed on each date --
    win_by_date = {}
    for tp in trade_pnl[m]:
        d = tp['date']
        if tp['pnl_pct'] > 0:
            win_by_date[d] = win_by_date.get(d, 0) + 1
    # Also count total closes per date
    close_count = {}
    for tp in trade_pnl[m]:
        close_count[tp['date']] = close_count.get(tp['date'], 0) + 1
    result['win_rate'] = [(d, win_by_date.get(d, 0) / close_count[d])
                          for d in sorted(close_count) if close_count[d] > 0]

    # -- 2. Profit/Loss ratio: avg profit / avg loss for trades closing on each date --
    pl_by_date = {}
    for tp in trade_pnl[m]:
        d = tp['date']; pnl = tp['pnl_pct']
        pl_by_date.setdefault(d, {'profits': [], 'losses': []})
        if pnl > 0: pl_by_date[d]['profits'].append(pnl)
        else: pl_by_date[d]['losses'].append(pnl)
    pl_ratio = {}
    for d in sorted(pl_by_date):
        prof = pl_by_date[d]['profits']
        loss = pl_by_date[d]['losses']
        avg_prof = np.mean(prof) if prof else 0
        avg_loss = abs(np.mean(loss)) if loss else 0
        if avg_loss > 0:
            pl_ratio[d] = avg_prof / avg_loss
    result['pl_ratio'] = sorted(pl_ratio.items())

    # -- 3. MRR daily --
    mrr_vals = [(e['date'], e['mrr']) for e in bareraw[m] if e.get('mrr') is not None]
    result['mrr'] = mrr_vals

    # -- 4. IC daily --
    ic_vals = [(e['date'], e['ic']) for e in bareraw[m] if e.get('ic') is not None]
    result['ic'] = ic_vals

    # -- 5. NDCG daily --
    ndcg_vals = [(e['date'], e['ndcg']) for e in bareraw[m] if e.get('ndcg') is not None]
    result['ndcg'] = ndcg_vals

    # -- 6. KS-p daily (lower = better) --
    ksp_vals = [(e['date'], e['ksp']) for e in bareraw[m] if e.get('ksp') is not None]
    result['ksp'] = ksp_vals

    # -- 7. Daily return %
    dr_vals = [(d, daily_rets[m].get(d, 0)) for d in all_dates if d in daily_rets[m]]
    result['daily_ret'] = dr_vals

    # -- 8. Sharpe: daily ret volatility-adjusted --
    # We'll compute this via rolling later

    # -- 9. Score std (conviction) --
    score_std_list = []
    for d in all_dates:
        # forward-fill from last rebalance
        pass  # computed in rolling stage

    return result

raw_data = {MSHORT[m]: compute_raw_indicators(m) for m in MODELS}

for ms in ['i54','i64','i6']:
    print(f'{ms}:')
    for name, vals in raw_data[ms].items():
        n = len(vals)
        if n > 0:
            print(f'  {name:12s}: {n} points, range [{vals[0][0]} ~ {vals[-1][0]}]')
""")

md("""## 3. Indicator Transformers: Rolling Window Functions

Each indicator can be summarized over a window using a function.
""")

code("""def roll_mean(vals, n):
    # vals: list of values. Return {index: rolling_mean}
    out = {}
    for i in range(len(vals)):
        s = max(0, i-n+1)
        out[i] = float(np.mean(vals[s:i+1]))
    return out

def roll_sharpe(daily_rets_list, n):
    # daily_rets_list: [(date, ret)]. Rolling Sharpe.
    rets = [r for _, r in daily_rets_list]
    out = {}
    for i in range(len(rets)):
        s = max(0, i-n+1)
        sub = rets[s:i+1]
        if len(sub) >= 5 and np.std(sub) > 0:
            out[i] = np.mean(sub) / np.std(sub) * np.sqrt(252)
    return out

class RollingIndicator:
    def __init__(self, name, raw_key, transform_fn, n, invert=False):
        self.name = name
        self.raw_key = raw_key
        self.transform_fn = transform_fn
        self.n = n
        self.invert = invert

    def compute(self, raw_vals):
        # raw_vals: [(date, value), ...] sorted by date.
        # Returns: {date_str: score_component}
        if not raw_vals:
            return {}
        values = [v for _, v in raw_vals]
        idx_map = roll_mean(values, self.n) if self.transform_fn == 'mean' else \
                  roll_sharpe(raw_vals, self.n) if self.transform_fn == 'sharpe' else \
                  {}
        dates = [d for d, _ in raw_vals]
        result = {}
        for i, d in enumerate(dates):
            if i in idx_map:
                v = idx_map[i]
                if self.invert:
                    v = -v
                result[d] = v
        return result

# Define candidate indicators
N_CANDIDATES = [3, 5, 10, 15, 20]

def make_candidates():
    cands = []
    # Win rate (mean, higher = better)
    for n in N_CANDIDATES:
        cands.append(RollingIndicator('win_rate', 'win_rate', 'mean', n, invert=False))
    # Profit/loss ratio (mean, higher = better)
    for n in N_CANDIDATES:
        cands.append(RollingIndicator('pl_ratio', 'pl_ratio', 'mean', n, invert=False))
    # MRR (mean, higher = better)
    for n in N_CANDIDATES:
        cands.append(RollingIndicator('mrr', 'mrr', 'mean', n, invert=False))
    # IC (mean, higher = better)
    for n in N_CANDIDATES:
        cands.append(RollingIndicator('ic', 'ic', 'mean', n, invert=False))
    # NDCG (mean, higher = better)
    for n in N_CANDIDATES:
        cands.append(RollingIndicator('ndcg', 'ndcg', 'mean', n, invert=False))
    # KS-p (mean, LOWER = better, so invert)
    for n in N_CANDIDATES:
        cands.append(RollingIndicator('ksp', 'ksp', 'mean', n, invert=True))
    # Daily return (Sharpe, higher = better)
    for n in N_CANDIDATES:
        cands.append(RollingIndicator('sharpe', 'daily_ret', 'sharpe', n, invert=False))
    return cands

ALL_CANDIDATES = make_candidates()
print(f'Total candidate indicators: {len(ALL_CANDIDATES)}')
for c in ALL_CANDIDATES:
    fn_short = 'mean' if c.transform_fn == 'mean' else 'sharpe'
    arrow = '<' if c.invert else '>'
    print(f'  {c.name:8s} n={c.n:2d} {fn_short:6s} health={arrow}')
""")

md("""## 4. Health Score Calculation

`health(t) = sum(w_i * indicator_i(t))` normalized to 0-100.
""")

code("""class HealthModel:
    def __init__(self, indicators, weights):
        self.indicators = indicators
        self.weights = np.array(weights)

    def compute(self, raw_data_dict):
        # raw_data_dict: {raw_key: [(date, value), ...]}
        # Returns: {date: health_score (0-100)} for each model
        # Compute each indicator's contribution
        contribs = []
        all_dates = set()
        for ind, w in zip(self.indicators, self.weights):
            raw_vals = raw_data_dict.get(ind.raw_key, [])
            series = ind.compute(raw_vals)
            # Scale to 0-1 within series
            if series:
                vals = np.array(list(series.values()))
                mn, mx = vals.min(), vals.max()
                if mx > mn:
                    for d in series:
                        series[d] = (series[d] - mn) / (mx - mn)
                elif mx == 0:
                    for d in series:
                        series[d] = 0.0
                else:
                    for d in series:
                        series[d] = 0.5
            contribs.append(series)
            all_dates.update(series.keys())

        # Weighted sum
        health = {}
        for d in sorted(all_dates):
            total = 0.0
            for series, w in zip(contribs, self.weights):
                total += series.get(d, 0) * w
            health[d] = total

        # Normalize to 0-100
        if health:
            vals = np.array(list(health.values()))
            mn, mx = vals.min(), vals.max()
            if mx > mn:
                for d in health:
                    health[d] = (health[d] - mn) / (mx - mn) * 100
            else:
                for d in health:
                    health[d] = 50.0
        return health

def forward_rets(df_model, la):
    # Forward cumulative return over next la days
    vals = list(df_model['ret_cum'])
    dates = list(df_model['date'])
    fwd = {}
    for i, d in enumerate(dates):
        if i + la < len(vals):
            fwd[d] = vals[i+la] - vals[i]
    return fwd

def evaluate_health(health, df_model, lookaheads=[1,3,5,10]):
    # Pearson r between health and forward returns. Returns avg r across lookaheads.
    r_sum = 0.0; r_cnt = 0
    for la in lookaheads:
        fwd = forward_rets(df_model, la)
        pairs = [(health[d], fwd[d]) for d in health if d in fwd and fwd[d] is not None]
        if len(pairs) < 5: continue
        h_vals = np.array([p[0] for p in pairs])
        f_vals = np.array([p[1] for p in pairs])
        if np.std(h_vals) == 0 or np.std(f_vals) == 0: continue
        r, p = pearsonr(h_vals, f_vals)
        if p < 0.05:
            r_sum += r
            r_cnt += 1
    return r_sum / max(r_cnt, 1), r_cnt
""")

md("""## 5. Search: Find Best (Indicators, Weights)

Strategy:
1. Start with a small random population
2. Evaluate each on forward-return correlation (avg r across models/lookaheads)
3. Keep best, mutate and crossover to generate new candidates
4. Repeat

This is essentially a genetic algorithm / random search over indicator combinations and weights.
""")

code("""random.seed(42)
np.random.seed(42)

LOOKAHEADS = [1, 3, 5, 10]
POPULATION_SIZE = 200
GENERATIONS = 5
SELECTION = 30  # keep top N each generation

# Build date-indexed DataFrame for forward return computation
df_rows = []
for d in all_dates:
    for m in MODELS:
        ret = cum_rets[m].get(d)
        if ret is None: continue
        df_rows.append({'date': d, 'model': MSHORT[m], 'ret_cum': ret})
df_all = pd.DataFrame(df_rows)
df_by_model = {ms: df_all[df_all['model'] == ms] for ms in ['i54','i64','i6']}

def random_candidate():
    # Create a random HealthModel
    n_indicators = random.randint(2, 6)
    selected = random.sample(ALL_CANDIDATES, n_indicators)
    weights = np.random.uniform(0.5, 2.0, n_indicators)
    weights = weights / weights.sum() * n_indicators
    return HealthModel(selected, weights)

def mutate_candidate(hp):
    # Slightly mutate a HealthModel
    indicators = list(hp.indicators)
    weights = list(hp.weights)
    for i in range(len(indicators)):
        if random.random() < 0.3:
            # Replace with random indicator
            indicators[i] = random.choice(ALL_CANDIDATES)
        if random.random() < 0.3:
            # Jitter weight
            weights[i] *= np.random.uniform(0.7, 1.3)
    # Maybe add or remove indicator
    if random.random() < 0.2 and len(indicators) < 8:
        indicators.append(random.choice(ALL_CANDIDATES))
        weights.append(np.random.uniform(0.5, 2.0))
    if random.random() < 0.2 and len(indicators) > 2:
        idx = random.randrange(len(indicators))
        indicators.pop(idx)
        weights.pop(idx)
    # Re-normalize
    w = np.array(weights)
    w = w / w.sum() * len(w)
    return HealthModel(indicators, w)

def fitness(hp):
    # Evaluate a HealthModel. Returns avg Pearson r across all models and lookaheads.
    r_sum = 0.0; r_cnt = 0
    for ms in ['i54','i64','i6']:
        raw = raw_data[ms]
        health = hp.compute(raw)
        r_avg, r_n = evaluate_health(health, df_by_model[ms], LOOKAHEADS)
        if r_n > 0:
            r_sum += r_avg
            r_cnt += 1
    if r_cnt == 0: return -999
    return r_sum / r_cnt

print(f'Population: {POPULATION_SIZE}, Generations: {GENERATIONS}')
print(f'Evaluating each candidate on {len(LOOKAHEADS)} lookaheads x 3 models')
print()

# Initial population
population = [random_candidate() for _ in range(POPULATION_SIZE)]
best_overall = None
best_fitness = -999
history = []

for gen in range(GENERATIONS):
    print(f'Generation {gen+1}/{GENERATIONS}...')
    scores = [(fitness(hp), hp) for hp in population]
    scores.sort(key=lambda x: x[0], reverse=True)

    gen_best = scores[0][1]
    gen_fit = scores[0][0]
    history.append(gen_fit)

    if gen_fit > best_fitness:
        best_fitness = gen_fit
        best_overall = gen_best
        print(f'  New best! fitness={gen_fit:.4f}')
    else:
        print(f'  Best this gen: {gen_fit:.4f} (overall: {best_fitness:.4f})')

    # Selection + mutation
    selected = [hp for _, hp in scores[:SELECTION]]
    population = list(selected)
    while len(population) < POPULATION_SIZE:
        parent = random.choice(selected)
        population.append(mutate_candidate(parent))

print(f'\\n=== Search Complete ===')
print(f'Best fitness (avg Pearson r): {best_fitness:.4f}')
print(f'Best model has {len(best_overall.indicators)} indicators:')
total_w = sum(best_overall.weights)
for ind, w in zip(best_overall.indicators, best_overall.weights):
    pct = w / total_w * 100
    arrow = 'lower=better' if ind.invert else 'higher=better'
    print(f'  {ind.name:8s} n={ind.n:2d}  w={w:.3f} ({pct:.0f}%)  {arrow}')
""")

md("""## 6. Validate Best Model""")

code("""# Re-evaluate best model in full detail
print("=== Best Model: Full Validation ===\\n")
for ms in ['i54','i64','i6']:
    raw = raw_data[ms]
    health = best_overall.compute(raw)
    sub = df_by_model[ms]
    print(f'--- {ms} ---')
    for la in LOOKAHEADS:
        fwd = forward_rets(sub, la)
        pairs = [(health[d], fwd[d]) for d in health if d in fwd]
        if len(pairs) < 5: continue
        h = np.array([p[0] for p in pairs])
        f = np.array([p[1] for p in pairs])
        pr, pp = pearsonr(h, f)
        sr, sp = spearmanr(h, f)
        sig = '***' if pp < 0.01 else ('**' if pp < 0.05 else ('*' if pp < 0.1 else ''))
        print(f'  la={la:2d}d  r={pr:+.4f}  p={pp:.4f}{sig}  rho={sr:+.4f}  n={len(pairs)}')
    print()
""")

md("""## 7. Visualize""")

code("""fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ai, ms in enumerate(['i54','i64','i6']):
    ax = axes[ai]
    raw = raw_data[ms]
    health = best_overall.compute(raw)
    dates = pd.to_datetime(list(health.keys()))
    vals = list(health.values())
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
    ax.set_title(f'{ms} | Health Score', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
plt.suptitle('Optimal Health Score (Weighted Indicators)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
""")

md("""## 8. Final Configuration""")

code("""print('=== FINAL CONFIGURATION ===\\n')
print(f'Method: weighted sum of rolling indicators, normalized to 0-100')
print(f'Total indicators: {len(best_overall.indicators)}\\n')
print(f'{"Indicator":10s} {"Window":8s} {"Weight":8s} {"Direction":15s}')
print('-' * 45)
for ind, w in zip(best_overall.indicators, best_overall.weights):
    arrow = 'lower=healthier' if ind.invert else 'higher=healthier'
    print(f'{ind.name:10s} n={ind.n:3d}     {w:.3f}    {arrow}')
print()
print(f'Thresholds: GREEN>=80, YELLOW>=60, ORANGE>=40, RED<40')
print()
for ms in ['i54','i64','i6']:
    raw = raw_data[ms]
    health = best_overall.compute(raw)
    last_d = list(health.keys())[-1]
    last_v = list(health.values())[-1]
    flag = 'GREEN' if last_v >= 80 else ('YELLOW' if last_v >= 60 else ('ORANGE' if last_v >= 40 else 'RED'))
    print(f'  [{flag:6s}] {ms}: {last_v:.1f} (as of {last_d})')
""")

nb.cells = cells
out_path = '/mnt/c/Users/xyl/Desktop/ETF/notebooks/model_degradation_experiment_v3.ipynb'
nbf.write(nb, out_path)
print(f'Done: {out_path}')
