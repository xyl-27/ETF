"""
v7: ONE set of (window, weight) across ALL models.
Maximize corr(health, future_win_rate).
Brute-force search over windows + optimize weights.
"""
import nbformat as nbf
from pathlib import Path
nb=nbf.v4.new_notebook()
nb.metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
             "language_info":{"name":"python","version":"3.12.0"}}
cs=[];md=lambda s:cs.append(nbf.v4.new_markdown_cell(s));cd=lambda s:cs.append(nbf.v4.new_code_cell(s))

md("# v7: Universal Health = weighted sum of rolling indicators. Find best (window, weight) for ALL models.")

md("## 1. Load Data")

cd("""import numpy as np, pandas as pd, json, matplotlib.pyplot as plt, itertools, random
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import warnings; warnings.filterwarnings('ignore')
_NB_DIR=Path().resolve()
for _p in [_NB_DIR,_NB_DIR.parent,Path('/mnt/c/Users/xyl/Desktop/ETF')]:
    if (_p/'output').exists(): ROOT=_p; break
else: ROOT=_NB_DIR
with open(str(ROOT/'output'/'backtest_state.json')) as f: state=json.load(f)
seqs=state.get('sequences',state)
ALL_MODELS={'i54':'search_itransformer_exp_54','i64':'search_itransformer_exp_64',
            'i6':'search_itransformer_exp_6','avg':'average','vote':'voting'}
MCOLOR={'i54':'#E24A33','i64':'#348ABD','i6':'#988ED5','avg':'#2ECC40','vote':'#FF851B'}
print('Loaded')
""")

cd("""# Parse data
raw_vars={}
for ms,m in ALL_MODELS.items():
    s=seqs[m]; eq=s.get('equity_curve',[])
    dates=[e['date'].strftime('%Y-%m-%d') if hasattr(e['date'],'strftime') else str(e['date'])[:10] for e in eq]
    vals=[e['total_value'] for e in eq]
    dr_vals=[(vals[i]/vals[i-1]-1)*100 for i in range(1,len(vals))]
    dr_dates=dates[1:]
    # Daily win rate
    daily_win=[1 if v>0 else 0 for v in dr_vals]
    raw_vars[ms]={'dates':dates,'dr_dates':dr_dates,'dr_vals':dr_vals,'daily_win':daily_win,'eq_vals':vals}
print('Parsed')
""")

md("## 2. Define Indicators + Targets")

cd("""def compute_indicator(ms, name, window):
    \"\"\"Compute a rolling indicator for one model. Returns dict {date: value}.\"\"\"
    r=raw_vars[ms]
    dw=r['daily_win']; dr=r['dr_vals']; dd=r['dr_dates']; eq=r['eq_vals']
    out={}
    for i,d in enumerate(dd):
        s=max(0,i-window+1)
        if name=='win_rate':
            out[d]=sum(dw[s:i+1])/len(dw[s:i+1])
        elif name=='avg_ret':
            out[d]=float(np.mean(dr[s:i+1]))
        elif name=='vol':
            if len(dr[s:i+1])>=3:
                out[d]=float(np.std(dr[s:i+1]))
        elif name=='sharpe':
            sub=dr[s:i+1]
            if len(sub)>=5 and np.std(sub)>0:
                out[d]=float(np.mean(sub)/np.std(sub)*np.sqrt(252))
        elif name=='ddepth':
            out[d]=(eq[i+1]/max(eq[:i+2])-1)*100
    return out

def compute_target(ms, lookahead):
    \"\"\"Future win rate. {date: value}\"\"\"
    r=raw_vars[ms]; dw=r['daily_win']; dd=r['dr_dates']
    out={}
    for i,d in enumerate(dd):
        if i+lookahead<len(dw):
            out[d]=sum(dw[i:i+lookahead])/lookahead
    return out

# Candidate indicator types
IND_NAMES=['win_rate','avg_ret','vol','sharpe','ddepth']
WINDOWS=[3,5,10,15,20]
FIXED_LA=5  # lookahead for target

# Precompute TARGETS for all models
TARGETS={ms:compute_target(ms,FIXED_LA) for ms in ALL_MODELS}
print('Targets ready')
""")

md("## 3. Universal Health Score")

cd("""class UniversalHealth:
    \"\"\"health = sum(w_i * indicator_i(window_i)). Normalized 0-100.\"\"\"
    def __init__(self, config):
        # config: [(name, window, weight, direction), ...]
        # direction: 1=higher=healthier, -1=lower=healthier
        self.config=config
    
    def score(self, ms):
        # Compute each indicator
        series_list=[]
        for name,window,weight,direction in self.config:
            s=compute_indicator(ms,name,window)
            if not s: continue
            vals=np.array(list(s.values()))
            if vals.max()-vals.min()<1e-10: continue
            # Normalize 0-1
            norm={d:(v-vals.min())/(vals.max()-vals.min()) for d,v in s.items()}
            if direction==-1:
                norm={d:1-v for d,v in norm.items()}
            series_list.append((norm,weight))
        # Weighted sum
        all_dates=set()
        for norm,_ in series_list: all_dates.update(norm.keys())
        health={}
        for d in sorted(all_dates):
            total=0.0; wsum=0.0
            for norm,w in series_list:
                if d in norm:
                    total+=norm[d]*abs(w)
                    wsum+=abs(w)
            health[d]=total/wsum if wsum>0 else 0
        # Normalize 0-100
        if health:
            hv=np.array(list(health.values()))
            if hv.max()-hv.min()>1e-10:
                for d in health: health[d]=(health[d]-hv.min())/(hv.max()-hv.min())*100
            else:
                for d in health: health[d]=50
        return health

def fitness(config):
    \"\"\"Average Pearson r across all models. Penalize variance across models.\"\"\"
    uh=UniversalHealth(config)
    rs=[]; model_rs={}
    for ms in ALL_MODELS:
        h=uh.score(ms); t=TARGETS[ms]
        pairs=[(h[d],t[d]) for d in h if d in t]
        if len(pairs)<5: continue
        hv=np.array([p[0] for p in pairs]); tv=np.array([p[1] for p in pairs])
        if np.std(hv)==0 or np.std(tv)==0: continue
        r,p=pearsonr(hv,tv)
        model_rs[ms]=r
        if p<0.05: rs.append(r)
    if len(rs)<2: return -999
    avg_r=np.mean(rs)
    std_r=np.std(rs)
    # Penalize high variance: score = avg_r - 0.5 * std_r
    return avg_r - 0.5 * std_r

# Test: equal-weighted win_rate at different windows
print('Testing equal-weighted combos...')
for w in WINDOWS:
    config=[('win_rate',w,1,1)]
    f=fitness(config)
    print(f'  win_rate n={w:2d}: r={f:+.4f}')
print()
# Combo: win_rate + sharpe
for w1 in [5,10]:
    for w2 in [10,15]:
        config=[('win_rate',w1,1,1),('sharpe',w2,1,1)]
        f=fitness(config)
        print(f'  win_rate n={w1:2d} + sharpe n={w2:2d}: r={f:+.4f}')
""")

md("## 4. Search: Find Best Config")

cd("""random.seed(42); np.random.seed(42)

def random_config():
    n=random.randint(2,5)
    names=random.sample(IND_NAMES,n)
    windows=random.choices(WINDOWS,k=n)
    weights=[random.uniform(0.5,2) for _ in range(n)]
    directions=[1 if random.random()<0.7 else -1 for _ in range(n)]
    # For ddepth and vol, direction should be -1
    for i,name in enumerate(names):
        if name in ('ddepth','vol'): directions[i]=-1
    return list(zip(names,windows,weights,directions))

def mutate(config):
    cfg=list(config)
    if random.random()<0.3:
        # replace one
        i=random.randrange(len(cfg))
        name=random.choice(IND_NAMES)
        w=random.choice(WINDOWS)
        wt=random.uniform(0.5,2)
        d=-1 if name in ('ddepth','vol') else 1
        cfg[i]=(name,w,wt,d)
    if random.random()<0.3:
        # jitter weight
        i=random.randrange(len(cfg))
        n,wn,wt,d=cfg[i]
        wt*=random.uniform(0.7,1.3)
        cfg[i]=(n,wn,wt,d)
    if random.random()<0.2 and len(cfg)<6:
        name=random.choice(IND_NAMES)
        cfg.append((name,random.choice(WINDOWS),random.uniform(0.5,2),
                     -1 if name in ('ddepth','vol') else 1))
    if random.random()<0.2 and len(cfg)>2:
        cfg.pop(random.randrange(len(cfg)))
    return cfg

POP=300; GENS=8; KEEP=40
pop=[random_config() for _ in range(POP)]
best_cfg=None; best_f=-999

print(f'Population {POP}, Generations {GENS}')
for gen in range(GENS):
    scores=[(fitness(c),c) for c in pop]
    scores.sort(key=lambda x:x[0],reverse=True)
    if scores[0][0]>best_f:
        best_f=scores[0][0]; best_cfg=scores[0][1]
        print(f'Gen {gen+1}: NEW BEST r={best_f:.4f}')
    else:
        print(f'Gen {gen+1}: best={scores[0][0]:.4f} (overall={best_f:.4f})')
    selected=[c for _,c in scores[:KEEP]]
    pop=list(selected)
    while len(pop)<POP:
        pop.append(mutate(random.choice(selected)))

print(f'\\nBest fitness (avg_r - 0.5*std_r): {best_f:.4f}')
print(f'Config ({len(best_cfg)} indicators):')
for name,w,wt,d in best_cfg:
    dir_s='higher=better' if d==1 else 'lower=better'
    print(f'  {name:10s} win={w:2d} wgt={wt:.3f}  {dir_s}')
""")

md("## 5. Validate Best Config")

cd("""uh=UniversalHealth(best_cfg)
print('=== Validation: Health vs Future Win Rate ===\\n')
for ms in ALL_MODELS:
    h=uh.score(ms); t=TARGETS[ms]
    pairs=[(h[d],t[d]) for d in h if d in t]
    if len(pairs)<5: continue
    hv=np.array([p[0] for p in pairs]); tv=np.array([p[1] for p in pairs])
    pr,pp=pearsonr(hv,tv)
    sr,sp=spearmanr(hv,tv)
    sig='***' if pp<0.01 else ('**' if pp<0.05 else ('*' if pp<0.1 else ''))
    print(f'{ms}: r={pr:+.3f}(p={pp:.4f}){sig} rho={sr:+.3f} n={len(pairs)}')
    print(f'     Latest: {list(h.values())[-1]:.1f}')
    print()
""")

md("## 6. Visualize")

cd("""fig,axes=plt.subplots(2,3,figsize=(18,8)); axes=axes.flatten()
for ai,ms in enumerate(ALL_MODELS):
    ax=axes[ai]
    h=uh.score(ms)
    dates=pd.to_datetime(list(h.keys())); vals=list(h.values())
    ax.plot(dates,vals,color=MCOLOR[ms],lw=2.5)
    ax.fill_between(dates,0,vals,alpha=0.15,color=MCOLOR[ms])
    ax.axhline(80,color='green',ls=':',alpha=0.7)
    ax.axhline(60,color='orange',ls=':',alpha=0.7)
    ax.axhline(40,color='red',ls=':',alpha=0.7)
    last_v=vals[-1]
    flag='GREEN' if last_v>=80 else ('YELLOW' if last_v>=60 else ('ORANGE' if last_v>=40 else 'RED'))
    ax.annotate(f'{last_v:.0f}({flag})',xy=(dates[-1],last_v),xytext=(10,10),
                textcoords='offset points',fontsize=9,fontweight='bold',color=MCOLOR[ms])
    ax.set_title(f'{ms}: Health={last_v:.0f}',fontsize=11,fontweight='bold')
    ax.set_ylim(0,105); ax.grid(True,alpha=0.3)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(),rotation=45,ha='right',fontsize=8)
plt.suptitle('Universal Health Score (same weights/windows for all models)',fontsize=14,fontweight='bold')
plt.tight_layout(); plt.show()
""")

md("## 7. Final Config")

cd("""print('=== FINAL UNIVERSAL CONFIG ===')
print(f'Target: future win rate (la={FIXED_LA}d)')
print(f'Indicators:')
for name,w,wt,d in best_cfg:
    dir_s='higher=healthier' if d==1 else 'lower=healthier'
    print(f'  {name:10s} window={w:2d} weight={wt:.3f} ({dir_s})')
print(f'\\nCurrent health:')
for ms in ALL_MODELS:
    h=uh.score(ms)
    ld=list(h.keys())[-1]; lv=list(h.values())[-1]
    flag='GREEN' if lv>=80 else ('YELLOW' if lv>=60 else ('ORANGE' if lv>=40 else 'RED'))
    print(f'  {ms:4s}: {lv:6.1f} [{flag:6s}] (as of {ld})')
""")

nb.cells=cs
out_path='/mnt/c/Users/xyl/Desktop/ETF/notebooks/model_degradation_experiment_v7.ipynb'
nbf.write(nb,out_path)
print(f'Done: {out_path}')
