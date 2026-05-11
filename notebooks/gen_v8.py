"""
v8: Compare universal health vs single indicators (win_rate, pl_ratio) directly.
Which predicts future win rate better?
"""
import nbformat as nbf
from pathlib import Path
nb=nbf.v4.new_notebook()
nb.metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
             "language_info":{"name":"python","version":"3.12.0"}}
cs=[];md=lambda s:cs.append(nbf.v4.new_markdown_cell(s));cd=lambda s:cs.append(nbf.v4.new_code_cell(s))

md("# v8: Universal Health vs Single Indicators (Win Rate, PnL Ratio)")

md("## 1. Load Data")

cd("""import numpy as np, pandas as pd, json, matplotlib.pyplot as plt
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

cd("""# Parse all data
raw_vars={}
for ms,m in ALL_MODELS.items():
    s=seqs[m]; eq=s.get('equity_curve',[])
    dates=[e['date'].strftime('%Y-%m-%d') if hasattr(e['date'],'strftime') else str(e['date'])[:10] for e in eq]
    vals=[e['total_value'] for e in eq]
    dr=[(vals[i]/vals[i-1]-1)*100 for i in range(1,len(vals))]
    dd=dates[1:]
    dw=[1 if v>0 else 0 for v in dr]
    # Trade PnL
    tr=s.get('trades',[])
    pnl_by_date={}
    prices={}
    for t in tr:
        if t['action']=='买入': prices[t['stock']]=t['price']
        elif t['action']=='卖出' and t['stock'] in prices:
            pnl=(t['price']-prices[t['stock']])/prices[t['stock']]*100
            d=t['date'].strftime('%Y-%m-%d') if hasattr(t['date'],'strftime') else str(t['date'])[:10]
            pnl_by_date.setdefault(d,[]).append(pnl)
            del prices[t['stock']]
    raw_vars[ms]={'dd':dd,'dr':dr,'dw':dw,'vals':vals,'dates':dates,'pnl':pnl_by_date}
print('Parsed')
""")

md("## 2. Compute All Indicators + Target")

cd("""WINS=[3,5,10,15,20]
LA=5  # lookahead for target

def roll(data, win, fn):
    out={}
    for i,d in enumerate(data['dd']):
        s=max(0,i-win+1)
        if fn=='mean': out[d]=float(np.mean(data['dr'][s:i+1]))
        elif fn=='win': out[d]=sum(data['dw'][s:i+1])/len(data['dw'][s:i+1])
        elif fn=='vol' and len(data['dr'][s:i+1])>=3: out[d]=float(np.std(data['dr'][s:i+1]))
        elif fn=='sharpe':
            sub=data['dr'][s:i+1]
            if len(sub)>=5 and np.std(sub)>0: out[d]=float(np.mean(sub)/np.std(sub)*np.sqrt(252))
        elif fn=='dd':
            out[d]=(data['vals'][i+1]/max(data['vals'][:i+2])-1)*100
    return out

def roll_pnl(data, win):
    out={}
    for i,d in enumerate(data['dd']):
        pnls=[]
        for j in range(max(0,i-win+1),i+1):
            if data['dd'][j] in data['pnl']: pnls.extend(data['pnl'][data['dd'][j]])
        if pnls:
            prof=[v for v in pnls if v>0]; loss=[v for v in pnls if v<0]
            out[d]=len(prof)/len(pnls) if pnls else 0  # trade win rate
    return out

def target(data):
    out={}
    for i,d in enumerate(data['dd']):
        if i+LA<len(data['dw']): out[d]=sum(data['dw'][i:i+LA])/LA
    return out

# Precompute all
all_indicators={}
for ms in ALL_MODELS:
    d=raw_vars[ms]
    inds={}
    for w in WINS:
        inds[f'wr_{w}']=roll(d,w,'win')
        inds[f'avgret_{w}']=roll(d,w,'mean')
        inds[f'vol_{w}']=roll(d,w,'vol')
        inds[f'shp_{w}']=roll(d,w,'sharpe')
        inds[f'dd_{w}']=roll(d,w,'dd')
        inds[f'twr_{w}']=roll_pnl(d,w)
    all_indicators[ms]=inds
TARGETS={ms:target(raw_vars[ms]) for ms in ALL_MODELS}
print('All indicators computed')
""")

md("## 3. Compare: Single Indicator vs Universal Health")

cd("""def corr_with_target(ms, series_dict, target_dict):
    \"\"\"Return Pearson r between indicator and target.\"\"\"
    rs={}
    for name,series in series_dict.items():
        t=target_dict[ms]
        pairs=[(series[d],t[d]) for d in series if d in t]
        if len(pairs)<5: continue
        hv=np.array([p[0] for p in pairs]); tv=np.array([p[1] for p in pairs])
        if np.std(hv)==0 or np.std(tv)==0: continue
        r,p=pearsonr(hv,tv)
        rs[name]=r
    return rs

# Get best single indicator per type
print('=== Best Single Indicator by Type (Avg r across models) ===')
single_results={}
for prefix,name in [('wr','Win Rate'),('avgret','Avg Return'),('vol','Volatility'),
                     ('shp','Sharpe'),('dd','Drawdown'),('twr','Trade Win Rate')]:
    best_r=-999; best_w=None
    for w in WINS:
        key=f'{prefix}_{w}'
        rs=[]
        for ms in ALL_MODELS:
            s=all_indicators[ms].get(key,{})
            t=TARGETS[ms]
            pairs=[(s[d],t[d]) for d in s if d in t]
            if len(pairs)<5: continue
            hv=np.array([p[0] for p in pairs]); tv=np.array([p[1] for p in pairs])
            if np.std(hv)==0 or np.std(tv)==0: continue
            r,p=pearsonr(hv,tv)
            rs.append(r)
        avg_r=np.mean(rs) if rs else -999
        if avg_r>best_r: best_r=avg_r; best_w=w
    single_results[prefix]=best_r
    print(f'{name:20s} best_w={best_w:2d} avg_r={best_r:+.4f}')

# Universal health (best config from v7)
class UniversalHealth:
    def __init__(self,config): self.config=config
    def score(self,ms):
        series_list=[]
        for name,window,weight,direction in self.config:
            key=f'{name}_{window}'
            s=all_indicators[ms].get(key,{})
            if not s: continue
            vals=np.array(list(s.values()))
            if vals.max()-vals.min()<1e-10: continue
            norm={d:(v-vals.min())/(vals.max()-vals.min()) for d,v in s.items()}
            if direction==-1: norm={d:1-v for d,v in norm.items()}
            series_list.append((norm,weight))
        all_dates=set()
        for norm,_ in series_list: all_dates.update(norm.keys())
        health={}
        for d in sorted(all_dates):
            total=0.0; wsum=0.0
            for norm,w in series_list:
                if d in norm: total+=norm[d]*abs(w); wsum+=abs(w)
            health[d]=total/wsum if wsum>0 else 0
        if health:
            hv=np.array(list(health.values()))
            if hv.max()-hv.min()>1e-10:
                for d in health: health[d]=(health[d]-hv.min())/(hv.max()-hv.min())*100
        return health

# v7 best config: avg_ret(3,0.514,1) ddepth(5,0.909,-1) win_rate(10,1.175,1) vol(3,1.946,-1)
BEST_CONFIG=[('avgret',3,0.514,1),('dd',5,0.909,-1),('wr',10,1.175,1),('vol',3,1.946,-1)]
uh=UniversalHealth(BEST_CONFIG)

print(f'\\n=== Universal Health vs Single Indicators ===')
print(f'{"Model":6s} {"Universal":10s} {"Best WR":10s} {"Best TWR":10s} {"Best AvgRet":10s} {"Best Sharpe":10s}')
print('-'*60)
for ms in ALL_MODELS:
    h=uh.score(ms); t=TARGETS[ms]
    pairs=[(h[d],t[d]) for d in h if d in t]
    if len(pairs)<5: continue
    hv=np.array([p[0] for p in pairs]); tv=np.array([p[1] for p in pairs])
    r_uni,_=pearsonr(hv,tv)
    
    # Best single indicators for this model
    r_wr=corr_with_target(ms,{k:v for k,v in all_indicators[ms].items() if k.startswith('wr_')},TARGETS)
    r_twr=corr_with_target(ms,{k:v for k,v in all_indicators[ms].items() if k.startswith('twr_')},TARGETS)
    r_ret=corr_with_target(ms,{k:v for k,v in all_indicators[ms].items() if k.startswith('avgret_')},TARGETS)
    r_shp=corr_with_target(ms,{k:v for k,v in all_indicators[ms].items() if k.startswith('shp_')},TARGETS)
    
    best_wr=max(r_wr.values()) if r_wr else 0
    best_twr=max(r_twr.values()) if r_twr else 0
    best_ret=max(r_ret.values()) if r_ret else 0
    best_shp=max(r_shp.values()) if r_shp else 0
    
    print(f'{ms:4s}  {r_uni:+.3f}     {best_wr:+.3f}     {best_twr:+.3f}      {best_ret:+.3f}      {best_shp:+.3f}')

# Summary
print(f'\\n=== Summary ===')
print(f'Average across models:')
r_uni_all=[]; r_wr_all=[]; r_twr_all=[]; r_ret_all=[]; r_shp_all=[]
for ms in ALL_MODELS:
    h=uh.score(ms); t=TARGETS[ms]
    pairs=[(h[d],t[d]) for d in h if d in t]
    hv=np.array([p[0] for p in pairs]); tv=np.array([p[1] for p in pairs])
    r_uni_all.append(pearsonr(hv,tv)[0])
    r_wr=corr_with_target(ms,{k:v for k,v in all_indicators[ms].items() if k.startswith('wr_')},TARGETS)
    r_twr=corr_with_target(ms,{k:v for k,v in all_indicators[ms].items() if k.startswith('twr_')},TARGETS)
    r_ret=corr_with_target(ms,{k:v for k,v in all_indicators[ms].items() if k.startswith('avgret_')},TARGETS)
    r_shp=corr_with_target(ms,{k:v for k,v in all_indicators[ms].items() if k.startswith('shp_')},TARGETS)
    r_wr_all.append(max(r_wr.values()) if r_wr else 0)
    r_twr_all.append(max(r_twr.values()) if r_twr else 0)
    r_ret_all.append(max(r_ret.values()) if r_ret else 0)
    r_shp_all.append(max(r_shp.values()) if r_shp else 0)
print(f'{"Universal Health":20s}: {np.mean(r_uni_all):+.3f}')
print(f'{"Best Win Rate":20s}: {np.mean(r_wr_all):+.3f}')
print(f'{"Best Trade WR":20s}: {np.mean(r_twr_all):+.3f}')
print(f'{"Best Avg Return":20s}: {np.mean(r_ret_all):+.3f}')
print(f'{"Best Sharpe":20s}: {np.mean(r_shp_all):+.3f}')

print(f'\\n=== Bar Chart: Universal Health Wins ===')
for ms in ALL_MODELS:
    h=uh.score(ms); t=TARGETS[ms]
    pairs=[(h[d],t[d]) for d in h if d in t]
    hv=np.array([p[0] for p in pairs]); tv=np.array([p[1] for p in pairs])
    r_uni,_=pearsonr(hv,tv)
    r_wr=corr_with_target(ms,{k:v for k,v in all_indicators[ms].items() if k.startswith('wr_')},TARGETS)
    best_single=max(r_wr.values()) if r_wr else -999
    win='YES' if r_uni>best_single else 'no'
    print(f'{ms:4s}: Uni={r_uni:+.3f} BestSingle={best_single:+.3f} Universal_better={win}')
""")

md("## 4. Best Win Rate Only (Best Window)")

cd("""print('=== Best Single Win Rate Window (avg across models) ===')
wr_results={}
for w in WINS:
    k=f'wr_{w}'; rs=[]
    for ms in ALL_MODELS:
        s=all_indicators[ms].get(k,{}); t=TARGETS[ms]
        pairs=[(s[d],t[d]) for d in s if d in t]
        if len(pairs)<5: continue
        hv=np.array([p[0] for p in pairs]); tv=np.array([p[1] for p in pairs])
        if np.std(hv)==0 or np.std(tv)==0: continue
        r,p=pearsonr(hv,tv)
        rs.append(r)
    wr_results[w]=np.mean(rs) if rs else -999
    print(f'  WR win={w:2d}: avg_r={wr_results[w]:+.4f}')
best_wr_w=max(wr_results, key=wr_results.get)
print(f'\\nBest WR window: {best_wr_w} (avg_r={wr_results[best_wr_w]:+.4f})')

print(f'\\n=== Best Avg PnL Window (avg across models) ===')
ret_results={}
for w in WINS:
    k=f'avgret_{w}'; rs=[]
    for ms in ALL_MODELS:
        s=all_indicators[ms].get(k,{}); t=TARGETS[ms]
        pairs=[(s[d],t[d]) for d in s if d in t]
        if len(pairs)<5: continue
        hv=np.array([p[0] for p in pairs]); tv=np.array([p[1] for p in pairs])
        if np.std(hv)==0 or np.std(tv)==0: continue
        r,p=pearsonr(hv,tv)
        rs.append(r)
    ret_results[w]=np.mean(rs) if rs else -999
    print(f'  AvgRet win={w:2d}: avg_r={ret_results[w]:+.4f}')
best_ret_w=max(ret_results, key=ret_results.get)
print(f'\\nBest AvgRet window: {best_ret_w} (avg_r={ret_results[best_ret_w]:+.4f})')
""")

nb.cells=cs
out_path='/mnt/c/Users/xyl/Desktop/ETF/notebooks/model_degradation_experiment_v8.ipynb'
nbf.write(nb,out_path)
print(f'Done: {out_path}')
