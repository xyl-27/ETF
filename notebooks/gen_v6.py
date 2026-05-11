"""
v6: Health = Ridge regression of trade features -> future return.
Clean version. All 5 models.
Features: wr_N, avgpnl_N, plratio_N, twr_N, sharpe_N, ddepth (NO ic/ndcg/mrr/ksp)
"""
import nbformat as nbf
from pathlib import Path
nb=nbf.v4.new_notebook()
nb.metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
             "language_info":{"name":"python","version":"3.12.0"}}
cs=[];md=lambda s:cs.append(nbf.v4.new_markdown_cell(s));cd=lambda s:cs.append(nbf.v4.new_code_cell(s))

md("# Model Degradation v6: Trade-Only Features + Ridge Regression")

md("## 1. Load Data")

cd("""import numpy as np, pandas as pd, json, matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, linregress
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score
import warnings; warnings.filterwarnings('ignore')
_NB_DIR=Path().resolve()
for _p in [_NB_DIR,_NB_DIR.parent,Path('/mnt/c/Users/xyl/Desktop/ETF')]:
    if (_p/'output').exists(): ROOT=_p; break
else: ROOT=_NB_DIR
with open(str(ROOT/'output'/'backtest_state.json')) as f: state=json.load(f)
seqs=state.get('sequences',state)
MODELS=['search_itransformer_exp_54','search_itransformer_exp_64',
        'search_itransformer_exp_6','average','voting']
MSHORT={m:k for m,k in zip(MODELS,['i54','i64','i6','avg','vote'])}
MCOLOR={'i54':'#E24A33','i64':'#348ABD','i6':'#988ED5','avg':'#2ECC40','vote':'#FF851B'}
ALL_MS=['i54','i64','i6','avg','vote']
""")

cd("""# Parse
bareraw={}; equities={}; trades={}
for m in MODELS:
    s=seqs[m]; met=s.get('metrics',{})
    by_date={}
    for ks,kr in [('ic','_rank_ic_raw'),('ndcg','_ndcg_raw'),('mrr','_mrr_raw'),('ksp','_ks_p_raw')]:
        for e in met.get(kr,[]):
            d=e['date']
            if d not in by_date: by_date[d]={'date':d}
            by_date[d][ks]=e['value']
    bareraw[m]=sorted(by_date.values(),key=lambda x:x['date'])
    eq=s.get('equity_curve',[])
    for e in eq:
        d=e['date'].strftime('%Y-%m-%d') if hasattr(e['date'],'strftime') else str(e['date'])[:10]
        e['date']=d
    equities[m]=sorted(eq,key=lambda x:x['date'])
    tr=[]
    for t in s.get('trades',[]):
        d=t['date'].strftime('%Y-%m-%d') if hasattr(t['date'],'strftime') else str(t['date'])[:10]
        tr.append({'date':d,'action':t.get('action'),'stock':t.get('stock'),
                   'score':t.get('score'),'price':t.get('price')})
    trades[m]=tr
all_dates=sorted(set(e['date'] for m in MODELS for e in equities[m]))
print(f'Dates: {all_dates[0]} ~ {all_dates[-1]} ({len(all_dates)} days)')
cum_rets={}
for m in MODELS:
    eq=equities[m]; i0=eq[0]['total_value']
    cum_rets[m]={e['date']:(e['total_value']/i0-1)*100 for e in eq}
daily_rets={}
for m in MODELS:
    eq=equities[m]; rets={}
    for i in range(1,len(eq)):
        rets[eq[i]['date']]=(eq[i]['total_value']/eq[i-1]['total_value']-1)*100
    daily_rets[m]=rets
trade_pnl={}
for m in MODELS:
    pnl=[]; prices={}
    for t in trades[m]:
        if t['action']==chr(20080)+chr(20837):
            prices[t['stock']]=t['price']
        elif t['action']==chr(21334)+chr(20986) and t['stock'] in prices:
            pnl_pct=(t['price']-prices[t['stock']])/prices[t['stock']]*100
            pnl.append({'date':t['date'],'stock':t['stock'],'pnl_pct':pnl_pct})
            del prices[t['stock']]
    trade_pnl[m]=pnl
print('Data loaded')
""")

md("## 2. Build Features")

cd("""WINS=[3,5,10,15,20]

def build(m,la=5):
    dates=[e['date'] for e in equities[m]]
    dr=[daily_rets[m].get(d,0) for d in dates]
    ev=[e['total_value'] for e in equities[m]]
    dw=[1 if v>0 else 0 for v in dr]
    pbd={}
    for tp in trade_pnl[m]: pbd.setdefault(tp['date'],[]).append(tp['pnl_pct'])
    Xr=[]; yr=[]; fd=[]
    for p,d in enumerate(dates):
        r={}
        for w in WINS:
            s=max(0,p-w+1)
            sub=dw[s:p+1]
            if sub: r[f'wr_{w}']=sum(sub)/len(sub)
        for w in WINS:
            s=max(0,p-w+1)
            sub=dr[s:p+1]
            if sub: r[f'avgret_{w}']=float(np.mean(sub))
        for w in WINS:
            tpnls=[]
            for i in range(max(0,p-w+1),p+1):
                if dates[i] in pbd: tpnls.extend(pbd[dates[i]])
            if tpnls:
                r[f'n_{w}']=len(tpnls)
                r[f'avgpnl_{w}']=float(np.mean(tpnls))
                prof=[v for v in tpnls if v>0]
                loss=[v for v in tpnls if v<0]
                r[f'twr_{w}']=len(prof)/len(tpnls)
                ap=np.mean(prof) if prof else 0
                al=abs(np.mean(loss)) if loss else 0
                r[f'plr_{w}']=ap/al if al>0 and ap>0 else (0 if ap<=0 else 5)
        for w in WINS:
            s=max(0,p-w+1); sub=dr[s:p+1]
            if len(sub)>=5 and np.std(sub)>0:
                r[f'shp_{w}']=float(np.mean(sub)/np.std(sub)*np.sqrt(252))
        for w in WINS:
            s=max(0,p-w+1); sub=dr[s:p+1]
            if len(sub)>=3: r[f'vol_{w}']=float(np.std(sub))
        r['dd']=(ev[p]/max(ev[:p+1])-1)*100
        if p+la<len(cum_rets[m]):
            fwd=cum_rets[m][dates[p+la]]-cum_rets[m][d]
            Xr.append(r); yr.append(fwd); fd.append(d)
    X=pd.DataFrame(Xr,index=fd); y=pd.Series(yr,index=fd,name='fwd_ret')
    return X,y

print('Building datasets (la=5, target=future return)...')
allX={}; ally={}
for m in MODELS:
    X,y=build(m,5)
    allX[MSHORT[m]]=X; ally[MSHORT[m]]=y
    print(f'  {MSHORT[m]:4s}: X {str(X.shape):10s} y mean={y.mean():+.3f} std={y.std():.3f}')
""")

md("## 3. Approach A: Pool All 5 Models into One Ridge")

cd("""Xp=pd.concat([allX[ms] for ms in ALL_MS]); yp=pd.concat([ally[ms] for ms in ALL_MS])
Xp=Xp.fillna(Xp.mean())
sc=StandardScaler(); Xs=sc.fit_transform(Xp)
rc=RidgeCV(alphas=np.logspace(-3,3,20),scoring='r2',cv=LeaveOneOut()); rc.fit(Xs,yp)
yp_cv=cross_val_predict(Ridge(alpha=rc.alpha_),Xs,yp,cv=LeaveOneOut())
print(f'=== Pooled Ridge ===')
print(f'Alpha={rc.alpha_:.4f} LOOCV R2={r2_score(yp,yp_cv):.4f} Samples={len(yp)}')
# Save for later use (before per-model analysis overwrites these globals)
_pool_rc=rc; _pool_sc=sc; _pool_Xs=Xs
coef=pd.DataFrame({'f':Xp.columns,'c':rc.coef_}).sort_values('c',ascending=False)
print('\\nTop:')
for _,r_ in coef.head(8).iterrows(): print(f'  +{r_["c"]:+.4f}  {r_["f"]}')
print('\\nBottom:')
for _,r_ in coef.tail(6).iterrows(): print(f'  {r_["c"]:+.4f}  {r_["f"]}')
""")

md("## 4. Approach B: Per-Model Ridge (5 separate models)")

cd("""per_models={}
for ms in ALL_MS:
    X=allX[ms].fillna(allX[ms].mean())
    if len(X)<5: continue
    sc=StandardScaler(); Xs=sc.fit_transform(X)
    y=ally[ms]
    rc=RidgeCV(alphas=np.logspace(-3,3,20),scoring='r2',cv=LeaveOneOut()); rc.fit(Xs,y)
    yp=cross_val_predict(Ridge(alpha=rc.alpha_),Xs,y,cv=LeaveOneOut())
    per_models[ms]={'model':rc,'scaler':sc,'Xcols':X.columns,'yp':yp}
    print(f'{ms}: alpha={rc.alpha_:.4f} LOOCV R2={r2_score(y,yp):.4f} n={len(y)}')
print(f'Avg LOOCV R2: {np.mean([r2_score(ally[ms], per_models[ms]["yp"]) for ms in per_models]):.4f}')
""")

md("## 5. Compare: Pooled vs Per-Model")

cd("""# Build health scores for pooled model
health_pool={}
yp_all=_pool_rc.predict(_pool_Xs)
idx=0
for ms in ALL_MS:
    n=len(ally[ms])
    h=yp_all[idx:idx+n]
    h=(h-h.min())/(h.max()-h.min()+1e-10)*100
    health_pool[ms]={d:h[i] for i,d in enumerate(ally[ms].index)}
    idx+=n

# Build df_by_model
dfr=[]
for d in all_dates:
    for m in MODELS:
        ret=cum_rets[m].get(d)
        if ret is None: continue
        dfr.append({'date':d,'model':MSHORT[m],'ret_cum':ret})
dfa=pd.DataFrame(dfr)
dfb={ms:dfa[dfa['model']==ms] for ms in ALL_MS}
def fwd(df,la):
    v=list(df['ret_cum']); d=list(df['date']); f={}
    for i,dd in enumerate(d):
        if i+la<len(v): f[dd]=v[i+la]-v[i]
    return f

print('=== Health vs Forward Return: Pooled Model ===')
for ms in ALL_MS:
    print(f'--- {ms} ---')
    for la in [1,3,5,10]:
        fwd_=fwd(dfb[ms],la)
        pairs=[(health_pool[ms][d],fwd_[d]) for d in health_pool[ms] if d in fwd_]
        if len(pairs)<5: continue
        h=np.array([p[0] for p in pairs]); f=np.array([p[1] for p in pairs])
        if np.std(h)==0 or np.std(f)==0: continue
        pr,pp=pearsonr(h,f)
        sig='*'*min(3,sum([pp<0.01,pp<0.05,pp<0.1]))
        print(f'  la={la:2d}d  r={pr:+.3f}(p={pp:.4f}){sig:3s} n={len(pairs)}')
    print()
""")

cd("""# Per-model health (skip models with bad fit)
print('=== Health vs Forward Return: Per-Model ===')
for ms in per_models:
    X=allX[ms].fillna(allX[ms].mean())
    Xs=per_models[ms]['scaler'].transform(X)
    yp=per_models[ms]['model'].predict(Xs)
    if yp.max()-yp.min()<1e-10: 
        print(f'--- {ms} --- (skipped, constant prediction)')
        continue
    h=(yp-yp.min())/(yp.max()-yp.min()+1e-10)*100
    hd={d:h[i] for i,d in enumerate(ally[ms].index)}
    print(f'--- {ms} ---')
    for la in [1,3,5,10]:
        fwd_=fwd(dfb[ms],la)
        pairs=[(hd[d],fwd_[d]) for d in hd if d in fwd_]
        if len(pairs)<5: continue
        hh=np.array([p[0] for p in pairs]); ff=np.array([p[1] for p in pairs])
        if np.std(hh)==0 or np.std(ff)==0: continue
        pr,pp=pearsonr(hh,ff)
        sig='*'*min(3,sum([pp<0.01,pp<0.05,pp<0.1]))
        print(f'  la={la:2d}d  r={pr:+.3f}(p={pp:.4f}){sig:3s} n={len(pairs)}')
    print()
""")

md("## 6. Visualize (Best Model)")

cd("""# Use pooled model for visualization
fig,axes=plt.subplots(2,3,figsize=(18,8)); axes=axes.flatten()
for ai,ms in enumerate(ALL_MS):
    ax=axes[ai]
    h=health_pool[ms]
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
plt.suptitle('Health Score (Pooled Ridge, Trade-Only Features)',fontsize=14,fontweight='bold')
plt.tight_layout(); plt.show()
""")

md("## 7. Final Config")

cd("""print('=== FINAL CONFIGURATION ===')
print('Method: Pooled Ridge Regression')
print(f'Alpha: {rc.alpha_:.4f}')
print(f'Target: future return (la=5d)')
print(f'Features: {Xp.shape[1]} (all trade-level)')
print(f'LOOCV R2: {r2_score(yp,yp_cv):.4f}')
print(f'Samples: {len(yp)} (pooled from {len(ALL_MS)} models)')
print('Thresholds: GREEN>=80 YELLOW>=60 ORANGE>=40 RED<40')
print()
print('Model  Health  Flag')
for ms in ALL_MS:
    hd=health_pool[ms]
    last_d=list(hd.keys())[-1]; last_v=list(hd.values())[-1]
    flag='GREEN' if last_v>=80 else ('YELLOW' if last_v>=60 else ('ORANGE' if last_v>=40 else 'RED'))
    print(f'  {ms:4s}  {last_v:6.1f}  {flag:6s}')
""")

nb.cells=cs
out_path='/mnt/c/Users/xyl/Desktop/ETF/notebooks/model_degradation_experiment_v6.ipynb'
nbf.write(nb,out_path)
print(f'Done: {out_path}')
