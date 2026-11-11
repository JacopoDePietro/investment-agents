"""Generates the per-agent analysis notebooks in notebooks/. Usage: python tools/make_notebooks.py"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETUP = """import sys, json, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
pd.set_option('display.max_rows', 80); pd.set_option('display.width', 200)
D = ROOT / 'data'
def ok(p):
    if not Path(p).exists(): print('File not available yet:', p); return False
    return True"""

def nb(cells):
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3 (.venv)", "language": "python"}},
            "cells": [{"cell_type": t, "metadata": {}, "source": s.strip("\n"),
                       **({"outputs": [], "execution_count": None} if t == "code" else {})} for t, s in cells]}

M, C = "markdown", "code"

DATA_NB = [
(M, "# 01 · Data agent\nDocumentation: `docs/agents/data_agent.md`. Run the cells in order with the `.venv` kernel."),
(C, SETUP),
(C, """uni = pd.read_parquet(D/'processed/universe.parquet')
px = pd.read_parquet(D/'processed/prices.parquet')
print('Assets in the universe:', len(uni), '· with data:', int(uni.has_data.sum()), '· eligible:', int(uni.eligible.sum()))
print('Prices from', px.index[0].date(), 'to', px.index[-1].date())
uni.groupby(['asset_class','region']).size().unstack(fill_value=0)"""),
(M, "## Coverage over time: assets with prices, by year and class"),
(C, """cov = px.notna().groupby(px.index.year).any().T.join(uni['asset_class']).groupby('asset_class').sum().T
cov.plot(figsize=(12,4)); plt.title('Assets with at least one price in the year'); plt.show(); cov.tail()"""),
(M, "## Quality: missing data, years of history, tickers without data"),
(C, """q = pd.DataFrame({'missing_pct': px.isna().mean().round(3), 'last': px.apply(lambda s: s.last_valid_index())}).join(uni[['asset_class','history_years']])
print('Without data:', uni.index[~uni.has_data].tolist())
q.sort_values('missing_pct', ascending=False).head(15)"""),
(M, "## FX rates (units of currency per 1 EUR)"),
(C, """fx = pd.read_parquet(D/'processed/fx.parquet')
(fx/fx.iloc[0]*100).plot(figsize=(12,4), logy=True); plt.title('FX rates, base 100'); plt.show()"""),
(M, "## Macro FRED"),
(C, """macro = pd.read_parquet(D/'processed/macro.parquet')
print(macro.apply(lambda s: s.first_valid_index()).astype(str))
macro[['DGS10','T10Y2Y','VIXCLS','NFCI','ECBDFR']].ffill().plot(subplots=True, figsize=(12,9)); plt.show()"""),
(M, "## SEC fundamentals: coverage and an example"),
(C, """f = pd.read_parquet(D/'processed/fundamentals.parquet')
print('Companies:', f.ticker.nunique(), '· rows:', len(f))
display(f.groupby('item').ticker.nunique().rename('companies'))
f.groupby(f.filed.dt.year).ticker.nunique().plot(kind='bar', figsize=(10,3), title='Companies with filings per year'); plt.show()
f[(f.ticker=='AAPL') & f.item.isin(['net_income','equity','assets'])].pivot_table(index='end', columns='item', values='val').tail(6)"""),
(M, "## Synthetic history of capital-efficient ETFs: reconstruction vs real prices"),
(C, """from core.config import load_config
from agents.portfolio_agent.agent import load
cfg = load_config(); recipes = cfg['synthetic_history']
pl, pe, vol, u2, rf = load(D/'processed', 'EUR')
display(u2.loc[list(recipes), ['synthetic_from']])
r = px.pct_change(fill_method=None); cash = macro['DFF'].reindex(px.index).ffill()/100/252
for t, rec in recipes.items():
    syn = sum(w*r[l] for l, w in rec['legs'].items()) + rec['cash']*cash - rec['fee']/252
    real = r[t].dropna(); s = syn.reindex(real.index)
    print(f"{t}: correlation {real.corr(s):.3f} · annual return real {real.mean()*252:+.1%} synthetic {s.mean()*252:+.1%} · tracking error {(real-s).std()*np.sqrt(252):.1%}")"""),
(C, """t = 'NTSX'; real = px[t].dropna(); syn = pl[t].loc[:real.index[0]]
ax = (pl[t]/pl[t].dropna().iloc[0]).plot(figsize=(12,4), label='series used (synthetic + real)', logy=True)
ax.axvline(real.index[0], color='grey', ls='--'); plt.title(f'{t}: the dashed line marks the ETF inception'); plt.legend(); plt.show()"""),
]

NEWS_NB = [
(M, "# 02 · News agent\nDocumentation: `docs/agents/news_agent.md`."),
(C, SETUP),
(C, """if ok(D/'news/news.parquet'):
    n = pd.read_parquet(D/'news/news.parquet')
    n['day'] = pd.to_datetime(n.fetched_at, utc=True).dt.date
    print('News items:', len(n), '· days:', n.day.nunique())
    display(n.groupby(['day','theme']).size().unstack(fill_value=0))"""),
(M, "## Sentiment and impact by theme"),
(C, """fig, ax = plt.subplots(1, 2, figsize=(13,4))
n.boxplot(column='sentiment', by='theme', ax=ax[0], rot=45); n.groupby('theme').impact.mean().plot(kind='bar', ax=ax[1], title='Average impact')
plt.suptitle(''); plt.tight_layout(); plt.show()"""),
(M, "## Sources and tickers"),
(C, """display(n.source_name.value_counts().head(20))
tk = n.explode('tickers').tickers.value_counts().head(25); display(tk.rename('mentions'))
print('Tickers proposed outside the universe:', n.explode('unknown_tickers').unknown_tickers.dropna().value_counts().head(20).to_dict())"""),
(M, "## Signals (impact-weighted sentiment, half-life 3 days)"),
(C, """s = pd.read_parquet(D/'processed/news_signals.parquet')
for k in ['ticker','asset_class','region']:
    print(k); display(s[s.kind==k].sort_values('score')[['key','score','weight','n_news']].round(3))"""),
(M, "## Latest high-impact news"),
(C, """n.sort_values(['fetched_at','impact'], ascending=False).query('impact >= 4')[['day','theme','title','tickers','sentiment','impact','horizon']].head(25)"""),
(M, "## Did sentiment anticipate prices? (meaningful only after weeks of history)"),
(C, """px = pd.read_parquet(D/'processed/prices.parquet')
e = n.explode('tickers').dropna(subset=['tickers'])
e['date'] = pd.to_datetime(e.fetched_at, utc=True).dt.tz_localize(None).dt.normalize()
def fwd(row, days=5):
    s = px[row.tickers].dropna(); s = s.loc[row.date:]
    return s.iloc[min(days, len(s)-1)] / s.iloc[0] - 1 if len(s) > 1 else np.nan
e['ret_5d'] = e.apply(fwd, axis=1)
print('Correlation sentiment / 5-day return:', round(e[['sentiment','ret_5d']].corr().iloc[0,1], 3), '· observations:', e.ret_5d.notna().sum())"""),
]

PORT = [
(M, "# 03 · Portfolio agent\nDocumentation: `docs/agents/portfolio_agent.md`. Diagnostic files are written at every agent run."),
(C, SETUP + """
P = D/'portfolio'
from core.config import load_config
from agents.portfolio_agent.agent import apply_profile
from agents.portfolio_agent.backtest import regime_profile
cfg = load_config(); pcfg = regime_profile(apply_profile(dict(cfg['portfolio'])), 'offense')
uni = pd.read_parquet(D/'processed/universe.parquet')"""),
(M, "## 1. Effective limits (base → growth → offense)"),
(C, """for k in ['class_caps','equity_total_cap','region_caps','ai_cap','hedge','risk','ml','ml_returns','mc_selection','mc_scenarios','choose_by']:
    print(f'{k:18}', pcfg[k] if k != 'hedge' else {kk: v for kk, v in pcfg[k].items() if kk != 'tickers'})"""),
(M, "## 2. Backtest metrics: full period and by horizon"),
(C, """m = pd.read_csv(P/'backtest_metrics.csv', index_col=0); display(m.round(3))
hz = {f.stem.replace('metrics_',''): pd.read_csv(f, index_col=0) for f in sorted(P.glob('metrics_*.csv'))}
pd.DataFrame({h: x['Return'] for h, x in hz.items()}).round(3)"""),
(M, "## 3. Equity curves and drawdowns"),
(C, """r = pd.read_parquet(P/'backtest_returns.parquet'); eq = (1+r.fillna(0)).cumprod()
fig, ax = plt.subplots(2, 1, figsize=(13,8), sharex=True, gridspec_kw={'height_ratios':[2,1]})
eq.plot(ax=ax[0], logy=True); ax[0].set_title('Value (base 1, EUR, log scale)')
(eq/eq.cummax()-1).plot(ax=ax[1], legend=False); ax[1].set_title('Drawdown'); plt.show()"""),
(C, """y = ((1+r).resample('YE').prod()-1); y.index = y.index.year; (y*100).round(1)"""),
(M, "## 4. Weights by asset class over time (choose the method)"),
(C, """METHOD = 'max_sharpe'
if ok(P/'class_weights_history.csv'):
    cw = pd.read_csv(P/'class_weights_history.csv', index_col=[0,1], parse_dates=[1]).loc[METHOD]
    cw.plot.area(figsize=(13,4), title=f'{METHOD}: weights by asset class'); plt.legend(bbox_to_anchor=(1,1)); plt.show()"""),
(C, """wh = pd.read_parquet(P/'weights_history.parquet')
last = wh.loc[METHOD].dropna(axis=1, how='all').tail(4).T
last[last.notna().any(axis=1)].fillna(0).sort_values(last.columns[-1], ascending=False).round(3).head(30)"""),
(M, "## 5. Turnover and costs"),
(C, """w = wh.loc[METHOD].fillna(0)
to = w.diff().abs().sum(axis=1); trades = (w.diff().abs() > 1e-4).sum(axis=1)
pd.DataFrame({'turnover': to.round(2), 'orders': trades, 'cost_%': (to*pcfg['cost_bps']/1e4 + trades*pcfg['commission_per_trade']/pcfg['capital'])*100}).tail(12).round(2)"""),
(M, "## 6. Factors: predictive power (IC) and weights used over time"),
(C, """ic = pd.read_parquet(P/'factor_ic.parquet')
s = ic.groupby(['family','factor']).ic.agg(['mean','std','count']); s['ICIR'] = s['mean']/s['std']
display(s.sort_values(['family','mean'], ascending=[True,False]).round(3))
roll = ic[ic.family=='equity'].pivot(index='date', columns='factor', values='ic').rolling(24).mean()
roll[['mom_12_1','risk_adj_mom','low_vol','gross_prof','fcf_yield','earnings_yield']].plot(figsize=(13,4), title='24-month rolling mean IC (equities)'); plt.axhline(0, color='k', lw=.5); plt.show()"""),
(C, """if ok(P/'factor_weights_history.csv'):
    fw = pd.read_csv(P/'factor_weights_history.csv', parse_dates=['date'])
    h = fw[fw.family=='equity'].pivot_table(index='date', columns='factor', values='weight').fillna(0)
    h.plot.area(figsize=(13,4), title='Factor weights used (equities)'); plt.legend(bbox_to_anchor=(1,1)); plt.show()"""),
(M, "## 7. The two LightGBM models: out-of-sample quality in the backtest"),
(C, """if ok(P/'ml_oos_ic.csv'):
    o = pd.read_csv(P/'ml_oos_ic.csv', parse_dates=['date'])
    display(o.groupby('model').ic.agg(['mean','count', lambda x: (x>0).mean()]).rename(columns={'<lambda_0>':'share_positive'}).round(3))
    o.pivot_table(index='date', columns='model', values='ic').rolling(8, min_periods=4).mean().plot(figsize=(13,4), title='Out-of-sample IC (rolling mean)'); plt.axhline(0, color='k', lw=.5); plt.show()
p = D/'experiments/ml_tuning.csv'
if ok(p): display(pd.read_csv(p).head(10).round(4))"""),
(M, "## 8. Context signals over time (AI bubble, high volatility, mode)"),
(C, """if ok(P/'context_history.csv'):
    ch = pd.read_csv(P/'context_history.csv', index_col=0, parse_dates=True)
    display(ch.astype(str).apply(lambda c: c.value_counts()).fillna(0))
    ch[[c for c in ['bubble','high_vol'] if c in ch]].astype(str).replace({'True':1,'False':0}).astype(float).plot(figsize=(13,2.5), drawstyle='steps-post'); plt.show()"""),
(M, "## 9. Today: scores, expected returns, method selection, proposal"),
(C, """if ok(P/'scores_today.csv'):
    sc = pd.read_csv(P/'scores_today.csv', index_col=0); display(sc.head(30).round(3))
if ok(P/'expected_returns_ml.csv'):
    mu = pd.read_csv(P/'expected_returns_ml.csv', index_col=0).iloc[:,0].sort_values()
    print('LightGBM expected returns (relative, annualised): lowest and highest'); display(pd.concat([mu.head(10), mu.tail(10)]).to_frame().join(uni[['asset_class','name']]).round(3))"""),
(C, """display(pd.read_csv(P/'mc_selection.csv', index_col=0).round(3))
tg = pd.read_csv(P/'target_portfolio.csv', index_col=0); display(tg.round(3))
print(json.dumps(json.loads((P/'context.json').read_text()), indent=1, default=str)[:3000])"""),
]

MOD = [
(M, "# 04 · Modeling agent\nDocumentation: `docs/agents/modeling_agent.md`."),
(C, SETUP + """
from core.config import load_config
cfg = load_config(); mcfg = cfg['modeling']
rep = json.loads((D/'modeling/risk_report.json').read_text())
print('Date:', rep['date'])"""),
(M, "## 1. Calibrated parameters"),
(C, """cal = pd.DataFrame(rep['calibration']['params']).T
cal['long_run_vol'] = np.sqrt(cal.theta); cal['vol_today'] = np.sqrt(cal.v0)
display(cal.round(3)); print('CIR yield:', {k: round(v,4) for k, v in rep['calibration']['rate'].items()})
print('Current regime:', 'crisis' if rep['calibration']['crisis_now'] else 'calm', '· share of crisis weeks:', round(rep['calibration']['crisis_share'],2))"""),
(M, "## 2. Portfolio factor exposures"),
(C, """pd.Series(rep['portfolio_beta']).round(3).to_frame('beta').T"""),
(M, "## 3. Monte Carlo results (1,000,000 paths)"),
(C, """pd.DataFrame(rep['monte_carlo']).round(4)"""),
(M, "## 4. Stress tests"),
(C, """display(pd.DataFrame(rep['stress_historical']).T)
display(pd.DataFrame(rep['stress_hypothetical']).T)"""),
(M, "## 5. Re-run a simulation here: 3-month factor distribution"),
(C, """from agents.portfolio_agent.agent import load
from agents.modeling_agent.calibrate import calibrate
from agents.modeling_agent.simulate import factor_terminal
_, pe, _, uni, _ = load(D/'processed', 'EUR'); macro = pd.read_parquet(D/'processed/macro.parquet')
c = calibrate(pe, macro, mcfg)
Fs = pd.DataFrame(factor_terminal(c, mcfg, 13, 20000, 1), columns=c['factors'] + ['RATE'])
display(Fs.describe(percentiles=[.01,.05,.5,.95]).T.round(3))
Fs[['SPY','GLD','HYG','BTC-USD']].clip(-0.6, 0.8).plot.hist(bins=80, alpha=.5, figsize=(12,4)); plt.title('13-week returns'); plt.show()"""),
(M, "## 6. Risk of each position of today's portfolio in the scenarios"),
(C, """from agents.modeling_agent.scenarios import asset_scenarios
tg = pd.read_csv(D/'portfolio/target_portfolio.csv', index_col=0); w = tg.weight_real/tg.weight_real.sum()
R = asset_scenarios(pe, macro, mcfg, list(w.index), pe.index[-1], n=20000)
q = np.quantile(R, .05, axis=0); cv = np.array([R[R[:,j] <= q[j], j].mean() for j in range(R.shape[1])])
port = R @ w.values; qp = np.quantile(port, .05); tail = port <= qp
contrib = (R[tail] * w.values).mean(axis=0)
t = pd.DataFrame({'weight': w, 'CVaR95_3m_asset': cv, 'contribution_to_tail_loss': contrib}).join(uni[['asset_class']])
print(f'Portfolio 3-month CVaR 95%: {port[tail].mean():.1%}')
t.sort_values('contribution_to_tail_loss').round(4)"""),
(M, "## 7. Sensitivity: how risk changes with the model assumptions"),
(C, """def cvar_port(mc):
    cc = calibrate(pe, macro, mc); from agents.modeling_agent import scenarios as S
    S._CACHE.clear(); RR = S.asset_scenarios(pe, macro, mc, list(w.index), pe.index[-1], n=20000)
    p = RR @ w.values; return p[p <= np.quantile(p, .05)].mean()
rows = {}
for lab, ch in [('base', {}), ('no risk premium', {'drift_shrink': 0.0}), ('full historical premium', {'drift_shrink': 1.0}),
                ('jumps x2', {'jumps': {**mcfg['jumps'], 'intensity': 1.0}}), ('no jumps', {'jumps': {**mcfg['jumps'], 'intensity': 0.0}}),
                ('crisis VIX threshold 20', {'crisis_vix': 20})]:
    rows[lab] = cvar_port({**mcfg, **ch})
pd.Series(rows).round(4).to_frame('portfolio 3-month CVaR95')"""),
]

CTRL = [
(M, "# 05 · Controller agent\nDocumentation: `docs/agents/controller_agent.md`."),
(C, SETUP + """
from core.config import load_config
cfg = load_config()
dec = json.loads((D/'controller/decision.json').read_text())
print('Date:', dec['date'], '· verdict:', dec['verdict'], '· active portfolio:', dec['active_portfolio'])"""),
(M, "## Outcome and margin of each check"),
(C, """rows = []
for lvl in ['hard','soft']:
    for x in dec[lvl]:
        v, l = x['value'], x['limit']
        margin = (v - l) if isinstance(v, (int, float)) and isinstance(l, (int, float)) else None
        rows.append({'level': lvl, 'check': x['check'], 'value': v, 'limit': l, 'margin': margin, 'ok': x['ok']})
pd.DataFrame(rows)"""),
(M, "## Verdict history"),
(C, """if ok(D/'controller/decisions.csv'): display(pd.read_csv(D/'controller/decisions.csv'))"""),
(M, "## What would happen with different thresholds"),
(C, """rep = json.loads((D/'modeling/risk_report.json').read_text()); mc = rep['monte_carlo'].get('1') or rep['monte_carlo'].get(1)
for s in [-0.12, -0.15, -0.18, -0.22, -0.25]:
    print(f"CVaR 95% threshold {s:+.0%}: {'pass' if mc['CVaR95'] >= s else 'BLOCK'} (value {mc['CVaR95']:+.1%})")
for s in [0.03, 0.05, 0.10, 0.12]:
    print(f"P(DD>20%) threshold {s:.0%}: {'pass' if mc['prob_dd20'] <= s else 'BLOCK'} (value {mc['prob_dd20']:.1%})")"""),
(M, "## Current thresholds"),
(C, """cfg['controller']"""),
]

for name, cells in [("01_data", DATA_NB), ("02_news", NEWS_NB), ("03_portfolio", PORT), ("04_modeling", MOD), ("05_controller", CTRL)]:
    (ROOT / "notebooks" / f"{name}.ipynb").write_text(json.dumps(nb(cells), indent=1))
    print("written", name)
