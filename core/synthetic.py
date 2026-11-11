"""Synthetic history before an ETF's inception, rebuilt from its recipe (legs + financing cost).
Needed for backtests because 'capital efficient' ETFs are recent. The synthetic segment is chained
to the first real price. Returns are in local currency (USD), then converted to EUR like every other asset."""
import numpy as np
import pandas as pd

def backfill(px_local: pd.DataFrame, vol: pd.DataFrame, macro: pd.DataFrame, recipes: dict):
    rets = px_local.pct_change(fill_method=None)
    cash = (macro["DFF"].reindex(px_local.index).ffill() / 100 / 252).fillna(0)
    info = {}
    for t, rec in recipes.items():
        if t not in px_local:
            continue
        legs = rec["legs"]
        if any(l not in rets for l in legs):
            continue
        syn = sum(w * rets[l] for l, w in legs.items()) + rec.get("cash", 0) * cash - rec.get("fee", 0) / 252
        first = px_local[t].first_valid_index()
        if first is None:
            continue
        s = syn.loc[:first].iloc[1:]
        s = s[s.notna()]
        avail = pd.concat([rets[l] for l in legs], axis=1).loc[:first].notna().all(axis=1)
        s = s[avail.reindex(s.index).fillna(False)]
        if len(s) < 60:
            continue
        # synthetic prices backwards from the first real price
        # p(d) = p(first) / product of (1 + r_j) for d < j <= first
        rev = (1 + s)[::-1].cumprod()[::-1]
        px_syn = px_local.loc[first, t] * (1 + s) / rev
        px_local.loc[px_syn.index, t] = px_syn.values
        vol.loc[px_syn.index, t] = vol[t].loc[first:].median()
        info[t] = str(px_syn.index[0].date())
    return px_local, vol, info
