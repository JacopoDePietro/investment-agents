"""Stress tests: historical crises replayed on the current portfolio and hypothetical factor shocks."""
import numpy as np
import pandas as pd

def historical(px_eur, weights, betas, windows, factor_cols):
    out = {}
    for name, (a, b) in windows.items():
        p = px_eur.loc[a:b]
        if len(p) < 5:
            continue
        r = p.iloc[-1] / p.iloc[0] - 1
        fac = r.reindex(factor_cols).fillna(0)
        contrib, proxied = {}, []
        for t, w in weights.items():
            if pd.notna(r.get(t)) and p[t].notna().mean() > 0.9:
                contrib[t] = w * r[t]
            else:                                    # asset did not exist yet: estimate with its betas
                contrib[t] = w * float((betas.loc[t, factor_cols] * fac).sum()); proxied.append(t)
        out[name] = {"port": sum(contrib.values()), "spy": float(r.get("SPY", np.nan)),
                     "proxied_share": float(weights[proxied].sum()) if proxied else 0.0,
                     "worst": sorted(contrib.items(), key=lambda x: x[1])[:3]}
    return out

def hypothetical(weights, betas, shocks, ai_tickers):
    out = {}
    for name, sh in shocks.items():
        s = pd.Series({k: v for k, v in sh.items() if k != "ai_extra"})
        imp = {}
        for t, w in weights.items():
            r = float((betas.loc[t].reindex(s.index).fillna(0) * s).sum())
            if t in ai_tickers:
                r += sh.get("ai_extra", 0.0)
            imp[t] = w * r
        out[name] = {"port": sum(imp.values()), "spy": sh.get("SPY", np.nan) + (sh.get("ai_extra", 0) * 0.3),
                     "worst": sorted(imp.items(), key=lambda x: x[1])[:3]}
    return out
