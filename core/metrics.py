"""Performance metrics on daily simple returns."""
import numpy as np
import pandas as pd

def perf(r: pd.Series, rf: pd.Series | None = None, periods=252) -> dict:
    r = r.dropna()
    if len(r) < 20:
        return {}
    ex = r - (rf.reindex(r.index).fillna(0) if rf is not None else 0)
    eq = (1 + r).cumprod()
    years = len(r) / periods
    dd = eq / eq.cummax() - 1
    down = ex[ex < 0].std() * np.sqrt(periods)
    cagr = eq.iloc[-1] ** (1 / years) - 1
    return {
        "CAGR": cagr,
        "Volatility": r.std() * np.sqrt(periods),
        "Sharpe": ex.mean() / ex.std() * np.sqrt(periods) if ex.std() > 0 else np.nan,
        "Sortino": ex.mean() * periods / down if down > 0 else np.nan,
        "MaxDrawdown": dd.min(),
        "Calmar": cagr / abs(dd.min()) if dd.min() < 0 else np.nan,
        "WorstMonth": (1 + r).resample("ME").prod().sub(1).min(),
        "CVaR95_daily": r[r <= r.quantile(0.05)].mean(),
    }
