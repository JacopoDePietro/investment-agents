"""Monte Carlo scenarios for the optimiser: simulated factors -> returns of the individual assets.
Used by the portfolio agent for the tail-loss (CVaR) cap and the min_cvar method."""
import numpy as np
import pandas as pd
from .calibrate import calibrate
from .simulate import factor_terminal

_CACHE = {}

def factor_sims(px_eur, macro, mcfg, t, n, weeks, seed=7):
    """Calibrates on data up to t and simulates the factors (cached: shared by all methods on the same date)."""
    key = (pd.Timestamp(t), n, weeks)
    if key not in _CACHE:
        cal = calibrate(px_eur.loc[:t], macro.loc[:t], mcfg)
        _CACHE.clear()
        _CACHE[key] = (cal["factors"] + ["RATE"], factor_terminal(cal, mcfg, weeks, n, seed))
    return _CACHE[key]

def asset_scenarios(px_eur, macro, mcfg, tickers, t, n=10000, weeks=13, years=5, seed=7):
    """Scenarios (n, assets) of returns over `weeks` weeks: factor betas (weekly regression
    over the last `years` years up to t) applied to the simulated factors + idiosyncratic risk."""
    cols, Fs = factor_sims(px_eur, macro, mcfg, t, n, weeks, seed)
    start = pd.Timestamp(t) - pd.DateOffset(years=years)
    wk = px_eur.loc[start:t].resample("W-FRI").last().pct_change(fill_method=None)
    dy = (macro[mcfg["rate_factor"]].loc[start:t] / 100).resample("W-FRI").last().diff().reindex(wk.index)
    X = wk[cols[:-1]].assign(RATE=dy)
    rng = np.random.default_rng(seed + 1)
    R = np.empty((n, len(tickers)))
    for j, a in enumerate(tickers):
        d = pd.concat([wk[a].rename("_y"), X], axis=1).dropna()
        if len(d) < 52:
            R[:, j] = rng.standard_normal(n) * wk[a].std() * np.sqrt(weeks); continue
        A = np.column_stack([np.ones(len(d)), d[cols].values])
        coef, *_ = np.linalg.lstsq(A, d["_y"].values, rcond=None)
        resid = np.std(d["_y"].values - A @ coef)
        R[:, j] = Fs @ coef[1:] + resid * np.sqrt(weeks) * rng.standard_normal(n)
    return R
