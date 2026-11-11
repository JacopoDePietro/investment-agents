"""LightGBM expected returns for the optimiser.
Target: 3-month return (in EUR) of each asset, comparable across equities, bonds, gold, etc.
Features: factor ranks, raw values (volatility, momentum, trend), asset class, optional macro context.
Walk-forward training; parameters chosen with time-series validation (tune) on past data only."""
import itertools, logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from .factors import FACTORS, RAW_COLS

log = logging.getLogger("portfolio_agent")
H = 63          # target horizon in business days

def macro_context(px_eur, macro, t):
    """Market context at date t (the same for all assets)."""
    v = macro["VIXCLS"].loc[:t].dropna()
    spy = px_eur["SPY"].loc[:t].dropna()
    hy = np.log(px_eur["HYG"] / px_eur["IEF"]).loc[:t].dropna()
    return {"m_vix": float(np.log(v.tail(20).mean())), "m_curve": float(macro["T10Y2Y"].loc[:t].dropna().iloc[-1]),
            "m_credit": float(hy.iloc[-1] - hy.iloc[-64]) if len(hy) > 64 else 0.0,
            "m_spy_12m": float(spy.iloc[-1] / spy.iloc[-252] - 1), "m_spy_trend": float(spy.iloc[-1] / spy.tail(200).mean() - 1)}

OPTIONS = {"use_macro": True, "demean": False}   # set from config (portfolio.ml_returns)

def frame(d, feats, raw, fam, uni, ctx):
    f = feats[d]
    X = f[FACTORS].copy()
    X[[f"raw_{c}" for c in RAW_COLS]] = raw[d].reindex(f.index)[RAW_COLS].values
    X["is_equity"] = (fam.reindex(f.index) == "equity").astype(float)
    X["asset_class"] = uni["asset_class"].reindex(f.index).astype("category").cat.codes
    if OPTIONS["use_macro"]:
        for k, v in ctx[d].items():
            X[k] = v
    return X

def target(d, fwd, idx):
    y = fwd[d].reindex(idx)
    lo, hi = y.quantile([0.01, 0.99])
    y = y.clip(lo, hi)
    return y - y.mean() if OPTIONS["demean"] else y

class MLReturns:
    def __init__(self, feats, raw, fwd, fam, uni, ctx, params, train_years=10, retrain_months=12):
        self.feats, self.raw, self.fwd, self.fam, self.uni, self.ctx = feats, raw, fwd, fam, uni, ctx
        self.params, self.years, self.every = params, train_years, retrain_months
        self.model, self.trained_at, self.oos = None, None, []

    def _fit(self, cutoff, start):
        dates = [d for d in self.feats if start <= d <= cutoff and d in self.fwd and d in self.ctx]
        if len(dates) < 36:
            return None
        X = pd.concat([frame(d, self.feats, self.raw, self.fam, self.uni, self.ctx) for d in dates])
        y = pd.concat([target(d, self.fwd, self.feats[d].index) for d in dates])
        ok = y.notna().values
        m = lgb.LGBMRegressor(objective="huber", alpha=0.1, subsample_freq=1, verbose=-1, random_state=0, **self.params)
        return m.fit(X[ok], y[ok])

    def predict(self, t, X_t):
        """Annualised expected return for every asset in X_t."""
        t = pd.Timestamp(t)
        if self.model is None or (t - self.trained_at).days > self.every * 30.5:
            cutoff = t - pd.tseries.offsets.BDay(H)              # only targets already realised at t
            m = self._fit(cutoff, cutoff - pd.DateOffset(years=self.years))
            if m is not None:
                self.model, self.trained_at = m, t
                log.info("Return LightGBM retrained at %s", t.date())
        if self.model is None:
            return None
        mu = pd.Series(self.model.predict(X_t), index=X_t.index) * (252 / H)
        if t in self.fwd:
            y = self.fwd[t].reindex(mu.index); ok = y.notna()
            if ok.sum() > 50:
                self.oos.append({"date": t, "ic": mu[ok].rank().corr(y[ok].rank())})
        return mu.clip(-0.5, 0.8)

GRID = {"num_leaves": [7, 15], "min_child_samples": [100, 500],
        "n_estimators_lr": [(150, 0.05), (400, 0.02)], "colsample_bytree": [0.7]}

def tune(feats, raw, fwd, fam, uni, ctx, val_years=(2011, 2012, 2013), train_years=10):
    """Time-series validation: for each validation year, train on the previous years
    (with a 3-month gap) and measure how well the forecasts rank the realised returns."""
    rows = []
    combos = list(itertools.product(*GRID.values()))
    for nl, mcs, (ne, lr), cs in combos:
        params = {"num_leaves": nl, "min_child_samples": mcs, "n_estimators": ne, "learning_rate": lr,
                  "colsample_bytree": cs, "subsample": 0.7, "reg_lambda": 5.0}
        ics, pooled = [], []
        for vy in val_years:
            vstart = pd.Timestamp(f"{vy}-01-01")
            cutoff = vstart - pd.tseries.offsets.BDay(H)
            m = MLReturns(feats, raw, fwd, fam, uni, ctx, params, train_years)._fit(cutoff, cutoff - pd.DateOffset(years=train_years))
            if m is None:
                continue
            vd = [d for d in feats if vstart <= d <= pd.Timestamp(f"{vy}-12-31") and d in fwd and d in ctx]
            P, Y = [], []
            for d in vd:
                X = frame(d, feats, raw, fam, uni, ctx); y = fwd[d].reindex(X.index)
                p = pd.Series(m.predict(X), index=X.index); ok = y.notna()
                ics.append(p[ok].rank().corr(y[ok].rank())); P.append(p[ok]); Y.append(y[ok])
            pooled.append(pd.concat(P).rank().corr(pd.concat(Y).rank()))
        rows.append({**params, "mean_ic": np.nanmean(ics), "ic_hit": np.mean(np.array(ics) > 0), "pooled_ic": np.nanmean(pooled)})
        log.info("Params %s -> IC %.3f, pooled IC %.3f", (nl, mcs, ne, lr), rows[-1]["mean_ic"], rows[-1]["pooled_ic"])
    res = pd.DataFrame(rows)
    res["score"] = res.mean_ic + 0.5 * res.pooled_ic
    return res.sort_values("score", ascending=False)
