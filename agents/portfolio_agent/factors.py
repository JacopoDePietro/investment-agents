"""Adaptive factor model: factor weights follow their past predictive power (IC)."""
import numpy as np
import pandas as pd

PRICE_FACTORS = ["mom_12_1", "mom_6_1", "mom_3_1", "risk_adj_mom", "low_vol", "trend_200",
                 "near_high", "reversal_1m", "low_beta"]
FUND_FACTORS = ["earnings_yield", "book_to_market", "fcf_yield", "gross_prof", "roe",
                "low_leverage", "low_asset_growth"]
FACTORS = PRICE_FACTORS + FUND_FACTORS
MOMENTUM = ["mom_12_1", "mom_6_1", "mom_3_1", "risk_adj_mom", "trend_200", "near_high"]
QUALITY_FALLBACK = {"equity": {"roe": 1 / 3, "gross_prof": 1 / 3, "low_leverage": 1 / 3}, "other": {"low_vol": 1.0}}
DEFAULT = {"mom_12_1": 0.4, "mom_6_1": 0.2, "low_vol": 0.3, "trend_200": 0.1}

def family(asset_class: pd.Series) -> pd.Series:
    """Equities are ranked against equities, everything else against everything else."""
    return asset_class.isin(["equity", "equity_etf"]).map({True: "equity", False: "other"})

class Fundamentals:
    """Point-in-time SEC fundamentals: at each date only data filed before that date is used."""
    def __init__(self, f: pd.DataFrame, max_age_days=550):
        self.max_age = pd.Timedelta(days=max_age_days)
        f = f.sort_values(["filed", "end"])
        a = f[f.item == "assets"].sort_values("end")
        a = a.assign(prev=a.groupby("ticker").val.shift(1))
        growth = a.dropna(subset=["prev"]).assign(item="asset_growth", val=lambda x: x.val / x.prev - 1)
        f = pd.concat([f, growth[f.columns]])
        self.tab = {it: g.pivot_table(index="filed", columns="ticker", values="val", aggfunc="last").sort_index()
                    for it, g in f.groupby("item")}
        self.age = {it: g.pivot_table(index="filed", columns="ticker", values="end", aggfunc="last").sort_index()
                    for it, g in f.groupby("item")}

    def _asof(self, item, t):
        tab = self.tab.get(item)
        if tab is None:
            return pd.Series(dtype=float)
        v = tab.loc[:t].ffill().iloc[-1] if len(tab.loc[:t]) else pd.Series(dtype=float)
        e = self.age[item].loc[:t].ffill().iloc[-1] if len(tab.loc[:t]) else pd.Series(dtype="datetime64[ns]")
        return v.where(pd.to_datetime(e) > t - self.max_age)

    def factors_at(self, t, price_usd: pd.Series) -> pd.DataFrame:
        g = {k: self._asof(k, t) for k in self.tab}
        mcap = (g["shares"] * price_usd.reindex(g["shares"].index))
        mcap = mcap.where(mcap > 1e8)
        eq = g.get("equity", pd.Series(dtype=float)); assets = g.get("assets", pd.Series(dtype=float))
        gp = g.get("gross_profit", pd.Series(dtype=float)).combine_first(g.get("operating_income", pd.Series(dtype=float)))
        return pd.DataFrame({
            "earnings_yield": g.get("net_income") / mcap,
            "book_to_market": eq / mcap,
            "fcf_yield": (g.get("cfo") - g.get("capex", 0).fillna(0) if "capex" in g else g.get("cfo")) / mcap,
            "gross_prof": gp / assets,
            "roe": g.get("net_income") / eq.where(eq > 0),
            "low_leverage": -g.get("lt_debt", pd.Series(dtype=float)).reindex(assets.index).fillna(0) / assets,
            "low_asset_growth": -g.get("asset_growth", pd.Series(dtype=float)),
        }).dropna(how="all")

def features_at(px: pd.DataFrame, i: int, mkt: pd.Series, fund: "Fundamentals" = None,
                px_local: pd.DataFrame = None) -> pd.DataFrame:
    """Indicators at row i using only data up to i."""
    p = px.iloc[max(0, i - 260): i + 1]
    if len(p) < 253:
        return pd.DataFrame()
    last = p.iloc[-1]
    r = np.log(p).diff().iloc[-252:]
    vol = r.std() * np.sqrt(252)
    m = np.log(mkt.iloc[max(0, i - 252): i + 1]).diff().iloc[1:]
    beta = r.apply(lambda c: c.cov(m) / m.var() if c.notna().sum() > 200 else np.nan)
    mom12 = p.iloc[-21] / p.iloc[-252] - 1
    f = pd.DataFrame({
        "mom_12_1": mom12, "mom_6_1": p.iloc[-21] / p.iloc[-126] - 1, "mom_3_1": p.iloc[-21] / p.iloc[-63] - 1,
        "risk_adj_mom": mom12 / vol, "low_vol": -vol, "trend_200": last / p.iloc[-200:].mean() - 1,
        "near_high": last / p.iloc[-252:].max() - 1, "reversal_1m": -(last / p.iloc[-21] - 1), "low_beta": -beta})
    f = f[last.notna()]
    if fund is not None and px_local is not None:
        fu = fund.factors_at(px.index[i], px_local.iloc[i])
        f = f.join(fu.reindex(f.index))
    return f.reindex(columns=FACTORS)

def rank_z(f: pd.DataFrame, fam: pd.Series) -> pd.DataFrame:
    """Normalised rank within each family (robust to outliers)."""
    out = f.copy()
    for g in fam.reindex(f.index).dropna().unique():
        idx = fam.reindex(f.index) == g
        out.loc[idx] = f.loc[idx].rank(pct=True) - 0.5
    return out

def build_panel(px: pd.DataFrame, uni: pd.DataFrame, horizon=63, mkt="SPY", fund=None, px_local=None):
    """Indicators at every month end + forward return over `horizon` days (to measure the IC)."""
    fam = family(uni["asset_class"])
    ends = px.groupby(px.index.to_period("M")).apply(lambda x: x.index[-1]).values
    pos = {d: i for i, d in enumerate(px.index)}
    feats, fwd, raw = {}, {}, {}
    for d in ends:
        i = pos[d]
        f = features_at(px, i, px[mkt], fund, px_local)
        if f.empty:
            continue
        feats[d] = rank_z(f, fam)
        raw[d] = f[RAW_COLS].astype(float)
        if i + horizon < len(px):
            fwd[d] = px.iloc[i + horizon] / px.iloc[i] - 1
    build_panel.raw = raw
    return feats, fwd, fam

RAW_COLS = ["mom_12_1", "mom_3_1", "low_vol", "trend_200", "near_high", "low_beta"]

def ic_table(feats, fwd, fam):
    """Spearman IC by date, factor and family."""
    rows = []
    for d, f in feats.items():
        if d not in fwd:
            continue
        y = fwd[d].reindex(f.index)
        for g in ("equity", "other"):
            idx = (fam.reindex(f.index) == g) & y.notna()
            if idx.sum() < 30:
                continue
            for c in FACTORS:
                ok = idx & f[c].notna()
                if ok.sum() > 30:
                    rows.append({"date": d, "family": g, "factor": c, "ic": f.loc[ok, c].rank().corr(y[ok].rank())})
    return pd.DataFrame(rows)

def weights_at(ic: pd.DataFrame, t, horizon_days=63, window_months=60, min_obs=24):
    """Factor weights at date t: mean IC over the last 5 years, using only forward returns already known at t."""
    known = ic[ic.date <= pd.Timestamp(t) - pd.tseries.offsets.BDay(horizon_days)]
    out = {}
    for g in ("equity", "other"):
        k = known[known.family == g]
        k = k[k.date >= k.date.max() - pd.DateOffset(months=window_months)] if len(k) else k
        mean = k.groupby("factor").ic.mean()
        n = k.groupby("factor").ic.size()
        pos = mean[(mean > 0) & (n >= min_obs)]
        out[g] = (pos / pos.sum()).to_dict() if len(pos) else DEFAULT
    return out

def dampen_momentum(w: dict, scale: float) -> dict:
    """High volatility: momentum loses weight and the freed weight goes to the independent factors."""
    out = {}
    for g, wg in w.items():
        mom = {k: v * scale for k, v in wg.items() if k in MOMENTUM}
        rest = {k: v for k, v in wg.items() if k not in MOMENTUM}
        freed = sum(v for k, v in wg.items() if k in MOMENTUM) - sum(mom.values())
        if not rest:
            rest = {k: 0.0 for k in QUALITY_FALLBACK[g]}
            base = QUALITY_FALLBACK[g]
        else:
            tot = sum(rest.values()); base = {k: v / tot for k, v in rest.items()}
        out[g] = {**mom, **{k: rest.get(k, 0) + freed * base[k] for k in base}}
    return out

def high_vol(macro: pd.DataFrame, t, threshold: float) -> bool:
    """High-volatility regime: average VIX of the last 4 weeks above the threshold."""
    v = macro["VIXCLS"].loc[:t].dropna().tail(20)
    return bool(len(v) and v.mean() > threshold)

def scores_at(feats_t: pd.DataFrame, fam: pd.Series, w: dict) -> pd.Series:
    s = pd.Series(0.0, index=feats_t.index)
    for g, wg in w.items():
        idx = fam.reindex(feats_t.index) == g
        for c, v in wg.items():
            s[idx] += v * feats_t.loc[idx, c].fillna(0)
    return s * 4   # scale similar to z-scores, so hold_bonus and news_tilt stay comparable
