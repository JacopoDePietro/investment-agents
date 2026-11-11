"""Calibration on historical series: Heston for the price factors, CIR for the 10-year yield,
calm/crisis regimes with different correlations, factor exposures (betas) of every asset."""
import numpy as np
import pandas as pd

def _cir_ols(v, dt):
    """dv = kappa*(theta - v)dt + sigma*sqrt(v) dW, estimated by OLS on the discretised version."""
    v = v[v > 0]
    x, y = v.iloc[:-1].values, v.diff().iloc[1:].values
    b, a = np.polyfit(x, y, 1)
    kappa = float(np.clip(-b / dt, 0.3, 25))
    theta = float(np.clip(a / (kappa * dt), x.min(), x.max()))
    resid = y - (a + b * x)
    sigma = float(np.clip(np.std(resid / np.sqrt(x)) / np.sqrt(dt), 0.01, 5))
    return kappa, theta, sigma

def nearest_pd(c, eps=1e-6):
    w, V = np.linalg.eigh((c + c.T) / 2)
    c2 = V @ np.diag(np.maximum(w, eps)) @ V.T
    d = np.sqrt(np.diag(c2)); return c2 / np.outer(d, d)

def calibrate(px_eur, macro, mcfg):
    dt = 1 / mcfg["steps_per_year"]
    start = px_eur.index[-1] - pd.DateOffset(years=mcfg["calibration_years"])
    px = px_eur.loc[start:]
    # only factors with at least 2 years of history at the date (e.g. Bitcoin does not exist before 2014)
    facs = [f for f in mcfg["factors"] if f in px and px[f].notna().sum() > 520]
    dly = np.log(px[facs]).diff()
    wk_ret = np.log(px[facs].resample("W-FRI").last()).diff()
    rv = (dly ** 2).resample("W-FRI").sum() * 52                   # annualised realised variance
    rf_now = macro["ECBDFR"].dropna().iloc[-1] / 100 if "ECBDFR" in macro else 0.0
    rf_hist = macro["ECBDFR"].loc[start:].mean() / 100 if "ECBDFR" in macro else 0.0

    params = {}
    for f in facs:
        v = rv[f].replace(0, np.nan).dropna()
        k, th, s = _cir_ols(v, dt)
        r = wk_ret[f].reindex(v.index)
        rho = float(np.clip(r.corr(v.diff()), -0.95, 0.95))
        simple = (np.exp(wk_ret[f].dropna()) - 1).mean() * 52
        mu = rf_now + mcfg["drift_shrink"] * (simple - rf_hist)
        params[f] = {"kappa": k, "theta": th, "sigma": s, "v0": float(v.tail(4).mean()), "rho": rho, "mu": float(mu)}

    y = macro[mcfg["rate_factor"]].dropna().loc[start:] / 100
    yw = y.resample("W-FRI").last().dropna()
    k, th, s = _cir_ols(yw, dt)
    rate = {"kappa": k, "theta": th, "sigma": s, "y0": float(yw.iloc[-1])}

    # regimes: a week is a crisis week if the average VIX exceeds the threshold
    vix = macro["VIXCLS"].resample("W-FRI").mean().reindex(wk_ret.index).ffill()
    crisis = (vix > mcfg["crisis_vix"]).astype(int)
    trans = pd.crosstab(crisis.iloc[:-1].values, crisis.iloc[1:].values, normalize="index").values
    z = wk_ret / np.sqrt(rv.shift(1).clip(lower=1e-6) / 52)          # standardised shocks
    z["RATE"] = yw.diff().reindex(z.index) / yw.diff().std()
    corr = {g: nearest_pd(z[crisis == g].dropna().corr().values) for g in (0, 1)}
    dv = rv.diff().join(yw.diff().rename("RATE"), how="inner").dropna()
    vol_corr = nearest_pd(dv.corr().values)
    return {"factors": facs, "params": params, "rate": rate, "trans": trans, "corr": corr,
            "vol_corr": vol_corr, "crisis_now": int(crisis.iloc[-1]), "rf": rf_now,
            "crisis_share": float(crisis.mean())}

def exposures(px_eur, macro, weights: pd.Series, mcfg, years=5):
    """Weekly regression of every asset on the factors: portfolio betas and idiosyncratic risk."""
    start = px_eur.index[-1] - pd.DateOffset(years=years)
    facs = [f for f in mcfg["factors"] if f in px_eur]
    wk = px_eur.loc[start:].resample("W-FRI").last().pct_change(fill_method=None)
    dy = (macro[mcfg["rate_factor"]] / 100).resample("W-FRI").last().diff().reindex(wk.index)
    X = wk[facs].assign(RATE=dy)
    betas, idio = {}, {}
    for t in weights.index:
        d = pd.concat([wk[t].rename("_y"), X], axis=1).dropna()
        if len(d) < 60:
            betas[t] = pd.Series(0.0, index=X.columns); idio[t] = wk[t].std(); continue
        A = np.column_stack([np.ones(len(d)), d[X.columns].values])
        coef, *_ = np.linalg.lstsq(A, d["_y"].values, rcond=None)
        betas[t] = pd.Series(coef[1:], index=X.columns)
        idio[t] = float(np.std(d["_y"].values - A @ coef))
    B = pd.DataFrame(betas).T
    bp = (B.mul(weights, axis=0)).sum()                               # portfolio betas
    idio_p = float(np.sqrt(((weights * pd.Series(idio)) ** 2).sum()))  # weekly idiosyncratic risk
    r2 = {t: None for t in weights.index}
    return bp, idio_p, B, pd.Series(idio)
