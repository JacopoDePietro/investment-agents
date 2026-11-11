"""Asset selection: factor score + choice by correlation cluster (diversification)."""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf

def zscore(s):
    return (s - s.mean()) / s.std() if s.std() > 0 else s * 0

def factor_scores(px: pd.DataFrame) -> pd.Series:
    """Static fallback score: 12-1 and 6-month momentum rewarded, volatility penalised. px = base-currency prices."""
    mom12 = px.iloc[-21] / px.iloc[-252] - 1
    mom6 = px.iloc[-21] / px.iloc[-126] - 1
    vol = np.log(px).diff().tail(252).std() * np.sqrt(252)
    trend = (px.iloc[-1] > px.tail(200).mean()).astype(float)
    return 0.4 * zscore(mom12) + 0.2 * zscore(mom6) - 0.3 * zscore(vol) + 0.1 * zscore(trend)

def pick(rets: pd.DataFrame, scores: pd.Series, n_clusters: int, per_cluster: int, max_n: int):
    """Clusters by correlation and takes the best of each cluster: similar assets do not enter together."""
    corr = rets.corr().fillna(0).values
    dist = np.sqrt(np.clip(0.5 * (1 - corr), 0, None)); np.fill_diagonal(dist, 0)
    lab = fcluster(linkage(squareform(dist, checks=False), "average"), n_clusters, "maxclust")
    df = pd.DataFrame({"score": scores.reindex(rets.columns).values, "cluster": lab}, index=rets.columns)
    chosen = df.sort_values("score", ascending=False).groupby("cluster").head(per_cluster)
    return chosen.sort_values("score", ascending=False).head(max_n)

def dcc_cov(x: np.ndarray, lam=0.94, a=0.02, b=0.97):
    """Dynamic DCC(1,1) correlations on residuals standardised by EWMA (RiskMetrics) volatility.
    Fixed parameters typical of the literature: stable and fast to estimate at every rebalance."""
    T, n = x.shape
    var = np.empty((T, n)); var[0] = x[:60].var(0) + 1e-10
    for t in range(1, T):
        var[t] = lam * var[t - 1] + (1 - lam) * x[t - 1] ** 2
    e = x / np.sqrt(var)
    S = np.corrcoef(e.T); Q = S.copy()
    for t in range(T):
        Q = (1 - a - b) * S + a * np.outer(e[t], e[t]) + b * Q
    d = np.sqrt(np.diag(Q)); R = Q / np.outer(d, d)
    vol_now = np.sqrt(lam * var[-1] + (1 - lam) * x[-1] ** 2)
    return R * np.outer(vol_now, vol_now)

def estimate(rets: pd.DataFrame, method="lw", blend=0.5):
    """Covariance: Ledoit-Wolf, or a Ledoit-Wolf + DCC blend (more reactive to regime changes).
    Expected returns shrunk towards the cross-sectional mean (more stable estimates)."""
    x = rets.fillna(0).values
    cov = LedoitWolf().fit(x).covariance_ * 252
    if method == "dcc":
        cov = (1 - blend) * cov + blend * dcc_cov(x) * 252
    mu = rets.mean().values * 252
    mu = 0.5 * mu + 0.5 * mu.mean()
    return mu, cov


def pick_by_class(rets, scores, meta, pcfg):
    """Quotas per asset class: each class brings enough candidates to fill its cap.
    Within each class the best asset of each correlation cluster is chosen."""
    caps, mw = pcfg["class_caps"], pcfg["max_weight"]
    out = []
    for cls, cap in caps.items():
        cols = [c for c in rets.columns if meta.loc[c, "asset_class"] == cls]
        if not cols:
            continue
        quota = min(len(cols), int(np.ceil(cap / mw)) + 1)
        if len(cols) <= quota:
            out.append(pd.DataFrame({"score": scores.reindex(cols), "cluster": cls}))
            continue
        ch = pick(rets[cols], scores, min(quota * 2, len(cols)), 1, quota)
        out.append(ch.assign(cluster=cls + "_" + ch.cluster.astype(str)))
    return pd.concat(out).sort_values("score", ascending=False)
