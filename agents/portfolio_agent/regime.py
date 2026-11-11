"""Market regime detection with a Hidden Markov Model on macro and market data (walk-forward)."""
import logging, warnings
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

log = logging.getLogger("portfolio_agent")
NAMES = ["calm", "normal", "crisis"]

def features(macro: pd.DataFrame, px_eur: pd.DataFrame, mkt="SPY") -> pd.DataFrame:
    """Weekly indicators: stress level and market trend."""
    m = macro.resample("W-FRI").last().ffill()
    spy = px_eur[mkt].resample("W-FRI").last()
    dly = np.log(px_eur[mkt]).diff()
    f = pd.DataFrame({
        "vix": np.log(m["VIXCLS"]),
        # credit stress: high yield vs Treasuries (ICE spreads on FRED only cover 3 years)
        "credit_stress": -(np.log(px_eur["HYG"] / px_eur["IEF"]).resample("W-FRI").last().diff(13)),
        "curve": m["T10Y2Y"],
        "nfci": m["NFCI"],
        "ret_4w": np.log(spy).diff(4),
        "vol_4w": (dly ** 2).resample("W-FRI").sum().rolling(4).sum().pow(0.5) * np.sqrt(13),
    })
    return f.dropna()

def regime_at(feat: pd.DataFrame, t, min_weeks=260, seed=0):
    """Fits the HMM on data up to t and returns (regime name, probability per regime)."""
    x = feat.loc[:t]
    if len(x) < min_weeks:
        return "normal", {"calm": 0.0, "normal": 1.0, "crisis": 0.0}
    z = ((x - x.mean()) / x.std()).values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hmm = GaussianHMM(n_components=3, covariance_type="full", n_iter=200, random_state=seed).fit(z)
    # labels ordered by mean VIX level: low = calm, high = crisis
    order = np.argsort(hmm.means_[:, list(x.columns).index("vix")])
    post = hmm.predict_proba(z)[-1]
    probs = {NAMES[r]: float(post[s]) for r, s in enumerate(order)}
    return max(probs, key=probs.get), probs

def monthly_regimes(feat, dates):
    """Regime at every month end (also used to rebalance immediately when a crisis starts)."""
    out = {}
    for t in dates:
        out[pd.Timestamp(t)] = regime_at(feat, t)
    return out
