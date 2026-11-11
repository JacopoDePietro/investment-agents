"""Walk-forward LightGBM: predicts the rank of the 3-month return from the factors and tilts the score.
It does not replace the optimiser: it only changes the selection score."""
import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from .factors import FACTORS

log = logging.getLogger("portfolio_agent")
PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_child_samples=200,
              subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, n_estimators=300, verbose=-1)

def _xy(f: pd.DataFrame, fam: pd.Series, uni: pd.DataFrame):
    X = f[FACTORS].copy()
    X["is_equity"] = (fam.reindex(f.index) == "equity").astype(float)
    X["asset_class"] = uni["asset_class"].reindex(f.index).astype("category").cat.codes
    return X

class Booster:
    """Retrains every `retrain_months` months on `train_years` of history with forward returns already known."""
    def __init__(self, feats, fwd, fam, uni, horizon_days=63, train_years=8, retrain_months=12):
        self.feats, self.fwd, self.fam, self.uni = feats, fwd, fam, uni
        self.h, self.years, self.every = horizon_days, train_years, retrain_months
        self.model, self.trained_at, self.oos = None, None, []

    def _train(self, t):
        cutoff = t - pd.tseries.offsets.BDay(self.h)
        dates = [d for d in self.feats if cutoff - pd.DateOffset(years=self.years) <= d <= cutoff and d in self.fwd]
        if len(dates) < 36:
            return None
        Xs, ys = [], []
        for d in dates:
            f = self.feats[d]; y = self.fwd[d].reindex(f.index)
            ok = y.notna()
            fam = self.fam.reindex(f.index)
            yr = y[ok].groupby(fam[ok]).rank(pct=True) - 0.5      # rank within the family
            Xs.append(_xy(f[ok], self.fam, self.uni)); ys.append(yr)
        m = lgb.LGBMRegressor(**PARAMS, random_state=0)
        m.fit(pd.concat(Xs), pd.concat(ys))
        return m

    def predict(self, t, f_t: pd.DataFrame) -> pd.Series:
        t = pd.Timestamp(t)
        if self.model is None or (t - self.trained_at).days > self.every * 30.5:
            m = self._train(t)
            if m is not None:
                self.model, self.trained_at = m, t
                log.info("Selection LightGBM retrained at %s", t.date())
        if self.model is None:
            return pd.Series(0.0, index=f_t.index)
        p = pd.Series(self.model.predict(_xy(f_t, self.fam, self.uni)), index=f_t.index)
        out = p.groupby(self.fam.reindex(f_t.index)).rank(pct=True) - 0.5
        if t in self.fwd:                                       # out-of-sample IC (diagnostics only)
            y = self.fwd[t].reindex(out.index)
            ok = y.notna()
            if ok.sum() > 50:
                self.oos.append({"date": t, "ic": out[ok].rank().corr(y[ok].rank())})
        return out * 4
