"""Risk mode: offensive profile by default, defensive profile when enough warning signals fire.
Signals are checked weekly using only data available at that date."""
import logging
import numpy as np
import pandas as pd
from . import regime as RG

log = logging.getLogger("portfolio_agent")

def signals_at(t, px_local, px_eur, macro, feat_hmm, tcfg, mkt="SPY"):
    """Value and state (on/off) of each signal at date t."""
    out = {}
    if tcfg.get("hmm_crisis_prob") is not None:
        _, probs = RG.regime_at(feat_hmm, t)
        out["hmm_crisis"] = (round(probs["crisis"], 2), probs["crisis"] >= tcfg["hmm_crisis_prob"])
    if tcfg.get("vix") is not None:
        v = macro["VIXCLS"].loc[:t].dropna().tail(20).mean()
        out["vix"] = (round(float(v), 1), bool(v > tcfg["vix"]))
    s = px_local[mkt].loc[:t].dropna()
    if tcfg.get("trend_ma_days"):
        gap = s.iloc[-1] / s.tail(tcfg["trend_ma_days"]).mean() - 1
        out["trend"] = (round(float(gap), 3), bool(gap < 0))
    if tcfg.get("market_drawdown") is not None:
        dd = s.iloc[-1] / s.tail(252).max() - 1
        out["market_dd"] = (round(float(dd), 3), bool(dd < tcfg["market_drawdown"]))
    if tcfg.get("credit_z") is not None:
        c = -(np.log(px_eur["HYG"] / px_eur["IEF"]).loc[:t].resample("W-FRI").last().diff(13)).dropna()
        z = (c.iloc[-1] - c.tail(260).mean()) / c.tail(260).std()
        out["credit"] = (round(float(z), 2), bool(z > tcfg["credit_z"]))
    return out

def signal_table(px_local, px_eur, macro, tcfg, start):
    """Weekly signals (on/off) from `start`, using only data available at each date."""
    feat = RG.features(macro, px_eur)
    weeks = pd.Series(px_local.index, index=px_local.index).loc[start:].resample("W-FRI").last().dropna()
    rows = []
    for t in weeks.values:
        t = pd.Timestamp(t)
        sig = signals_at(t, px_local, px_eur, macro, feat, tcfg)
        rows.append({"date": t, **{k: v[1] for k, v in sig.items()}})
    return pd.DataFrame(rows).set_index("date")

def modes_from_signals(sig: pd.DataFrame, rcfg, use=None):
    """Applies the hysteresis rule to a signal table (optionally to a subset of signals)."""
    cols = [c for c in (use or sig.columns) if c in sig]
    n = sig[cols].sum(axis=1) if cols else pd.Series(0, index=sig.index)
    mode, calm, out = "offense", 0, []
    for k in n.values:
        if mode == "offense" and cols and k >= rcfg["enter"]:
            mode, calm = "defense", 0
        elif mode == "defense":
            calm = calm + 1 if k <= rcfg["exit_max"] else 0
            if calm >= rcfg["exit_weeks"]:
                mode = "offense"
        out.append(mode)
    return pd.DataFrame({"n_signals": n.values, "mode": out}, index=sig.index).join(sig)

def mode_history(px_local, px_eur, macro, rcfg, start):
    """Weekly mode series: switch to defense with at least `enter` signals,
    back to offense after `exit_weeks` consecutive weeks with at most `exit_max` signals."""
    sig = signal_table(px_local, px_eur, macro, rcfg["triggers"], start)
    h = modes_from_signals(sig, rcfg)
    log.info("Risk mode: %.0f%% of weeks in defense, %d switches",
             100 * (h["mode"] == "defense").mean(), int((h["mode"] != h["mode"].shift()).sum() - 1))
    return h
