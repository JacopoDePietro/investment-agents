"""Calibration of the defensive (anti-crisis) profile on separate periods.
Requires the daily returns of 'always offense' and 'always defense' (tools/experiment.py).
The switching strategy is approximated by using, day by day, the profile decided the previous week,
plus a cost per switch. The rule is chosen on the training period and validated on the following one."""
import sys, itertools, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from core.config import load_config
from core.metrics import perf
from agents.portfolio_agent.agent import load
from agents.portfolio_agent import risk_mode as RM

SETS = {"all": None, "slow": ["trend", "credit", "market_dd"], "stress": ["hmm_crisis", "vix", "credit"],
        "trend_dd": ["trend", "market_dd"], "vix_trend": ["vix", "trend"], "credit_trend": ["credit", "trend"]}

def switched(off, dfn, modes, cost):
    m = modes["mode"].reindex(off.index, method="ffill").shift(1).fillna("offense")   # decision taken the previous week
    r = off.where(m == "offense", dfn)
    sw = (m != m.shift()).astype(float); sw.iloc[0] = 0
    return r - sw * cost, int(sw.sum())

def main(train_end="2016-12-31", method="max_diversification", cost=0.004):
    cfg = load_config(); d = cfg["paths"]["data"]; ex = d / "experiments"
    off = pd.read_parquet(ex / "returns_always_offense.parquet")[method]
    dfn = pd.read_parquet(ex / "returns_always_defense.parquet")[method]
    idx = off.index.intersection(dfn.index); off, dfn = off.loc[idx], dfn.loc[idx]
    px_local, px_eur, _, _, _ = load(d / "processed", "EUR")
    macro = pd.read_parquet(d / "processed" / "macro.parquet")
    rcfg = cfg["portfolio"]["risk_mode"]
    sig = RM.signal_table(px_local, px_eur, macro, rcfg["triggers"], idx[0])
    rows = []
    for (sname, use), enter, exit_w in itertools.product(SETS.items(), [1, 2, 3], [2, 4, 8]):
        cols = use or list(sig.columns)
        if enter > len(cols):
            continue
        rc = {"enter": enter, "exit_max": enter - 1, "exit_weeks": exit_w}
        modes = RM.modes_from_signals(sig, rc, cols)
        r, nsw = switched(off, dfn, modes, cost)
        a, b = r.loc[:train_end], r.loc[train_end:]
        rows.append({"signals": sname, "enter": enter, "exit_weeks": exit_w, "switches": nsw,
                     "train_sharpe": perf(a)["Sharpe"], "train_cagr": perf(a)["CAGR"], "train_dd": perf(a)["MaxDrawdown"],
                     "test_sharpe": perf(b)["Sharpe"], "test_cagr": perf(b)["CAGR"], "test_dd": perf(b)["MaxDrawdown"]})
    res = pd.DataFrame(rows)
    ref = {k: perf(x.loc[:train_end] if k.startswith("train") else x.loc[train_end:])
           for k, x in (("train_off", off), ("test_off", off), ("train_def", dfn), ("test_def", dfn))}
    res = res.sort_values("train_sharpe", ascending=False)
    res.to_csv(ex / "risk_mode_tuning.csv", index=False)
    print("References:", {k: (round(v["Sharpe"], 2), round(v["CAGR"], 3), round(v["MaxDrawdown"], 3)) for k, v in ref.items()})
    print(res.head(10).round(3).to_string(index=False))
    best = res.iloc[0]
    beats_train = best.train_sharpe > ref["train_off"]["Sharpe"]
    beats_test = best.test_sharpe > ref["test_off"]["Sharpe"]
    verdict = {"best": best.to_dict(), "beats_offense_train": bool(beats_train), "beats_offense_test": bool(beats_test),
               "reference": {k: {m: float(v[m]) for m in ("Sharpe", "CAGR", "MaxDrawdown")} for k, v in ref.items()}}
    (ex / "risk_mode_tuning.json").write_text(json.dumps(verdict, indent=1, default=float))
    print("Best training rule beats 'always offense': training", beats_train, "· validation", beats_test)

if __name__ == "__main__":
    main()
