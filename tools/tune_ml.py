"""Hyper-parameter search for the expected-return LightGBM, using data up to 2013 only.
Saves the best ones to data/experiments/ml_params.json (read when portfolio.ml_returns.params is 'auto')."""
import sys, json, logging, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from core.config import load_config
from agents.portfolio_agent.agent import load
from agents.portfolio_agent.backtest import factor_model
from agents.portfolio_agent import ml_returns as MR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
cfg = load_config(); d = cfg["paths"]["data"]
px_local, px_eur, vol, uni, rf = load(d / "processed", "EUR")
macro = pd.read_parquet(d / "processed" / "macro.parquet")
fund = pd.read_parquet(d / "processed" / "fundamentals.parquet")
feats, ic, fam, _, fwd = factor_model(px_eur, px_local, uni, fund, d / "portfolio")
raw = factor_model.raw
ctx = {t: MR.macro_context(px_eur, macro, t) for t in feats}
fwd_train = {t: v for t, v in fwd.items() if t + pd.tseries.offsets.BDay(MR.H) <= pd.Timestamp("2013-12-31")}
allres = []
for use_macro, demean in [(True, False), (False, False), (False, True), (True, True)]:
    MR.OPTIONS.update(use_macro=use_macro, demean=demean)
    r = MR.tune(feats, raw, fwd_train, fam, uni, ctx).assign(use_macro=use_macro, demean=demean)
    allres.append(r)
res = pd.concat(allres).sort_values("score", ascending=False)
res.to_csv(d / "experiments" / "ml_tuning.csv", index=False)
print(res[["num_leaves","min_child_samples","n_estimators","use_macro","demean","mean_ic","ic_hit","pooled_ic","score"]].round(4).head(12).to_string(index=False))
best = res.iloc[0][["num_leaves", "min_child_samples", "n_estimators", "learning_rate", "colsample_bytree", "subsample", "reg_lambda"]].to_dict()
best = {k: (int(v) if k in ("num_leaves", "min_child_samples", "n_estimators") else float(v)) for k, v in best.items()}
best["_options"] = {"use_macro": bool(res.iloc[0].use_macro), "demean": bool(res.iloc[0].demean)}
best["_validation"] = {"mean_ic": float(res.iloc[0].mean_ic), "pooled_ic": float(res.iloc[0].pooled_ic), "ic_hit": float(res.iloc[0].ic_hit)}
(d / "experiments" / "ml_params.json").write_text(json.dumps(best, indent=1))
print("Best parameters:", best)
