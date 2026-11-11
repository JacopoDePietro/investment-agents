"""Backtest variant comparison. Usage: python tools/experiment.py <variant>
Results in data/experiments/ (metrics by horizon and daily returns)."""
import sys, logging, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from core.config import load_config
from agents.portfolio_agent.agent import load, apply_profile, horizons
from agents.portfolio_agent.backtest import run_backtest

VARIANTS = {
    "switching": lambda p: None,                                            # current configuration
    "always_offense": lambda p: p.update(risk_mode={**p["risk_mode"], "triggers": {}, "rebalance_on_switch": False}),
    "always_defense": lambda p: p.update(risk_mode={**p["risk_mode"], "enter": 0, "exit_max": -1, "rebalance_on_switch": False}),
    "no_ml": lambda p: p.update(ml={"enabled": False}),
    "off_no_mc": lambda p: (VARIANTS["always_offense"](p), p.update(mc_scenarios={"enabled": False})),
    "off_no_stockfilter": lambda p: (VARIANTS["always_offense"](p), p.update(single_stocks_require_fundamentals=False)),
    "off_no_capeff": lambda p: (VARIANTS["always_offense"](p), p.update(exclude_groups=p["exclude_groups"] + ["capital_efficient"])),
}

def _mlr(lam, gamma, end):
    def f(p):
        VARIANTS["always_offense"](p)
        p["ml_returns"] = {**p["ml_returns"], "blend": lam, "gamma": gamma}
        p["methods"] = ["mean_variance", "max_sharpe", "max_diversification"]
        if end: p["backtest_end"] = end
    return f

for lam in (0.5, 1.0):
    for g in (2, 5):
        VARIANTS[f"mlr_l{lam}_g{g}_train"] = _mlr(lam, g, "2013-12-31")
        VARIANTS[f"mlr_l{lam}_g{g}_full"] = _mlr(lam, g, None)

def run(name):
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s [{name}] %(message)s", stream=sys.stdout)
    cfg = load_config(); d = cfg["paths"]["data"]; out = d / "experiments"; out.mkdir(exist_ok=True)
    p = apply_profile(dict(cfg["portfolio"])); p["ai_tickers"] = cfg["ai_bubble"]["tickers"]
    p["methods"] = ["max_diversification"]
    VARIANTS[name](p)
    px_local, px_eur, vol, uni, rf = load(d / "processed", "EUR")
    macro = pd.read_parquet(d / "processed" / "macro.parquet")
    fund = pd.read_parquet(d / "processed" / "fundamentals.parquet")
    daily, _, to, extra = run_backtest(px_eur, px_local, vol, uni, p, rf, macro, fund, d / "portfolio", cfg["modeling"])
    r = px_eur.pct_change(fill_method=None).loc[daily.index]
    daily = daily.join(pd.DataFrame({"SPY": r["SPY"], "60/40": 0.6 * r["SPY"].fillna(0) + 0.4 * r["IEF"].fillna(0)}))
    daily.to_parquet(out / f"returns_{name}.parquet")
    pd.concat(horizons(daily, rf), names=["horizon"]).to_csv(out / f"horizons_{name}.csv")
    if extra.get("modes") is not None:
        extra["modes"].to_csv(out / f"modes_{name}.csv")
    print(f"{name} completed")

if __name__ == "__main__":
    run(sys.argv[1])
