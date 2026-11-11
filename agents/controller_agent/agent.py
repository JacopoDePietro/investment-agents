"""Controller agent: checks the proposal against configurable thresholds: APPROVED, APPROVED WITH WARNINGS or BLOCKED.
When blocked, the last approved portfolio stays active. It never places orders."""
import json, logging, shutil
from datetime import datetime
import pandas as pd
from core.obsidian import write_note

log = logging.getLogger("controller_agent")

def run(cfg):
    c, d, vault = cfg["controller"], cfg["paths"]["data"], cfg["paths"]["vault"]
    out = d / "controller"; out.mkdir(parents=True, exist_ok=True)
    tg = pd.read_csv(d / "portfolio" / "target_portfolio.csv", index_col=0)
    ctx = json.loads((d / "portfolio" / "context.json").read_text())
    risk = json.loads((d / "modeling" / "risk_report.json").read_text())
    w = tg["weight_real"] / tg["weight_real"].sum()
    mc = risk["monte_carlo"].get("1") or risk["monte_carlo"].get(1)
    hard, soft = [], []

    def check(ok, level, name, value, limit):
        (hard if level == "hard" else soft).append({"check": name, "value": value, "limit": limit, "ok": bool(ok)})

    h = c["hard"]
    check(mc["CVaR95"] >= h["max_cvar95_1y"], "hard", "1-year CVaR 95% (Monte Carlo)", round(mc["CVaR95"], 3), h["max_cvar95_1y"])
    check(mc["prob_dd20"] <= h["max_prob_dd20_1y"], "hard", "1-year P(drawdown > 20%)", round(mc["prob_dd20"], 3), h["max_prob_dd20_1y"])
    for name, lim in h["max_stress_loss"].items():
        v = risk["stress_historical"].get(name, {}).get("port")
        if v is not None:
            check(v >= lim, "hard", f"Historical crisis {name}", round(v, 3), lim)
    worst_h = min((v["port"] for v in risk["stress_hypothetical"].values()), default=0)
    check(worst_h >= h["max_hypothetical_loss"], "hard", "Worst hypothetical scenario", round(worst_h, 3), h["max_hypothetical_loss"])
    viol = ctx.get("violations") or {}
    check(not (h["block_on_violations"] and viol), "hard", "Optimiser constraints", viol or "none", "no breaches")
    check(w.max() <= h["max_single_weight"], "hard", "Max single-asset weight", round(w.max(), 3), h["max_single_weight"])

    s = c["soft"]
    prev_path = out / "approved_portfolio.csv"
    if prev_path.exists():
        prev = pd.read_csv(prev_path, index_col=0)["weight_real"]
        prev = prev / prev.sum()
        to = float((w.reindex(w.index.union(prev.index)).fillna(0) - prev.reindex(w.index.union(prev.index)).fillna(0)).abs().sum())
        check(to <= s["max_turnover"], "soft", "Turnover vs last approved", round(to, 2), s["max_turnover"])
    check(mc["prob_underperform_spy"] <= s["max_prob_underperform_spy_1y"], "soft", "1-year P(underperforming the S&P 500)",
          round(mc["prob_underperform_spy"], 2), s["max_prob_underperform_spy_1y"])
    hit = _bad_news(d, w, s["news"])
    check(hit["weight"] <= s["news"]["max_weight_affected"], "soft", "Weight on assets with very negative news",
          f"{hit['weight']:.1%} {hit['tickers']}", s["news"]["max_weight_affected"])

    failed_hard = [x for x in hard if not x["ok"]]
    failed_soft = [x for x in soft if not x["ok"]]
    verdict = "BLOCKED" if failed_hard else ("APPROVED WITH WARNINGS" if failed_soft else "APPROVED")
    if verdict != "BLOCKED":
        shutil.copy(d / "portfolio" / "target_portfolio.csv", prev_path)
    dec = {"date": ctx.get("date"), "decided_at": datetime.now().isoformat(timespec="seconds"),
           "verdict": verdict, "hard": hard, "soft": soft,
           "active_portfolio": "new proposal" if verdict != "BLOCKED" else
                               ("last approved" if prev_path.exists() else "none")}
    (out / "decision.json").write_text(json.dumps(dec, indent=1, default=str))
    hist = out / "decisions.csv"
    pd.DataFrame([{"date": dec["date"], "verdict": verdict, "failed": "; ".join(x["check"] for x in failed_hard + failed_soft)}]) \
        .to_csv(hist, mode="a", header=not hist.exists(), index=False)
    _note(vault, dec)
    log.info("Verdict: %s", verdict)
    return {"verdict": verdict, "hard_failed": len(failed_hard), "soft_failed": len(failed_soft)}

def _bad_news(d, w, ncfg):
    p = d / "news" / "news.parquet"
    if not p.exists():
        return {"weight": 0.0, "tickers": []}
    n = pd.read_parquet(p)
    n = n[pd.to_datetime(n.fetched_at, utc=True) > pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)]
    n = n[(n.impact >= ncfg["min_impact"]) & (n.sentiment <= ncfg["max_sentiment"])]
    hit = sorted({t for ts in n.tickers for t in ts if t in w.index})
    return {"weight": float(w.reindex(hit).sum()) if hit else 0.0, "tickers": hit}

def _note(vault, dec):
    icon = {"APPROVED": "✅", "APPROVED WITH WARNINGS": "⚠️", "BLOCKED": "⛔"}[dec["verdict"]]
    rows = lambda xs: "\n".join(f"| {'✅' if x['ok'] else '❌'} | {x['check']} | {x['value']} | {x['limit']} |" for x in xs)
    body = (f"# Control {dec['date']}\n\n## {icon} {dec['verdict']}\n\nActive portfolio: **{dec['active_portfolio']}**\n\n"
            "## Blocking checks\n| | Check | Value | Limit |\n|---|---|---|---|\n" + rows(dec["hard"]) +
            "\n\n## Warnings\n| | Check | Value | Limit |\n|---|---|---|---|\n" + rows(dec["soft"]) +
            "\n\nThresholds in `config/settings.yaml`, section `controller`. No orders are ever placed.\n")
    write_note(vault, "Reports", f"control_{dec['date']}", {"tags": "[report, controller_agent]", "verdict": dec["verdict"]}, body)
