"""Modeling agent: 1M+ Monte Carlo paths, stress tests and risk metrics for the proposed portfolio."""
import json, logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core.obsidian import write_note
from agents.portfolio_agent.agent import load
from .cflib_bridge import load_cflib
from .calibrate import calibrate, exposures
from .simulate import simulate
from .stress import historical, hypothetical

log = logging.getLogger("modeling_agent")

def summarize(sim, years, cap):
    p, s = sim["port"], sim["spy"]
    q = lambda x, a: float(np.quantile(x, a))
    var95, var99 = q(p, 0.05), q(p, 0.01)
    return {
        "years": years, "paths": int(len(p)),
        "mean": float(p.mean()), "median": q(p, 0.5), "p05": var95, "p95": q(p, 0.95),
        "VaR95": var95, "VaR99": var99,
        "CVaR95": float(p[p <= var95].mean()), "CVaR99": float(p[p <= var99].mean()),
        "prob_loss": float((p < 0).mean()),
        "prob_dd10": float((sim["mdd"] < -0.10).mean()), "prob_dd20": float((sim["mdd"] < -0.20).mean()),
        "prob_dd30": float((sim["mdd"] < -0.30).mean()), "median_mdd": q(sim["mdd"], 0.5),
        "prob_below_80_from_start": float((sim["min_w"] <= 0.8).mean()),
        "prob_underperform_spy": float((p < s).mean()),
        "spy_median": q(s, 0.5), "spy_CVaR95": float(s[s <= q(s, 0.05)].mean()),
        "spy_prob_dd20": float((sim["mdd_spy"] < -0.20).mean()),
        "vol_median": q(sim["vol"], 0.5),
        "wealth_p05": cap * (1 + var95), "wealth_median": cap * (1 + q(p, 0.5)), "wealth_p95": cap * (1 + q(p, 0.95)),
    }

def run(cfg):
    mcfg, d, vault = cfg["modeling"], cfg["paths"]["data"], cfg["paths"]["vault"]
    out = d / "modeling"; out.mkdir(parents=True, exist_ok=True)
    CIR, QT, jmp_normal, first_passage = load_cflib(mcfg["cflib_path"])
    _, px_eur, _, uni, _ = load(d / "processed", cfg.get("base_currency", "EUR"))
    macro = pd.read_parquet(d / "processed" / "macro.parquet")
    tg = pd.read_csv(d / "portfolio" / "target_portfolio.csv", index_col=0)
    w = tg["weight_real"] / tg["weight_real"].sum()
    cap = cfg["portfolio"]["capital"]

    cal = calibrate(px_eur, macro, mcfg)
    bp, idio_p, B, idio = exposures(px_eur, macro, w, mcfg)
    log.info("Portfolio betas: %s · weekly idiosyncratic risk %.4f", bp.round(2).to_dict(), idio_p)

    results, sims = {}, {}
    for h in mcfg["horizons_years"]:
        sim, secs = simulate(cal, bp, idio_p, mcfg, h, CIR, QT, jmp_normal,
                             mcfg["n_paths"], mcfg["batch"], mcfg["seed"] + h)
        results[h] = summarize(sim, h, cap); results[h]["seconds"] = round(secs, 1)
        # cross-check with the analytical first-passage formula (geometric Brownian motion)
        lr = np.log1p(sim["port"]); mu_a = lr.mean() / h; sig_a = max(results[h]["vol_median"], 1e-4)
        results[h]["first_passage_80_gbm"] = float(first_passage(1.0, h, mu_a + 0.5 * sig_a ** 2, sig_a, 0.8))
        sims[h] = sim

    st_h = historical(px_eur, w, B, mcfg["stress_historical"], [c for c in B.columns if c != "RATE"])
    st_x = hypothetical(w, B, mcfg["stress_hypothetical"], set(cfg.get("ai_bubble", {}).get("tickers", [])))
    report = {"date": str(px_eur.index[-1].date()), "calibration": {k: v for k, v in cal.items() if k in ("params", "rate", "crisis_now", "crisis_share", "rf")},
              "portfolio_beta": bp.to_dict(), "idio_weekly": idio_p, "monte_carlo": results,
              "stress_historical": st_h, "stress_hypothetical": st_x}
    (out / "risk_report.json").write_text(json.dumps(report, indent=1, default=float))
    _chart(sims[mcfg["horizons_years"][0]], vault)
    _note(vault, report, cap)
    r1 = results[mcfg["horizons_years"][0]]
    return {"paths": r1["paths"], "CVaR95_1y": round(r1["CVaR95"], 3), "prob_dd20_1y": round(r1["prob_dd20"], 3),
            "seconds": {h: results[h]["seconds"] for h in results}}

def _chart(sim, vault):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    bins = np.linspace(-0.6, 0.8, 120)
    for x, lab, c in ((sim["port"], "Portfolio", "#2a78d6"), (sim["spy"], "S&P 500", "#eb6834")):
        a1.hist(np.clip(x, bins[0], bins[-1]), bins=bins, density=True, histtype="step", lw=2, color=c, label=lab)
    a1.set_title("1-year return (EUR)", loc="left", fontsize=11); a1.legend(frameon=False)
    a1.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    bins2 = np.linspace(-0.7, 0, 100)
    for x, lab, c in ((sim["mdd"], "Portfolio", "#2a78d6"), (sim["mdd_spy"], "S&P 500", "#eb6834")):
        a2.hist(x, bins=bins2, density=True, histtype="step", lw=2, color=c, label=lab)
    a2.set_title("Max drawdown from peak within 1 year", loc="left", fontsize=11)
    a2.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    for a in (a1, a2):
        a.spines[["top", "right"]].set_visible(False); a.set_yticks([]); a.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb"); fig.tight_layout()
    (vault / "attachments").mkdir(parents=True, exist_ok=True)
    fig.savefig(vault / "attachments" / "montecarlo_1y.png", dpi=130); plt.close(fig)

def _note(vault, rep, cap):
    pct = lambda x: f"{x:+.1%}"
    lines = [f"# Portfolio risk {rep['date']}\n",
             "Monte Carlo: correlated Heston factors (Andersen QE scheme) with calm/crisis regimes, "
             "CIR yield, Merton jumps. Returns in EUR.\n",
             "| Horizon | Paths | Median | 5th percentile | CVaR 95% | CVaR 99% | P(loss) | P(DD>20%) | P(worse than SPY) | Time |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for h, r in rep["monte_carlo"].items():
        lines.append(f"| {h} years | {r['paths']:,} | {pct(r['median'])} | {pct(r['p05'])} | {pct(r['CVaR95'])} | {pct(r['CVaR99'])} | "
                     f"{r['prob_loss']:.0%} | {r['prob_dd20']:.0%} | {r['prob_underperform_spy']:.0%} | {r['seconds']:.0f}s |")
    r = rep["monte_carlo"][min(rep["monte_carlo"])]
    lines += [f"\nWith {cap:,.0f} EUR, in one year: 5% of cases below **{r['wealth_p05']:,.0f}**, median **{r['wealth_median']:,.0f}**, "
              f"5% of cases above **{r['wealth_p95']:,.0f}** EUR.",
              f"S&P 500 on the same scenarios: median {pct(r['spy_median'])}, CVaR 95% {pct(r['spy_CVaR95'])}, P(DD>20%) {r['spy_prob_dd20']:.0%}.",
              f"Check: P(falling below 80% of capital) Monte Carlo {r['prob_below_80_from_start']:.1%} · first-passage formula {r['first_passage_80_gbm']:.1%}.\n",
              "![[montecarlo_1y.png]]\n", "## Historical crises replayed on today's portfolio",
              "| Crisis | Portfolio | S&P 500 | Assets estimated via betas | Worst contributors |", "|---|---|---|---|---|"]
    for k, v in rep["stress_historical"].items():
        worst = ", ".join(f"{t} {c:+.1%}" for t, c in v["worst"])
        lines.append(f"| {k} | {pct(v['port'])} | {pct(v['spy'])} | {v['proxied_share']:.0%} | {worst} |")
    lines += ["\n## Hypothetical scenarios", "| Scenario | Portfolio | Worst contributors |", "|---|---|---|"]
    for k, v in rep["stress_hypothetical"].items():
        lines.append(f"| {k} | {pct(v['port'])} | " + ", ".join(f"{t} {c:+.1%}" for t, c in v["worst"]) + " |")
    c = rep["calibration"]
    lines += ["\n## Heston calibration (historical)", "| Factor | kappa | theta (vol) | sigma | vol today | rho | drift |", "|---|---|---|---|---|---|---|"]
    for f, p in c["params"].items():
        lines.append(f"| {f} | {p['kappa']:.2f} | {np.sqrt(p['theta']):.1%} | {p['sigma']:.2f} | {np.sqrt(p['v0']):.1%} | {p['rho']:+.2f} | {p['mu']:+.1%} |")
    rt = c["rate"]
    lines.append(f"\n10-year yield CIR: kappa {rt['kappa']:.2f}, theta {rt['theta']:.2%}, sigma {rt['sigma']:.3f}, today {rt['y0']:.2%}. "
                 f"Current regime: {'crisis' if c['crisis_now'] else 'calm'} · share of crisis weeks in history {c['crisis_share']:.0%}.")
    write_note(vault, "Reports", f"risk_{rep['date']}", {"tags": "[report, modeling_agent]"}, "\n".join(lines))
