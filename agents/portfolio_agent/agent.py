"""Portfolio agent: walk-forward backtest of the allocation methods and today's proposed portfolio."""
import json, logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core.currency import to_base
from core.synthetic import backfill
from core.metrics import perf
from core.obsidian import write_note
from .backtest import predict_mu, run_backtest, build_weights, eligible, date_context, violations, regime_profile, momentum_scale
from . import regime as RG
from . import risk_mode as RM
from . import factors as F

log = logging.getLogger("portfolio_agent")
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # validated categorical palette, fixed order

def load(d, base, recipes=None):
    px = pd.read_parquet(d / "prices.parquet")
    vol = pd.read_parquet(d / "volume.parquet")
    uni = pd.read_parquet(d / "universe.parquet")
    fx = pd.read_parquet(d / "fx.parquet")
    grid = pd.bdate_range(px.index[0], px.index[-1])          # business days common to all markets
    px_local = px.reindex(grid).ffill(limit=5)
    vol = vol.reindex(grid).fillna(0)
    if recipes is None:
        from core.config import load_config
        recipes = load_config().get("synthetic_history", {})
    if recipes:
        macro = pd.read_parquet(d / "macro.parquet")
        px_local, vol, info = backfill(px_local, vol, macro, recipes)
        uni["synthetic_from"] = pd.Series(info)
    fpath = d / "fundamentals.parquet"
    uni["has_fundamentals"] = uni.index.isin(pd.read_parquet(fpath, columns=["ticker"]).ticker.unique()) if fpath.exists() else False
    px_eur = to_base(px_local, fx, base)
    macro = pd.read_parquet(d / "macro.parquet")
    rf = (macro.get("ECBDFR", pd.Series(0.0, index=grid)).reindex(grid).ffill().fillna(0) / 100 / 252)
    return px_local, px_eur, vol, uni, rf

def news_adjustment(d, uni, tilt):
    """News score per asset: the ticker signal, otherwise the average of its asset class and region."""
    p = d / "news_signals.parquet"
    if tilt == 0 or not p.exists():
        return None
    s = pd.read_parquet(p)
    by = {k: g.set_index("key")["score"] for k, g in s.groupby("kind")}
    tk, ac, rg = by.get("ticker", pd.Series()), by.get("asset_class", pd.Series()), by.get("region", pd.Series())
    cls = uni["asset_class"].replace({"equity_etf": "equity", "alternative": "equity"})
    fallback = 0.5 * cls.map(ac).fillna(0) + 0.5 * uni["region"].map(rg).fillna(0)
    return tilt * tk.reindex(uni.index).fillna(fallback)

def apply_profile(pcfg):
    """Overrides the limits with those of the selected risk profile."""
    prof = pcfg.get("profiles", {}).get(pcfg.get("risk_profile", "balanced"), {}) or {}
    for k, v in prof.items():
        if isinstance(v, dict):
            pcfg[k] = {**pcfg.get(k, {}), **v}
        else:
            pcfg[k] = v
    return pcfg

def run(cfg):
    pcfg, base = apply_profile(dict(cfg["portfolio"])), cfg.get("base_currency", "EUR")
    pcfg["ai_tickers"] = cfg.get("ai_bubble", {}).get("tickers", [])
    d, vault = cfg["paths"]["data"], cfg["paths"]["vault"]
    out = d / "portfolio"; out.mkdir(parents=True, exist_ok=True)
    px_local, px_eur, vol, uni, rf = load(d / "processed", base)
    macro = pd.read_parquet(d / "processed" / "macro.parquet")
    fpath = d / "processed" / "fundamentals.parquet"
    fund_df = pd.read_parquet(fpath) if fpath.exists() else None

    daily, hist, turnover, extra = run_backtest(px_eur, px_local, vol, uni, pcfg, rf, macro, fund_df, out, cfg["modeling"])
    r_eur = px_eur.pct_change(fill_method=None).loc[daily.index]
    bench = pd.DataFrame({"SPY": r_eur["SPY"], "VT": r_eur["VT"],
                          "60/40": 0.6 * r_eur["SPY"].fillna(0) + 0.4 * r_eur["IEF"].fillna(0)})
    allr = daily.join(bench)
    metrics = pd.DataFrame({c: perf(allr[c], rf) for c in allr}).T
    metrics["Turnover_yr"] = pd.Series(turnover)
    metrics.to_csv(out / "backtest_metrics.csv")
    allr.to_parquet(out / "backtest_returns.parquet")
    pd.DataFrame({(m, t): w for m, h in hist.items() for t, w in h.items()}).T.to_parquet(out / "weights_history.parquet")
    save_diagnostics(out, extra, hist, uni)

    if pcfg["choose_by"].lower() == "multi_horizon":
        # mean Sharpe over 1, 2, 5 years and the full period: rewards methods that work on every horizon
        hz = horizons(allr, rf)
        labs = [l for l in ("1y", "2y", "5y") if l in hz] + [l for l in hz if l.startswith("since")][-1:]
        metrics["MultiHorizonSharpe"] = pd.concat([hz[l]["Sharpe"] for l in labs], axis=1).mean(axis=1)
        metrics.to_csv(out / "backtest_metrics.csv")
        key = "MultiHorizonSharpe"
    else:
        key = {c.lower(): c for c in metrics.columns}[pcfg["choose_by"].lower()]

    # Today's proposed portfolio, including the news signal
    t = px_eur.index[-1]
    cols = eligible(px_local, vol, uni, t, pcfg)
    adj = news_adjustment(d / "processed", uni, pcfg["news_tilt"])
    reg_now = RG.regime_at(RG.features(macro, px_eur), t)
    ctx = date_context(px_eur, macro, t, pcfg, reg_now)
    if "risk_mode" in pcfg:
        ctx["signals"] = RM.signals_at(t, px_local, px_eur, macro, RG.features(macro, px_eur), pcfg["risk_mode"]["triggers"])
        if extra.get("modes") is not None:
            ctx["mode"] = extra["modes"]["mode"].iloc[-1]
            extra["modes"].to_csv(out / "risk_mode_history.csv")
        else:
            ctx["mode"] = "offense"
    pcfg_t = regime_profile(pcfg, ctx.get("mode", ctx.get("regime", "normal")))
    base, fw_now = None, None
    rets_all = px_eur.pct_change(fill_method=None)
    if "ic" in extra:
        extra["ic"].to_parquet(out / "factor_ic.parquet")
        fw_now = F.weights_at(extra["ic"], t)
        sc = momentum_scale(pcfg_t, ctx)
        if sc < 1:
            fw_now = F.dampen_momentum(fw_now, sc)
        fund = F.Fundamentals(fund_df) if fund_df is not None else None
        fam = F.family(uni["asset_class"])
        ft = F.rank_z(F.features_at(px_eur, len(px_eur) - 1, px_eur["SPY"], fund, px_local), fam)
        base = F.scores_at(ft, fam, fw_now)
        parts = pd.DataFrame({"factors": base})
        if extra.get("booster") is not None:
            a = pcfg_t.get("ml", {}).get("tilt", 0.5)
            pred = extra["booster"].predict(t, ft).reindex(base.index).fillna(0)
            parts["lightgbm_selection"] = pred
            base = (1 - a) * base + a * pred
        if adj is not None:
            parts["news"] = adj.reindex(base.index).fillna(0)
        parts["final_score"] = base + (parts["news"] if "news" in parts else 0)
        parts.join(uni[["asset_class", "region", "name"]]).sort_values("final_score", ascending=False).to_csv(out / "scores_today.csv")
    ctx.update(macro=macro, mcfg=cfg["modeling"])
    if extra.get("ml_returns") is not None:
        ctx["mu_ml"] = predict_mu(extra["ml_returns"], t, extra["feats"], px_eur, macro)
        if ctx["mu_ml"] is not None:
            ctx["mu_ml"].rename("mu_ml").to_csv(out / "expected_returns_ml.csv")
    w_all, chosen = build_weights(px_eur, rets_all, uni, cols, pcfg_t, pcfg["methods"], t, adj, base, ctx)
    # Monte Carlo simulations decide which methods are acceptable; among them the backtest decides
    best, mc_table = mc_select(w_all, metrics[key], px_eur, macro, cfg, pcfg)
    mc_table.to_csv(out / "mc_selection.csv")
    log.info("Chosen method (Monte Carlo filter + backtest %s): %s", pcfg["choose_by"], best)
    w = w_all[best].sort_values(ascending=False)
    cap = pcfg["capital"]
    price = px_eur.loc[t, w.index]
    qty = w * cap / price
    if not pcfg["fractional_shares"]:
        qty = np.floor(qty)
    target = pd.DataFrame({"weight": w, "value_eur": qty * price, "price_eur": price, "quantity": qty,
                           "currency": uni.loc[w.index, "currency"], "asset_class": uni.loc[w.index, "asset_class"],
                           "region": uni.loc[w.index, "region"], "name": uni.loc[w.index, "name"],
                           "news_adj": (adj.reindex(w.index) if adj is not None else 0)})
    target["weight_real"] = target.value_eur / cap
    target.to_csv(out / "target_portfolio.csv")
    sub = r_eur.loc[:, w.index].tail(pcfg["lookback_days"])
    avg_corr = sub.corr().values[np.triu_indices(len(w), 1)].mean()

    _chart(allr[[best, "SPY", "60/40", "VT"]], best, vault)
    _chart(allr[[best, "SPY", "60/40", "VT"]].tail(252), best, vault, "portfolio_12m.png", "last 12 months")
    recent = horizons(allr, rf)
    for lab, mm in recent.items():
        mm.to_csv(out / f"metrics_{lab}.csv")
    win = rets_all.loc[:t, w.index].tail(pcfg["lookback_days"])
    viol = violations(w, uni, pcfg_t, ctx, cov=win.cov() * 252, R=win.fillna(0))
    (out / "context.json").write_text(json.dumps({"date": str(t.date()), **{k: v for k, v in ctx.items() if k not in ("macro", "mcfg", "mu_ml")},
                                                  "violations": viol, "factor_weights": fw_now}, default=str, indent=1))
    _note(vault, metrics, best, target, avg_corr, cap, pcfg_t, t, ctx, viol, recent, mc_table)
    if extra:
        _factor_note(vault, extra["ic"], fw_now, t)
    return {"method": best, "positions": len(target), "invested": round(target.value_eur.sum(), 2),
            "sharpe": round(metrics.loc[best, "Sharpe"], 2), "maxdd": round(metrics.loc[best, "MaxDrawdown"], 3)}

def save_diagnostics(out, extra, hist, uni):
    """Intermediate backtest results for analysis (notebook 03_portfolio)."""
    rows = [{"date": t, "family": g, "factor": f, "weight": v}
            for t, fw in (extra.get("factor_weights") or {}).items() for g, d in fw.items() for f, v in d.items()]
    if rows:
        pd.DataFrame(rows).to_csv(out / "factor_weights_history.csv", index=False)
    ctx = pd.DataFrame({t: {k: v for k, v in c.items() if not isinstance(v, (dict, pd.Series))}
                        for t, c in (extra.get("context") or {}).items()}).T
    if len(ctx):
        ctx.to_csv(out / "context_history.csv")
    oos = []
    if extra.get("booster") is not None:
        oos += [{**r, "model": "lightgbm_selection"} for r in extra["booster"].oos]
    if extra.get("ml_returns") is not None:
        oos += [{**r, "model": "lightgbm_returns"} for r in extra["ml_returns"].oos]
    if oos:
        pd.DataFrame(oos).to_csv(out / "ml_oos_ic.csv", index=False)
    cls = {m: pd.DataFrame({t: w.groupby(uni.loc[w.index, "asset_class"]).sum() for t, w in h.items()}).T.fillna(0)
           for m, h in hist.items()}
    pd.concat(cls, names=["method", "date"]).to_csv(out / "class_weights_history.csv")

def mc_select(w_all, score, px_eur, macro, cfg, pcfg):
    """Simulates the portfolio of every method (1 year) and keeps only those within the risk limits.
    Among those, picks the best in the backtest; if none passes, the one with the smallest tail loss."""
    from agents.modeling_agent.cflib_bridge import load_cflib
    from agents.modeling_agent.calibrate import calibrate, exposures
    from agents.modeling_agent.simulate import simulate
    from agents.modeling_agent.agent import summarize
    sel = pcfg["mc_selection"]; mcfg = cfg["modeling"]
    CIR, QT, jmp, _ = load_cflib(mcfg["cflib_path"])
    cal = calibrate(px_eur, macro, mcfg)
    rows = {}
    for m, w in w_all.items():
        w = w / w.sum()
        bp, idio, _, _ = exposures(px_eur, macro, w, mcfg)
        sim, _ = simulate(cal, bp, idio, mcfg, 1, CIR, QT, jmp, sel["n_paths"], mcfg["batch"], mcfg["seed"])
        r = summarize(sim, 1, pcfg["capital"])
        rows[m] = {"backtest_score": float(score.get(m, np.nan)), "mc_median": r["median"], "mc_CVaR95": r["CVaR95"],
                   "mc_prob_dd20": r["prob_dd20"], "mc_prob_loss": r["prob_loss"]}
    tab = pd.DataFrame(rows).T
    tab["passes"] = (tab.mc_CVaR95 >= sel["max_cvar95_1y"]) & (tab.mc_prob_dd20 <= sel["max_prob_dd20_1y"])
    ok = tab[tab.passes]
    best = ok.backtest_score.idxmax() if len(ok) else tab.mc_CVaR95.idxmax()
    tab["chosen"] = tab.index == best
    return best, tab

HORIZONS = {"1y": 252, "2y": 504, "5y": 1260}

def horizons(allr, rf):
    """Metrics over the last 1, 2, 5 years, year to date and the full backtest."""
    out = {}
    wins = {k: allr.tail(n) for k, n in HORIZONS.items() if len(allr) >= n}
    wins["ytd"] = allr.loc[f"{allr.index[-1].year}-01-01":]
    if allr.index[0].year < 2014:
        wins["since 2014"] = allr.loc["2014-01-01":]
    wins[f"since {allr.index[0].year}"] = allr
    for lab, x in wins.items():
        mm = pd.DataFrame({c: perf(x[c], rf) for c in x}).T
        mm["Return"] = (1 + x[mm.index]).prod() - 1
        out[lab] = mm
    return out

def _chart(r, best, vault, fname="portfolio_backtest.png", label=None):
    eq = (1 + r.fillna(0)).cumprod() * 100
    dd = eq / eq.cummax() - 1
    # end labels: push apart labels whose end values are within ~6% of each other
    offsets, prev = {}, None
    for c, v in eq.iloc[-1].sort_values().items():
        offsets[c] = 0 if prev is None or v / prev[1] > 1.06 else offsets[prev[0]] + 11
        prev = (c, v)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    for i, c in enumerate(eq.columns):
        lw = 2.4 if c == best else 1.6
        a1.plot(eq.index, eq[c], color=COLORS[i], lw=lw, label=c)
        a1.annotate(f"{c} {eq[c].iloc[-1]:.0f}", (eq.index[-1], eq[c].iloc[-1]), xytext=(6, offsets[c]),
                    textcoords="offset points", va="center", fontsize=9, color="#0b0b0b")
        a2.plot(dd.index, dd[c] * 100, color=COLORS[i], lw=1.4)
    if label is None:
        a1.set_yscale("log"); a1.set_ylabel("Value (base 100, EUR, log scale)")
    else:
        a1.set_ylabel("Value (base 100, EUR)")
    a2.set_ylabel("Drawdown %")
    a1.legend(loc="upper left", frameon=False)
    a1.set_title(f"Backtest walk-forward{' ' + label if label else ''}: {best} vs benchmarks", loc="left", fontsize=12)
    for a in (a1, a2):
        a.grid(axis="y", color="#e5e4e0", lw=0.8); a.spines[["top", "right"]].set_visible(False)
        a.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb"); fig.tight_layout()
    (vault / "attachments").mkdir(parents=True, exist_ok=True)
    fig.savefig(vault / "attachments" / fname, dpi=130); plt.close(fig)

def _pct(x): return "" if pd.isna(x) else f"{x:.1%}"

def horizon_table(recent, best):
    labs = list(recent)
    keys = [k for k in [best, "max_sharpe", "min_cvar", "SPY", "VT", "60/40"] if k in recent[labs[0]].index]
    head = "| Strategy | " + " | ".join(labs) + " |\n|---|" + "---|" * len(labs)
    rows = []
    for k in keys:
        cells = []
        for lab in labs:
            r = recent[lab].loc[k]
            ann = f" ({r['CAGR']:+.1%}/a)" if lab not in ("1y", "ytd") else ""
            cells.append(f"{r['Return']:+.1%}{ann} · {r['Sharpe']:.2f} · {r['MaxDrawdown']:.0%}")
        rows.append(f"| {k}{' ⭐' if k == best else ''} | " + " | ".join(cells) + " |")
    return head + "\n" + "\n".join(rows)

def _note(vault, m, best, tg, avg_corr, cap, pcfg, t, ctx, viol, recent, mc_table=None):
    cols = ["CAGR", "Volatility", "Sharpe", "Sortino", "MaxDrawdown", "Calmar", "WorstMonth", "Turnover_yr"]
    rows = []
    for name, r in m.iterrows():
        mark = " ⭐" if name == best else ""
        rows.append(f"| {name}{mark} | " + " | ".join(
            f"{r[c]:.2f}" if c in ("Sharpe", "Sortino", "Calmar", "Turnover_yr") else _pct(r[c]) for c in cols) + " |")
    alloc = tg.groupby("asset_class").weight_real.sum().sort_values(ascending=False)
    reg = tg.groupby("region").weight_real.sum().sort_values(ascending=False)
    pos = "\n".join(f"| [[{i.replace('^', '_')}]] | {r['name'][:28] if r['name'] else ''} | {r.asset_class} | {r.region} | "
                    f"{r.weight_real:.1%} | {r.value_eur:.0f} | {r.quantity:.3f} | {r.currency} | {r.news_adj:+.2f} |"
                    for i, r in tg.iterrows())
    body = f"""# Proposed portfolio {pd.Timestamp(t).date()}

Profile: **{pcfg.get('risk_profile', 'balanced')}** · Method: **{best}** (best {pcfg['choose_by']} in the backtest among those passing the simulations) · Capital: {cap:,.0f} EUR · Invested: {tg.value_eur.sum():,.0f} EUR
Positions: {len(tg)} · Average pairwise correlation (3 years): **{avg_corr:+.2f}**

> Proposal to be validated by the controller agent. No orders are ever placed.

Risk mode: **{ {'offense': 'OFFENSE', 'defense': 'DEFENSE'}.get(ctx.get('mode'), 'n/a') }** (automatic switching {'on' if pcfg.get('risk_mode', {}).get('enabled') else 'off'}) · signals: {ctx.get('signals', {})}
Market regime (HMM): **{ctx.get('regime', 'n/a')}** {({k: round(v, 2) for k, v in ctx.get('regime_probs', {}).items()})}
Safety rules today: AI bubble **{'ON' if ctx['bubble'] else 'no'}** {ctx['bubble_info']} · high volatility (momentum reduced) **{'yes' if ctx['high_vol'] else 'no'}**
Crash-hedge share: **{tg.index.isin(pcfg['hedge']['tickers']).astype(float) @ tg.weight_real:.1%}** · AI assets: **{tg.index.isin(pcfg['ai_tickers']).astype(float) @ tg.weight_real:.1%}** · Constraint breaches: {viol or 'none'}

## Method selection with simulations (1 year)
Limits: CVaR 95% not worse than {pcfg['mc_selection']['max_cvar95_1y']:.0%}, P(drawdown > 20%) at most {pcfg['mc_selection']['max_prob_dd20_1y']:.0%}.

| Method | Backtest score | MC median | MC CVaR 95% | P(DD>20%) | Outcome |
|---|---|---|---|---|---|
""" + ("\n".join(f"| {k}{' ⭐' if r.chosen else ''} | {r.backtest_score:.2f} | {r.mc_median:+.1%} | {r.mc_CVaR95:+.1%} | {r.mc_prob_dd20:.1%} | {'pass' if r.passes else 'rejected'} |"
               for k, r in mc_table.iterrows()) if mc_table is not None else "") + f"""

## Positions
| Asset | Name | Class | Region | Weight | EUR | Quantity | Currency | News |
|---|---|---|---|---|---|---|---|---|
{pos}

## By asset class
""" + "\n".join(f"- {k}: {v:.1%}" for k, v in alloc.items()) + "\n\n## By region\n" + \
        "\n".join(f"- {k}: {v:.1%}" for k, v in reg.items()) + f"""

## Walk-forward backtest since {pcfg['backtest_start']} (EUR, net of costs)
| Method | CAGR | Volatility | Sharpe | Sortino | Max DD | Calmar | Worst month | Turnover/year |
|---|---|---|---|---|---|---|---|---|
""" + "\n".join(rows) + """

![[portfolio_backtest.png]]

## Results by horizon
Cumulative return (annualised in brackets beyond 1 year) · Sharpe · max drawdown.

""" + horizon_table(recent, best) + """

![[portfolio_12m.png]]

## Known limitations
- The S&P 500 list is today's constituents (survivorship bias): the backtest is optimistic on US equities.
- The news signal only affects today's portfolio: there is no news history for the backtest.
"""
    write_note(vault, "Reports", f"portfolio_{pd.Timestamp(t).date()}", {"tags": "[report, portfolio_agent]",
               "method": best}, body)


def _factor_note(vault, ic, fw_now, t):
    s = ic.groupby(["family", "factor"]).ic.agg(["mean", "std", "count"])
    s["ICIR"] = s["mean"] / s["std"]
    lines = [f"# Selection factors {pd.Timestamp(t).date()}\n",
             "IC = rank correlation between the factor at month end and the return over the following 3 months.",
             "Positive = the factor predicted well. Today's weights only use the last 5 years.\n"]
    for fam in ("equity", "other"):
        lines += [f"\n## {fam}\n| Factor | Mean IC | ICIR | Weight today |\n|---|---|---|---|"]
        for fac, r in s.loc[fam].sort_values("mean", ascending=False).iterrows():
            lines.append(f"| {fac} | {r['mean']:+.3f} | {r.ICIR:+.2f} | {fw_now.get(fam, {}).get(fac, 0):.0%} |")
    write_note(vault, "Reports", "selection_factors", {"tags": "[report, factors]"}, "\n".join(lines))
