"""Walk-forward backtest: at every rebalance, selection and optimisation use past data only.
Safety rules: minimum allocation to crash-hedge ETFs, cap on AI assets (tighter during a bubble),
volatility cap, risk budget per class, CVaR cap, reduced momentum when volatility is high."""
import logging
import numpy as np
import pandas as pd
from .selection import factor_scores, pick_by_class, estimate
from .optimizers import allocate, set_risk, min_cvar, cap_cvar, cvar
from . import factors as F
from . import regime as RG
from . import risk_mode as RM
from .ml import Booster
from . import ml_returns as MR
import json as _json
import pickle, hashlib

log = logging.getLogger("portfolio_agent")
EQUITY = ("equity", "equity_etf")

def _idx(tick, cond):
    return [i for i, t in enumerate(tick) if cond(t)]

def constraints(meta: pd.DataFrame, pcfg: dict, ctx: dict = None):
    """Linear constraints (caps) and risk constraints for the selected assets."""
    ctx = ctx or {}
    tick = list(meta.index)
    groups = []
    for cls, cap in pcfg["class_caps"].items():
        idx = _idx(tick, lambda t: meta.loc[t, "asset_class"] == cls)
        if idx: groups.append((idx, cap))
    eq = _idx(tick, lambda t: meta.loc[t, "asset_class"] in EQUITY)
    if eq: groups.append((eq, pcfg["equity_total_cap"]))
    for reg, cap in pcfg["region_caps"].items():
        idx = _idx(tick, lambda t: meta.loc[t, "region"] == reg)
        if idx: groups.append((idx, cap))
    ai = set(pcfg.get("ai_tickers", []))
    ai_idx = _idx(tick, lambda t: t in ai)
    if ai_idx:
        groups.append((ai_idx, pcfg["ai_cap_bubble"] if ctx.get("bubble") else pcfg["ai_cap"]))
    floors = []
    hedge = set(pcfg["hedge"]["tickers"])
    h_idx = _idx(tick, lambda t: t in hedge)
    if h_idx:
        lo = pcfg["hedge"]["min_total_bubble" if ctx.get("bubble") else "min_total"]
        floors.append((h_idx, min(lo, len(h_idx) * pcfg["max_weight"])))
    rk = pcfg.get("risk", {})
    budgets = []
    for key, b in rk.get("risk_budget", {}).items():
        idx = eq if key == "equity_all" else _idx(tick, lambda t: meta.loc[t, "asset_class"] == key)
        if idx: budgets.append((idx, b))
    return [(0, pcfg["max_weight"])] * len(tick), groups, floors, budgets

def eligible(px_local, vol, uni, t, pcfg):
    """Investable assets at date t: enough history, liquid, not excluded."""
    lb = pcfg["lookback_days"]
    win = px_local.loc[:t].tail(lb)
    ok = (win.notna().mean() > 0.9) & win.iloc[-1].notna()
    dv = (px_local.loc[:t].tail(60) * vol.loc[:t].tail(60)).median()
    liquid = (dv >= pcfg["min_dollar_volume"]) | uni["asset_class"].reindex(px_local.columns).eq("fx")
    allowed = uni["tradable"] & ~uni["group"].isin(pcfg["exclude_groups"])
    if pcfg.get("single_stocks_require_fundamentals"):
        allowed &= (uni["asset_class"] != "equity") | uni.get("has_fundamentals", False)
    return ok[ok & liquid.reindex(ok.index).fillna(False) & allowed.reindex(ok.index).fillna(False)].index

def bubble_signal(px_eur, t, pcfg):
    """AI bubble: the semiconductor ETF is far above its 200-day average
    or its 12-month return is extreme relative to its own history (z-score)."""
    b = pcfg["bubble"]; s = px_eur[b["etf"]].loc[:t].dropna()
    if len(s) < 252 * 3:
        return False, {}
    gap = s.iloc[-1] / s.tail(200).mean() - 1
    r12 = s.pct_change(252).dropna()
    z = (r12.iloc[-1] - r12.tail(252 * 5).mean()) / r12.tail(252 * 5).std()
    on = bool(gap > b["ma_gap"] or z > b["z_threshold"])
    return on, {"gap_ma200": round(float(gap), 3), "z_12m": round(float(z), 2)}

def build_weights(px_eur, rets, uni, cols, pcfg, methods, t, score_adj=None, base_scores=None, ctx=None):
    """Selects candidates and optimises with every method. Returns {method: pd.Series of weights}."""
    ctx = ctx or {}
    win = rets.loc[:t, cols].tail(pcfg["lookback_days"])
    if base_scores is not None:
        scores = base_scores.reindex(cols).fillna(base_scores.min())
    else:
        scores = factor_scores(px_eur.loc[:t, cols].ffill())
    if score_adj is not None:
        scores = scores + score_adj.reindex(scores.index).fillna(0)
    chosen = pick_by_class(win, scores, uni.loc[cols], pcfg)
    # crash-hedge ETFs: the best ones (by score) of each asset class are always included
    hedge = [h for h in pcfg["hedge"]["tickers"] if h in cols]
    if hedge:
        hs = pd.DataFrame({"score": scores.reindex(hedge), "cls": uni.loc[hedge, "asset_class"]})
        extra = hs.sort_values("score", ascending=False).groupby("cls").head(pcfg["hedge"].get("per_class", 2))
        chosen = pd.concat([chosen, extra[["score"]].assign(cluster="hedge")])
        chosen = chosen[~chosen.index.duplicated()]
    sel = list(chosen.index)
    mu, cov = estimate(win[sel], pcfg.get("covariance", "lw"))
    if ctx.get("mu_ml") is not None:      # LightGBM expected returns, blended with historical ones
        lam = pcfg["ml_returns"].get("blend", 0.7)
        mml = ctx["mu_ml"].reindex(sel)
        if MR.OPTIONS.get("demean"):           # relative forecast: add back the historical average level
            mml = mml + mu.mean()
        mu = np.where(mml.notna(), lam * mml.fillna(0).values + (1 - lam) * mu, mu)
    rk = dict(pcfg.get("risk", {}))
    mcs = pcfg.get("mc_scenarios", {})
    if mcs.get("enabled") and ctx.get("macro") is not None:
        # Monte Carlo scenarios (Heston) over `weeks` weeks: drive the CVaR cap and min_cvar
        from agents.modeling_agent.scenarios import asset_scenarios
        R = asset_scenarios(px_eur, ctx["macro"], ctx["mcfg"], sel, t, mcs.get("n", 10000), mcs.get("weeks", 13))
        rk["cvar_cap_daily"] = rk.get("cvar_cap_mc")
    else:
        R = win[sel].fillna(0).values
    thresh = pcfg["min_position_value"] / pcfg["capital"]
    out = {}
    for m in methods:
        w = _optimize(m, sel, mu, cov, R, uni.loc[sel], pcfg, ctx, rk)
        keep = w[w >= thresh].index
        if 0 < len(keep) < len(sel):   # drop positions that are too small and re-optimise
            i = [sel.index(k) for k in keep]
            w = _optimize(m, list(keep), mu[i], cov[np.ix_(i, i)], R[:, i], uni.loc[keep], pcfg, ctx, rk)
        w = w[w > 1e-6]; w = w / w.sum()
        bad = violations(w, uni, {**pcfg, "risk": rk}, ctx, cov=pd.DataFrame(cov, sel, sel), R=pd.DataFrame(R, columns=sel))
        if bad:
            log.debug("%s %s: constraints not fully met %s", pd.Timestamp(t).date(), m, bad)
        out[m] = add_core(w, pcfg, px_eur, t)
    return out, chosen

def add_core(w, pcfg, px_eur, t):
    """Core-satellite: a fixed share in a global equity ETF, the strategy manages the rest."""
    core = pcfg.get("core") or {}
    cw = core.get("weight", 0)
    if cw <= 0:
        return w
    tick = next((c for c in core["tickers"] if c in px_eur and pd.notna(px_eur[c].loc[:t].iloc[-1])), None)
    if tick is None:
        return w
    w = w * (1 - cw)
    w[tick] = w.get(tick, 0) + cw
    return w

def _optimize(m, sel, mu, cov, R, meta, pcfg, ctx, rk):
    bounds, groups, floors, budgets = constraints(meta, pcfg, ctx)
    set_risk(cov, floors, rk.get("vol_cap"), budgets)
    w = min_cvar(R, bounds, groups) if m == "min_cvar" else allocate(m, mu, cov, bounds, groups,
                                                                       pcfg.get("ml_returns", {}).get("gamma", 3.0))
    w = cap_cvar(w, R, bounds, groups, rk.get("cvar_cap_daily"))
    return pd.Series(w, index=sel)

def violations(w, uni, pcfg, ctx=None, tol=0.005, cov=None, R=None):
    """Returns the limits that are breached (also used by the controller agent)."""
    ctx = ctx or {}
    cls = w.groupby(uni.loc[w.index, "asset_class"]).sum()
    reg = w.groupby(uni.loc[w.index, "region"]).sum()
    bad = {f"class:{k}": round(v, 3) for k, v in cls.items() if v > pcfg["class_caps"].get(k, 1) + tol}
    bad.update({f"region:{k}": round(v, 3) for k, v in reg.items() if v > pcfg["region_caps"].get(k, 1) + tol})
    eq = sum(cls.get(c, 0) for c in EQUITY)
    if eq > pcfg["equity_total_cap"] + tol: bad["equity_total"] = round(eq, 3)
    if w.max() > pcfg["max_weight"] + tol: bad["max_weight"] = round(w.max(), 3)
    ai = w[w.index.isin(pcfg.get("ai_tickers", []))].sum()
    ai_cap = pcfg["ai_cap_bubble"] if ctx.get("bubble") else pcfg["ai_cap"]
    if ai > ai_cap + tol: bad["ai_cap"] = round(ai, 3)
    h = w[w.index.isin(pcfg["hedge"]["tickers"])].sum()
    h_min = pcfg["hedge"]["min_total_bubble" if ctx.get("bubble") else "min_total"]
    if h < h_min - tol: bad["hedge_min"] = round(h, 3)
    rk = pcfg.get("risk", {})
    if cov is not None:
        c = cov.loc[w.index, w.index].values; vol = float(np.sqrt(w.values @ c @ w.values))
        if rk.get("vol_cap") and vol > rk["vol_cap"] + tol: bad["vol"] = round(vol, 3)
    if R is not None and rk.get("cvar_cap_daily"):
        cv = cvar(w.values, R[w.index].values)
        if cv < rk["cvar_cap_daily"] - 0.001: bad["cvar"] = round(cv, 4)
    return bad

def apply_band(target, prev, band):
    """No-trade band: small weight changes do not generate orders."""
    if band <= 0:
        return target[target > 0]
    w = target.where((target - prev).abs() >= band, prev)
    w = w.where(target > 0, 0)            # exits decided by the selection are always executed
    w = w[w > 0]
    return w / w.sum()

def rebalance_dates(idx, start, freq):
    rule = "ME" if freq == "monthly" else "QE"
    d = pd.Series(idx, index=idx).loc[start:].resample(rule).last().dropna()
    return list(d.values)

def factor_model(px_eur, px_local, uni, fund_df, cache_dir=None):
    """Factor panel with on-disk cache (recomputed only when prices, universe or fundamentals change)."""
    fund = F.Fundamentals(fund_df) if fund_df is not None else None
    key = hashlib.md5(f"v2|{px_eur.index[-1]}|{len(px_eur.columns)}|{len(fund_df) if fund_df is not None else 0}|"
                      f"{','.join(F.FACTORS)}".encode()).hexdigest()
    path = cache_dir / f"panel_{key}.pkl" if cache_dir else None
    if path and path.exists():
        feats, fwd, fam, ic, raw = pickle.loads(path.read_bytes())
        log.info("Factor panel loaded from cache")
    else:
        feats, fwd, fam = F.build_panel(px_eur, uni, fund=fund, px_local=px_local)
        raw = F.build_panel.raw
        ic = F.ic_table(feats, fwd, fam)
        if path:
            for old in cache_dir.glob("panel_*.pkl"): old.unlink()
            path.write_bytes(pickle.dumps((feats, fwd, fam, ic, raw)))
    factor_model.raw = raw
    return feats, ic, fam, fund, fwd

def regime_profile(pcfg, regime):
    """Overrides limits and rules with those of the active profile (one-level merge for dictionaries).
    With risk_mode the profiles are 'offense' and 'defense'; otherwise the HMM regime profiles."""
    if regime in ("offense", "defense") and "risk_mode" in pcfg:
        prof = pcfg["risk_mode"].get(regime, {}) or {}
    elif pcfg.get("regimes", {}).get("enabled"):
        prof = pcfg["regimes"].get(regime, {}) or {}
    else:
        return pcfg
    out = dict(pcfg)
    for k, v in prof.items():
        out[k] = {**pcfg.get(k, {}), **v} if isinstance(v, dict) and isinstance(pcfg.get(k), dict) else v
    return out

def date_context(px_eur, macro, t, pcfg, regime=None):
    bub, info = bubble_signal(px_eur, t, pcfg)
    hv = F.high_vol(macro, t, pcfg["momentum_dampening"]["vix"])
    ctx = {"bubble": bub, "bubble_info": info, "high_vol": hv}
    if regime is not None:
        ctx["regime"], ctx["regime_probs"] = regime
    return ctx

def momentum_scale(pcfg_t, ctx):
    s = pcfg_t.get("momentum_scale", 1.0)
    if ctx.get("high_vol"):
        s = min(s, pcfg_t["momentum_dampening"]["scale"])
    return s

def schedule(idx, pcfg, regimes, modes=None):
    """Regular rebalances + dates when the risk mode (or the crisis regime) changes."""
    dates = [pd.Timestamp(d) for d in rebalance_dates(idx, pcfg["backtest_start"], pcfg["rebalance"])]
    if modes is not None and pcfg["risk_mode"].get("rebalance_on_switch"):
        sw = modes[modes["mode"] != modes["mode"].shift()].index[1:]
        dates += [pd.Timestamp(d) for d in sw if d >= dates[0]]
        return sorted(set(dates))
    if regimes and pcfg.get("regimes", {}).get("rebalance_on_switch"):
        prev = None
        for t in sorted(regimes):
            r = regimes[t][0]
            if prev is not None and (r == "crisis") != (prev == "crisis") and t >= dates[0]:
                dates.append(t)
            prev = r
    return sorted(set(dates))

def run_backtest(px_eur, px_local, vol, uni, pcfg, rf, macro=None, fund_df=None, cache_dir=None, mcfg=None):
    rets = px_eur.pct_change(fill_method=None)
    methods = pcfg["methods"]
    cap = pcfg["capital"]
    band, bonus = pcfg.get("no_trade_band", 0.0), pcfg.get("hold_bonus", 0.0)
    adaptive = pcfg.get("selection_model", "adaptive") == "adaptive"
    feats = ic = fam = fwd = None
    if adaptive:
        feats, ic, fam, _, fwd = factor_model(px_eur, px_local, uni, fund_df, cache_dir)
    use_mode = pcfg.get("risk_mode", {}).get("enabled") and macro is not None
    fixed_offense = "risk_mode" in pcfg and not use_mode
    modes = RM.mode_history(px_local, px_eur, macro, pcfg["risk_mode"], pcfg["backtest_start"]) if use_mode else None
    use_reg = (not use_mode) and (not fixed_offense) and pcfg.get("regimes", {}).get("enabled") and macro is not None
    regimes = {}
    if use_reg:
        feat_r = RG.features(macro, px_eur)
        ends = pd.Series(rets.index, index=rets.index).loc[pcfg["backtest_start"]:].resample("ME").last().dropna()
        regimes = RG.monthly_regimes(feat_r, [pd.Timestamp(x) for x in ends.values])
    mlc = pcfg.get("ml", {})
    booster = Booster(feats, fwd, fam, uni) if adaptive and mlc.get("enabled") else None
    mret = ml_returns_model(pcfg, feats, fwd, fam, uni, px_eur, macro, cache_dir) if adaptive and macro is not None else None
    dates = schedule(rets.index, pcfg, regimes, modes)
    if pcfg.get("backtest_end"):
        dates = [d for d in dates if d <= pd.Timestamp(pcfg["backtest_end"])]
    daily = {m: pd.Series(0.0, index=rets.index) for m in methods}
    hist = {m: {} for m in methods}
    held = {m: pd.Series(dtype=float) for m in methods}
    turnover = {m: [] for m in methods}
    fw_hist, ctx_hist = {}, {}
    for k, t in enumerate(dates):
        cols = eligible(px_local, vol, uni, t, pcfg)
        end = dates[k + 1] if k + 1 < len(dates) else rets.index[-1]
        period = rets.loc[t:end].iloc[1:]
        reg = regimes.get(t) if use_reg else None
        ctx = date_context(px_eur, macro, t, pcfg, reg) if macro is not None else {}
        ctx_hist[t] = dict(ctx)
        ctx.update(macro=macro, mcfg=mcfg)
        if use_mode:
            row = modes.loc[:t].iloc[-1] if len(modes.loc[:t]) else None
            ctx["mode"] = row["mode"] if row is not None else "offense"
            ctx["n_signals"] = int(row["n_signals"]) if row is not None else 0
        if fixed_offense:
            ctx["mode"] = "offense"
        pcfg_t = regime_profile(pcfg, ctx.get("mode", ctx.get("regime", "normal")))
        ctx_hist[t].update(mode=ctx.get("mode"), n_signals=ctx.get("n_signals"))
        base = None
        ft = feats.get(t) if adaptive else None
        if ft is None and adaptive:                          # date not at month end: use the latest available panel
            prev_d = [d for d in feats if d <= t]
            ft = feats[prev_d[-1]] if prev_d else None
        if adaptive and ft is not None:
            fw = F.weights_at(ic, t)
            sc = momentum_scale(pcfg_t, ctx)
            if sc < 1:
                fw = F.dampen_momentum(fw, sc)
            fw_hist[t] = fw
            base = F.scores_at(ft, fam, fw)
            if booster is not None:
                a = pcfg_t.get("ml", mlc).get("tilt", 0.5)
                base = (1 - a) * base + a * booster.predict(t, ft).reindex(base.index).fillna(0)
        if mret is not None and ft is not None:
            ctx["mu_ml"] = predict_mu(mret, t, feats, px_eur, macro)
        for m in methods:
            # holding bonus: an asset already held only leaves if it really deteriorates
            adj = pd.Series(bonus, index=held[m].index) if bonus and len(held[m]) else None
            w = build_weights(px_eur, rets, uni, cols, pcfg_t, [m], t, adj, base, ctx)[0][m]
            prev = held[m].reindex(w.index.union(held[m].index)).fillna(0)
            w = apply_band(w.reindex(prev.index).fillna(0), prev, band)
            to = (w - prev).abs().sum()
            trades = int(((w - prev).abs() > 1e-4).sum())
            cost = to * pcfg["cost_bps"] / 1e4 + trades * pcfg["commission_per_trade"] / cap
            turnover[m].append(to); hist[m][t] = w
            R = period[w.index].fillna(0).values
            if len(R):
                # weights drift with the market: position value = initial weight x cumulative growth
                growth = np.cumprod(1 + R, axis=0)
                val = growth * w.values
                tot = val.sum(axis=1)
                pr = np.diff(np.concatenate([[1.0], tot])) / np.concatenate([[1.0], tot[:-1]])
                pr[0] -= cost
                daily[m].loc[period.index] = pr
                held[m] = pd.Series(val[-1] / tot[-1], index=w.index)
            else:
                held[m] = w
        if k % 8 == 0:
            log.info("Backtest %s (%d/%d) mode=%s regime=%s bubble=%s high_vol=%s", t.date(), k + 1, len(dates),
                     ctx.get("mode"), ctx.get("regime"), ctx.get("bubble"), ctx.get("high_vol"))
    first = rets.loc[dates[0]:].index[1]
    daily = pd.DataFrame(daily).loc[first:pcfg.get("backtest_end")]
    to_year = {m: np.sum(v) / (len(daily) / 252) for m, v in turnover.items()}
    extra = {"context": ctx_hist, "regimes": regimes, "booster": booster, "modes": modes, "ml_returns": mret}
    if adaptive:
        extra.update(ic=ic, factor_weights=fw_hist, feats=feats, fam=fam)
    return daily, hist, to_year, extra


def ml_returns_model(pcfg, feats, fwd, fam, uni, px_eur, macro, cache_dir):
    """LightGBM expected-return model with the parameters chosen by time-series validation."""
    mc = pcfg.get("ml_returns", {})
    if not mc.get("enabled"):
        return None
    params = mc.get("params", "auto")
    if params == "auto":
        f = cache_dir.parent / "experiments" / "ml_params.json" if cache_dir else None
        params = _json.loads(f.read_text()) if f and f.exists() else \
            {"num_leaves": 15, "min_child_samples": 100, "n_estimators": 150, "learning_rate": 0.05,
             "colsample_bytree": 0.7, "subsample": 0.7, "reg_lambda": 5.0,
             "_options": {"use_macro": False, "demean": True}}
    params = dict(params)
    MR.OPTIONS.update(params.pop("_options", {}))
    params.pop("_validation", None)
    raw = getattr(factor_model, "raw", None)
    ctxs = {t: MR.macro_context(px_eur, macro, t) for t in feats}
    return MR.MLReturns(feats, raw, fwd, fam, uni, ctxs, params, mc.get("train_years", 10))

def predict_mu(mret, t, feats, px_eur, macro):
    d = max(x for x in feats if x <= t)
    if d not in mret.ctx:
        mret.ctx[d] = MR.macro_context(px_eur, macro, d)
    return mret.predict(t, MR.frame(d, mret.feats, mret.raw, mret.fam, mret.uni, mret.ctx))
