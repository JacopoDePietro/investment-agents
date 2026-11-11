"""Data agent: builds the universe, downloads, validates, stores Parquet files and documents in Obsidian."""
import logging
import numpy as np
import pandas as pd
from core.config import ROOT
from core.obsidian import write_note
from core.currency import currency_of, fx_tickers
from .sources import fetch_prices, fetch_fred
from .quality import check_prices
from .universe import build_universe
from . import fundamentals

log = logging.getLogger("data_agent")

def run(cfg):
    d, vault = cfg["paths"]["data"], cfg["paths"]["vault"]
    (d / "processed").mkdir(parents=True, exist_ok=True)
    uni = build_universe(ROOT, d)
    start = cfg["history_start"]

    log.info("Universe: %d assets", len(uni))
    close, volume = fetch_prices(list(uni.index), start, cfg.get("download_batch", 100))
    missing = sorted(set(uni.index) - set(close.columns))
    if missing:
        log.warning("%d tickers without data: %s", len(missing), missing)

    quality = check_prices(close)
    years = {t: (close[t].dropna().index[-1] - close[t].dropna().index[0]).days / 365 for t in close}
    uni["has_data"] = uni.index.isin(close.columns)
    uni["history_years"] = pd.Series(years).round(1)
    uni["eligible"] = uni["history_years"].fillna(0) >= cfg.get("min_history_years", 3)
    uni["currency"] = [currency_of(t) for t in uni.index]
    uni["tradable"] = uni["currency"] != "NONE"

    base = cfg.get("base_currency", "EUR")
    fxmap = fx_tickers(uni.loc[uni.has_data, "currency"], base)
    fx_close, _ = fetch_prices(sorted(set(fxmap.values())), start)
    fx = pd.DataFrame({("GBP" if c == "GBp" else c): fx_close[t] for c, t in fxmap.items() if t in fx_close})
    fx.to_parquet(d / "processed" / "fx.parquet")
    log.info("%s FX rates downloaded: %s", base, sorted(fx.columns))

    close.to_parquet(d / "processed" / "prices.parquet")
    volume.to_parquet(d / "processed" / "volume.parquet")
    np.log(close).diff().dropna(how="all").to_parquet(d / "processed" / "log_returns.parquet")
    uni.to_parquet(d / "processed" / "universe.parquet")

    macro, failed = {}, []
    for sid in cfg["fred_series"]:
        try:
            macro[sid] = fetch_fred(sid, start)
        except Exception as e:
            log.warning("FRED %s failed: %s", sid, e); failed.append(sid)
    if macro:
        pd.DataFrame(macro).to_parquet(d / "processed" / "macro.parquet")

    try:
        fcfg = cfg.get("fundamentals", {})
        fundamentals.update(d, uni, fcfg.get("sec_user_agent", "investment-agents research tool"),
                            fcfg.get("refresh_days", 7))
    except Exception as e:
        log.warning("SEC fundamentals not updated: %s", e)

    _write_vault(vault, close, uni, quality, macro, cfg, failed, missing)
    log.info("Data agent completed")
    return {"universe": len(uni), "with_data": len(close.columns),
            "eligible": int(uni["eligible"].sum()), "missing": len(missing), "macro_failed": failed}

def _tag(s):
    return str(s).lower().replace(" ", "_")

def _write_vault(vault, close, uni, quality, macro, cfg, failed, missing):
    # Correlations on weekly returns (aligns markets with different trading hours)
    wk = np.log(close.resample("W-FRI").last()).diff().tail(156)
    corr = wk.corr(min_periods=52)
    daily = np.log(close).diff().tail(252)
    for t in close.columns:
        s, q, m = close[t].dropna(), quality[t], uni.loc[t]
        r = daily[t].dropna()
        vol = r.std() * np.sqrt(252) if len(r) > 20 else float("nan")
        ret1y = s.iloc[-1] / s.iloc[-min(252, len(s))] - 1
        c = corr[t].drop(t).dropna().sort_values()
        tags = f"[asset, {_tag(m.asset_class)}, {_tag(m.region)}" + (f", {_tag(m.sector)}" if m.sector else "") + "]"
        body = (f"# {t} {m['name']}\n\n"
                f"Group: {m.group} · Asset class: {m.asset_class} · Region: {m.region}"
                + (f" · Sector: {m.sector}" if m.sector else "") + "\n\n"
                f"| Metric | Value |\n|---|---|\n"
                f"| Last price | {s.iloc[-1]:.2f} |\n| 1-year return | {ret1y:.1%} |\n"
                f"| 1-year volatility | {vol:.1%} |\n| Years of history | {m.history_years} |\n"
                f"| Missing data | {q['missing_pct']}% |\n\n"
                "## Diversifiers: lowest correlation (3 years, weekly)\n"
                + "\n".join(f"- [[{k}]] {v:+.2f}" for k, v in c.head(5).items())
                + "\n\n## Most correlated\n"
                + "\n".join(f"- [[{k}]] {v:+.2f}" for k, v in c.tail(5).items()))
        write_note(vault, "Assets", t.replace("^", "_"),
                   {"tags": tags, "asset_class": m.asset_class, "region": m.region,
                    "eligible": bool(m.eligible), "stale": q["stale"]}, body)

    for sid, s in macro.items():
        write_note(vault, "Macro", sid, {"tags": "[macro]"},
                   f"# {cfg['fred_series'][sid]} ({sid})\n\nLast value: **{s.iloc[-1]:.2f}** on {s.index[-1].date()}")

    ok = uni[uni.has_data]
    by_class = ok.groupby("asset_class").size().to_frame("n").join(
        ok[ok.eligible].groupby("asset_class").size().rename("eligible")).fillna(0).astype(int)
    by_region = ok.groupby("region").size()
    write_note(vault, "Agents", "Universe", {"tags": "[universe]"},
               "# Universe\n\n## By asset class\n| Asset class | Assets | With enough history |\n|---|---|---|\n"
               + "\n".join(f"| {k} | {r.n} | {r.eligible} |" for k, r in by_class.iterrows())
               + "\n\n## By region\n| Region | Assets |\n|---|---|\n"
               + "\n".join(f"| {k} | {v} |" for k, v in by_region.items()))

    issues = [t for t, q in quality.items() if q["stale"] or q["jumps_over_25pct"] > 3 or q["missing_pct"] > 30]
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    body = (f"# Data agent report {today}\n\n- Assets in the universe: {len(uni)}\n"
            f"- Assets with data: {len(close.columns)} · eligible: {int(uni.eligible.sum())}\n"
            f"- Tickers without data: {', '.join(missing) or 'none'}\n"
            f"- Macro series: {len(macro)} (failed: {failed or 'none'})\n"
            f"- To check: {', '.join(f'[[{t}]]' for t in issues[:50]) or 'none'}\n"
            "\nSee [[Universe]].\n")
    write_note(vault, "Reports", f"data_{today}", {"tags": "[report, data_agent]"}, body)
