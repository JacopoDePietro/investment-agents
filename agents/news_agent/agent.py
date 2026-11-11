"""News agent: recent market news by theme, classified by Gemini and linked to the assets in the universe."""
import hashlib, logging, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from core.obsidian import write_note
from .gemini_client import get_client, search_news, analyze_headlines, NoSearchQuota
from .rss import fetch_headlines

log = logging.getLogger("news_agent")

PROMPT = """You are a buy-side financial analyst. Search Google for the most important news of the last {hours} hours
on the topic "{theme}": {desc}. Today is {today}. Prefer primary and international sources
(Reuters, Bloomberg, FT, WSJ, CNBC, central bank and government releases).
Select at most {n} news items with potential impact on financial markets.
For each item list the affected assets using ONLY Yahoo Finance tickers.
Prefer tickers from this list when relevant: {hubs}.
Single stocks are also allowed (e.g. NVDA, ENI.MI, 7203.T).

Reply ONLY with a JSON array, no text before or after, of objects shaped like:
{{"title": str (in English), "summary": str (max 2 sentences, in English), "date": "YYYY-MM-DD",
  "tickers": [str], "asset_classes": [one or more of equity, bond, commodity, fx, crypto, real_estate, volatility],
  "regions": [one or more of US, Europe, Italy, AsiaPacific, Emerging, Global],
  "sentiment": number from -1 (very negative for the listed assets) to 1 (very positive),
  "impact": integer from 1 (marginal) to 5 (market shock),
  "horizon": "days" | "weeks" | "months"}}"""

RSS_PROMPT = """You are a buy-side financial analyst. Below are recent headlines (last {hours} hours, today is {today})
about the topic "{theme}": {desc}.
Select at most {n} headlines with the highest potential impact on financial markets. Skip duplicates,
opinion pieces and clickbait. For each selected headline return its "index" exactly as given.
Write "title" as a clean English headline and "summary" as max 2 English sentences on why it matters for markets,
using only what the headline states plus well-established context. Do not invent figures.
Use ONLY Yahoo Finance tickers in "tickers". Prefer tickers from this list when relevant: {hubs}.
Single stocks are also allowed (e.g. NVDA, ENI.MI, 7203.T).
asset_classes: equity, bond, commodity, fx, crypto, real_estate, volatility.
regions: US, Europe, Italy, AsiaPacific, Emerging, Global.
sentiment: -1 (very negative for the listed assets) to 1 (very positive). impact: 1 (marginal) to 5 (market shock).

Headlines:
{headlines}"""

def _collect(client, models, mode, theme, t, ncfg, today, hubs):
    """Returns (items, sources per item) according to the configured mode."""
    if mode in ("auto", "search"):
        prompt = PROMPT.format(hours=ncfg["lookback_hours"], theme=theme, desc=t["desc"], today=today,
                               n=ncfg["max_items_per_theme"], hubs=hubs)
        items, sources = search_news(client, models, prompt)
        return items, [[s["uri"] for s in sources[:5]]] * len(items), "search"
    heads = fetch_headlines(t["queries"], ncfg["lookback_hours"], ncfg.get("max_headlines_per_theme", 60))
    if not heads:
        return [], [], "rss"
    listing = "\n".join(f"[{i}] {h['published']} | {h['source']} | {h['title']}" for i, h in enumerate(heads))
    prompt = RSS_PROMPT.format(hours=ncfg["lookback_hours"], theme=theme, desc=t["desc"], today=today,
                               n=ncfg["max_items_per_theme"], hubs=hubs, headlines=listing)
    items = analyze_headlines(client, models, prompt)
    srcs = []
    for it in items:
        i = it.get("index")
        h = heads[i] if isinstance(i, int) and 0 <= i < len(heads) else None
        it["date"] = h["published"][:10] if h else today
        it["source_name"] = h["source"] if h else ""
        srcs.append([h["link"]] if h else [])
    return items, srcs, "rss"

def run(cfg):
    ncfg, d, vault = cfg["news"], cfg["paths"]["data"], cfg["paths"]["vault"]
    uni = pd.read_parquet(d / "processed" / "universe.parquet")
    valid = set(uni.index)
    hubs = ", ".join(uni[uni.asset_class != "equity"].index[:150])
    client = get_client()
    today = datetime.now().strftime("%Y-%m-%d")

    models = [ncfg["model"], *ncfg.get("fallback_models", [])]
    mode = ncfg.get("mode", "auto")
    rows = []
    for theme, t in ncfg["themes"].items():
        try:
            items, srcs, used = _collect(client, models, mode, theme, t, ncfg, today, hubs)
        except NoSearchQuota:
            if mode == "search":
                log.error("No quota for Gemini search: use mode auto or rss"); break
            log.warning("Gemini search has no quota: switching to RSS feeds for the rest of the run")
            mode = "rss"
            try:
                items, srcs, used = _collect(client, models, mode, theme, t, ncfg, today, hubs)
            except Exception as e:
                log.warning("Theme %s failed with RSS feeds too: %s", theme, str(e)[:200]); continue
        except Exception as e:
            log.warning("Theme %s failed: %s", theme, str(e)[:200]); time.sleep(ncfg["pause_seconds"]); continue
        log.info("Theme %s (%s): %d news items", theme, used, len(items))
        for it, src in zip(items, srcs):
            tick = [str(x).strip() for x in it.get("tickers", [])]
            tick = [x if "." in x or "=" in x or "^" in x else x.upper() for x in tick]
            rows.append({
                "id": hashlib.md5(it.get("title", "").lower().encode()).hexdigest()[:12],
                "fetched_at": datetime.now(timezone.utc), "theme": theme, "method": used,
                "date": it.get("date"), "title": it.get("title", ""), "summary": it.get("summary", ""),
                "source_name": it.get("source_name", ""),
                "tickers": [x for x in tick if x in valid],
                "unknown_tickers": [x for x in tick if x not in valid],
                "asset_classes": it.get("asset_classes", []), "regions": it.get("regions", []),
                "sentiment": float(np.clip(_num(it.get("sentiment")), -1, 1)),
                "impact": int(np.clip(_num(it.get("impact"), 1), 1, 5)),
                "horizon": it.get("horizon", ""), "sources": src})
        time.sleep(ncfg["pause_seconds"])

    if not rows:
        log.warning("No news collected"); return {"news": 0}
    new = pd.DataFrame(rows)
    hist_path = d / "news" / "news.parquet"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    hist = pd.read_parquet(hist_path) if hist_path.exists() else pd.DataFrame()
    allnews = pd.concat([hist, new]).drop_duplicates("id", keep="first")
    allnews.to_parquet(hist_path)

    signals = _signals(allnews, ncfg["half_life_days"])
    signals.to_parquet(d / "processed" / "news_signals.parquet")
    todays = allnews[pd.to_datetime(allnews.fetched_at, utc=True).dt.tz_convert(None).dt.strftime("%Y-%m-%d") == today]
    _write_vault(vault, todays, signals, today)
    return {"news_new": int((~new.id.isin(hist.get("id", []))).sum()), "news_total": len(allnews),
            "signals": len(signals)}

def _num(x, default=0.0):
    try: return float(x)
    except (TypeError, ValueError): return default

def _signals(news, half_life):
    """Sentiment weighted by impact with exponential decay, per ticker, asset class and region."""
    now = pd.Timestamp.now(tz="UTC")
    age = (now - pd.to_datetime(news.fetched_at, utc=True)).dt.total_seconds() / 86400
    w = news.impact * 0.5 ** (age / half_life)
    out = []
    for kind, col in [("ticker", "tickers"), ("asset_class", "asset_classes"), ("region", "regions")]:
        e = news.assign(w=w, key=news[col]).explode("key").dropna(subset=["key"])
        if e.empty: continue
        g = e.groupby("key").apply(lambda x: pd.Series({
            "score": (x.sentiment * x.w).sum() / x.w.sum(), "weight": x.w.sum(), "n_news": len(x)}),
            include_groups=False)
        out.append(g.assign(kind=kind))
    return pd.concat(out).reset_index(names="key") if out else pd.DataFrame()

def _write_vault(vault, new, signals, today):
    lines = [f"# News {today}\n"]
    for theme, g in new.groupby("theme"):
        lines.append(f"\n## {theme}\n")
        for _, r in g.sort_values("impact", ascending=False).iterrows():
            links = " ".join(f"[[{t.replace('^', '_')}]]" for t in r.tickers)
            arrow = "🟢" if r.sentiment > 0.2 else "🔴" if r.sentiment < -0.2 else "⚪"
            src = f" · [{r.get('source_name') or 'source'}]({r.sources[0]})" if len(r.sources) else ""
            lines.append(f"- {arrow} **{r.title}** (impact {r.impact}/5, {r.horizon})\n"
                         f"  {r.summary}\n  {links}{src}")
    if not signals.empty:
        t = signals[signals.kind == "ticker"].sort_values("score")
        lines.append("\n## Signals by asset (weighted sentiment)\n| Asset | Score | News |\n|---|---|---|")
        for _, r in pd.concat([t.head(10), t.tail(10)]).drop_duplicates("key").iterrows():
            lines.append(f"| [[{r.key.replace('^', '_')}]] | {r.score:+.2f} | {int(r.n_news)} |")
        c = signals[signals.kind != "ticker"]
        lines.append("\n## Signals by asset class and region\n| Kind | Key | Score |\n|---|---|---|")
        lines += [f"| {r.kind} | {r.key} | {r.score:+.2f} |" for _, r in c.iterrows()]
    write_note(vault, "News", f"news_{today}", {"tags": "[news, news_agent]"}, "\n".join(lines))
