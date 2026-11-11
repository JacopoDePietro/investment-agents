# News agent

Code: `agents/news_agent/` · Notebook: `notebooks/02_news.ipynb`

Collects market-moving news, has Gemini classify it and turns it into a numeric sentiment signal per asset, asset class and region.

## Steps
1. **Mode** (`news.mode`): `rss` (default: Google News RSS + Gemini), `search` (Gemini with Google Search grounding, paid tier), `auto` (search, falls back to rss on quota errors).
2. **Headlines** (`rss.fetch_headlines`): 4-6 English queries per theme, last 36 hours, de-duplicated, at most 60 per theme.
3. **Classification** (`gemini_client.analyze_headlines`): Gemini picks at most 8 headlines per theme and returns schema-constrained JSON (headline index, title, summary, Yahoo tickers, asset classes, regions, sentiment −1…+1, impact 1-5, horizon). Models are tried in order; the next one is used on 404, 429 or 503.
4. **Cleaning**: tickers outside the universe are kept separately; a title hash prevents double counting.
5. **Signal** (`_signals`): weight = impact × 0.5^(age in days / 3); score = Σ sentiment × weight / Σ weight, per ticker, asset class and region.

## How it is used
- Portfolio agent: score adjustment = 0.25 × ticker score (or the average of its asset class and region scores), **live portfolio only**.
- Controller: warning if more than 8% of the portfolio is on assets with recent news of impact ≥ 4 and sentiment ≤ −0.5.
- Not used in the backtest or the ML models (no news history).

## Outputs
`data/news/news.parquet` (cumulative history), `data/processed/news_signals.parquet` (key, score, weight, n_news, kind), `vault/News/news_<date>.md`.
