"""Free sources: Yahoo Finance (prices) and FRED (macro, public CSV, no key)."""
import io, logging, time
import pandas as pd
import requests
import yfinance as yf

log = logging.getLogger("data_agent")

def fetch_prices(tickers, start, batch=100):
    closes, vols = [], []
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        for attempt in range(3):
            try:
                df = yf.download(chunk, start=start, auto_adjust=True, progress=False,
                                 threads=True, group_by="column")
                closes.append(df["Close"]); vols.append(df["Volume"])
                break
            except Exception as e:
                log.warning("Batch %d failed (attempt %d): %s", i // batch, attempt + 1, e)
                time.sleep(10 * (attempt + 1))
        log.info("Downloaded %d/%d tickers", min(i + batch, len(tickers)), len(tickers))
        time.sleep(2)  # avoid Yahoo rate limits
    close = pd.concat(closes, axis=1).dropna(how="all").dropna(axis=1, how="all")
    volume = pd.concat(vols, axis=1).reindex(columns=close.columns)
    missing = [t for t in tickers if t not in close.columns]
    if missing and len(missing) < len(tickers):      # second attempt, one at a time, for transient errors
        time.sleep(5)
        for t in missing:
            try:
                df = yf.download(t, start=start, auto_adjust=True, progress=False)
                if len(df):
                    close[t] = df["Close"].squeeze(); volume[t] = df["Volume"].squeeze()
            except Exception:
                pass
            time.sleep(0.5)
        log.info("Second attempt: recovered %d of %d", sum(t in close for t in missing), len(missing))
    return close, volume

def fetch_fred(series_id, start):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=60, headers={"User-Agent": "curl/8.0"})
            r.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))
    s = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True).iloc[:, 0]
    return pd.to_numeric(s, errors="coerce").dropna().loc[start:].rename(series_id)
