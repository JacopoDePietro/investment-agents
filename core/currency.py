"""Quote currency inferred from the Yahoo ticker suffix, and conversion to the base currency."""
import numpy as np
import pandas as pd

SUFFIX = {".MI": "EUR", ".PA": "EUR", ".DE": "EUR", ".AS": "EUR", ".MC": "EUR", ".BR": "EUR",
          ".L": "GBp", ".SW": "CHF", ".T": "JPY", ".HK": "HKD", ".KS": "KRW", ".AX": "AUD",
          ".NS": "INR", ".SI": "SGD", ".CO": "DKK", ".ST": "SEK", ".OL": "NOK", ".TO": "CAD"}

def currency_of(ticker: str) -> str:
    if ticker.endswith(("=X", "=F")) or ticker.startswith("^"):
        return "NONE"           # FX pairs, futures and indices: signals only, not investable
    for suf, ccy in SUFFIX.items():
        if ticker.endswith(suf):
            return ccy
    return "USD"

def fx_tickers(currencies, base="EUR"):
    """Yahoo tickers quoting 'units of currency per 1 base unit', e.g. EURUSD=X."""
    out = {}
    for c in set(currencies) - {"NONE", base}:
        real = "GBP" if c == "GBp" else c
        out[c] = f"{base}{real}=X"
    return out

def to_base(prices: pd.DataFrame, fx: pd.DataFrame, base="EUR") -> pd.DataFrame:
    """Converts every column to the base currency. fx columns = currencies, values = currency per 1 base unit."""
    fx = fx.reindex(prices.index).ffill()
    out = prices.copy()
    for t in prices.columns:
        c = currency_of(t)
        if c in ("NONE", base):
            continue
        rate = fx.get("GBP" if c == "GBp" else c)
        if rate is None:
            out[t] = np.nan; continue
        out[t] = prices[t] / rate / (100 if c == "GBp" else 1)
    return out
