"""Data quality checks: gaps, outliers, stale data."""
import pandas as pd

def check_prices(close: pd.DataFrame) -> dict:
    rets = close.pct_change(fill_method=None)
    report = {}
    for t in close.columns:
        s = close[t].dropna()
        report[t] = {
            "first": s.index.min().date() if len(s) else None,
            "last": s.index.max().date() if len(s) else None,
            "missing_pct": round(close[t].isna().mean() * 100, 2),
            "jumps_over_25pct": int((rets[t].abs() > 0.25).sum()),
            "stale": bool(len(s) and (pd.Timestamp.today() - s.index.max()).days > 5),
        }
    return report
