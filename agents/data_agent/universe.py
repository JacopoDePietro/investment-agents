"""Builds the universe: dynamic S&P 500 list + static groups, with metadata."""
import io, logging
import pandas as pd
import requests
import yaml

log = logging.getLogger("data_agent")
SP500_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"

def build_universe(root, data_dir) -> pd.DataFrame:
    spec = yaml.safe_load((root / "config" / "universe.yaml").read_text())
    rows = []
    sp = spec.get("dynamic", {}).get("sp500", {})
    if sp.get("enabled"):
        cache = data_dir / "raw" / "sp500_constituents.csv"
        try:
            r = requests.get(SP500_URL, timeout=30); r.raise_for_status()
            cache.parent.mkdir(parents=True, exist_ok=True); cache.write_text(r.text)
        except Exception as e:
            log.warning("S&P 500 list not downloaded (%s), using cache", e)
        if cache.exists():
            df = pd.read_csv(cache)
            for _, x in df.iterrows():
                rows.append({"ticker": x["Symbol"].replace(".", "-"), "name": x["Security"],
                             "group": "sp500", "asset_class": sp["asset_class"],
                             "region": sp["region"], "sector": x["GICS Sector"]})
    for group, g in spec.get("static", {}).items():
        for t in g["tickers"]:
            rows.append({"ticker": str(t), "name": "", "group": group, "asset_class": g["asset_class"],
                         "region": g["region"], "sector": ""})
    return pd.DataFrame(rows).drop_duplicates("ticker").set_index("ticker")
