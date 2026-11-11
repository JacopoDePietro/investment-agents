"""Free historical fundamentals from the SEC (XBRL companyfacts) with filing dates: no look-ahead bias.
Covers US companies with a known CIK (S&P 500). Saves a compact table to data/processed/fundamentals.parquet."""
import logging, time
from datetime import datetime, timedelta
import pandas as pd
import requests

log = logging.getLogger("data_agent")
URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
CONCEPTS = {   # internal name: alternative us-gaap concepts in order of preference
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "assets": ["Assets"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "lt_debt": ["LongTermDebt", "LongTermDebtNoncurrent"],
}
FLOW = {"net_income", "revenue", "gross_profit", "operating_income", "cfo", "capex"}

def _extract(js):
    g = js.get("facts", {}).get("us-gaap", {})
    rows = []
    for name, alts in CONCEPTS.items():
        for c in alts:
            units = g.get(c, {}).get("units", {}).get("USD")
            if not units:
                continue
            for u in units:
                if u.get("form") not in ("10-K", "10-K/A"):
                    continue
                if name in FLOW:      # annual values only (duration ~1 year)
                    if "start" not in u or not 330 <= (pd.Timestamp(u["end"]) - pd.Timestamp(u["start"])).days <= 400:
                        continue
                rows.append({"item": name, "end": u["end"], "filed": u["filed"], "val": u["val"]})
            break
    sh = js.get("facts", {}).get("dei", {}).get("EntityCommonStockSharesOutstanding", {}).get("units", {}).get("shares", [])
    rows += [{"item": "shares", "end": u["end"], "filed": u["filed"], "val": u["val"]} for u in sh]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["end"] = pd.to_datetime(df["end"]); df["filed"] = pd.to_datetime(df["filed"])
    # for each item and period end keep the first filing (the value as it was known at the time)
    return df.sort_values("filed").drop_duplicates(["item", "end"], keep="first")

def update(data_dir, uni, user_agent, refresh_days=7):
    path = data_dir / "processed" / "fundamentals.parquet"
    if path.exists() and datetime.fromtimestamp(path.stat().st_mtime) > datetime.now() - timedelta(days=refresh_days):
        log.info("SEC fundamentals updated less than %d days ago: skipping download", refresh_days)
        return pd.read_parquet(path)
    sp = pd.read_csv(data_dir / "raw" / "sp500_constituents.csv")
    sp["ticker"] = sp["Symbol"].str.replace(".", "-", regex=False)
    sp = sp[sp.ticker.isin(uni.index)]
    s = requests.Session(); s.headers["User-Agent"] = user_agent
    out, fail = [], 0
    for i, r in enumerate(sp.itertuples()):
        try:
            resp = s.get(URL.format(cik=int(r.CIK)), timeout=30)
            resp.raise_for_status()
            df = _extract(resp.json())
            if len(df):
                out.append(df.assign(ticker=r.ticker))
        except Exception:
            fail += 1
        time.sleep(0.15)                          # SEC limit: at most 10 requests per second
        if (i + 1) % 100 == 0:
            log.info("SEC fundamentals: %d/%d companies", i + 1, len(sp))
    f = pd.concat(out, ignore_index=True)
    f.to_parquet(path)
    log.info("SEC fundamentals: %d companies, %d rows, %d errors", f.ticker.nunique(), len(f), fail)
    return f
