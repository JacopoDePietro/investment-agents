# Data agent

Code: `agents/data_agent/` · Notebook: `notebooks/01_data.ipynb`

Builds the universe, downloads prices, volumes, FX, macro data and fundamentals, checks quality, stores Parquet files and writes one note per asset to the vault. It is the first agent: if it fails, the pipeline stops.

## Steps
1. **Universe** (`universe.build_universe`): S&P 500 list with GICS sector and CIK from the `datasets/s-and-p-500-companies` GitHub repository (cached in `data/raw/`), plus the static groups of `config/universe.yaml`, each with `asset_class` and `region`. Tickers with `.` become `-` (Yahoo format).
2. **Prices and volumes** (`sources.fetch_prices`): Yahoo Finance, adjusted for splits and dividends, batches of 100 with 3 attempts; missing tickers are retried one by one.
3. **Metadata**: years of history, `eligible` (≥ 3 years), quote currency from the ticker suffix (`core/currency.py`: .MI/.PA/.DE → EUR, .L → GBp, .T → JPY, …), `tradable` = false for FX pairs, futures and indices (signals only).
4. **FX**: `EUR<ccy>=X` for every currency; `core/currency.to_base` converts prices to EUR (pence divided by 100).
5. **Macro** (`sources.fetch_fred`): 15 public FRED series.
6. **Fundamentals** (`fundamentals.update`, at most weekly): SEC XBRL `companyfacts` for every S&P 500 CIK, 10-K values only (net income, revenue, gross profit, operating income, operating cash flow, capex, total assets, equity, long-term debt, shares outstanding). For every item and period end the **first filing** is kept, so the backtest sees the value as it was known at the time.
7. **Quality** (`quality.check_prices`): missing data, daily jumps above ±25%, stale prices.
8. **Vault notes**: one note per asset (1-year return and volatility, 5 least and most correlated assets on 3-year weekly returns), macro notes, universe summary, daily report.

## Post-processing used by the other agents (`portfolio_agent.agent.load`)
- common business-day grid, prices forward-filled for at most 5 days (local holidays);
- **synthetic history** (`core/synthetic.backfill`) for ETFs listed in `synthetic_history`: return = Σ leg weight × leg return + cash weight × Fed Funds/252 − fee/252, chained backwards to the first real price;
- conversion to EUR and a `has_fundamentals` flag.

## Outputs
| File | Content |
|---|---|
| `data/processed/prices.parquet`, `volume.parquet` | local-currency prices and volumes (dates × tickers) |
| `data/processed/universe.parquet` | name, group, asset_class, region, sector, has_data, history_years, eligible, currency, tradable |
| `data/processed/fx.parquet`, `macro.parquet` | FX rates per EUR, FRED series |
| `data/processed/fundamentals.parquet` | item, end, filed, val, ticker |

## Limitations
- Yahoo may rename or drop tickers; a few tickers have no data.
- S&P 500 membership is today's (survivorship bias).
- Synthetic histories validated against real prices: correlation 0.92-0.99 for NTSX, NTSI, NTSE, RSST, GDE; weaker for RSBT (0.69) and optimistic for RSSB (+3.5 pts/yr).
