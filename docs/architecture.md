# Architecture and parameter reference

`run.py` executes the agents in order: **data → news → portfolio → modeling → controller**. If the data agent fails, the others do not run. Every agent reads `config/settings.yaml`, writes machine-readable outputs to `data/` and human-readable Markdown notes to `vault/`.

## Data flow
| Producer | File | Consumers |
|---|---|---|
| data | `data/processed/prices.parquet`, `volume.parquet`, `fx.parquet`, `macro.parquet`, `universe.parquet`, `fundamentals.parquet` | all |
| news | `data/news/news.parquet`, `data/processed/news_signals.parquet` | portfolio (live score), controller |
| portfolio | `data/portfolio/target_portfolio.csv`, `context.json`, backtest files, diagnostics | modeling, controller, notebooks |
| modeling | `data/modeling/risk_report.json` | controller |
| controller | `data/controller/decision.json`, `approved_portfolio.csv`, `decisions.csv` | you |

The portfolio agent also calls the modeling code directly (calibration and simulation) for the Monte Carlo scenarios used inside the optimiser and for the method-selection filter.

## Effective portfolio limits
Three layers, each overriding the previous one (dictionaries are merged one level deep):
1. base values in `portfolio`;
2. the risk profile `portfolio.profiles.<risk_profile>` (default `growth`);
3. the risk mode profile `portfolio.risk_mode.offense` (always active while automatic switching is off).

| Limit | Effective value |
|---|---|
| Max weight per asset | 10% |
| Single stocks / equity ETFs | 70% / 70%, together max 90% |
| Bonds | 15% |
| Commodities / alternatives / FX | 20% / 15% / 10% |
| Real estate / infrastructure / crypto | 10% / 10% / 5% |
| Capital-efficient ETFs | 40% |
| Regions | US 60%, Europe 30%, Asia-Pacific 20%, Italy 15%, Emerging 15%, Canada 10% |
| AI-related assets | 35% (10% when the AI-bubble rule fires) |
| Crash-hedge ETFs, minimum | 5% (10% with bubble) |
| Max annual volatility | 18% |
| Risk budgets | equities ≤ 92% of total risk, crypto ≤ 10%, commodities ≤ 30% |
| Tail-loss cap (3-month Monte Carlo scenarios) | CVaR 95% ≥ -16% |

## Main switches
| Key | Default | Meaning |
|---|---|---|
| `portfolio.methods` | 6 methods | allocation methods compared in the backtest |
| `portfolio.choose_by` | `multi_horizon` | mean Sharpe over 1y, 2y, 5y and full period |
| `portfolio.mc_selection` | 200k paths, CVaR ≥ -20%, P(DD>20%) ≤ 10% | Monte Carlo filter on today's candidates |
| `portfolio.mc_scenarios.enabled` | true | Heston scenarios for the CVaR cap and `min_cvar` |
| `portfolio.ml_returns` | enabled, blend 1.0, gamma 2 | LightGBM expected returns |
| `portfolio.ml` | enabled, tilt 0.7 (offense) | LightGBM selection score |
| `portfolio.covariance` | `dcc` | 50% Ledoit-Wolf + 50% DCC |
| `portfolio.risk_mode.enabled` | false | signal-based offense/defense switching |
| `portfolio.single_stocks_require_fundamentals` | false | single stocks only with SEC fundamentals |
| `modeling.cflib_path` | null | external library; null = built-in `sde.py` |

## Runtime (Intel i5 quad-core, 8 GB RAM)
| Step | Time |
|---|---|
| Data download | ~6 min |
| Factor panel (first run, then cached) | ~5 min |
| Portfolio backtest 2008-today, 6 methods, quarterly | ~35 min |
| Monte Carlo 1M paths, 1y + 5y | ~3-5 min |
