# Portfolio agent

Code: `agents/portfolio_agent/` · Notebook: `notebooks/03_portfolio.ipynb`

1. Runs a **walk-forward backtest** since 2008 of 6 allocation methods, redoing selection and optimisation at every rebalance with data available at that date only.
2. Builds **today's portfolio** with every method, has each one simulated, and picks the method.
3. Writes the proposal, metrics and diagnostics.

## Code map
| File | Role |
|---|---|
| `agent.py` | orchestration, layered limits (`apply_profile`), Monte Carlo method selection (`mc_select`), horizons, reports, diagnostics |
| `backtest.py` | backtest loop, weight construction (`build_weights`), constraints, eligibility, AI-bubble signal, no-trade band, factor-panel cache |
| `factors.py` | factor definitions, point-in-time fundamentals, IC table, adaptive weights, momentum dampening |
| `ml.py` | selection LightGBM (`Booster`) |
| `ml_returns.py` | expected-return LightGBM (`MLReturns`) and its time-series validation (`tune`) |
| `selection.py` | per-class quotas with correlation clustering, Ledoit-Wolf + DCC covariance |
| `optimizers.py` | SLSQP with analytic gradients, risk constraints, HRP, CVaR LP, CVaR cap |
| `regime.py`, `risk_mode.py` | HMM regimes and signal-based offense/defense switching (optional) |

## 1. Investable assets at date t (`eligible`)
Prices on ≥ 90% of the previous 756 days and at t; 60-day median traded value ≥ 1,000,000 (FX exempt); tradable and not in `exclude_groups`.

## 2. Factors (`features_at`, `Fundamentals.factors_at`)
p = EUR prices, index −k = k days ago.

| Factor | Definition |
|---|---|
| mom_12_1 / mom_6_1 / mom_3_1 | p[−21] / p[−252, −126, −63] − 1 |
| risk_adj_mom | mom_12_1 / annual volatility |
| low_vol | −std(252-day log returns) × √252 |
| trend_200 | p[0] / 200-day mean − 1 |
| near_high | p[0] / 252-day max − 1 |
| reversal_1m | −(p[0] / p[−21] − 1) |
| low_beta | −252-day beta to SPY |
| earnings_yield, book_to_market, fcf_yield | net income, equity, (operating cash flow − capex) / market cap |
| gross_prof, roe | gross (or operating) profit / assets; net income / positive equity |
| low_leverage, low_asset_growth | −long-term debt / assets; −(assets / prior-year assets − 1) |

Market cap = shares outstanding × USD price (≥ 100M); filings older than 550 days are ignored. Factors are converted to percentile ranks − 0.5 within each family (equities vs everything else).

## 3. Adaptive factor weights (`ic_table`, `weights_at`)
Monthly Spearman IC between each factor and the next 63-day return, per family. At date t only ICs whose forward return is already realised (≤ t − 63 days) over the last 60 months are used; weight ∝ positive mean IC. If the 4-week average VIX exceeds 25, momentum-type weights are scaled by 0.3 and the freed weight goes to the other factors. Factor score = 4 × Σ weight × rank.

## 4. Selection LightGBM (`ml.Booster`)
Inputs: the 16 ranks, family, asset class. Target: within-family percentile rank of the 63-day return. 8 years of training, retrained yearly. Final score = 0.3 × factor score + 0.7 × LightGBM + 0.5 holding bonus + (live only) 0.25 × news.

## 5. Candidates (`pick_by_class`)
Per asset class with a cap: quota = min(n, ⌈cap / 10%⌉ + 1). Within the class, hierarchical clustering on distance √(0.5 (1 − ρ)) into min(2 × quota, n) clusters, best score per cluster. Plus the 2 best crash-hedge ETFs per asset class.

## 6. Estimates
- **Covariance**: 50% Ledoit-Wolf + 50% DCC(1,1) (a = 0.02, b = 0.97) on EWMA-standardised residuals (λ = 0.94), 3-year window.
- **Expected returns** (`MLReturns`): LightGBM with Huber loss on factor ranks + 6 raw values + family + asset class; target = 63-day return clipped at the 1st/99th percentile **minus the cross-sectional mean** (no macro features: with them the model learned market timing and failed validation). μ = forecast × 252/63 + historical mean of the candidates.
- **Scenarios**: Heston/CIR calibrated on data up to t, 10,000 13-week factor scenarios mapped to assets through 5-year weekly betas plus idiosyncratic noise.

## 7. Optimisation
w ≥ 0, Σw = 1, w ≤ 10%; linear caps per class, total equity, region, AI assets; minimum crash-hedge weight; w'Σw ≤ vol²; risk contribution of each group ≤ budget × variance.

| Method | Objective |
|---|---|
| equal_weight | equal weights projected onto the constraints |
| max_sharpe | max μ'w / √(w'Σw) |
| mean_variance | max μ'w − γ/2 w'Σw, γ = 2 |
| risk_parity | equal risk contributions |
| max_diversification | max Σ w_i σ_i / √(w'Σw) |
| min_cvar | min scenario CVaR 95% (Rockafellar-Uryasev LP, ≤ 4,000 scenarios) |

Afterwards, if the scenario CVaR breaches the cap, the portfolio is moved towards the minimum-CVaR portfolio by the smallest amount needed (bisection). Positions below EUR 100 are dropped and the problem is re-solved.

## 8. Backtest
Quarter ends since 2008. No-trade band of 2 points; cost = turnover × 10 bp + orders × EUR 1 / capital; weights drift with prices between rebalances. Benchmarks: SPY, VT, 60/40, all in EUR.

## 9. Today's choice (`mc_select`)
Score = mean Sharpe over 1y, 2y, 5y and the full period. Each method's portfolio is simulated (200,000 one-year paths); methods with CVaR 95% < −20% or P(drawdown > 20%) > 10% are rejected; the best score among the rest wins.

## Outputs (`data/portfolio/`)
`target_portfolio.csv`, `backtest_returns.parquet`, `backtest_metrics.csv`, `metrics_<horizon>.csv`, `weights_history.parquet`, `class_weights_history.csv`, `factor_ic.parquet`, `factor_weights_history.csv`, `ml_oos_ic.csv`, `context_history.csv`, `scores_today.csv`, `expected_returns_ml.csv`, `mc_selection.csv`, `context.json`.

## Diagnostics from the latest full run
| Model | Mean out-of-sample IC (2008-today) | Dates with positive IC |
|---|---|---|
| Expected-return LightGBM | 0.065 | 73% |
| Selection LightGBM | 0.021 | 63% |
