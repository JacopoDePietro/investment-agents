# Experiment log

Chronological record of what was tried, what was kept and why. Unless stated otherwise: walk-forward backtest in EUR, net of costs, `max_diversification` or the method indicated. `tools/experiment.py` reproduces the variants.

| # | Change | Result | Decision |
|---|---|---|---|
| 1 | Static score: momentum + low volatility | Low volatility had a **negative** IC on equities (−0.032) | replaced by adaptive IC-weighted factors |
| 2 | Monthly rebalancing | Turnover 7-10× per year: fixed commissions on a small account wiped out returns | quarterly + 2-point no-trade band + holding bonus (turnover ~1.5×) |
| 3 | Selection by top momentum only | Infeasible constraints (90% equities against a 65% cap) | per-class quotas with correlation clustering |
| 4 | Balanced vs growth risk profile | The equity cap was the main drag on returns | growth profile |
| 5 | Permanent crash-protection rules (hedge floor, AI cap, vol cap, CVaR cap) | −3 pts/yr, smaller drawdowns | kept, with lighter values in the offensive profile |
| 6 | HMM regimes, DCC covariance, selection LightGBM | Gain came almost entirely from LightGBM; regimes and DCC roughly neutral | all kept, HMM used only as a signal |
| 7 | Signal-based offense/defense switching | Best rule on 2008-2016 lost to "always offense" on 2017-2026 (Sharpe 0.55 vs 0.69) | **off** (`risk_mode.enabled: false`) |
| 8 | Single stocks only with SEC fundamentals | Helped the last year, cost ~2 pts/yr since 2008 | **off** |
| 9 | Capital-efficient ETFs with synthetic history | Negligible effect | kept as an option in the universe |
| 10 | Monte Carlo scenarios in the optimiser | −0.8 pts/yr over the full period, better on 5 years | kept (simulations must drive the decisions) |
| 11 | Expected-return LightGBM with macro features, absolute target | Negative validation IC on 2011-2013 for every parameter set | rejected |
| 12 | Expected-return LightGBM, no macro, cross-sectional target | Validation IC 0.070 (61% positive); OOS backtest IC 0.065 | kept; with `mean_variance` (γ = 2, blend 1.0 chosen on 2008-2013) |
| 13 | Method choice by full-period Sharpe | Ignored weak recent horizons | `multi_horizon` + Monte Carlo filter |

## Why max_diversification lags in equity rallies
It maximises Σ w_i σ_i / σ_p and ignores expected returns. Equities are correlated with each other, so it keeps few of them and favours volatile, idiosyncratic or low-return diversifiers. `mean_variance` with LightGBM expected returns addresses exactly this.
