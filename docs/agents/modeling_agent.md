# Modeling agent

Code: `agents/modeling_agent/` · Notebook: `notebooks/04_modeling.ipynb`

1. Simulates the proposed portfolio with **1,000,000 paths** over 1 and 5 years.
2. Replays historical crises and hypothetical shocks on today's portfolio.
3. Provides scenarios to the portfolio agent (`scenarios.py`): CVaR cap, `min_cvar`, method-selection filter.

## Code map
| File | Role |
|---|---|
| `sde.py` | built-in Andersen QE scheme for CIR variance, Merton jumps, first-passage probability |
| `cflib_bridge.py` | chooses the backend (external library via `cflib_path`, otherwise `sde.py`); vectorised Heston log-price step |
| `calibrate.py` | model calibration and portfolio factor exposures |
| `simulate.py` | batched, multi-process Monte Carlo; short-horizon factor scenarios |
| `stress.py` | historical and hypothetical stress tests |
| `scenarios.py` | asset-level scenarios for the optimiser |

## Calibration (last 12 years)
Price factors in EUR: SPY, VGK, EEM, EWJ, HYG, GLD, DBC, BTC-USD (those with ≥ 2 years of history).
- Weekly realised variance v = Σ daily squared log returns × 52.
- CIR variance dynamics by OLS on Δv = a + b v: κ = −b/Δt (bounded 0.3-25), θ = a/(κΔt), σ = std(residual/√v)/√Δt.
- v0 = last 4 weeks; ρ = correlation between weekly return and Δv.
- Drift μ = current ECB rate + 0.5 × (historical return − average ECB rate).
- 10-year yield: CIR by the same regression.
- Regimes: crisis week if average VIX > 25; transition matrix from history; separate correlation matrices of standardised shocks per regime; correlated variance shocks.

## Simulation
Per weekly step: Markov regime per path; correlated price shocks (regime Cholesky); correlated variance shocks fed to the QE scheme; Heston log-return
X = −ΔI/2 + (ρ/η)(ΔV − κ(θΔt − ΔI)) + √((1−ρ²)ΔI) ξ + μΔt;
Merton jumps (0.5/year, mean −6%, sd 5%) with compensator on equity factors; CIR yield path. Portfolio return = Σ β_f (e^X_f − 1) + β_rate Δy + idiosyncratic noise.

## Metrics
Median, percentiles, VaR/CVaR 95% and 99%, probabilities of loss and of 10/20/30% drawdowns, probability of underperforming the S&P 500, terminal wealth; cross-check of the probability of touching 80% of capital with the analytical first-passage formula.

## Validation of the built-in backend
| Check | Simulated | Theory |
|---|---|---|
| E[V₁], κ = 3 | 0.04239 | 0.04249 |
| E[∫V], κ = 3 | 0.05577 | 0.05584 |
| E[V₁], κ = 25 | 0.03609 | 0.03600 |
| E[exp(jumps)] over 1 year | 0.9718 | 0.97187 |

On the same portfolio the built-in backend and an external course library gave 1-year CVaR 95% of −19.1% and −19.4% and P(DD > 20%) of 5.6% and 6.0%.

## Sensitivity (3-month CVaR 95% of a sample portfolio)
Base −11.5%; no risk premium −12.1%; full historical premium −11.0%; jumps doubled or removed ±0.1 pt; crisis threshold VIX 20 −11.7%. Results depend more on the drift assumption than on jumps.

## Limitations
κ often hits its upper bound (weekly realised variance is noisy); the drift is deliberately conservative and excludes the selection edge; idiosyncratic residuals are independent across assets.
