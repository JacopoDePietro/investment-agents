# Controller agent

Code: `agents/controller_agent/agent.py` · Notebook: `notebooks/05_controller.ipynb`

Checks the proposal against fixed thresholds and returns **APPROVED**, **APPROVED WITH WARNINGS** or **BLOCKED**. When blocked, the last approved portfolio stays active. It never places orders.

| Level | Check | Default limit |
|---|---|---|
| hard | 1-year CVaR 95% from the 1M-path Monte Carlo | ≥ −22% |
| hard | 1-year probability of a drawdown above 20% | ≤ 12% |
| hard | GFC 2008 / Covid 2020 replayed on today's portfolio | ≥ −35% / ≥ −30% |
| hard | worst hypothetical scenario (AI bubble, rate shock, oil shock) | ≥ −20% |
| hard | optimiser constraint breaches; max single weight | none; ≤ 12% |
| soft | turnover vs last approved portfolio | ≤ 80% |
| soft | 1-year probability of underperforming the S&P 500 | ≤ 70% |
| soft | weight on assets with very negative recent news | ≤ 8% |

Outputs: `data/controller/decision.json`, `decisions.csv` (history), `approved_portfolio.csv`, `vault/Reports/control_<date>.md`. Thresholds: section `controller` of `config/settings.yaml`.
