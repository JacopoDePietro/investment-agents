"""Long-only allocation methods with asset-class, region and risk constraints."""
import numpy as np
from scipy.optimize import minimize, linprog
from scipy import sparse
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

RISK = {}   # current risk constraints: set by set_risk before each optimisation

def set_risk(cov=None, floors=(), vol_cap=None, budgets=()):
    """floors: [(idx, minimum)] · vol_cap: max annual volatility · budgets: [(idx, max share of total risk)]"""
    RISK.clear(); RISK.update(cov=cov, floors=list(floors), vol_cap=vol_cap, budgets=list(budgets))

def _risk_cons(n):
    cons = []
    for idx, lo in RISK.get("floors", []):
        g = np.zeros(n); g[idx] = 1.0
        cons.append({"type": "ineq", "fun": lambda w, idx=idx, lo=lo: w[idx].sum() - lo, "jac": lambda w, g=g: g})
    C = RISK.get("cov")
    if C is not None and RISK.get("vol_cap"):
        v2 = RISK["vol_cap"] ** 2
        cons.append({"type": "ineq", "fun": lambda w: v2 - w @ C @ w, "jac": lambda w: -2 * C @ w})
    if C is not None:
        for idx, b in RISK.get("budgets", []):
            m = np.zeros(n); m[idx] = 1.0
            cons.append({"type": "ineq",
                         "fun": lambda w, m=m, b=b: b * (w @ C @ w) - (m * w) @ (C @ w),
                         "jac": lambda w, m=m, b=b: 2 * b * (C @ w) - (m * (C @ w) + C @ (m * w))})
    return cons

def _solve(obj, n, bounds, groups, w0=None, jac=None):
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(n)}]
    for idx, cap in groups:
        g = np.zeros(n); g[idx] = -1.0
        cons.append({"type": "ineq", "fun": lambda w, idx=idx, cap=cap: cap - w[idx].sum(),
                     "jac": lambda w, g=g: g})
    cons += _risk_cons(n)
    w0 = np.full(n, 1 / n) if w0 is None else w0
    res = minimize(obj, w0, jac=jac, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-10})
    w = np.clip(res.x if res.success or res.x is not None else w0, 0, None)
    return w / w.sum()

def project(target, bounds, groups):
    """Feasible weights closest to a target (for heuristic methods such as HRP or 1/vol)."""
    t = np.asarray(target, float)
    w = _solve(lambda w: ((w - t) ** 2).sum() * 1e3, len(t), bounds, groups, w0=t,
               jac=lambda w: 2e3 * (w - t))
    if not _feasible(w, groups):   # infeasible start: restart from equal weights
        w = _solve(lambda w: ((w - t) ** 2).sum() * 1e3, len(t), bounds, groups, w0=None,
                   jac=lambda w: 2e3 * (w - t))
    return w

def _feasible(w, groups, tol=0.005):
    return all(w[idx].sum() <= cap + tol for idx, cap in groups) and \
        all(w[idx].sum() >= lo - tol for idx, lo in RISK.get("floors", []))

def hrp_weights(cov):
    """Hierarchical Risk Parity (López de Prado)."""
    sd = np.sqrt(np.diag(cov)); corr = cov / np.outer(sd, sd)
    dist = np.sqrt(np.clip(0.5 * (1 - corr), 0, None)); np.fill_diagonal(dist, 0)
    order = list(leaves_list(linkage(squareform(dist, checks=False), "single")))
    w = np.ones(len(cov)); clusters = [order]
    while clusters:
        clusters = [c[s:e] for c in clusters for s, e in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
        for i in range(0, len(clusters), 2):
            a, b = clusters[i], clusters[i + 1]
            va, vb = (_ivp_var(cov, a), _ivp_var(cov, b))
            alpha = 1 - va / (va + vb)
            w[a] *= alpha; w[b] *= 1 - alpha
    return w / w.sum()

def _ivp_var(cov, idx):
    c = cov[np.ix_(idx, idx)]; iv = 1 / np.diag(c); iv /= iv.sum()
    return iv @ c @ iv

def allocate(method, mu, cov, bounds, groups, gamma=3.0):
    n = len(mu); sd = np.sqrt(np.diag(cov))
    pv = lambda w: w @ cov @ w
    if method == "equal_weight":
        return project(np.full(n, 1 / n), bounds, groups)
    if method == "inverse_vol":
        return project((1 / sd) / (1 / sd).sum(), bounds, groups)
    if method == "hrp":
        return project(hrp_weights(cov), bounds, groups)
    if method == "min_variance":
        return _solve(lambda w: pv(w) * 1e4, n, bounds, groups, jac=lambda w: 2e4 * (cov @ w))
    if method in ("max_sharpe", "max_diversification"):
        a = mu if method == "max_sharpe" else sd
        def f(w):
            s = np.sqrt(pv(w) + 1e-12); return -(w @ a) / s
        def g(w):
            s = np.sqrt(pv(w) + 1e-12); return -(a / s - (w @ a) * (cov @ w) / s ** 3)
        return _solve(f, n, bounds, groups, jac=g)
    if method == "mean_variance":   # max expected return with a risk penalty (risk aversion gamma)
        return _solve(lambda w: -(w @ mu) + 0.5 * gamma * pv(w), n, bounds, groups,
                      jac=lambda w: -mu + gamma * (cov @ w))
    if method == "risk_parity":
        def obj(w):
            rc = w * (cov @ w); return ((rc / rc.sum() - 1 / n) ** 2).sum() * 1e4
        return _solve(obj, n, bounds, groups)
    raise ValueError(method)


def cvar(w, R, alpha=0.95):
    """CVaR (mean of the worst losses) of the scenario returns R (S, n)."""
    pr = R @ w; q = np.quantile(pr, 1 - alpha)
    return pr[pr <= q].mean()

def min_cvar(R, bounds, groups, alpha=0.95, max_scen=4000, seed=0):
    """Minimum CVaR via linear programming (Rockafellar-Uryasev), same linear constraints.
    Sparse matrices; above `max_scen` scenarios a random sample is used."""
    if len(R) > max_scen:
        R = R[np.random.default_rng(seed).choice(len(R), max_scen, replace=False)]
    S, n = R.shape
    c = np.concatenate([np.zeros(n), [1.0], np.full(S, 1 / ((1 - alpha) * S))])
    blocks = [sparse.hstack([sparse.csr_matrix(-R), sparse.csr_matrix(-np.ones((S, 1))), -sparse.identity(S)])]
    b = [np.zeros(S)]
    extra = []
    for idx, cap in groups:
        row = np.zeros(n); row[idx] = 1; extra.append((row, cap))
    for idx, lo in RISK.get("floors", []):
        row = np.zeros(n); row[idx] = -1; extra.append((row, -lo))
    if extra:
        blocks.append(sparse.hstack([sparse.csr_matrix(np.array([e[0] for e in extra])),
                                     sparse.csr_matrix((len(extra), 1 + S))]))
        b.append(np.array([e[1] for e in extra]))
    Aeq = sparse.csr_matrix(np.concatenate([np.ones(n), np.zeros(1 + S)])[None])
    bnds = list(bounds) + [(None, None)] + [(0, None)] * S
    res = linprog(c, A_ub=sparse.vstack(blocks).tocsr(), b_ub=np.concatenate(b), A_eq=Aeq, b_eq=[1],
                  bounds=bnds, method="highs")
    if not res.success:
        return np.full(n, 1 / n)
    w = np.clip(res.x[:n], 0, None); return w / w.sum()

def cap_cvar(w, R, bounds, groups, cap, alpha=0.95):
    """If the CVaR breaches the cap, move the portfolio towards the minimum-CVaR one just enough."""
    if cap is None or cvar(w, R, alpha) >= cap:
        return w
    w0 = min_cvar(R, bounds, groups, alpha)
    if cvar(w0, R, alpha) < cap:
        return w0                                   # cap unreachable: best possible
    lo, hi = 0.0, 1.0                               # lam = share of the original portfolio
    for _ in range(25):
        mid = (lo + hi) / 2
        if cvar(mid * w + (1 - mid) * w0, R, alpha) >= cap: lo = mid
        else: hi = mid
    return lo * w + (1 - lo) * w0
