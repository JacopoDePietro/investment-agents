"""Built-in stochastic-process toolkit used when no external library is configured.

Implements, from the published algorithms, the same interfaces the modeling agent expects:
- CIR process parameters (kappa, sigma, theta, initial value);
- QT_cir_evol: Quadratic-Exponential (QE) scheme of L. Andersen (2008), "Simple and efficient
  simulation of the Heston stochastic volatility model", for the CIR variance and its time integral;
- jmp_normal: compound-Poisson jumps with normal log-jump sizes (Merton 1976) and their compensator;
- first_passage: probability that a geometric Brownian motion hits a lower barrier before t.
"""
from math import exp, log, sqrt
import numpy as np
from scipy.stats import norm


class CIR:
    def __init__(self, kappa, sigma, theta, ro):
        self.kappa, self.sigma, self.theta, self.ro = kappa, sigma, theta, ro


def QT_cir_evol(rand, cir, L, dt, Nt, DT, N, psi_c=1.5):
    """Simulates N CIR paths with L steps of size dt using the QE scheme.
    Returns (V, I) sampled every DT on Nt+1 points: variance and integrated variance, shape (Nt+1, N).
    `rand` must provide normal(size=(L, N)) and uniform(size=N) (one uniform draw per step)."""
    k, s, th = cir.kappa, cir.sigma, cir.theta
    e = exp(-k * dt)
    Z = rand.normal(loc=0.0, scale=1.0, size=(L, N))
    V = np.empty((L + 1, N)); I = np.zeros((L + 1, N))
    V[0] = cir.ro
    for n in range(L):
        v = V[n]
        m = th + (v - th) * e
        s2 = v * s * s * e * (1 - e) / k + th * s * s * (1 - e) ** 2 / (2 * k)
        m_safe = np.maximum(m, 1e-12)
        psi = s2 / m_safe ** 2
        U = rand.uniform(low=0.0, high=1.0, size=N)
        quad = psi <= psi_c
        # quadratic branch: V' = a (b + Z)^2
        inv = 2.0 / np.where(quad, psi, 1.0)
        b2 = np.where(quad, inv - 1 + np.sqrt(np.maximum(inv * (inv - 1), 0)), 0.0)
        a = m_safe / (1 + b2)
        v_quad = a * (np.sqrt(b2) + Z[n]) ** 2
        # exponential branch: point mass at zero with probability p, exponential tail otherwise
        p = np.where(quad, 0.0, (psi - 1) / (psi + 1))
        beta = (1 - p) / m_safe
        v_exp = np.where(U <= p, 0.0, np.log(np.maximum((1 - p) / np.maximum(1 - U, 1e-300), 1.0)) / beta)
        V[n + 1] = np.where(quad, v_quad, v_exp)
        I[n + 1] = I[n] + 0.5 * dt * (V[n] + V[n + 1])
    idx = [min(int(round(j * DT / dt)), L) for j in range(Nt + 1)]
    return V[idx], I[idx]


class jmp_normal:
    """Merton jumps: N ~ Poisson(lambda * dt) jumps per step, each with log-size ~ Normal(m, eta)."""
    def __init__(self, lmbda, sigma=0.0, m=0.0, eta=0.0):
        self._lambda, self._sigma, self._m, self._eta = lmbda, sigma, m, eta

    @property
    def intensity(self):
        return self._lambda

    def do_jmp(self, rand, Dt, J):
        n = rand.poisson(lam=self._lambda * Dt, size=J)
        return n * self._m + np.sqrt(n) * self._eta * rand.normal(0.0, 1.0, size=J)

    def compensator(self):
        """1 - E[exp(jump)]: added as lambda * dt * compensator so prices stay martingales."""
        return 1.0 - exp(self._m + 0.5 * self._eta ** 2)


def first_passage(So, t, r, sigma, Bl):
    """P(min_{s<=t} S_s <= Bl) for dS = r S dt + sigma S dW, S_0 = So > Bl."""
    if t < 1e-3:
        return 0.0
    mu = r - 0.5 * sigma * sigma
    x = log(Bl / So)
    sd = sigma * sqrt(t)
    return float(norm.cdf((x - mu * t) / sd) + exp(2 * mu * x / sigma ** 2) * norm.cdf((x + mu * t) / sd))
