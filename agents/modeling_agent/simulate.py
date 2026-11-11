"""Batched Monte Carlo: correlated Heston factors (QE scheme) with regimes, CIR yield, Merton jumps."""
import logging, os, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .cflib_bridge import heston_logret, load_cflib

log = logging.getLogger("modeling_agent")

class _FeedRand:
    """Random generator that feeds QT_cir_evol normals already correlated across factors (uniforms from the real generator)."""
    def __init__(self, z, rng): self.z, self.rng = z, rng
    def normal(self, loc=0.0, scale=1.0, size=None): return self.z
    def uniform(self, low=0.0, high=1.0, size=None): return self.rng.uniform(low, high, size)

def _corr_mul(ch, z):
    """Applies the Cholesky factor to z (F, L, N) with one matrix multiplication (BLAS)."""
    F = z.shape[0]
    return (ch @ z.reshape(F, -1)).reshape(z.shape)

def simulate(cal, bp, idio_p, mcfg, years, CIR, QT, jmp_normal, n_paths, batch, seed):
    """Splits the paths into batches and simulates them in parallel processes."""
    L = int(years * mcfg["steps_per_year"])
    batch = max(5000, int(batch * 52 / L))                           # smaller batches on longer horizons
    sizes = [min(batch, n_paths - i) for i in range(0, n_paths, batch)]
    workers = min(mcfg.get("workers", os.cpu_count() // 2), len(sizes))
    seeds = np.random.SeedSequence(seed).spawn(len(sizes))
    args = [(cal, bp, idio_p, mcfg, years, n, sq) for n, sq in zip(sizes, seeds)]
    t0 = time.time(); parts = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, part in enumerate(ex.map(_batch, args)):
            parts.append(part)
            if (i + 1) % max(1, len(sizes) // 5) == 0 or i + 1 == len(sizes):
                log.info("Horizon %d years: %d/%d paths (%.0fs, %d processes)",
                         years, sum(sizes[:i + 1]), n_paths, time.time() - t0, workers)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}, time.time() - t0

def _batch(a):
    cal, bp, idio_p, mcfg, years, N, seq = a
    CIR, QT, jmp_normal, _ = load_cflib(mcfg["cflib_path"])
    facs = cal["factors"]; F = len(facs)
    L = int(years * mcfg["steps_per_year"]); dt = 1 / mcfg["steps_per_year"]
    ch = {g: np.linalg.cholesky(cal["corr"][g]) for g in (0, 1)}
    chv = np.linalg.cholesky(cal["vol_corr"])
    P = cal["trans"]
    cirs = {f: CIR(kappa=p["kappa"], sigma=p["sigma"], theta=p["theta"], ro=p["v0"]) for f, p in cal["params"].items()}
    r = cal["rate"]; cir_y = CIR(kappa=r["kappa"], sigma=r["sigma"], theta=r["theta"], ro=r["y0"])
    jcfg = mcfg["jumps"]; jmp = jmp_normal(lmbda=jcfg["intensity"], sigma=0.0, m=jcfg["mean"], eta=jcfg["std"])
    bvec = np.array([bp.get(f, 0.0) for f in facs]); b_rate = bp.get("RATE", 0.0)
    out = {k: [] for k in ("port", "spy", "mdd", "mdd_spy", "vol", "min_w")}
    rng = np.random.default_rng(seq)
    if True:
        # regime per path (weekly Markov chain)
        reg = np.empty((L, N), dtype=np.int8); reg[0] = cal["crisis_now"]
        u = rng.random((L, N))
        for n in range(1, L):
            p_crisis = np.where(reg[n - 1] == 1, P[1, 1], P[0, 1])
            reg[n] = (u[n] < p_crisis)
        zp = rng.standard_normal((F + 1, L, N))
        xi = np.where(reg[None] == 1, _corr_mul(ch[1], zp), _corr_mul(ch[0], zp)); del zp
        zv = _corr_mul(chv, rng.standard_normal((F + 1, L, N)))
        port = np.zeros((L, N)); spy = None
        for i, f in enumerate(facs):
            p = cal["params"][f]
            V, I = QT(_FeedRand(zv[i], rng), cirs[f], L, dt, L, dt, N)
            X = heston_logret(V, I, xi[i], cirs[f], p["rho"], dt) + p["mu"] * dt
            if f in jcfg["equity_factors"]:
                X += np.stack([jmp.do_jmp(rng, dt, N) for _ in range(L)]) + dt * jmp.intensity * jmp.compensator()
            R = np.expm1(X)
            port += bvec[i] * R
            if f == "SPY": spy = R
            del V, I, X
        Y, _ = QT(_FeedRand(zv[F], rng), cir_y, L, dt, L, dt, N)
        port += b_rate * np.diff(Y, axis=0)
        port += idio_p * rng.standard_normal((L, N))
        port = np.maximum(port, -0.99)
        W = np.cumprod(1 + port, axis=0); S = np.cumprod(1 + spy, axis=0)
        out["port"].append(W[-1] - 1); out["spy"].append(S[-1] - 1)
        out["mdd"].append((W / np.maximum.accumulate(np.vstack([np.ones(N), W]), axis=0)[1:] - 1).min(0))
        out["mdd_spy"].append((S / np.maximum.accumulate(np.vstack([np.ones(N), S]), axis=0)[1:] - 1).min(0))
        out["vol"].append(np.log1p(port).std(0) * np.sqrt(mcfg["steps_per_year"]))
        out["min_w"].append(np.minimum(W.min(0), 1.0))
    return {k: np.concatenate(v) for k, v in out.items()}


def factor_terminal(cal, mcfg, weeks, N, seed):
    """Cumulative factor returns over `weeks` weeks (N scenarios) with the same dynamics as the
    simulazione completa: Heston QE (CFLib) correlato, regimi, salti di Merton, tasso CIR.
    Returns an (N, F+1) matrix: price factors and change of the 10-year yield."""
    CIR, QT, jmp_normal, _ = load_cflib(mcfg["cflib_path"])
    facs = cal["factors"]; F = len(facs); L = weeks; dt = 1 / mcfg["steps_per_year"]
    rng = np.random.default_rng(seed)
    ch = {g: np.linalg.cholesky(cal["corr"][g]) for g in (0, 1)}
    chv = np.linalg.cholesky(cal["vol_corr"]); P = cal["trans"]
    jc = mcfg["jumps"]; jmp = jmp_normal(lmbda=jc["intensity"], sigma=0.0, m=jc["mean"], eta=jc["std"])
    reg = np.empty((L, N), dtype=np.int8); reg[0] = cal["crisis_now"]
    u = rng.random((L, N))
    for n in range(1, L):
        reg[n] = u[n] < np.where(reg[n - 1] == 1, P[1, 1], P[0, 1])
    zp = rng.standard_normal((F + 1, L, N))
    xi = np.where(reg[None] == 1, _corr_mul(ch[1], zp), _corr_mul(ch[0], zp))
    zv = _corr_mul(chv, rng.standard_normal((F + 1, L, N)))
    out = np.empty((N, F + 1))
    for i, f in enumerate(facs):
        p = cal["params"][f]
        cir = CIR(kappa=p["kappa"], sigma=p["sigma"], theta=p["theta"], ro=p["v0"])
        V, I = QT(_FeedRand(zv[i], rng), cir, L, dt, L, dt, N)
        X = heston_logret(V, I, xi[i], cir, p["rho"], dt) + p["mu"] * dt
        if f in jc["equity_factors"]:
            X += np.stack([jmp.do_jmp(rng, dt, N) for _ in range(L)]) + dt * jmp.intensity * jmp.compensator()
        out[:, i] = np.expm1(X.sum(0))
    r = cal["rate"]; cy = CIR(kappa=r["kappa"], sigma=r["sigma"], theta=r["theta"], ro=r["y0"])
    Y, _ = QT(_FeedRand(zv[F], rng), cy, L, dt, L, dt, N)
    out[:, F] = Y[-1] - Y[0]
    return out
