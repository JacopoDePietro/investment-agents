"""Stochastic-process backend: an external CFLib-compatible library if configured, otherwise the built-in sde.py."""
import logging, sys
import warnings
import numpy as np
warnings.filterwarnings("ignore", message=".*where.*used without.*out")

if not hasattr(np, "float"):      # some older libraries use np.float, removed in numpy >= 1.24
    np.float = float

_WARNED = []

def load_cflib(path=None):
    """Returns (CIR, QT_cir_evol, jmp_normal, first_passage).
    If `path` points to a folder containing a CFLib-compatible package it is used; otherwise the
    built-in implementation in sde.py (same interfaces, written from the published algorithms)."""
    if path:
        try:
            if path not in sys.path:
                sys.path.insert(0, path)
            from CFLib.CIR import CIR, QT_cir_evol
            from CFLib.jumps import jmp_normal
            from CFLib.BS_first_passage import first_passage
            return CIR, QT_cir_evol, jmp_normal, first_passage
        except ImportError:
            if not _WARNED:
                logging.getLogger("modeling_agent").warning("External library not found in %s: using built-in sde.py", path)
                _WARNED.append(1)
    from . import sde
    return sde.CIR, sde.QT_cir_evol, sde.jmp_normal, sde.first_passage

def heston_logret(V, I, xi, cir, rho, Dt):
    """Heston log-price step given the simulated variance path (Broadie-Kaya style conditional step):
        X = -DI/2 + (rho/eta) * (nu_n - nu_{n-1} - k*(theta*Dt - DI)) + sqrt((1-rho^2) DI) * xi
    vectorised over all paths.
    V, I: varianza e varianza integrata (L, N) da QT_cir_evol; xi: normali (L-1, N).
    Returns the per-step log-returns (L-1, N), without drift."""
    k, th, eta = cir.kappa, cir.theta, cir.sigma
    DI = np.diff(I, axis=0)
    dV = np.diff(V, axis=0)
    return -0.5 * DI + (rho / eta) * (dV - k * (th * Dt - DI)) + np.sqrt(np.maximum((1 - rho * rho) * DI, 0)) * xi
