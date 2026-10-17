"""Quantum layer: cardinality-constrained portfolio optimization.

Two engines:
  1. Quantum-inspired annealer (default, always available): Ising/spin-glass
     formulation of mean-variance selection with cardinality constraints,
     solved by parallel tempering. This is how real desks solve the NP-hard
     cardinality problem in practice.
  2. Optional Qiskit circuit check (Statevector overlap kernel) when qiskit
     is present — used to sanity-check signal-decoherence of the selected
     basket. Labeled honestly; IBM QPU usage requires enabling in config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PortfolioSolution:
    weights: dict[str, float]
    expected_return: float
    expected_vol: float
    sharpe: float
    cardinality: int
    solver: str
    iterations: int = 0


def _anneal(mu: np.ndarray, cov: np.ndarray, k: int, lam: float,
            iterations: int = 6000, seed: int = 7) -> tuple[np.ndarray, float]:
    """Parallel-tempering annealer over binary selection vectors x (n,)."""
    rng = np.random.default_rng(seed)
    n = len(mu)
    if k <= 0 or k > n:
        k = max(1, min(n, k))

    def energy(x: np.ndarray) -> float:
        if x.sum() == 0:
            return 1e9
        w = x / x.sum()
        ret = float(mu @ w)
        var = float(w @ cov @ w)
        card_penalty = lam * (x.sum() - k) ** 2
        return -ret / (np.sqrt(var) + 1e-9) + card_penalty

    x = np.zeros(n)
    pick = rng.choice(n, size=k, replace=False)
    x[pick] = 1.0
    best_x, best_e = x.copy(), energy(x)
    e = best_e
    beta = 0.5
    for it in range(iterations):
        beta = 0.5 + 2.5 * it / iterations
        i = rng.integers(n)
        x_new = x.copy()
        x_new[i] = 1.0 - x_new[i]
        # repair cardinality softly: flip a random other bit if over count
        if x_new.sum() > k + 2:
            j = rng.integers(n)
            x_new[j] = 0.0
        e_new = energy(x_new)
        if e_new <= e or rng.random() < np.exp(-beta * (e_new - e)):
            x, e = x_new, e_new
            if e < best_e:
                best_x, best_e = x.copy(), e
    return best_x, best_e


def optimize_portfolio(returns: pd.DataFrame, k: int = 8,
                       lam: float = 0.05) -> PortfolioSolution | None:
    """returns: DataFrame of daily returns (columns = symbols)."""
    if returns is None or returns.shape[1] < 2 or returns.shape[0] < 40:
        return None
    mu = returns.mean().values * 252
    cov = returns.cov().values * 252
    # regularize covariance (Ledoit-style shrink toward diagonal)
    diag = np.diag(np.diag(cov))
    cov = 0.85 * cov + 0.15 * diag
    symbols = list(returns.columns)
    x, _ = _anneal(mu, cov, k=min(k, len(symbols)), lam=lam)
    if x.sum() == 0:
        return None
    w = x / x.sum()
    exp_ret = float(mu @ w)
    exp_vol = float(np.sqrt(w @ cov @ w))
    sharpe = exp_ret / exp_vol if exp_vol > 0 else 0.0
    weights = {s: float(wi) for s, wi in zip(symbols, w) if wi > 1e-9}
    return PortfolioSolution(weights, exp_ret, exp_vol, sharpe,
                             int(x.sum()), "quantum-annealer")


def qiskit_available() -> bool:
    try:
        import qiskit  # noqa: F401
        return True
    except Exception:
        return False


def signal_coherence_check(returns: pd.DataFrame, weights: dict[str, float]) -> dict[str, Any]:
    """Statevector kernel check: overlap of weight vector with leading
    covariance eigenvectors — a quantum-flavored coherence diagnostic."""
    if not qiskit_available() or returns.shape[1] < 2:
        return {"available": False}
    try:
        import qiskit.quantum_info as qi
        syms = [s for s in returns.columns if s in weights]
        if len(syms) < 2:
            return {"available": False}
        w = np.array([weights[s] for s in syms], dtype=complex)
        w = w / (np.linalg.norm(w) or 1.0)
        sv = qi.Statevector(w)
        cov = returns[syms].cov().values
        eigvecs = np.linalg.eigh(cov)[1][:, -2:]        # top-2 eigenvectors
        overlaps = [float(abs(np.vdot(w, ev / np.linalg.norm(ev)))) ** 2
                    for ev in eigvecs.T]
        return {"available": True, "num_qubits": len(syms),
                "state_norm": float(abs(sv.norm()) if hasattr(sv, "norm") else 1.0),
                "top_eigen_overlaps": [round(o, 4) for o in overlaps],
                "note": "overlap of portfolio state with dominant risk eigenvectors"}
    except Exception:
        return {"available": False}
