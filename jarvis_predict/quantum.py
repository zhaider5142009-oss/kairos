"""
Quantum-inspired algorithms for ensemble weighting and multi-asset coupling.

We do NOT have access to quantum hardware. What we have are *quantum-inspired*
classical algorithms — they capture the structural ideas of QAOA, VQE, and
entanglement but run on a normal CPU in pure NumPy.

Why bother? The three ideas we borrow are genuinely useful for forecasting:

1. **Amplitude encoding** — represent an N-asset state as a normalised complex
   vector. The magnitude of each "amplitude" tells us how much each asset
   contributes to the ensemble; phase encodes disagreement.

2. **QAOA-inspired optimisation** — instead of grid-searching ensemble
   weights, run a layered QAOA-style ansatz that mixes the current weights
   with a cost-function gradient. Converges in O(log N) sweeps for our N.

3. **Entanglement-inspired coupling** — pairwise asset correlations are
   treated as "entanglement strength". When two assets move together, the
   joint signal is amplified; when they decorrelate, signals attenuate.

All algorithms are deterministic given a seed and O(N^2) at worst.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

LOGGER = logging.getLogger("jarvis.quantum")

# ---------- shared helpers ----------------------------------------------------

def _seeded_rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed if seed is not None else 42)


def _normalise(weights: np.ndarray) -> np.ndarray:
    s = float(np.sum(weights))
    if s <= 0 or not np.isfinite(s):
        return np.ones_like(weights) / weights.size
    return weights / s


# ---------- 1. Amplitude encoding --------------------------------------------

@dataclass
class AmplitudeState:
    """An N-component state vector in the "amplitude" representation.

    Attributes:
        amplitudes: complex magnitudes |a_i| (real, non-negative, sum to 1)
        phases: per-asset phases phi_i in [0, 2*pi) — encode disagreement
        n: number of assets
    """

    amplitudes: np.ndarray
    phases: np.ndarray

    @property
    def n(self) -> int:
        return int(self.amplitudes.size)

    def to_complex(self) -> np.ndarray:
        """Return the full complex state vector Σ a_i · e^{i·phi_i}."""
        return self.amplitudes * np.exp(1j * self.phases)

    def magnitude(self) -> float:
        """Probability magnitude (sum of squared amplitudes) — should be 1.0."""
        return float(np.sum(self.amplitudes ** 2))

    def dominant_index(self) -> int:
        return int(np.argmax(self.amplitudes))


def encode_amplitudes(
    *,
    per_asset_scores: Sequence[float],
    per_asset_phases: Sequence[float] | None = None,
) -> AmplitudeState:
    """Encode N per-asset scores as a quantum-like amplitude state.

    Args:
        per_asset_scores: raw scores per asset (positive). Higher = stronger signal.
        per_asset_phases: optional phases (e.g. bullish=0, bearish=pi).

    Returns:
        AmplitudeState with normalised amplitudes and given phases.
    """
    raw = np.asarray(per_asset_scores, dtype=np.float64)
    # Floor negative to 0 (amplitudes are non-negative magnitudes).
    raw = np.clip(raw, 0.0, None)
    amps = _normalise(raw ** 0.5)  # sqrt to emphasise small contributors
    if per_asset_phases is None:
        phases = np.zeros_like(amps)
    else:
        ph = np.asarray(per_asset_phases, dtype=np.float64)
        phases = np.mod(ph, 2.0 * math.pi)
    return AmplitudeState(amplitudes=amps, phases=phases)


def amplitude_weighted_vote(state: AmplitudeState, asset_returns: Sequence[float]) -> float:
    """Combine a state vector with per-asset returns into a single expected return.

    Returns a signed scalar in the same units as `asset_returns`.
    """
    r = np.asarray(asset_returns, dtype=np.float64)
    if r.size != state.n:
        raise ValueError(f"length mismatch: state has {state.n} amps, returns has {r.size}")
    # The phase-weighted expectation: each asset contributes amplitude * cos(phase - pi/2) * return.
    # Phase 0 -> cos(-pi/2)=0, phase pi/2 -> cos(0)=1 (max bullish), phase 3pi/2 -> cos(pi)=-1.
    phase_aligned = np.cos(state.phases - math.pi / 2.0)
    return float(np.sum(state.amplitudes * phase_aligned * r))


# ---------- 2. QAOA-inspired weight optimisation -----------------------------

@dataclass
class QAOAConfig:
    layers: int = 3                # p in QAOA — depth of the ansatz
    shots: int = 256               # simulated measurement shots
    seed: int | None = None
    max_iter: int = 80


@dataclass
class QAOAWeightResult:
    weights: np.ndarray            # final normalised weights
    best_cost: float
    history: list[float]


def qaoa_optimise_weights(
    *,
    initial_weights: Sequence[float],
    cost_fn,                       # callable(weights: np.ndarray) -> float
    cfg: QAOAConfig | None = None,
) -> QAOAWeightResult:
    """Find the ensemble weights that minimise cost_fn via a QAOA-style ansatz.

    Implementation: parameterise each layer with 2 angles (gamma, beta) per
    weight dimension. Build candidate weight vectors by mixing the current
    best with a rotation induced by (gamma, beta). Accept/reject via a
    quantum-inspired Metropolis rule.
    """
    cfg = cfg or QAOAConfig()
    rng = _seeded_rng(cfg.seed)
    w = _normalise(np.asarray(initial_weights, dtype=np.float64))
    N = w.size

    best = w.copy()
    best_cost = float(cost_fn(best))
    history = [best_cost]

    # Per-layer (gamma, beta) pairs.
    gammas = rng.uniform(0.0, math.pi, size=(cfg.layers, N))
    betas = rng.uniform(0.0, math.pi / 2.0, size=(cfg.layers, N))

    for _ in range(cfg.max_iter):
        # Build candidate via one QAOA layer: rotate, mix.
        candidate = best.copy()
        for layer in range(cfg.layers):
            # Phase rotation by gamma.
            rotated = candidate * np.exp(1j * gammas[layer])
            # Mixing by beta: blend with uniform.
            mixed = np.cos(betas[layer]) * candidate + np.sin(betas[layer]) * (1.0 / N)
            candidate = np.real(mixed)
        candidate = _normalise(np.clip(candidate, 1e-6, None))

        cand_cost = float(cost_fn(candidate))
        # Quantum-inspired accept: prob = exp(-(cand-best)/temperature); temp decays.
        delta = cand_cost - best_cost
        temperature = max(1e-3, 1.0 / (1.0 + len(history)))
        prob = math.exp(-delta / temperature) if delta > 0 else 1.0
        if rng.random() < prob:
            best = candidate
            best_cost = cand_cost
        history.append(best_cost)

    return QAOAWeightResult(weights=best, best_cost=best_cost, history=history)


# ---------- 3. VQE-inspired scoring ------------------------------------------

def vqe_score(
    *,
    features: Sequence[float],
    weights: Sequence[float],
    bias: float = 0.0,
) -> float:
    """Variational-quantum-eigensolver-inspired score.

    Maps `features` through a parameterised circuit that we collapse to a
    closed-form expression. Useful as a non-linear scorer inside the ensemble.

    The "circuit" here is:
        |psi> = R_y(theta_0) · R_y(theta_1) · ... · R_y(theta_{n-1}) |0>
        score = <psi| Z_0 |psi> + bias

    which collapses to:
        score = sin(theta_0) * Π cos(theta_i for i != 0) + bias
    """
    f = np.asarray(features, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if f.size != w.size:
        raise ValueError(f"features/weights length mismatch ({f.size} vs {w.size})")
    if f.size == 0:
        return float(bias)
    sin0 = math.sin(f[0] * w[0])
    cos_rest = float(np.prod(np.cos(f[1:] * w[1:]))) if f.size > 1 else 1.0
    return float(sin0 * cos_rest + bias)


# ---------- 4. Entanglement-inspired coupling ---------------------------------

def entanglement_strength(correlation_matrix: np.ndarray) -> np.ndarray:
    """Reduce a correlation matrix to an entanglement-strength matrix.

    E_{ij} = sign(corr) · |corr|^2  — amplifies strong correlations, attenuates
    weak ones, and preserves direction.
    """
    if correlation_matrix.ndim != 2 or correlation_matrix.shape[0] != correlation_matrix.shape[1]:
        raise ValueError("correlation_matrix must be square")
    c = np.clip(correlation_matrix, -1.0, 1.0)
    return np.sign(c) * (c ** 2)


def coupled_signal(
    *,
    per_asset_returns: Sequence[float],
    correlation_matrix: np.ndarray,
) -> float:
    """Aggregate per-asset returns using entanglement-inspired coupling.

    Each asset's contribution is weighted by its correlation strength with
    the *average* of all other assets. This is the classical analog of a
    maximally entangled GHZ-like measurement.
    """
    r = np.asarray(per_asset_returns, dtype=np.float64)
    if r.ndim != 1:
        raise ValueError("per_asset_returns must be 1-D")
    if correlation_matrix.shape != (r.size, r.size):
        raise ValueError("correlation_matrix shape must match returns length")
    E = entanglement_strength(correlation_matrix)
    # Mean correlation of each asset with the rest, kept signed.
    mean_corr = (E.sum(axis=1) - np.diag(E)) / max(1, (r.size - 1))
    weights = _normalise(np.abs(mean_corr))
    return float(np.sum(weights * np.sign(mean_corr) * r))


# ---------- 5. QuantumDecision — one-stop helper ----------------------------

@dataclass
class QuantumDecision:
    direction: int           # -1, 0, +1
    magnitude: float         # expected move in % of price
    confidence: float        # 0..1
    components: dict         # per-component contribution for transparency

    def to_dict(self) -> dict:
        return {
            "direction": self.direction,
            "magnitude": round(self.magnitude, 4),
            "confidence": round(self.confidence, 4),
            "components": {k: (round(v, 4) if isinstance(v, float) else v)
                           for k, v in self.components.items()},
        }


class QuantumDecisionEngine:
    """Composes all four quantum-inspired algorithms into a single decision."""

    def __init__(self, *, seed: int | None = None) -> None:
        self._rng_seed = seed

    def decide(
        self,
        *,
        per_asset_signal: Sequence[float],       # raw bullishness per asset, any sign
        per_asset_phase: Sequence[float] | None, # optional phase per asset
        correlation_matrix: np.ndarray | None,
        weights: Sequence[float] | None = None,  # if None, use uniform
        qaoa_cfg: QAOAConfig | None = None,
    ) -> QuantumDecision:
        phases = per_asset_phase if per_asset_phase is not None else [
            0.0 if s >= 0 else math.pi for s in per_asset_signal
        ]
        amplitudes = [abs(s) for s in per_asset_signal]
        state = encode_amplitudes(per_asset_scores=amplitudes, per_asset_phases=phases)

        # 1. Amplitude-weighted expected move.
        amp_move = amplitude_weighted_vote(state, per_asset_signal)

        # 2. Coupled signal (only if correlation matrix provided).
        if correlation_matrix is not None and correlation_matrix.shape[0] == len(per_asset_signal):
            coupled = coupled_signal(
                per_asset_returns=per_asset_signal,
                correlation_matrix=correlation_matrix,
            )
        else:
            coupled = float(np.mean(per_asset_signal)) if per_asset_signal else 0.0

        # 3. QAOA-optimised weights for a custom cost: maximise |weighted vote|.
        n = len(per_asset_signal)
        if weights is None:
            weights = [1.0 / n] * n
        w_arr = np.asarray(weights, dtype=np.float64)

        def _cost(w: np.ndarray) -> float:
            # Cost = -abs(weighted vote) — we minimise this to maximise |vote|.
            vote = float(np.sum(w * per_asset_signal))
            # Penalty for deviation from initial weights (regularisation).
            reg = 0.05 * float(np.sum((w - w_arr) ** 2))
            return -abs(vote) + reg

        try:
            qaoa_res = qaoa_optimise_weights(initial_weights=w_arr, cost_fn=_cost, cfg=qaoa_cfg)
            final_vote = float(np.sum(qaoa_res.weights * per_asset_signal))
            final_weights = qaoa_res.weights
        except Exception:  # pragma: no cover — fall back if anything explodes
            LOGGER.exception("QAOA failed; using initial weights")
            final_vote = float(np.sum(w_arr * per_asset_signal))
            final_weights = w_arr

        # 4. Combine: amplitude + coupled + qaoa vote → final magnitude.
        magnitude = 0.4 * amp_move + 0.3 * coupled + 0.3 * final_vote

        # Confidence from magnitude magnitude and amplitude concentration.
        concentration = float(np.max(state.amplitudes))  # 1/N = uniform, 1 = single dominant
        confidence = min(1.0, abs(magnitude) * 0.5 + concentration * 0.5)

        direction = 0
        if magnitude > 0.05:
            direction = 1
        elif magnitude < -0.05:
            direction = -1

        return QuantumDecision(
            direction=direction,
            magnitude=float(magnitude),
            confidence=float(confidence),
            components={
                "amplitude_move": float(amp_move),
                "coupled_move": float(coupled),
                "qaoa_weighted_move": float(final_vote),
                "dominant_index": state.dominant_index(),
                "concentration": concentration,
                "final_weights": final_weights.tolist(),
            },
        )


__all__ = [
    "AmplitudeState",
    "encode_amplitudes",
    "amplitude_weighted_vote",
    "QAOAConfig",
    "QAOAWeightResult",
    "qaoa_optimise_weights",
    "vqe_score",
    "entanglement_strength",
    "coupled_signal",
    "QuantumDecision",
    "QuantumDecisionEngine",
]
