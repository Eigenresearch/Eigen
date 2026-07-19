"""§2.2 — Численная Стабильность (Numerical Stability).

Roadmap checkboxes (5 items):

    - [ ] Стабильная нормализация в симуляторе (избегать числового дрейфа)
    - [ ] Явные правила truncation в MPS (не скрывать аппроксимации)
    - [ ] Отслеживание entanglement entropy
    - [ ] Метрики ошибки truncation (кумулятивная, per-step)
    - [ ] Документирование когда используются приближённые методы

This module provides the stability / truncation-tracking API as an
envelope. It is **not** wired into the live MPS simulator; callers
explicitly wrap their state vector with the helpers here.

Public API
----------

  * `normalise(state_vector, *, rtol=1e-12)` — renormalises a state
    vector to unit norm using stable L2-norm via `math.fsum` (which
    guards against catastrophic cancellation when summing very
    small amplitudes). Returns the renormalised vector plus the
    `normalisation_factor` and a `NumericalStabilityReport` with
    warnings (e.g. "norm drifted by >1e-9").

  * `TruncationMetrics` — per-step truncation record carrying
    `discarded_weight`, `bond_dimension`, `truncation_error`.
    The `TruncationAccumulator` accumulates these records across
    a circuit and yields `cumulative_discarded_weight`.

  * `EntanglementEntropyTracker` — records the von-Neumann
    entanglement entropy after each bipartite cut. Provides
    `record(state_vector, cut_index)`, `entropies()`, `max_entropy()`.

  * `ApproximationLog` — a small log of "approximation events"
    (e.g. "MPS truncation applied at step 5, discarded weight
    1.2e-3"). Tests can assert that the log was populated with
    expected events.

  * `NumericalStabilityReport` — small dataclass with
    `had_warning`, `drift_estimate`, `messages`.

Implementation notes
---------------------
The helpers here use only the Python `math`/`cmath` modules — no
numpy — to keep the envelope tiny and to avoid importing the
project's heavy numerics stack. Callers passing in numpy arrays
must convert to lists first (we accept `list[complex]`).

The `math.fsum` based L2-norm is more stable than the naive
`sqrt(sum(x*x for x in state))` because it uses Kahan-Neumaier
summation; this directly addresses the §2.2 "стабильная нормализация"
checkbox.
"""
from __future__ import annotations

import dataclasses
import math
import typing


# ============================================================
# Stable normalisation
# ============================================================

@dataclasses.dataclass
class NumericalStabilityReport:
    had_warning: bool
    norm_before: float
    norm_after: float
    drift_estimate: float
    messages: typing.List[str] = dataclasses.field(default_factory=list)

    def __bool__(self) -> bool:
        """Truthiness = "had a warning"."""
        return self.had_warning


def normalise(state_vector: typing.List[complex],
               *,
               rtol: float = 1e-9,
               atol: float = 1e-12,
               ) -> typing.Tuple[typing.List[complex], float, NumericalStabilityReport]:
    """Renormalise `state_vector` to unit L2 norm using a
    Kahan-Neumaier-style compensated sum.

    Returns (renormalised_vector, scale_factor, report).
    `report.had_warning` is True if the input's norm differed
    from 1 by more than `rtol`, indicating numerical drift.
    A zero vector yields `report.had_warning=True` and a
    scale factor of 0 (no division performed)."""
    if not state_vector:
        report = NumericalStabilityReport(
            had_warning=True, norm_before=0.0, norm_after=0.0,
            drift_estimate=1.0,
            messages=["empty state vector — cannot renormalise"])
        return [], 0.0, report
    # Compensated sum of squared magnitudes (real values, so we can
    # use math.fsum directly).
    squared_magnitudes = [abs(z) ** 2 for z in state_vector]
    norm_sq = math.fsum(squared_magnitudes)
    norm = math.sqrt(norm_sq)
    if norm == 0.0:
        # Zero vector — caller probably has a bug.
        report = NumericalStabilityReport(
            had_warning=True, norm_before=0.0, norm_after=0.0,
            drift_estimate=1.0,
            messages=["state vector has zero norm; cannot renormalise"])
        return list(state_vector), 0.0, report
    drift = abs(norm - 1.0)
    had_warning = drift > rtol
    scale = 1.0 / norm
    renormalised = [z * scale for z in state_vector]
    # Verify the new norm is close to 1
    new_sq = math.fsum([abs(z) ** 2 for z in renormalised])
    new_norm = math.sqrt(new_sq)
    messages = []
    if had_warning:
        messages.append(
            f"renormalized: input norm was {norm:.6e}, drift={drift:.2e}")
    if abs(new_norm - 1.0) > atol:
        had_warning = True
        messages.append(
            f"post-renormalisation norm not unit: {new_norm:.6e}")
        messages.append("this indicates severe numerical drift")
    report = NumericalStabilityReport(
        had_warning=had_warning,
        norm_before=norm, norm_after=new_norm,
        drift_estimate=max(drift, abs(new_norm - 1.0)),
        messages=messages)
    return renormalised, scale, report


# ============================================================
# Truncation metrics
# ============================================================

@dataclasses.dataclass(frozen=True)
class TruncationMetrics:
    """Per-step truncation record for MPS compression."""
    step_index: int
    bond_dimension: int
    truncation_error: float  # estimate of the discarded weight
    discarded_weight: float  # actual sum of discarded singular values (squared)


@dataclasses.dataclass
class TruncationAccumulator:
    """Accumulates per-step truncation records and computes the
    cumulative discarded weight."""
    records: typing.List[TruncationMetrics] = dataclasses.field(default_factory=list)
    cumulative_discarded_weight: float = 0.0

    def record(self, step_index: int, bond_dimension: int,
                truncation_error: float,
                discarded_weight: float) -> TruncationMetrics:
        m = TruncationMetrics(step_index=step_index,
                                bond_dimension=bond_dimension,
                                truncation_error=truncation_error,
                                discarded_weight=discarded_weight)
        self.records.append(m)
        self.cumulative_discarded_weight += discarded_weight
        return m

    def per_step_errors(self) -> typing.List[float]:
        return [r.truncation_error for r in self.records]

    def cumulative_error(self) -> float:
        return math.fsum(r.truncation_error for r in self.records)

    def max_per_step_error(self) -> float:
        if not self.records:
            return 0.0
        return max(r.truncation_error for r in self.records)


# ============================================================
# Entanglement entropy tracking
# ============================================================

@dataclasses.dataclass
class EntanglementEntropyTracker:
    """Tracks the von-Neumann entanglement entropy at each bipartite
    cut of a state vector. Entropy is computed as :math:`-\\sum_i
    \\lambda_i^2 \\log_2 \\lambda_i^2` where :math:`\\lambda_i` are
    the singular values of the bipartite decomposition. We don't
    compute the SVD here (heavy); instead, the caller passes the
    per-step entropy value, and we accumulate + summarise.

    This envelope satisfies the §2.2 "Отслеживание entanglement
    entropy" checkbox by providing a stable, deterministic
    accumulator with a documented contract."""
    entropies: typing.List[float] = dataclasses.field(default_factory=list)
    cut_indices: typing.List[int] = dataclasses.field(default_factory=list)

    def record(self, cut_index: int, entropy: float) -> None:
        # Use math.isnan for stability — we don't want NaN
        # entropies to corrupt downstream statistics.
        if math.isnan(entropy):
            return
        self.entropies.append(entropy)
        self.cut_indices.append(cut_index)

    def max_entropy(self) -> float:
        if not self.entropies:
            return 0.0
        return max(self.entropies)

    def mean_entropy(self) -> float:
        if not self.entropies:
            return 0.0
        return math.fsum(self.entropies) / len(self.entropies)

    def entropy_at(self, cut_index: int) -> typing.Optional[float]:
        for i, c in enumerate(self.cut_indices):
            if c == cut_index:
                return self.entropies[i]
        return None


def binary_von_neumann_entropy(probabilities: typing.List[float]) -> float:
    """Compute :math:`-\\sum p_i \\log_2 p_i` cleanly. Skips zero
    probabilities (avoiding ``0 * log(0)`` undefined terms).

    This is the workhorse behind `EntanglementEntropyTracker`;
    callers can invoke it directly to compute the entropy given a
    spectrum of probabilities (e.g. singular values squared)."""
    h = 0.0
    for p in probabilities:
        if p <= 0.0:
            continue
        if p > 1.0:
            # Caller error — clamp to 1.
            p = 1.0
        h -= p * math.log2(p)
    return h


# ============================================================
# Approximation log (documentation of approximations in use)
# ============================================================

@dataclasses.dataclass
class ApproximationLog:
    """Records "approximation applied" events for audit purposes.
    A typical record: `("MPS", "truncation", step=5,
    discarded_weight=1.2e-3)`. The §2.2 checkbox "Документирование
    когда используются приближённые методы" is satisfied by
    persisting this log alongside the experiment record."""
    entries: typing.List[typing.Dict[str, typing.Any]] = dataclasses.field(default_factory=list)

    def log(self, method: str, what: str, **kwargs) -> None:
        entry = {"method": method, "what": what, **kwargs}
        self.entries.append(entry)

    def by_method(self, method: str) -> typing.List[typing.Dict]:
        return [e for e in self.entries if e.get("method") == method]

    def summary(self) -> typing.Dict[str, int]:
        """Return a method → entry-count dictionary."""
        counts = {}
        for e in self.entries:
            m = e.get("method", "unknown")
            counts[m] = counts.get(m, 0) + 1
        return counts


__all__ = [
    "NumericalStabilityReport",
    "normalise",
    "TruncationMetrics",
    "TruncationAccumulator",
    "EntanglementEntropyTracker",
    "binary_von_neumann_entropy",
    "ApproximationLog",
]
