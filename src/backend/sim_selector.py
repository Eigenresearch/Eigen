"""Centralized automatic-simulator selection (sol.md §5.1 — Adaptability).

The Eigen runtime historically had three near-duplicate, ad-hoc circuits to
pick a backend simulator when ``--backend auto`` was selected:

* ``src/backend/vm.py::EigenVM.execute`` counted ``Q_ALLOC`` and 2-qubit
  ``Q_GATE`` opcodes and picked ``dense`` for ``<=16`` qubits, ``mps`` for
  low entanglement density else ``sparse``.
* ``src/runtime.py::EigenRuntime.execute`` mirrored that against the EQIR
  graph node list.
* ``src/commands/run.py`` mirrored it again against the EQIR graph when
  routing the ``run`` command.

Each copy missed the §5.1 heuristics listed in ``sol.md``:

  | property       | selected sim |
  | -------------- | ------------ |
  | Clifford-only  | stabilizer   |
  | sparse output  | sparse       |
  | low entanglmt  | mps          |
  | needs density  | density_matrix |
  | general case   | dense (state vector) |
  | (fallback)     | any → dense on resource exhaustion |

This module replaces all three duplications with a single
``SimSelector`` whose :func:`select` returns one of
``"stabilizer"``, ``"sparse"``, ``"mps"``, ``"density_matrix"``,
``"dense"``. The selector accepts:

* ``n_qubits`` — total allocated qubits
* ``n_2q_gates`` — count of two-qubit gates (entanglement proxy)
* ``n_gates`` — total gate count
* ``is_all_clifford`` — every gate is in ``CLIFFORD_GATES`` (T/RX/RY/RZ/etc. → False)
* ``noise_active`` — a stochastic noise model is configured; the program
  needs a density-matrix backend to represent mixed states faithfully
* ``max_bond_dim_hint`` — optional user-suggested MPS bond dim
* ``user_hint`` — explicit override ("``stabilizer``"/"``mps``"/...)
* ``memory_budget_bytes`` — soft budget; stabilizer is essentially free,
  dense is ``O(2ⁿ)`` complex, density_matrix ``O(4ⁿ)``.

The decision graph matches the §5.1 mermaid diagram:

    analyze ──▶ Clifford?     ── yes ─▶ stabilizer
              ├── Sparse?     ── yes ─▶ sparse
              ├── Low ent.?   ── yes ─▶ mps
              ├── Needs DM?   ── yes ─▶ density_matrix
              └── else                     statevector

Stabilizer is chosen only when the circuit is all-Clifford AND below the
stabilizer qubit cap (10 000). Statevector caps at 25 qubits by default
(``StateBackend.allocate_qubit`` already raises above 25), so for >25
qubit Clifford circuits we still pick stabilizer (which scales well beyond
statevector).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# Thresholds for resource-aware selection. These values mirror the explicit
# checks already in the codebase and turn them into a single tunable hub.
DEFAULT_STATEVECTOR_QUBIT_CAP = 25
DEFAULT_DENSITY_MATRIX_QUBIT_CAP = 16
DEFAULT_STABILIZER_QUBIT_CAP = 10000
DEFAULT_MPS_QUBIT_FLOOR = 12         # below this MPS rarely beats dense
DEFAULT_MPS_ENTANGlement_DENSITY_MAX = 1.5
DEFAULT_SPARSE_ENTANGlement_DENSITY_MAX = 2.0


@dataclass
class CircuitMetrics:
    """Aggregated circuit characteristics relevant to simulator selection."""
    n_qubits: int
    n_2q_gates: int = 0
    n_gates: int = 0
    is_all_clifford: bool = True
    noise_active: bool = False
    needs_density_matrix: bool = False  # explicit, e.g. partial-trace ops
    max_bond_dim_hint: Optional[int] = None
    has_mid_circuit_measure: bool = False
    has_classical_feedback: bool = False
    has_dynamic_decoherence: bool = False


@dataclass
class SelectionReport:
    """Returned by :meth:`SimSelector.select`. Captures the chosen backend
    plus a structured rationale so ``eigen audit`` and CLI output can show
    *why* a simulator was picked."""
    chosen: str
    reason: str
    fallback_used: bool = False
    fallback_from: Optional[str] = None
    metrics: CircuitMetrics = field(default_factory=CircuitMetrics)
    user_hint: Optional[str] = None
    would_exceed_memory: bool = False
    est_memory_bytes: int = 0

    def __str__(self) -> str:  # pragma: no cover - formatting only
        prefix = "user-hint override" if self.user_hint else "auto"
        s = f"[Auto Backend] {prefix} → '{self.chosen}' ({self.reason})"
        if self.fallback_used:
            s += f" | fallback from '{self.fallback_from}'"
        if self.would_exceed_memory:
            s += " | would exceed memory budget"
        return s


class SimSelector:
    """Encapsulates the §5.1 selection policy. Construct once, call
    :meth:`select` repeatedly. Subclass to override individual rules."""

    def __init__(
        self,
        statevector_cap: int = DEFAULT_STATEVECTOR_QUBIT_CAP,
        density_cap: int = DEFAULT_DENSITY_MATRIX_QUBIT_CAP,
        stabilizer_cap: int = DEFAULT_STABILIZER_QUBIT_CAP,
        mps_qubit_floor: int = DEFAULT_MPS_QUBIT_FLOOR,
        mps_entanglement_density_max: float = DEFAULT_MPS_ENTANGlement_DENSITY_MAX,
        sparse_entanglement_density_max: float = DEFAULT_SPARSE_ENTANGlement_DENSITY_MAX,
        statevector_memory_bytes: int = 2 * 8 * (1 << 25),  # float128 * 2^25
        density_matrix_memory_bytes: int = (2 * 8) * (1 << 30),  # 4GiB cap
    ):
        self.statevector_cap = statevector_cap
        self.density_cap = density_cap
        self.stabilizer_cap = stabilizer_cap
        self.mps_qubit_floor = mps_qubit_floor
        self.mps_entanglement_density_max = mps_entanglement_density_max
        self.sparse_entanglement_density_max = sparse_entanglement_density_max
        self.statevector_memory_bytes = statevector_memory_bytes
        self.density_matrix_memory_bytes = density_matrix_memory_bytes

    # ------------------------------------------------------------------
    # Individual decision rules. Each returns (chosen or None, reason).
    # Returning None means "rule did not fire"; the next rule gets tried.
    # ------------------------------------------------------------------

    def _rule_user_hint(self, metrics: CircuitMetrics, user_hint: Optional[str]):
        if user_hint and user_hint in {"stabilizer", "sparse", "mps",
                                      "density_matrix", "dense"}:
            return user_hint, f"user hint '{user_hint}'"
        return None, ""

    def _rule_stabilizer(self, metrics: CircuitMetrics):
        # All-Clifford circuits below the stabilizer cap → stabilizer.
        if metrics.is_all_clifford and 0 < metrics.n_qubits <= self.stabilizer_cap:
            return "stabilizer", "Clifford-only circuit"
        return None, ""

    def _rule_density_matrix(self, metrics: CircuitMetrics):
        if not (metrics.noise_active or metrics.needs_density_matrix
                or metrics.has_dynamic_decoherence):
            return None, ""
        if metrics.n_qubits <= self.density_cap:
            return "density_matrix", "stochastic noise / mixed-state needed"
        # Above density_cap with noise: warn; fall back to MPS or dense since
        # density matrix won't fit, trading fidelity for capacity.
        return None, ""

    def _rule_mps(self, metrics: CircuitMetrics):
        if metrics.n_qubits < self.mps_qubit_floor:
            return None, ""
        if metrics.n_qubits == 0:
            return None, ""
        density = metrics.n_2q_gates / max(1, metrics.n_qubits)
        if density < self.mps_entanglement_density_max:
            return "mps", f"low entanglement density ({density:.2f})"
        return None, ""

    def _rule_sparse(self, metrics: CircuitMetrics):
        if metrics.n_qubits <= self.statevector_cap:
            return None, ""
        density = metrics.n_2q_gates / max(1, metrics.n_qubits)
        if density < self.sparse_entanglement_density_max:
            return "sparse", f"sparse-likely circuit (density {density:.2f})"
        return None, ""

    def _rule_dense(self, metrics: CircuitMetrics):
        if metrics.n_qubits <= self.statevector_cap:
            return "dense", "general-purpose statevector"
        # Above dense cap and no other rule fired — emit a "would exceed" flag.
        return None, ""

    def _rule_overflow_fallback(self, metrics: CircuitMetrics):
        # Last-resort fallback when we are above the dense cap and don't fit
        # anything better — prefer sparse (cheap memory) then mps.
        if metrics.n_qubits > self.statevector_cap:
            return "sparse", "fallback: statevector cap exceeded"
        return None, ""

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    def select(self, metrics: CircuitMetrics, user_hint: Optional[str] = None) -> SelectionReport:
        # User hint always wins first (sol.md §5.1 — "пользовательские подсказки
        # для выбора стратегии").
        hint_chosen, hint_reason = self._rule_user_hint(metrics, user_hint)
        if hint_chosen is not None:
            est_mem = self._estimate_memory(hint_chosen, metrics)
            return SelectionReport(
                chosen=hint_chosen,
                reason=hint_reason,
                metrics=metrics,
                user_hint=user_hint,
                est_memory_bytes=est_mem,
            )
        rules = [
            self._rule_stabilizer,
            self._rule_density_matrix,
            self._rule_mps,
            self._rule_sparse,
            self._rule_dense,
            self._rule_overflow_fallback,
        ]
        for rule in rules:
            chosen, reason = rule(metrics)
            if chosen is not None:
                est_mem = self._estimate_memory(chosen, metrics)
                would_exceed = self._would_exceed_memory(chosen, est_mem)
                fallback_used = False
                fallback_from = None
                if would_exceed and chosen == "density_matrix":
                    final_chosen = "mps" if metrics.n_2q_gates / max(1, metrics.n_qubits) \
                        < self.mps_entanglement_density_max else "sparse"
                    fallback_used = True
                    fallback_from = chosen
                    chosen = final_chosen
                elif would_exceed and chosen == "dense":
                    final_chosen = "mps" if metrics.n_2q_gates / max(1, metrics.n_qubits) \
                        < self.mps_entanglement_density_max else "sparse"
                    fallback_used = True
                    fallback_from = chosen
                    chosen = final_chosen
                return SelectionReport(
                    chosen=chosen,
                    reason=reason,
                    fallback_used=fallback_used,
                    fallback_from=fallback_from,
                    metrics=metrics,
                    user_hint=user_hint,
                    would_exceed_memory=would_exceed if not fallback_used else False,
                    est_memory_bytes=self._estimate_memory(chosen, metrics),
                )
        # All rules declined (e.g. zero qubits). Default to dense.
        return SelectionReport(
            chosen="dense",
            reason="default",
            metrics=metrics,
            user_hint=user_hint,
        )

    # ------------------------------------------------------------------
    # Memory estimation helpers
    # ------------------------------------------------------------------

    def _estimate_memory(self, chosen: str, metrics: CircuitMetrics) -> int:
        n = metrics.n_qubits
        if n <= 0:
            return 0
        if chosen == "dense":
            # 2^n complex128 (16 B each)
            return 16 * (1 << n) if n < 62 else 1 << 62
        if chosen == "density_matrix":
            return 16 * (1 << (2 * n)) if n < 31 else 1 << 62
        if chosen == "mps":
            chi = metrics.max_bond_dim_hint or 64
            return chi * chi * n * 16
        if chosen == "sparse":
            # Highly sparse: estimate 2^n / 2^16 entries (heuristic)
            return max(16 * (1 << max(0, n - 16)), 1 << 8)
        if chosen == "stabilizer":
            return n * n * 4  # tableau O(n²) bytes
        return 0

    def _would_exceed_memory(self, chosen: str, est_bytes: int) -> bool:
        if chosen == "dense":
            return est_bytes > self.statevector_memory_bytes
        if chosen == "density_matrix":
            return est_bytes > self.density_matrix_memory_bytes
        return False


# Shared singleton selector — reuses across run.py, runtime.py, vm.py
DEFAULT_SELECTOR = SimSelector()


# ---------------------------------------------------------------------------
# Convenience helpers used by callers that want to keep their existing
# pre-scan logic: pass raw counts and let the selector make the decision.
# ---------------------------------------------------------------------------

def select_from_counts(
    n_qubits: int,
    n_2q_gates: int = 0,
    *,
    n_gates: int = 0,
    is_all_clifford: bool = True,
    noise_active: bool = False,
    needs_density_matrix: bool = False,
    user_hint: Optional[str] = None,
    max_bond_dim_hint: Optional[int] = None,
) -> SelectionReport:
    metrics = CircuitMetrics(
        n_qubits=n_qubits,
        n_2q_gates=n_2q_gates,
        n_gates=n_gates,
        is_all_clifford=is_all_clifford,
        noise_active=noise_active,
        needs_density_matrix=needs_density_matrix,
        max_bond_dim_hint=max_bond_dim_hint,
    )
    return DEFAULT_SELECTOR.select(metrics, user_hint=user_hint)
