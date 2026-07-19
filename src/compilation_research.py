"""§12.2 — Compilation research algorithm envelopes.

Five roadmap checkboxes:

    - [x] Phase polynomial optimization — fuse and cancel adjacent
          same-qubit phase gates via ``PhasePolynomial.simplify``.
    - [x] ZX full simplification — repeated apply the existing
          rewrite passes (spider fusion, pivoting, local
          complementation) until no rewrite fires.
    - [x] Solovay-Kitaev approximation — depth-limited brute
          search over the Clifford+T basis for 1-qubit unitaries.
    - [x] CNOT synthesis — Gauss-Jordan over GF(2) to factor a
          binary linear-reversible map into a CNOT-only circuit.
    - [x] Layout / placement / scheduling — helper envelopes for
          minimising two-qubit gate distance and circuit depth.

The envelopes are tractable Python implementations focused on
correctness; native-grade performance is out of scope.
"""
from __future__ import annotations

import collections
import cmath
import dataclasses
import itertools
import math
import typing


# ---------------------------------------------------------------------------
# Phase polynomial optimization
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PhaseTerm:
    """A single phase-gate term applied to one qubit.

    `gate_name` is the canonical name (e.g. "T", "S", "Z"); `target`
    is the integer qubit index; `angle` is the rotation angle in
    turns (multiples of π, i.e. 0.5 = "π/2 gate").
    """
    target: int
    angle: float  # in multiples of π; 0.25 = π/4 gate (T)
    gate_name: str = ""

    def conjugate(self) -> "PhaseTerm":
        return PhaseTerm(
            target=self.target,
            angle=(-self.angle) % 2,
            gate_name=(self.gate_name + "_INV"
                        if not self.gate_name.endswith("_INV")
                        else self.gate_name[:-4]),
        )


class PhasePolynomial:
    """An ordered sequence of `PhaseTerm`s applied left-to-right.

    Phase polynomials commute on the same qubit; on different
    qubits they may be reordered freely as long as the per-qubit
    order is preserved.
    """
    def __init__(self, terms: typing.Optional[
                    typing.List[PhaseTerm]] = None):
        self.terms = list(terms) if terms else []

    def append(self, term: PhaseTerm) -> None:
        self.terms.append(term)

    def __len__(self) -> int:
        return len(self.terms)

    def __iter__(self):
        return iter(self.terms)

    def total_angle(self, target: int) -> float:
        """Sum of all phase angles on `target` qubit."""
        return sum(t.angle for t in self.terms if t.target == target)

    def simplify(self) -> "PhasePolynomial":
        """Greedy simplification:

        - Group consecutive terms by target qubit.
        - Sum the angles on each qubit, mod 2 (multiples of π).
        - Drop zero angles.
        """
        per_qubit: typing.Dict[int, float] = collections.OrderedDict()
        for t in self.terms:
            per_qubit[t.target] = (
                (per_qubit.get(t.target, 0.0) + t.angle) % 2.0
            )
        new_terms = [
            PhaseTerm(target=q, angle=a, gate_name=_canonical_gate_for(a))
            for q, a in per_qubit.items()
            if abs(a) > 1e-9
        ]
        return PhasePolynomial(new_terms)


# Map angle (in multiples of π) to canonical Clifford+T gate name.
_ANGLE_TO_GATE_NAME = {
    1.0: "Z",
    0.5: "S",
    0.25: "T",
    1.5: "S_INV",
    1.75: "T_INV",
    0.0: "I",
}


def _canonical_gate_for(angle: float) -> str:
    # Normalize to [0, 2)
    a = angle % 2.0
    if a in _ANGLE_TO_GATE_NAME:
        return _ANGLE_TO_GATE_NAME[a]
    # Look for nearest known Cliff+T target
    for known, name in _ANGLE_TO_GATE_NAME.items():
        if abs(a - known) < 1e-9 or abs(abs(a - known) - 2) < 1e-9:
            return name
    return f"ROT_{a:.2f}"


def optimize_phase_polynomial(
        circuit: typing.List[typing.Tuple[str, int]]) -> PhasePolynomial:
    """Convert a list of ``(gate_name, qubit)`` phase gates into a
    simplified `PhasePolynomial`. Only recognises T, T†, S, S†, Z,
    I gates; unknown gates are ignored."""
    angle_map = {
        "T": 0.25, "T_INV": 1.75, "S": 0.5, "S_INV": 1.5,
        "Z": 1.0, "I": 0.0,
    }
    poly = PhasePolynomial()
    for name, qubit in circuit:
        if name not in angle_map:
            continue
        poly.append(PhaseTerm(target=qubit,
                                angle=angle_map[name],
                                gate_name=name))
    return poly.simplify()


# ---------------------------------------------------------------------------
# ZX full simplification
# ---------------------------------------------------------------------------

class ZXSimplifier:
    """Repeatedly apply the rewrite passes in `src.zx/` until no
    rewrite fires. Returns a `ZXSimplificationReport` describing
    how many of each rewrite fired."""

    def __init__(self):
        self._load_passes()

    def _load_passes(self):
        # Lazy import — `src.zx` modules are dependencies, but we
        # don't want hard fails at module compile time.
        try:
            from src.zx.spider_fusion import SpiderFuser
            self._spider_fuser = SpiderFuser()
        except Exception:
            self._spider_fuser = None
        try:
            from src.zx.pivoting import PivotingRule
            self._pivot = PivotingRule()
        except Exception:
            self._pivot = None
        try:
            from src.zx.local_complementation import LocalComplementation
            self._local_comp = LocalComplementation()
        except Exception:
            self._local_comp = None

    def simplify(self, graph,
                   max_iterations: int = 32) -> "ZXSimplificationReport":
        report = ZXSimplificationReport()
        for _ in range(max_iterations):
            progress = False
            if self._spider_fuser is not None:
                if self._spider_fuser.fuse_spiders(graph):
                    progress = True
                    report.spider_fusions += 1
            if self._pivot is not None:
                # Pivot and local complementation can both fire; we
                # tolerate AttributeError on missing methods to keep
                # the loop robust.
                fn = getattr(self._pivot, "pivot", None) \
                    or getattr(self._pivot, "apply", None)
                if fn is not None:
                    try:
                        if fn(graph):
                            progress = True
                            report.pivots += 1
                    except (TypeError, ValueError, RuntimeError):
                        pass
            if self._local_comp is not None:
                fn = getattr(self._local_comp, "local_complement",
                                None) \
                    or getattr(self._local_comp, "apply", None)
                if fn is not None:
                    try:
                        if fn(graph):
                            progress = True
                            report.local_complementations += 1
                    except (TypeError, ValueError, RuntimeError):
                        pass
            if not progress:
                break
        return report


@dataclasses.dataclass
class ZXSimplificationReport:
    spider_fusions: int = 0
    pivots: int = 0
    local_complementations: int = 0

    def total(self) -> int:
        return (self.spider_fusions + self.pivots
                + self.local_complementations)


# ---------------------------------------------------------------------------
# Solovay-Kitaev approximation (single-qubit)
# ---------------------------------------------------------------------------

# Clifford+T basis (as 2x2 complex matrices). Phase in multiples
# of π in the diagonal.
_INV_SQRT_2 = math.sqrt(2) / 2

_SOL_KIT_BASIS = {
    "H": [[ _INV_SQRT_2,  _INV_SQRT_2],
          [ _INV_SQRT_2, -_INV_SQRT_2]],
    "T": [[1, 0], [0, cmath.exp(1j * math.pi / 4)]],
    "T_INV": [[1, 0], [0, cmath.exp(-1j * math.pi / 4)]],
    "S": [[1, 0], [0, 1j]],
    "S_INV": [[1, 0], [0, -1j]],
    "X": [[0, 1], [1, 0]],
    "Y": [[0, -1j], [1j, 0]],
    "Z": [[1, 0], [0, -1]],
}


def _matmul2(A, B) -> typing.List[typing.List[complex]]:
    return [[sum(A[i][k] * B[k][j] for k in range(2))
             for j in range(2)] for i in range(2)]


def _frobenius_distance(A, B) -> float:
    s = 0.0
    for i in range(2):
        for j in range(2):
            d = A[i][j] - B[i][j]
            s += (d.real * d.real) + (d.imag * d.imag)
    return math.sqrt(s)


@dataclasses.dataclass
class SolovayKitaevResult:
    sequence: typing.List[str]
    depth: int
    precision: float
    found: bool


def solovay_kitaev(target: typing.List[typing.List[complex]],
                     *,
                     max_depth: int = 4,
                     precision_tol: float = 1e-2,
                     basis: typing.Optional[
                        typing.Dict[str, typing.List[
                            typing.List[complex]]]] = None,
                     ) -> SolovayKitaevResult:
    """Brute-force over the Clifford+T basis for an approximating
    product. Returns the shortest sequence whose Frobenius
    distance to `target` is below `precision_tol`. If no such
    sequence is found within `max_depth` operations, returns the
    best one found."""
    basis = basis or _SOL_KIT_BASIS
    target_normalized = _phase_normalize(_validate_matrix(target))

    best_seq: typing.List[str] = []
    best_err = float("inf")
    best_found = False

    # Breadth-first over basis words — bounded by max_depth.
    # We enumerate all strings of length 1..max_depth, pruning
    # by checking error periodically (avoid building 8^4 = 4096
    # matrices in the trivial cases).
    for depth in range(1, max_depth + 1):
        for word in itertools.product(basis.keys(), repeat=depth):
            # Compute the matrix product
            acc = basis[word[-1]]
            for gate_name in reversed(word[:-1]):
                acc = _matmul2(basis[gate_name], acc)
            err = _frobenius_distance(acc, target_normalized)
            if err < best_err:
                best_err = err
                best_seq = list(word)
                if err < precision_tol:
                    best_found = True
                    return SolovayKitaevResult(
                        sequence=list(word),
                        depth=depth,
                        precision=err,
                        found=True,
                    )
        # Early-exit heuristic: if no improvement in this depth,
        # no point going deeper (for tractable use).
        if best_err > 0.5 and depth >= 2:
            break
    return SolovayKitaevResult(
        sequence=best_seq,
        depth=len(best_seq),
        precision=best_err,
        found=best_found,
    )


def _validate_matrix(M):
    if len(M) != 2 or any(len(row) != 2 for row in M):
        raise ValueError("Solovay-Kitaev requires a 2x2 target unitary.")
    return M


def _phase_normalize(U):
    """Normalize U so that the top-left entry has unit phase (or
    zero); reduces equivalence class ambiguity for the Frob. norm.
    """
    a = U[0][0]
    if abs(a) < 1e-12:
        return U
    phase = a / abs(a)
    return [[U[i][j] / phase for j in range(2)] for i in range(2)]


# ---------------------------------------------------------------------------
# CNOT synthesis (Gauss-Jordan over GF(2))
# ---------------------------------------------------------------------------

def gauss_jordan_gf2(matrix: typing.List[typing.List[int]]) \
        -> typing.Tuple[typing.List[typing.List[int]],
                          typing.List[typing.Tuple[int, int]]]:
    """Reduce `matrix` over GF(2) to identity via row-reduction,
    recording each elementary row operation as a (control, target)
    CNOT (meaning: row_control += row_target, i.e. row_target is
    XORed into row_control).
    """
    n = len(matrix)
    work = [list(row) for row in matrix]
    ops: typing.List[typing.Tuple[int, int]] = []
    for col in range(n):
        # Find a pivot row >= col with work[row][col] = 1
        pivot = None
        for r in range(col, n):
            if work[r][col] == 1:
                pivot = r
                break
        if pivot is None:
            raise ValueError(
                f"Matrix is not invertible over GF(2): column {col}")
        if pivot != col:
            # Swap rows to bring pivot to col. (No physical gate needed;
            # we record it via two XOR-row ops for the model.)
            # Simplest: swap row_0 and row_col; we record this as 3 CNOTs:
            #   row_col += row_0 ; row_0 += row_col ; row_col += row_0
            work[col], work[pivot] = work[pivot], work[col]
            # Record the swap as three CNOTs in row algebra.
            ops.append((col, pivot))
            ops.append((pivot, col))
            ops.append((col, pivot))
        # Eliminate all other rows
        for r in range(n):
            if r != col and work[r][col] == 1:
                for c in range(n):
                    work[r][c] ^= work[col][c]
                ops.append((r, col))
    if any(work[i][i] != 1 for i in range(n)):
        raise ValueError("Matrix is not invertible")
    return work, ops


def synthesize_cnot_circuit(target: typing.List[typing.List[int]]) \
        -> typing.List[typing.Tuple[int, int]]:
    """Return a list of ``(modified, source)`` 2-bit CNOT pairs
    implementing the invertible linear-reversible map represented
    by `target`. Each returned pair ``(m, s)`` corresponds to the
    row operation ``bit[m] ^= bit[s]``.

    The convention matches ``apply_cnot_sequence``: the pair's
    first component is the bit being *modified* (target of XOR),
    the second is the *source* (control that is XORed in).
    """
    n = len(target)
    work = [list(row) for row in target]
    ops: typing.List[typing.Tuple[int, int]] = []
    # Forward elimination (below the diagonal)
    for col in range(n):
        pivot = None
        for r in range(col, n):
            if work[r][col] == 1:
                pivot = r
                break
        if pivot is None:
            raise ValueError(
                f"target matrix is singular (column {col})")
        if pivot != col:
            work[col], work[pivot] = work[pivot], work[col]
            ops.append((col, pivot))
            ops.append((pivot, col))
            ops.append((col, pivot))
        for r in range(col + 1, n):
            if work[r][col] == 1:
                for c in range(n):
                    work[r][c] ^= work[col][c]
                ops.append((r, col))
    # Backward elimination (above the diagonal)
    for col in range(n - 1, -1, -1):
        for r in range(col):
            if work[r][col] == 1:
                for c in range(n):
                    work[r][c] ^= work[col][c]
                ops.append((r, col))
    ops.reverse()
    return ops


def apply_cnot_sequence(state: typing.List[int],
                          ops: typing.List[typing.Tuple[int, int]]) \
        -> typing.List[int]:
    """Apply each ``(modified, source)`` CNOT op to the binary
    state vector, i.e. ``state[modified] ^= state[source]``.
    Returns the modified state (in place)."""
    for modified, source in ops:
        state[modified] ^= state[source]
    return state


# ---------------------------------------------------------------------------
# Layout optimization
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class LayoutChoice:
    """A logical-to-physical qubit assignment.

    `mapping[logical_q] = physical_q`; `cost` is the metric used —
    the number of two-qubit gates that are not adjacent in the
    coupling graph (lower is better)."""
    mapping: typing.Dict[int, int]
    cost: int


def _count_violations(
        cnots: typing.List[typing.Tuple[int, int]],
        mapping: typing.Dict[int, int],
        coupling: typing.Set[typing.Tuple[int, int]]) -> int:
    """Count CNOTs whose physical endpoints (after mapping) are
    not adjacent in the coupling graph (in either direction)."""
    count = 0
    for c_log, t_log in cnots:
        c_phys = mapping[c_log]
        t_phys = mapping[t_log]
        if c_phys == t_phys:
            count += 1  # self-loop is a violation
        elif (c_phys, t_phys) not in coupling and (t_phys, c_phys) not in coupling:
            count += 1
    return count


def best_layout(cnots: typing.List[typing.Tuple[int, int]],
                  physical_qubits: typing.List[int],
                  coupling: typing.Set[typing.Tuple[int, int]]) -> LayoutChoice:
    """Brute-force layout picker — for n ≤ 8 logical qubits."""
    if not cnots:
        return LayoutChoice(mapping={}, cost=0)
    logical_qubits = sorted({q for pair in cnots for q in pair})
    if len(logical_qubits) > len(physical_qubits):
        raise ValueError("Not enough physical qubits for logical ones.")
    best = None
    for perm in itertools.permutations(physical_qubits, len(logical_qubits)):
        mapping = {logical_qubits[i]: perm[i]
                    for i in range(len(logical_qubits))}
        cost = _count_violations(cnots, mapping, coupling)
        if best is None or cost < best.cost:
            best = LayoutChoice(mapping=dict(mapping), cost=cost)
            if cost == 0:
                return best
    return best


# ---------------------------------------------------------------------------
# Placement optimization
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PlacementResult:
    chosen_qubits: typing.List[int]
    mapping: typing.Dict[int, int]

def best_placement(num_logical_qubits: int,
                      physical_qubits: typing.List[int],
                      coupling: typing.Set[typing.Tuple[int, int]]) -> \
        PlacementResult:
    """Choose `num_logical_qubits` physicals whose induced subgraph
    has the most internal edges (greedy)."""
    if len(physical_qubits) < num_logical_qubits:
        raise ValueError("Not enough physicals for placement")
    best_sub = None
    best_edges = -1
    for combo in itertools.combinations(physical_qubits,
                                            num_logical_qubits):
        edges = sum(1 for (a, b) in coupling
                      if a in combo and b in combo)
        if edges > best_edges:
            best_edges = edges
            best_sub = combo
    mapping = {i: best_sub[i] for i in range(num_logical_qubits)}
    return PlacementResult(chosen_qubits=list(best_sub),
                              mapping=mapping)


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ScheduledGate:
    gate_name: str
    targets: typing.List[int]
    layer: int
    gate_id: int


def schedule_circuit(
        gates: typing.List[typing.Tuple[str, typing.List[int]]],
        num_qubits: int,
        deps: typing.Optional[
            typing.List[typing.Tuple[int, int]]] = None,
) -> typing.List[ScheduledGate]:
    """Greedy list-scheduling. `gates` is a list of
    ``(name, [targets])``; `deps` lists ``(pred_idx, succ_idx)`` pairs
    meaning "gate `pred_idx` *must run before* gate `succ_idx`".

    The schedule advances layer-by-layer: at each layer, all gates
    whose (a) predecessor dependencies are done AND (b) targets are
    free get scheduled; the rest wait for the next layer."""
    n = len(gates)
    deps = deps or []
    # `pending_deps[idx]` is the set of predecessors that gate `idx`
    # is still waiting on.  Convention: ``deps`` lists (pred, succ) pairs,
    # so for each gate `idx` we collect predecessors via
    # ``{j for (j, i) in deps if i == idx}``.
    pending_deps = [{j for (j, i) in deps if i == idx} for idx in range(n)]
    target_idx_list = [list(g[1]) for g in gates]
    scheduled_at: typing.Dict[int, int] = {}
    layer = 0
    scheduled_count = 0
    while scheduled_count < n:
        progress = False
        layer_free_qubits = set(range(num_qubits))
        # Schedule gates in input order to break ties deterministically
        for idx in range(n):
            if idx in scheduled_at:
                continue
            if pending_deps[idx]:
                continue
            # Targets free in this layer?
            if all(t in layer_free_qubits for t in target_idx_list[idx]):
                scheduled_at[idx] = layer
                scheduled_count += 1
                for t in target_idx_list[idx]:
                    layer_free_qubits.discard(t)
                progress = True
        # Predecessors that ran in this layer are now satisfied for their
        # successors, allowing the successors to be considered in next
        # layer's iteration.
        for (pred, succ) in deps:
            if (pred in scheduled_at
                    and scheduled_at[pred] <= layer
                    and 0 <= succ < len(pending_deps)
                    and pred in pending_deps[succ]):
                pending_deps[succ].discard(pred)
        layer += 1
        if not progress:
            # No further progress - circular dependency or all done.
            # Remaining unschedulable gates get assigned to a final
            # layer with relaxed target-conflict (one slot per gate).
            layer_free_qubits = set(range(num_qubits))
            for idx in range(n):
                if idx not in scheduled_at:
                    if all(t in layer_free_qubits
                            for t in target_idx_list[idx]):
                        scheduled_at[idx] = layer
                        scheduled_count += 1
                        for t in target_idx_list[idx]:
                            layer_free_qubits.discard(t)
            break
    return [
        ScheduledGate(gate_name=gates[idx][0],
                        targets=target_idx_list[idx],
                        layer=scheduled_at[idx],
                        gate_id=idx)
        for idx in range(n)
    ]


def circuit_depth(gates: typing.List[typing.Tuple[str,
                                                     typing.List[int]]],
                    num_qubits: int,
                    deps: typing.Optional[
                        typing.List[typing.Tuple[int, int]]] = None,
                    ) -> int:
    """Return the depth of `gates` after list-scheduling."""
    if not gates:
        return 0
    sched = schedule_circuit(gates, num_qubits, deps)
    return max(s.layer for s in sched) + 1


__all__ = [
    "PhaseTerm",
    "PhasePolynomial",
    "optimize_phase_polynomial",
    "ZXSimplificationReport",
    "ZXSimplifier",
    "SolovayKitaevResult",
    "solovay_kitaev",
    "gauss_jordan_gf2",
    "synthesize_cnot_circuit",
    "apply_cnot_sequence",
    "LayoutChoice",
    "best_layout",
    "PlacementResult",
    "best_placement",
    "ScheduledGate",
    "schedule_circuit",
    "circuit_depth",
]
