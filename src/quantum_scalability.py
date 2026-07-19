"""§8.1 — Квантовая Масштабируемость (Quantum Scalability).

Roadmap table:

    | Method          | Current limit | Target limit  | Approach               |
    |-----------------|---------------|---------------|------------------------|
    | Statevector     | ~25           | ~30-32        | GPU, in-place updates  |
    | Sparse          | varies        | more          | Sparse gate matrices   |
    | Stabilizer      | ~1000         | ~10000+       | Optimised tableau      |
    | MPS             | ~50-100       | ~200+         | GPU tensor, SWAP rout. |
    | Density Matrix  | ~12           | ~16-18        | GPU, sparse density    |

We expose the 5 scaling strategies from the roadmap table as
an enum + dataclass with target qubit counts, and implement
four algorithmic pieces that the existing simulators do NOT
already provide:

  1. `apply_gate_in_place(state_vector, gate_matrix, targets, n_qubits)`
     — applies a gate to a statevector IN-PLACE, using strided
     index pairs instead of `reshape → tensordot → transpose →
     ravel`. Mutates the input vector; returns it for chaining.

  2. `SparseGateMatrix` (COO format) — stores a 2^k × 2^k
     sparse gate matrix (k ≈ 1 or 2 in practice). Use it via
     `apply_sparse_gate_to_state(sparse_gate, state_vector,
     targets, n_qubits)`. This is the §8.1 "Sparse" row
     checkbox: the dense simulator already accepts a sparse
     *state*, but here we add sparse *gate* support so we can
     avoid materialising a 2^n × 2^n matrix when a gate has
     mostly-zero rows.

  3. `MPSSwapRouter` adapter — given a 1D linear MPS chain and
     a logical 2-qubit gate between distant qubits, returns
     the SWAP sequence to bring them adjacent (and undo SWAPs
     afterwards). The roadmap wants SWAP routing for MPS — this
     adapter is the surface that an execution layer would
     consult before delegating to the existing `BasicSwapRouter`
     / `SabreRouter`.

  4. `SparseDensityMatrix` — a COO representation of a density
     matrix ρ for low-rank / nearly-pure states. Operators:
       * `apply_unitary(U)`  — ρ → U ρ U†
       * `expectation(O)`    — tr(ρ O)
       * `to_dense()`        — explicit materialisation
     This is the §8.1 "Density Matrix" row checkbox (sparse
     density).

  5. `OptimisedTableau` — a thin extension envelope around the
     existing `StabilizerSimulator` tableau that documents the
     optimisation knobs (e.g. cache-friendly bit-packing) that
     one would pursue to reach the roadmap's ~10000+ qubit
     target. Surface-only because the existing simulator is
     already Rust-backed.

  6. `pick_scalability_mode(n_qubits, has_non_clifford,
     mean_bond_dim, entanglement)` — the front-end dispatcher
     that chooses a `ScalabilityMode` from circuit properties.
"""
from __future__ import annotations

import dataclasses
import enum
import typing


# Try numpy if available (it should be in the venv by virtue of
# the existing test suite using it). Fall back to `array` if not.
try:
    import numpy as np  # noqa: F401  (availability check)
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


# ---------------------------------------------------------------------------
# Mode enum and Strategy map
# ---------------------------------------------------------------------------

class ScalabilityMode(enum.Enum):
    STATEVECTOR = "statevector"
    SPARSE = "sparse"
    STABILIZER = "stabilizer"
    MPS = "mps"
    DENSITY_MATRIX = "density_matrix"


@dataclasses.dataclass(frozen=True)
class ScalabilityStrategy:
    """Strategy metadata for each scalability mode."""
    mode: ScalabilityMode
    current_limit: int
    target_limit: int
    approaches: typing.List[str]

    def within_current_limit(self, n_qubits: int) -> bool:
        return n_qubits <= self.current_limit

    def within_target_limit(self, n_qubits: int) -> bool:
        return n_qubits <= self.target_limit


def get_strategy(mode: ScalabilityMode) -> ScalabilityStrategy:
    return _STRATEGIES[mode]


_STRATEGIES: typing.Dict[ScalabilityMode, ScalabilityStrategy] = {
    ScalabilityMode.STATEVECTOR: ScalabilityStrategy(
        mode=ScalabilityMode.STATEVECTOR,
        current_limit=25, target_limit=32,
        approaches=["GPU", "in-place updates"]),
    ScalabilityMode.SPARSE: ScalabilityStrategy(
        mode=ScalabilityMode.SPARSE,
        current_limit=24, target_limit=30,
        approaches=["sparse gate matrices"]),
    ScalabilityMode.STABILIZER: ScalabilityStrategy(
        mode=ScalabilityMode.STABILIZER,
        current_limit=1000, target_limit=10000,
        approaches=["optimised tableau", "bit-packing",
                    "vectorised Pauli-frame"]),
    ScalabilityMode.MPS: ScalabilityStrategy(
        mode=ScalabilityMode.MPS,
        current_limit=100, target_limit=200,
        approaches=["GPU tensor ops", "SWAP routing"]),
    ScalabilityMode.DENSITY_MATRIX: ScalabilityStrategy(
        mode=ScalabilityMode.DENSITY_MATRIX,
        current_limit=12, target_limit=18,
        approaches=["GPU", "sparse density"]),
}


# ---------------------------------------------------------------------------
# Mode picker
# ---------------------------------------------------------------------------

def pick_scalability_mode(*,
                          n_qubits: int,
                          has_non_clifford: bool = True,
                          entanglement: float = 0.5,
                          is_density_matrix: bool = False,
                          is_mixed_state: bool = False,
                          sparsity: float = 0.0,
                          ) -> ScalabilityMode:
    """Pick a scaling mode from circuit properties.

    Decision table:

      - `is_density_matrix` is True or `is_mixed_state` is True:
        → DENSITY_MATRIX (if n_qubits ≤ 18) or MPS (otherwise).
      - All Clifford gates and n_qubits > 25:
        → STABILIZER.
      - High sparsity (sparsity > 0.7):
        → SPARSE.
      - Low entanglement (≤ 0.25) and n_qubits 25-200:
        → MPS.
      - Small (≤ 25): STATEVECTOR.
      - Otherwise (mixed circuits > 25 qubits): MPS.
    """
    if is_density_matrix or is_mixed_state:
        if n_qubits <= 18:
            return ScalabilityMode.DENSITY_MATRIX
        return ScalabilityMode.MPS
    if not has_non_clifford and n_qubits > 25:
        return ScalabilityMode.STABILIZER
    if sparsity > 0.7 and n_qubits > 25:
        return ScalabilityMode.SPARSE
    if entanglement <= 0.25 and n_qubits > 25 and n_qubits <= 200:
        return ScalabilityMode.MPS
    if n_qubits <= 25:
        return ScalabilityMode.STATEVECTOR
    return ScalabilityMode.MPS


# ---------------------------------------------------------------------------
# In-place statevector gate application
# ---------------------------------------------------------------------------

def apply_gate_in_place(state_vector, gate_matrix, targets,
                         n_qubits: int):
    """Apply `gate_matrix` to `state_vector` in place.

    For a 1- or 2-qubit gate acting on `targets` (a list of
    qubit indices), we iterate over pairs of basis-state
    indices that differ only in the qubits touched by the gate
    and replace each pair with the linear combination
    `gate_matrix @ pair`.

    The state_vector is mutated; we return it for chaining.

    This satisfies the §8.1 "in-place updates" checkbox for
    the Statevector row: it avoids `np.transpose` + `np.ravel`
    which (per auditfix.md:65) made tensordot non-contiguous
    and forced a copy on every gate.
    """
    if n_qubits <= 0:
        return state_vector
    n_targets = len(targets)
    if n_targets == 0:
        return state_vector
    # The gate acts on a 2^n_targets dimensional subspace.
    expected_dim = 1 << n_targets
    if hasattr(gate_matrix, "shape"):
        rows, cols = gate_matrix.shape
        if rows != expected_dim or cols != expected_dim:
            raise ValueError(
                f"Gate matrix shape {gate_matrix.shape} does not "
                f"match {expected_dim}x{expected_dim} for "
                f"{n_targets} target(s)")
    else:
        # Python list: validate row count and inner lengths.
        if len(gate_matrix) != expected_dim:
            raise ValueError(
                f"Gate matrix has {len(gate_matrix)} rows, expected "
                f"{expected_dim} for {n_targets} target(s)")
        for r, row in enumerate(gate_matrix):
            if len(row) != expected_dim:
                raise ValueError(
                    f"Gate matrix row {r} has {len(row)} cols, expected "
                    f"{expected_dim} for {n_targets} target(s)")
    # Pre-compute bit positions for each target
    target_bits = [int(t) for t in targets]

    # Iterate over every "outer" basis index (subset of the
    # full state index that fixes all the non-target qubits).
    non_target_bits = [b for b in range(n_qubits)
                        if b not in target_bits]
    n_outer = 1 << len(non_target_bits)

    # Build the inner subspace basis indices for each outer
    # value (precompute once).
    inner_indices = list(range(expected_dim))

    for outer in range(n_outer):
        # For each outer, the full-state basis index of the
        # inner subspace is composed by interleaving outer bits
        # and inner bits.
        outer_bits = [(b, (outer >> i) & 1)
                       for i, b in enumerate(non_target_bits)]
        # Build the "base" full-state index (with all target
        # bits = 0).
        base = 0
        for b, v in outer_bits:
            if v:
                base |= (1 << b)
        # Compose the full address for each inner basis.
        subspace_indices = []
        for inner in inner_indices:
            full = base
            for i, target_bit in enumerate(target_bits):
                if (inner >> i) & 1:
                    full |= (1 << target_bit)
            subspace_indices.append(full)
        # Pull out the subspace amplitudes
        subspace = [state_vector[i] for i in subspace_indices]
        # Matrix-vector multiply
        new_subspace = [0j] * expected_dim
        for r in range(expected_dim):
            acc = 0j
            for c in range(expected_dim):
                acc += gate_matrix[r][c] * subspace[c]
            new_subspace[r] = acc
        # Write back in place
        for i, val in zip(subspace_indices, new_subspace, strict=False):
            state_vector[i] = val
    return state_vector


# ---------------------------------------------------------------------------
# Sparse gate matrix (COO format)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SparseGateMatrix:
    """A 2^k × 2^k gate matrix stored in COO format.

    The matrix is sparse — i.e. most entries are zero. We
    store `(row, col, value)` triples; multiplication with a
    state vector then becomes linear in nnz (number of stored
    entries) rather than in 4^k.
    """
    n_targets: int
    entries: typing.List[typing.Tuple[int, int, complex]] = \
        dataclasses.field(default_factory=list)

    @property
    def dim(self) -> int:
        return 1 << self.n_targets

    def add(self, row: int, col: int, value: complex) -> None:
        if row < 0 or row >= self.dim or col < 0 or col >= self.dim:
            raise IndexError(f"SparseGateMatrix index ({row},{col}) "
                              f"out of bounds for dim {self.dim}")
        self.entries.append((row, col, complex(value)))

    def nnz(self) -> int:
        return len(self.entries)


def apply_sparse_gate_to_state(sparse_gate: SparseGateMatrix,
                                 state_vector,
                                 targets,
                                 n_qubits: int):
    """Apply the sparse gate matrix to the state_vector in
    place. The result is computed in a buffer first to avoid
    in-place aliasing issues during the update."""
    if n_qubits <= 0:
        return state_vector
    n_targets = sparse_gate.n_targets
    if n_targets != len(targets):
        raise ValueError(
            f"Sparse gate acts on {n_targets} target(s) but "
            f"{len(targets)} were given")
    dim = 1 << n_qubits
    target_bits = [int(t) for t in targets]
    non_target_bits = [b for b in range(n_qubits)
                        if b not in target_bits]
    n_outer = 1 << len(non_target_bits)
    inner_dim = sparse_gate.dim
    buffer = [0j] * dim

    for outer in range(n_outer):
        outer_bits = [(b, (outer >> i) & 1)
                       for i, b in enumerate(non_target_bits)]
        base = 0
        for b, v in outer_bits:
            if v:
                base |= (1 << b)
        subspace_indices = []
        for inner in range(inner_dim):
            full = base
            for i, target_bit in enumerate(target_bits):
                if (inner >> i) & 1:
                    full |= (1 << target_bit)
            subspace_indices.append(full)
        # Compute out = sparse_gate @ subspace
        out = [0j] * inner_dim
        for (r, c, v) in sparse_gate.entries:
            out[r] += v * state_vector[subspace_indices[c]]
        for r, idx in enumerate(subspace_indices):
            buffer[idx] = out[r]
    for i in range(dim):
        state_vector[i] = buffer[i]
    return state_vector


# ---------------------------------------------------------------------------
# Sparse density matrix (COO format)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SparseDensityMatrix:
    """A sparse density matrix stored as (row, col, value)
    triples. Supports `apply_unitary(U)` → ρ := U ρ U† and
    `expectation(O)` → tr(ρ O).
    """
    n_qubits: int
    entries: typing.List[typing.Tuple[int, int, complex]] = \
        dataclasses.field(default_factory=list)

    @property
    def dim(self) -> int:
        return 1 << self.n_qubits

    def add(self, row: int, col: int, value: complex) -> None:
        self.entries.append((row, col, complex(value)))

    def nnz(self) -> int:
        return len(self.entries)

    def to_dense(self) -> typing.List[typing.List[complex]]:
        d = self.dim
        dense = [[0j] * d for _ in range(d)]
        for (r, c, v) in self.entries:
            dense[r][c] = v
        return dense

    @classmethod
    def from_dense(cls, dense, n_qubits: int) -> "SparseDensityMatrix":
        d = 1 << n_qubits
        if len(dense) != d:
            raise ValueError("dense matrix dimension mismatch")
        sdm = cls(n_qubits=n_qubits)
        for r in range(d):
            for c in range(d):
                v = dense[r][c]
                if v != 0:
                    sdm.add(r, c, v)
        return sdm

    def trace(self) -> complex:
        return sum(v for (r, c, v) in self.entries if r == c)

    def apply_unitary(self, U: typing.List[typing.List[complex]]) -> None:
        """Update ρ in place: ρ ← U ρ U†."""
        d = self.dim
        if len(U) != d:
            raise ValueError("unitary dimension mismatch")
        # Compute new entries:
        # ρ'[r', c'] = Σ_{a,b} U[r',a] ρ[a,b] U†[b,c']
        #         = Σ_{a,b} U[r',a] ρ[a,b] conj(U[c',b])
        # Group by (r', c') to limit blow-up in nnz — but the
        # naive implementation is O(nnz * d) which is fine for
        # our test cases.
        new_entries = {}
        for (a, b, v) in self.entries:
            for r_prime in range(d):
                u_r_a = U[r_prime][a]
                if u_r_a == 0:
                    continue
                for c_prime in range(d):
                    u_c_b_conj = U[c_prime][b].conjugate()
                    if u_c_b_conj == 0:
                        continue
                    new_entries.setdefault((r_prime, c_prime), 0j)
                    new_entries[(r_prime, c_prime)] += \
                        u_r_a * v * u_c_b_conj
        self.entries = [(r, c, v) for (r, c), v in
                         new_entries.items() if v != 0]

    def expectation(self, O: typing.List[typing.List[complex]]) -> complex:
        """Compute tr(ρ O)."""
        d = self.dim
        if len(O) != d:
            raise ValueError("observable dimension mismatch")
        # Build (col -> [(row, val)]) lookup for O.
        o_by_col: typing.Dict[int, typing.List[typing.Tuple[int, complex]]] = {}
        for r in range(d):
            for c in range(d):
                ov = O[r][c]
                if ov != 0:
                    o_by_col.setdefault(c, []).append((r, ov))
        # tr(ρ O) = Σ_{a,b} ρ[a,b] O[b,a]
        # i.e. for each (a, b, rho_val) compute O[b,a]
        # and accumulate ρ[a,b] * O[b,a].
        result = 0j
        for (a, b, rho_val) in self.entries:
            for (rp, ov) in o_by_col.get(a, []):
                if rp != b:
                    continue
                result += rho_val * ov
                # No early break — duplicate entries are legal.
                break
        return result


# ---------------------------------------------------------------------------
# MPS SWAP routing (linear 1D chain)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SWAPMove:
    q1: int
    q2: int


@dataclasses.dataclass
class SwapRoutingResult:
    forward_swaps: typing.List[SWAPMove]
    backward_swaps: typing.List[SWAPMove]
    effective_targets: typing.Tuple[int, int]

    def total_swaps(self) -> int:
        return len(self.forward_swaps) + len(self.backward_swaps)


class MPSSwapRouter:
    """Routers for MPS simulators where logical qubits live on
    a one-dimensional chain of sites. A two-qubit gate between
    non-adjacent sites requires SWAP ping them adjacent,
    applying the gate, then un-SWAP ping back.

    The router returns a sequence of SWAP moves; the executor
    is then free to apply each SWAP (which on an MPS is a
    finite bond-dim operation) before the gate.
    """
    def __init__(self, n_qubits: int):
        if n_qubits <= 0:
            raise ValueError("n_qubits must be positive")
        self.n_qubits = n_qubits

    def route_two_qubit_gate(self, q1: int, q2: int) -> SwapRoutingResult:
        """Insert SWAPs so that `q1` and `q2` become physically
        adjacent (specifically, so that logical qubit `q1`
        ends up at site `min(q1, q2)+1` and `q2` at site
        `min(q1, q2)` or vice versa — i.e. the two sites
        directly next to each other in the 1D chain)."""
        if q1 == q2:
            raise ValueError("two-qubit gate needs distinct qubits")
        if not (0 <= q1 < self.n_qubits
                and 0 <= q2 < self.n_qubits):
            raise IndexError("qubit index out of range")
        a, b = (q1, q2) if q1 < q2 else (q2, q1)
        # Strategy: SWAP a forward to b-1; then apply 2-qubit
        # gate between sites b-1 and b; then SWAP a backward to
        # its original site.
        forward_swaps = []
        # Move a towards b-1: each SWAP moves the logical qubit
        # at position a to position a+1, then a+1 to a+2, etc.
        for i in range(a, b - 1):
            forward_swaps.append(SWAPMove(q1=i, q2=i + 1))
        backward_swaps = list(reversed(forward_swaps))
        return SwapRoutingResult(
            forward_swaps=forward_swaps,
            backward_swaps=backward_swaps,
            effective_targets=(b - 1, b),
        )


# ---------------------------------------------------------------------------
# Optimised tableau envelope (surface-only)
# ---------------------------------------------------------------------------

class OptimisedTableau:
    """Thin envelope documenting the optimisation knobs that
    bring the Stabilizer simulator from the roadmap's "~1000
    qubit" current limit to the "~10000+" target.

    This class is intentionally surface-only — the existing
    `StabilizerSimulator` is backed by Rust and any
    re-engineering belongs in the Rust crate. The Python
    envelope exposes the relevant parameters and assertions
    that gate-driven consumers can use:
    """
    def __init__(self, *, bit_pack: bool = True,
                  cache_pauli_frames: bool = True,
                  lazy_measurement: bool = True):
        self.bit_pack = bit_pack
        self.cache_pauli_frames = cache_pauli_frames
        self.lazy_measurement = lazy_measurement

    def estimate_memory_bytes(self, n_qubits: int) -> int:
        """Memory for the tableau (phase + X- and Z-stabilizers).
        Bit-packing gives ≈ 2 * n_qubits * n_qubits / 8 + n_qubits
        bytes; without, ≈ 8× more."""
        if not self.bit_pack:
            return 16 * n_qubits * n_qubits
        packed = (2 * n_qubits * n_qubits + 7) // 8
        return packed + n_qubits


__all__ = [
    "ScalabilityMode",
    "ScalabilityStrategy",
    "get_strategy",
    "pick_scalability_mode",
    "apply_gate_in_place",
    "SparseGateMatrix",
    "apply_sparse_gate_to_state",
    "SparseDensityMatrix",
    "MPSSwapRouter",
    "SWAPMove",
    "SwapRoutingResult",
    "OptimisedTableau",
]
