"""
P3 §12.1 — Entanglement witness (surface-level).

Roadmap (`sol.md` "12.1 Quantum Research Tools"):
    - [ ] **Entanglement witness** — обнаружение запутанности
    - [ ] **Quantum state tomography** — воссоздание состояния (NOT
        implemented in this envelope — left for the next round).

An entanglement witness is an observable W such that Tr[W · σ] >= 0
for all separable states but Tr[W · σ_target] < 0 for the target
entangled state. A "negative witness value" thus certifies
entanglement.

We expose the canonical Bell-state projector witness:

    W = ⟨Φ⁺|ρ|Φ⁺⟩ - 1/4

For a Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2 the witness value is 1 - 1/4
= 3/4 (positive certifies entanglement). For a separable product
state with no overlap onto |Φ⁺⟩ the witness is 0; for a full-rank
pseudopure state with |Φ⁺⟩ overlap == 1/4 (the separable bound) the
witness is 0.

We additionally expose the CHSH inequality value
    S = sqrt(2 * (<XX>**2 + <YY>**2))
which evaluates to the Tsirelson bound 2*sqrt(2) ≈ 2.83 for a
maximally entangled Bell state, providing an independent
entanglement certification that doesn't rely on knowing which Bell
state was prepared.
"""
from __future__ import annotations

import math
import typing


def prepare_bell_state(sim, q0_name: str, q1_name: str) -> None:
    """Prepare the Bell state |Φ⁺⟩ = (|00⟩ + |11⟩) / √2 by applying
    H(q0) followed by CNOT(q0, q1). Both qubits must already be
    allocated on the simulator.
    """
    sim.H(q0_name)
    sim.CNOT(q0_name, q1_name)


def _bell_projection_probability(sim) -> float:
    """Compute |<Φ⁺|ψ>|^2 from the simulator's state vector.

    In the standard 2-qubit basis ordering (|00>, |01>, |10>, |11>),
    |Φ⁺⟩ = (|00⟩ + |11⟩)/√2. So <Φ⁺|ψ> = (ψ_00 + ψ_11)/√2 and the
    squared projection is |ψ_00 + ψ_11|^2 / 2.
    """
    state = sim.get_state_vector()
    if len(state) < 4:
        return 0.0
    bell_amp = (state[0] + state[3]) / math.sqrt(2.0)
    return abs(bell_amp) ** 2


def bell_state_witness(sim, q0_name: str, q1_name: str) -> float:
    """Compute the entanglement witness value:
        W = ⟨Φ⁺|ρ|Φ⁺⟩ - 1/4

    Returns a float in [-1/4, 3/4]. W > 0 certifies entanglement.

    The `qubit_name` arguments aren't used in the implementation
    (the witness is computed on the global 2-qubit state), but
    they're kept in the signature for API symmetry with other
    research functions and for future single-qubit-relabeling
    extensions.
    """
    if sim is None:
        raise TypeError("bell_state_witness requires a simulator")
    p = _bell_projection_probability(sim)
    return p - 1.0 / 4.0


def _expectation_zz(sim: typing.Any, name_a: str, name_b: str) -> float:
    """Compute ⟨ZZ⟩ directly from the simulator's amplitudes.

    ZZ = σ_z ⊗ σ_z. The eigenvectors are the computational-basis
    states |xy⟩ with eigenvalue (-1)^(x ⊕ y). The expectation is
    sum over all basis states of |ψ_i|^2 * (-1)^(parity(i)).
    """
    state = sim.get_state_vector()
    if not state:
        return 0.0
    a_index = sim.qubit_map[name_a]
    b_index = sim.qubit_map[name_b]
    total = 0.0
    out = 0.0
    for i, amp in enumerate(state):
        p = abs(amp) ** 2
        bit_a = (i >> a_index) & 1
        bit_b = (i >> b_index) & 1
        eigenvalue = (-1) ** ((bit_a + bit_b) & 1)
        out += p * eigenvalue
        total += p
    return out / total if total > 0 else 0.0


def _clone_sim(sim: typing.Any):
    """Build a fresh dense QuantumSimulator with the same qubit
    allocations AND the same state vector as `sim`. Used so we can
    apply basis-change gates to a clone without mutating `sim`.
    """
    from src.simulator import QuantumSimulator
    clone = QuantumSimulator(sim_type="dense", seed=None)
    for n, _idx in sorted(sim.qubit_map.items(), key=lambda kv: kv[1]):
        clone.allocate_qubit(n)
    clone.state_vector = list(sim.get_state_vector())
    return clone


def _expectation_xx(sim: typing.Any, name_a: str, name_b: str) -> float:
    """Compute ⟨XX⟩ by switching both qubits into the X basis (via
    Hadamard) on a clone simulator and measuring ⟨ZZ⟩ there.
    """
    clone = _clone_sim(sim)
    clone.H(name_a)
    clone.H(name_b)
    return _expectation_zz(clone, name_a, name_b)


def _expectation_xz(sim: typing.Any, name_a: str, name_b: str) -> float:
    """Compute ⟨XZ⟩ by switching qubit a into X basis (via H on a
    only) on a clone and measuring ⟨ZZ⟩ there.
    """
    clone = _clone_sim(sim)
    clone.H(name_a)
    return _expectation_zz(clone, name_a, name_b)


def _expectation_zx(sim: typing.Any, name_a: str, name_b: str) -> float:
    """Compute ⟨ZX⟩ by switching qubit b into X basis (via H on b
    only) on a clone and measuring ⟨ZZ⟩ there.
    """
    clone = _clone_sim(sim)
    clone.H(name_b)
    return _expectation_zz(clone, name_a, name_b)


def _expectation_yy(sim: typing.Any, name_a: str, name_b: str) -> float:
    """Compute ⟨YY⟩ by switching both qubits into the Y basis (on a
    clone) and measuring ⟨ZZ⟩. Y-basis = S† · H · Z · H · S, so to
    measure Y on a qubit we pre-multiply by S†then H then read out Z.
    """
    clone = _clone_sim(sim)
    SDG = [[1.0, 0.0], [0.0, -1.0j]]
    clone.apply_1qubit_gate(name_a, SDG)
    clone.apply_1qubit_gate(name_b, SDG)
    clone.H(name_a)
    clone.H(name_b)
    return _expectation_zz(clone, name_a, name_b)


def chsh_inequality_value(sim: typing.Any, q0_name: str, q1_name: str,
                          *, angles: typing.Optional[typing.Tuple[float,
                                                                 ...]] = None,
                          ) -> float:
    """Compute the CHSH inequality value:

        S = |⟨A B⟩ + ⟨A B'⟩ + ⟨A' B⟩ - ⟨A' B'⟩|

    with the canonical Bell-state-maximizing measurement choice:
        A  = X       (q0)
        A' = Z       (q0)
        B  = (X+Z)/√2  (q1)
        B' = (X-Z)/√2  (q1)

    For the Bell state |Φ⁺⟩:
        ⟨XX⟩ = 1, ⟨ZZ⟩ = 1, ⟨XZ⟩ = 0, ⟨ZX⟩ = 0
        → S = √2 · (1 + 1) = 2√2 ≈ 2.83
    which exceeds the classical (separable-state) bound of 2, so
    CHSH violation certifies entanglement in a state-independent
    way (unlike `bell_state_witness` which assumes |Φ⁺⟩ specifically).

    The `angles` keyword is reserved for a future full-circle
    parameterization; the current implementation uses the canonical
    optimal measurement choice for |Φ⁺⟩.
    """
    xx = _expectation_xx(sim, q0_name, q1_name)
    zz = _expectation_zz(sim, q0_name, q1_name)
    xz = _expectation_xz(sim, q0_name, q1_name)
    zx = _expectation_zx(sim, q0_name, q1_name)
    sqrt2 = math.sqrt(2.0)
    e_ab = (xx + xz) / sqrt2
    e_abp = (xx - xz) / sqrt2
    e_apb = (zx + zz) / sqrt2
    e_apbp = (zx - zz) / sqrt2
    return abs(e_ab + e_abp + e_apb - e_apbp)
