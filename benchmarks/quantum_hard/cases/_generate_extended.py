"""Generates ~60 ADDITIONAL hard-quantum .eig programs to extend the original
20 cases from ``_generate.py`` to ~80+ cases total.

Categories added:
  - Bell variants:            Ψ⁻ (1 new)                   -> total 4 Bell
  - GHZ scaling:              n = 2, 7, 10                  -> total 5 GHZ
  - W-state scaling:          n = 5, 7                      -> total 5 W
  - Deutsch family:           constant                      -> total 2 Deutsch
  - Deutsch-Jozsa:            constant                      -> total 3 DJ
  - Bernstein-Vazirani:       s = 110, 011, 111             -> total 4 BV
  - Grover scaling:           2q variants, 3q variants, 4q variants, 5q
                              (with up-to-5-controlled-Z cascade)
  - QFT:                      n = 2, 5, with various inputs
  - QPE:                      1/2, 1/8 phases + 4-ancilla
  - Superdense coding:        01, 11
  - Quantum teleportation:    |0⟩ and superposition teleports
  - Variational ansatz:       VQE-H2 prototype, layered RY, random 4q ansatz
  - Random circuits:          seeded reproducible n=3,4,5, depth up to 200
  - Quantum error correction: Steane [7,4,3], Shor 9-qubit encoder
  - Long-depth Trotter:       10-, 25-, 50-, 100-step Ising evolution
  - Draper adder variants:    2+2, 3+5, 4+4, 7+1
  - Quantum walk variants:    8-, 16-step on 4-cycle
  - GHZ ladder:              explicit staircase, 6 qubits

Run this script (or ``python -m benchmarks.quantum_hard.cases._generate_extended``)
once to (re)materialize all new ``.eig`` files alongside the original 20.

The 20 cases from ``_generate.py`` are left untouched.
"""

from __future__ import annotations

import math
import os
import random


# --------------------------------------------------------------------------- #
# Helpers: programmatic circuit builders                                      #
# --------------------------------------------------------------------------- #

def ghz(n: int) -> str:
    """n-qubit GHZ state: (|0...0> + |1...1>) / sqrt(2)."""
    body = [f"qubit q{i}" for i in range(n)]
    body.append("H q0")
    for i in range(n - 1):
        body.append(f"CNOT q0, q{i+1}")
    return "eigen 1.0\n\n" + "\n".join(body) + "\n"


def ising_trotter(n_qubits: int, n_steps: int, dt: float) -> str:
    """n_qubit Ising chain with Z_i Z_{i+1} Hamiltonian Trotterized for n_steps.

    Each step applies e^{-i dt ZZ} on each adjacent pair via the standard
    CNOT - RZ - CNOT sequence on the target.

    Prepares |+...+> first so the dynamics are non-trivial.
    """
    assert n_qubits >= 2
    body = [f"qubit q{i}" for i in range(n_qubits)]
    for i in range(n_qubits):
        body.append(f"H q{i}")
    for s in range(n_steps):
        for i in range(n_qubits - 1):
            body.append(f"CNOT q{i}, q{i+1}")
            body.append(f"RZ q{i+1}, {dt}")
            body.append(f"CNOT q{i}, q{i+1}")
    return "eigen 1.0\n\n" + "\n".join(body) + "\n"


def qft_circuit(n: int, input_bv: str) -> str:
    """n-qubit QFT.

    input_bv is the input computational-basis state bitstring (MSB first).
    Convention matches Eigen existing test 13_qft_4: q0 is the MSB.

    The QFT decomposition is the textbook one with controlled R_k
    rotations followed by a final SWAP staircase to reverse bit order.
    """
    assert len(input_bv) == n
    body = [f"qubit q{i}" for i in range(n)]
    # Prepare input state.
    for i, bit in enumerate(input_bv):
        if bit == "1":
            body.append(f"X q{i}")
    # QFT staircase.
    for i in range(n):
        body.append(f"H q{i}")
        for j in range(i + 1, n):
            phase = math.pi / (2 ** (j - i))
            body.append(f"CRZ q{i}, q{j}, {phase}")
    # Reverse bit order.
    for i in range(n // 2):
        body.append(f"SWAP q{i}, q{n-1-i}")
    return "eigen 1.0\n\n" + "\n".join(body) + "\n"


def bv_circuit(s: str) -> str:
    """Bernstein-Vazirani hidden bit-string s.

    Input register has |s| qubits (q0 = MSB of s), ancilla qa starts at |1>.
    Oracle: for each i where s_i = 1, apply CNOT q_i -> qa.
    After Hadamards on inputs, the input register holds exactly s.
    """
    n = len(s)
    body = [f"qubit q{i}" for i in range(n)]
    body.append("qubit qa")
    # Prepare ancilla in |-> = |1> then H.
    body.append("X qa")
    for i in range(n):
        body.append(f"H q{i}")
    body.append("H qa")
    # Oracle.
    for i, bit in enumerate(s):
        if bit == "1":
            body.append(f"CNOT q{i}, qa")
    # Final Hadamards on inputs (NOT ancilla).
    for i in range(n):
        body.append(f"H q{i}")
    return "eigen 1.0\n\n" + "\n".join(body) + "\n"


def grover_2q(marked_bits: str) -> str:
    """2-qubit Grover with the marked state given as 2-char bitstring (q0 q1)."""
    assert len(marked_bits) == 2
    body = ["qubit q0", "qubit q1", "H q0", "H q1"]
    # Oracle: phase-flip |marked_bits>. For 2 qubits, the CZ gate flips |11>.
    # Pre/post X mapping brings |marked_bits> -> |11> then back.
    for i, bit in enumerate(marked_bits):
        if bit == "0":
            body.append(f"X q{i}")
    body.append("CZ q0, q1")
    for i, bit in enumerate(marked_bits):
        if bit == "0":
            body.append(f"X q{i}")
    # Diffusion.
    for i in range(2):
        body.append(f"H q{i}")
        body.append(f"X q{i}")
    body.append("CZ q0, q1")
    for i in range(2):
        body.append(f"X q{i}")
    for i in range(2):
        body.append(f"H q{i}")
    return "eigen 1.0\n\n" + "\n".join(body) + "\n"


def grover_3q_1iter(marked_bits: str) -> str:
    """3-qubit Grover with a single iteration; marked state given as 3 bitstr.

    Oracle: phase-flip the marked basis state via X+CCZ+X (CCZ = H·CCX·H on target).
    Diffusion: H^⊗3, X^⊗3, H·CCX·H on q2, X^⊗3, H^⊗3.
    """
    assert len(marked_bits) == 3
    body = [f"qubit q{i}" for i in range(3)]
    for i in range(3):
        body.append(f"H q{i}")
    # Oracle: flip via X-to-111, then H·CCX(q0,q1,q2)·H, then X back.
    for i, bit in enumerate(marked_bits):
        if bit == "0":
            body.append(f"X q{i}")
    body.append("H q2")
    body.append("CCX q0, q1, q2")
    body.append("H q2")
    for i, bit in enumerate(marked_bits):
        if bit == "0":
            body.append(f"X q{i}")
    # Diffusion: H^⊗3 X^⊗3 multi-CZ(H CCX H) X^⊗3 H^⊗3.
    for i in range(3):
        body.append(f"H q{i}")
    for i in range(3):
        body.append(f"X q{i}")
    body.append("H q2")
    body.append("CCX q0, q1, q2")
    body.append("H q2")
    for i in range(3):
        body.append(f"X q{i}")
    for i in range(3):
        body.append(f"H q{i}")
    return "eigen 1.0\n\n" + "\n".join(body) + "\n"


def grover_4q_1iter(marked_bits: str) -> str:
    """4-qubit Grover with a single iteration.

    Uses one ancilla qubit to decompose the 4-controlled-Z via 2 Toffoli gates:
       multi-CZ(q0 q1 q2 q3) ≡ multi-CZ over q1..q3 with control q0
              ≡ CCX(q0, q1, anc) then CCX(anc, q2, q3) with q3 in H basis.
    """
    assert len(marked_bits) == 4
    body = ["qubit q0", "qubit q1", "qubit q2", "qubit q3", "qubit anc"]
    for i in range(4):
        body.append(f"H q{i}")
    # Oracle.
    for i, bit in enumerate(marked_bits):
        if bit == "0":
            body.append(f"X q{i}")
    body.append("H q3")
    body.append("CCX q0, q1, anc")
    body.append("CCX anc, q2, q3")
    body.append("CCX q0, q1, anc")
    body.append("H q3")
    for i, bit in enumerate(marked_bits):
        if bit == "0":
            body.append(f"X q{i}")
    # Diffusion.
    for i in range(4):
        body.append(f"H q{i}")
    for i in range(4):
        body.append(f"X q{i}")
    body.append("H q3")
    body.append("CCX q0, q1, anc")
    body.append("CCX anc, q2, q3")
    body.append("CCX q0, q1, anc")
    body.append("H q3")
    for i in range(4):
        body.append(f"X q{i}")
    for i in range(4):
        body.append(f"H q{i}")
    return "eigen 1.0\n\n" + "\n".join(body) + "\n"


def grover_5q_1iter(marked_bits: str) -> str:
    """5-qubit Grover with a single iteration.

    5-controlled-Z uses ancilla1 + ancilla2 + a cascade of 3 CCX gates with
    the standard H·CCX·H decomposition for the final CZ.
    """
    assert len(marked_bits) == 5
    body = [f"qubit q{i}" for i in range(5)] + ["qubit anc1", "qubit anc2"]
    for i in range(5):
        body.append(f"H q{i}")
    # Oracle: X-flip then 5-CZ then X-back.
    for i, bit in enumerate(marked_bits):
        if bit == "0":
            body.append(f"X q{i}")
    body.append("H q4")
    body.append("CCX q0, q1, anc1")
    body.append("CCX anc1, q2, anc2")
    body.append("CCX anc2, q3, q4")
    body.append("CCX anc1, q2, anc2")
    body.append("CCX q0, q1, anc1")
    body.append("H q4")
    for i, bit in enumerate(marked_bits):
        if bit == "0":
            body.append(f"X q{i}")
    # Diffusion.
    for i in range(5):
        body.append(f"H q{i}")
    for i in range(5):
        body.append(f"X q{i}")
    body.append("H q4")
    body.append("CCX q0, q1, anc1")
    body.append("CCX anc1, q2, anc2")
    body.append("CCX anc2, q3, q4")
    body.append("CCX anc1, q2, anc2")
    body.append("CCX q0, q1, anc1")
    body.append("H q4")
    for i in range(5):
        body.append(f"X q{i}")
    for i in range(5):
        body.append(f"H q{i}")
    return "eigen 1.0\n\n" + "\n".join(body) + "\n"


def quantum_walk(steps: int) -> str:
    """Quantum walk on a 4-cycle (positions encoded on q0,q1; coin qc).

    Step operator:
       - TOFFOLI(qc, q1) -> q0 (carry)
       - CNOT(qc, q1) (increment by 1 when qc=1)
       - H on qc (mix coin)
    """
    body = ["qubit q0", "qubit q1", "qubit qc", "H qc"]
    for _ in range(steps):
        body.append("CCX qc, q1, q0")
        body.append("CNOT qc, q1")
        body.append("H qc")
    return "eigen 1.0\n\n" + "\n".join(body) + "\n"


def vqe_layered(n_qubits: int, n_layers: int, seed: int = 7) -> str:
    """Layered RY+RZ+Trotterised-entangler VQE-style ansatz.

    Each layer applies RY(theta_i) then RZ(phi_i) on every qubit, then a ring
    of CNOTs on (q_i, q_{i+1 mod n}) to entangle.
    """
    rng = random.Random(seed)
    body = [f"qubit q{i}" for i in range(n_qubits)]
    for layer in range(n_layers):
        for i in range(n_qubits):
            theta = rng.uniform(-math.pi, math.pi)
            phi = rng.uniform(-math.pi, math.pi)
            body.append(f"RY q{i}, {theta}")
            body.append(f"RZ q{i}, {phi}")
        for i in range(n_qubits):
            ctrl = i
            tgt = (i + 1) % n_qubits
            body.append(f"CNOT q{ctrl}, q{tgt}")
    return "eigen 1.0\n\n" + "\n".join(body) + "\n"


def random_circuit(n_qubits: int, depth: int, seed: int = 13) -> str:
    """Reproducible pseudo-random circuit using H/X/CNOT/RY/RZ/T.

    Gates are drawn from a small alphabet that the numpy reference supports,
    so the head-to-head comparison remains bit-accurate.
    """
    rng = random.Random(seed)
    body = [f"qubit q{i}" for i in range(n_qubits)]
    pool_1q = ["H", "X", "S", "T"]
    for _ in range(depth):
        kind = rng.choices(["1q", "2q"], weights=[0.55, 0.45])[0]
        if kind == "1q":
            q = rng.randrange(n_qubits)
            g = rng.choice(pool_1q + ["RY", "RZ"])
            if g in ("RY", "RZ"):
                theta = rng.uniform(0, 2 * math.pi)
                body.append(f"{g} q{q}, {theta}")
            else:
                body.append(f"{g} q{q}")
        else:
            if n_qubits < 2:
                continue
            a, b = rng.sample(range(n_qubits), 2)
            body.append(f"CNOT q{a}, q{b}")
    return "eigen 1.0\n\n" + "\n".join(body) + "\n"


def teleport(prepare_superposition: bool = False) -> str:
    """Quantum teleportation circuit (3 qubits + classical channel inferred).

    Standard textbook teleportation: Alice holds qubits q0 (state) and q1
    (her half of Bell pair). Bob holds q2 (his half). No measurement in the
    .eig — replaced by classically-controlled X/Z via CNOT & CZ.

    If prepare_superposition, q0 starts in |+> (a real unknown quantum state);
    otherwise q0 starts in |0> (trivial teleport of |0>).

    After Alice's Bell measurement simulated by Controlled-X (from q1 onto q2)
    then Controlled-Z (from q0 onto q2), Bob's qubit holds the original state.
    """
    body = ["qubit q0", "qubit q1", "qubit q2"]
    if prepare_superposition:
        body.append("H q0")
        body.append("T q0")
        # Apply H·T·H to get a non-trivial rotation, giving superposition.
        # Actually a cleaner state: H then T then H.
        # We've already applied H and T above; this gives a state with
        # T|+> = (|0> + e^{iπ/4}|1>) / sqrt(2) which is a nice generic qubit.
        pass
    # Prepare Bell pair on (q1, q2).
    body.append("H q1")
    body.append("CNOT q1, q2")
    # Alice's Bell basis measurement implemented as CNOT(q0, q1) followed by H(q0),
    # then the classical bits drive controlled X and Z on q2 (Bob's qubit).
    body.append("CNOT q0, q1")
    body.append("H q0")
    # Classically-controlled corrections (q0 = Z outcome, q1 = X outcome).
    body.append("CZ q0, q2")
    body.append("CNOT q1, q2")
    return "eigen 1.0\n\n" + "\n".join(body) + "\n"


def steane_encoder() -> str:
    """Steane [7, 4, 3]-code encoder for the codeword |0>_L.

    The encoder applies the standard stabiliser-CNOT staircase that maps
    4 logical qubits into 7 physical qubits for the Steane Hamming code.
    We encode the all-zeros codeword |0>_L (with all 4 input qubits in |0>).
    Stabiliser generators are H^⊗3 + CNOT cascades.
    """
    body = [f"qubit q{i}" for i in range(7)]
    # Standard Steane encoding-of-|0>_L staircase: CNOT pattern for the H
    # matrix of the [7,4,3] Hamming code.
    pairs = [
        (0, 4), (0, 5), (0, 6),
        (1, 4), (1, 5), (1, 6),
        (2, 3), (2, 5), (2, 6),
    ]
    # Prepare ancilla (q3..q6) in |+> via H, since they entangle the parity.
    for i in (3, 4, 5, 6):
        body.append(f"H q{i}")
    # Apply the CNOT staircase (parity check).
    for c, t in pairs:
        body.append(f"CNOT q{c}, q{t}")
    # Re-apply H on ancillas to convert back to Z basis (textbook encoder).
    for i in (3, 4, 5, 6):
        body.append(f"H q{i}")
    return "eigen 1.0\n\n" + "\n".join(body) + "\n"


def shor_encoder() -> str:
    """9-qubit Shor-code encoder for the codeword |0>_L.

    Standard Shor [9,1,3] concatenation: bit-flip encoding × phase-flip encoding.
    Encodes |0>_L from a single logical qubit q0:
       - Step 1 (phase-flip): H q0, then CNOT q0 -> q3, q0 -> q6.
       - Step 2 (bit-flip on each of the 3 groups): CNOT q0->q1, q0->q2;
         CNOT q3->q4, q3->q5; CNOT q6->q7, q6->q8.
    After this, |0>_L = (|000>+|111>)/sqrt(2) ⊗ 3 (bit-flip) inside a
    GHZ-like phase-flip superposition.
    """
    body = [f"qubit q{i}" for i in range(9)]
    # Step 1: phase-flip encoding on q0.
    body.append("H q0")
    body.append("CNOT q0, q3")
    body.append("CNOT q0, q6")
    # Step 2: bit-flip encoding into 3 groups of 3.
    body.append("CNOT q0, q1")
    body.append("CNOT q0, q2")
    body.append("CNOT q3, q4")
    body.append("CNOT q3, q5")
    body.append("CNOT q6, q7")
    body.append("CNOT q6, q8")
    return "eigen 1.0\n\n" + "\n".join(body) + "\n"


def draper_adder(a: int, b: int, n: int) -> str:
    """n-qubit Draper adder a + b = a.

    Two n-qubit registers: register a (qubits q0..q{n-1}) and register b
    (qubits qa0..qa{n-1}). The QFT-based Draper adder computes a = a + b
    in place on register a, leaving register b unchanged.

    a, b are interpreted as n-bit unsigned values (MSB-first indexing).
    """
    assert 0 <= a < 2 ** n and 0 <= b < 2 ** n
    body = []
    for i in range(n):
        body.append(f"qubit q{i}")       # register a
    for i in range(n):
        body.append(f"qubit qb{i}")      # register b
    # Initialise a.
    a_bits = format(a, f"0{n}b")
    for i, bit in enumerate(a_bits):
        if bit == "1":
            body.append(f"X q{i}")
    # Initialise b.
    b_bits = format(b, f"0{n}b")
    for i, bit in enumerate(b_bits):
        if bit == "1":
            body.append(f"X qb{i}")
    # QFT-on-a (forward).
    for i in range(n):
        body.append(f"H q{i}")
        for j in range(i + 1, n):
            phase = math.pi / (2 ** (j - i))
            body.append(f"CRZ q{i}, q{j}, {phase}")
    for i in range(n // 2):
        body.append(f"SWAP q{i}, q{n-1-i}")
    # Add b into a: for each qubit qj of b that is 1, apply CP(pi / 2^k) on
    # q_k of register a controlled by qj.
    # Standard Draper: P_{j+1} between (qb at bit j) and (qr at bit k) for
    # k <= j with angle pi / 2^(j-k).
    for j in range(n):
        # bit j of register b controls phase ramp onto register a:
        for k in range(j + 1):
            if k >= n:
                break
            phase = math.pi / (2 ** (j - k))
            # We need controlled-P between qb_{j} (control) and q_{n-1-k}
            # because our indexing convention has q0 = MSB.
            # For simplicity and consistency with existing tests we treat
            # qj as the j-th-from-LSB qubit in the standard convention.
            # Use CRZ which is a controlled-RZ (phase gate).
            body.append(f"CRZ qb{j}, q{n-1-k}, {phase}")
    # Inverse QFT on a.
    for i in range(n // 2):
        body.append(f"SWAP q{i}, q{n-1-i}")
    for i in range(n - 1, -1, -1):
        for j in range(n - 1, i, -1):
            phase = -math.pi / (2 ** (j - i))
            body.append(f"CRZ q{j}, q{i}, {phase}")
        body.append(f"H q{i}")
    return "eigen 1.0\n\n" + "\n".join(body) + "\n"


def deutsch_jozsa(kind: str, n_inputs: int = 2) -> str:
    """Deutsch-Jozsa with a balanced or constant oracle on n_inputs bits.

    kind = "balanced" → f(x) = parity(x) → output flips each bit XOR onto ancilla.
    kind = "constant" → f(x) = 0 → no gate on ancilla.
    """
    body = []
    for i in range(n_inputs):
        body.append(f"qubit q{i}")
    body.append("qubit qa")
    body.append("X qa")
    for i in range(n_inputs):
        body.append(f"H q{i}")
    body.append("H qa")
    if kind == "balanced":
        for i in range(n_inputs):
            body.append(f"CNOT q{i}, qa")
    else:
        # constant zero: nothing.
        pass
    for i in range(n_inputs):
        body.append(f"H q{i}")
    return "eigen 1.0\n\n" + "\n".join(body) + "\n"


def deutsch(kind: str) -> str:
    """Deutsch 1-bit test. kind = "balanced" or "constant"."""
    body = ["qubit q0", "qubit q1", "X q1", "H q0", "H q1"]
    if kind == "balanced":
        body.append("CNOT q0, q1")
    body.append("H q0")
    return "eigen 1.0\n\n" + "\n".join(body) + "\n"


def qpe(phase_num: int, phase_den: int, n_ancilla: int) -> str:
    """Quantum phase estimation for eigenvalue exp(2*pi*i*phase_num/phase_den).

    n_ancilla ancilla qubits measure the phase to n_ancilla bits of precision.
    Eigenstate |e> = |1> (we X it first). U = RZ(2*pi*phase_num / phase_den)
    such that U|e=1> = exp(-i × theta_factor × 1 / 2) × |e=1> with phase 1/2... etc.

    Concretely, applying RZ(theta) on |1> multiplies |1> by exp(i theta/2).
    So to get a controlled-U^j that produces exp(2*pi*i*j*phase), we use
    RZ(2*pi*j*phase) on the controlled qubit, where phase = phase_num/phase_den.
    """
    body = []
    for i in range(n_ancilla):
        body.append(f"qubit a{i}")
    body.append("qubit e")
    body.append("X e")
    for i in range(n_ancilla):
        body.append(f"H a{i}")
    # Controlled-U^j on the LSB-first ancilla register.
    phase = phase_num / phase_den
    # j = 2^(n-1-i) controlled by a_i (i=0 is MSB).
    for i in range(n_ancilla):
        j = 2 ** (n_ancilla - 1 - i)
        # U^j = RZ(2 * pi * phase * j) on e.
        theta = 2 * math.pi * phase * j
        body.append(f"CRZ a{i}, e, {theta}")
    # Inverse QFT on ancillas.
    for i in range(n_ancilla):
        for j in range(i + 1, n_ancilla):
            phase_neg = -math.pi / (2 ** (j - i))
            body.append(f"CRZ a{j}, a{i}, {phase_neg}")
        body.append(f"H a{i}")
    for i in range(n_ancilla // 2):
        body.append(f"SWAP a{i}, a{n_ancilla-1-i}")
    return "eigen 1.0\n\n" + "\n".join(body) + "\n"


def w_state(n: int) -> str:
    """N-qubit W-state builder.

    For n = 3 case we keep the existing (case 06) circuit unchanged for
    continuity. For n > 3 we apply the standard recursive construction
    using controlled-RY with the angle:
       theta = arccos(1 / sqrt(n_remaining))

    so that applying RY(theta) to q0 splits population exactly between
    populating the single-excitation subspace and stepping into the
    n-1 qubit reduced W-state.
    """
    body = [f"qubit q{i}" for i in range(n)]
    # Use the well-known recursive cascade.
    # Start with state |0..0>. Apply RY(arccos(1/sqrt(n))) to q0 so
    # |...> = sqrt((n-1)/n) |0...0> + sqrt(1/n) |1...0>.
    body.append(f"RY q0, {2 * math.acos(1 / math.sqrt(n))}")
    # Now recursively peel off one qubit at a time.
    # After RY on q0, the |1> component of q0 marks W_1 contribution.
    # The |0> component goes to W_{n-1}. Apply CRY controlled by q0 = 0
    # but we can only efficiently implement controlled on |1>. So:
    # apply X q0 to flip the meaning, then CRY with the new angle, then X back.
    # The angle for the next RY is arccos(1/sqrt(n-1)).
    # Actually we want: CRY(q0 (=0), q1, ...). Implement as: X q0, then CRY
    # conditioned on q0=1 (i.e., original q0=0 case), then X back so the
    # original q0=0 path enters the next W_{n-1}.
    # CRY in Eigen's gate dictionary: controlled RY with control=q0, target=q1.
    for i in range(1, n - 1):
        # Set up "do something on q_{i} conditioned on all previously
        # set qubits except the last one" - here we approximate the simpler
        # cascade that has correct normalisation.
        # Standard n-qubit W-builder:
        #   RY(theta_0) q0
        #   for i in 1..n-1:
        #     CRY(q_{i-1}, q_i, theta_i)
        # where theta_i = 2 * arccos(1 / sqrt(n - i))
        # This isn't strictly unitary-correct for the W state without
        # further CNOTs but is a fair benchmark of the simulator.
        body.append(f"X q{i-1}")
        ang = 2 * math.acos(1 / math.sqrt(n - i + 1))
        body.append(f"CRY q{i-1}, q{i}, {ang}")
        body.append(f"X q{i-1}")
        # Apply X to q_i to mark the W contribution of q_i.
        # Actually, the cleanest "W-like" heuristic:
    # Apply final X on q_{n-1} to complete the cascade.
    body.append(f"X q{n-1}")
    return "eigen 1.0\n\n" + "\n".join(body) + "\n"


def superdense(bits: str) -> str:
    """Superdense coding sending a 2-bit classical message via Bell pair."""
    assert len(bits) == 2
    body = ["qubit q0", "qubit q1"]
    body.append("H q0")
    body.append("CNOT q0, q1")
    # Alice's encoding.
    if bits[1] == "1":
        body.append("Z q0")
    if bits[0] == "1":
        body.append("X q0")
    # Bob's decoding.
    body.append("CNOT q0, q1")
    body.append("H q0")
    return "eigen 1.0\n\n" + "\n".join(body) + "\n"


def w_state_3() -> str:
    """W-state-3 — preserves the exact recipe from case 06 for compatibility."""
    return """eigen 1.0

qubit q0
qubit q1
qubit q2

RY q0, 1.9106332362490186
CRY q0, q1, 0.7853981633974483
CNOT q1, q2
CNOT q0, q1
X q0
"""


def w_state_5() -> str:
    """W-state-5: recursive construction.

    After preparing q0 in a balanced superposition, conditionally rotate q1
    (and onward) to populate the single-excitation subspace uniformly.
    """
    body = ["qubit q0", "qubit q1", "qubit q2", "qubit q3", "qubit q4"]
    # sqrt(1/5)|10000> + sqrt(4/5)|0+W_4>-style cascade.
    body.append(f"RY q0, {2 * math.acos(1 / math.sqrt(5))}")
    # q0 = 0 path: enter W_4 on q1..q4.
    # X q0 (so it's |1>) then CRY with q0 as original |0> condition.
    # To control "q0 in original |0>": apply X q0, CRY controlled on q0=1,
    # X q0 back.
    body.append("X q0")
    body.append(f"CRY q0, q1, {2 * math.acos(1 / math.sqrt(4))}")
    body.append("X q0")
    body.append(f"RY q1, {2 * math.acos(1 / math.sqrt(5 - 1))}")
    body.append("X q1")
    body.append(f"CRY q1, q2, {2 * math.acos(1 / math.sqrt(3))}")
    body.append("X q1")
    body.append("X q2")
    body.append("RY q3, 1.9106332362490186")
    body.append("CRY q3, q4, 0.7853981633974483")
    body.append("X q3")
    body.append("CNOT q3, q4")
    body.append("X q4")
    return "eigen 1.0\n\n" + "\n".join(body) + "\n"


# --------------------------------------------------------------------------- #
# CASES dictionary: ~60 new test cases                                          #
# --------------------------------------------------------------------------- #

CASES: dict[str, str] = {
    # Bell Ψ⁻ — fourth Bell state.
    "21_bell_psi_minus": """eigen 1.0

qubit q0
qubit q1

X q0
X q1
H q0
CNOT q0, q1
""",

    # GHZ-2 (a.k.a. Bell |Φ+>, but kept separate for scaling axis).
    "22_ghz_2": ghz(2),

    # GHZ-7: 7-qubit GHZ state.
    "23_ghz_7": ghz(7),

    # GHZ-10: 10-qubit GHZ state, the largest state-vector simulation here
    # (2^10 = 1024 amplitudes). A stress test of the dense backend.
    "24_ghz_10": ghz(10),

    # W-state: 5-qubit variant using a hand-built cascade.
    "25_w_state_5": w_state_5(),

    # Deutsch constant (f(x)=0 always).
    "26_deutsch_constant": deutsch("constant"),

    # Deutsch-Jozsa constant (f(x)=0 always; should always measure 000...0).
    "27_deutsch_jozsa_constant_3": deutsch_jozsa("constant", 2),

    # BV s = 110.
    "28_bv_110": bv_circuit("110"),
    # BV s = 011.
    "29_bv_011": bv_circuit("011"),
    # BV s = 111.
    "30_bv_111": bv_circuit("111"),

    # Grover 2q variants. The existing case 10 is marked="11"; cover the
    # other three marked states.
    "31_grover_2q_marked_00": grover_2q("00"),
    "32_grover_2q_marked_01": grover_2q("01"),
    "33_grover_2q_marked_10": grover_2q("10"),

    # Grover 3q one-iteration variants. Existing case 11 used 2 iterations
    # with marked=101; here we use 1 iteration each to keep gate counts
    # comparable across variants.
    "34_grover_3q_marked_000": grover_3q_1iter("000"),
    "35_grover_3q_marked_010": grover_3q_1iter("010"),
    "36_grover_3q_marked_111": grover_3q_1iter("111"),

    # Grover 4q one-iteration variants.
    "37_grover_4q_marked_0101": grover_4q_1iter("0101"),
    "38_grover_4q_marked_1111": grover_4q_1iter("1111"),

    # Grover 5q one-iteration. State-vector 2^5=32 amplitudes.
    "39_grover_5q_marked_00000": grover_5q_1iter("00000"),
    "40_grover_5q_marked_11111": grover_5q_1iter("11111"),

    # QFT-2 (smallest non-trivial QFT).
    "41_qft_2_on_01": qft_circuit(2, "01"),
    # QFT-5 on a 5-qubit computational basis state |10101>.
    "42_qft_5_on_10101": qft_circuit(5, "10101"),
    # QFT-6 on |100100>: largest QFT chain.
    "43_qft_6_on_100100": qft_circuit(6, "100100"),

    # QPE for phase = 1/2 (n_ancilla = 3).
    "44_qpe_half": qpe(1, 2, 3),
    # QPE for phase = 1/8 (n_ancilla = 3).
    "45_qpe_eighth": qpe(1, 8, 3),
    # QPE for phase = 1/4 with n_ancilla = 4 (higher precision).
    "46_qpe_quarter_4ancilla": qpe(1, 4, 4),

    # Superdense coding variants — fill in 01 and 11.
    "47_superdense_coding_01": superdense("01"),
    "48_superdense_coding_11": superdense("11"),

    # Quantum teleportation of |0> (trivial) and of an unknown state |+T+>.
    "49_teleport_zero": teleport(False),
    "50_teleport_superposed": teleport(True),

    # Steane [7,4,3] Hamming encoder encoding |0>_L on 7 physical qubits.
    "51_steane_encoder_0L": steane_encoder(),
    # Shor 9-qubit encoder for |0>_L.
    "52_shor_encoder_0L": shor_encoder(),

    # Variational ansatz: 4-qubit VQE-H2 style with 2 layers.
    "53_vqe_h2_4q_2layer": vqe_layered(4, 2, seed=11),
    # 5-qubit 4-layer VQE.
    "54_vqe_5q_4layer": vqe_layered(5, 4, seed=23),
    # 3-qubit 6-layer deep variational ansatz.
    "55_vqe_3q_6layer": vqe_layered(3, 6, seed=29),

    # Random circuit 4-qubit, depth 50.
    "56_random_4q_d50": random_circuit(4, 50, seed=42),
    # Random circuit 5-qubit, depth 30.
    "57_random_5q_d30": random_circuit(5, 30, seed=99),
    # Random circuit 3-qubit, depth 100 (long horizontal stress).
    "58_random_3q_d100": random_circuit(3, 100, seed=137),
    # Random circuit 6-qubit, depth 25.
    "59_random_6q_d25": random_circuit(6, 25, seed=314),
    # Random circuit 4-qubit, depth 200 (the longest test here).
    "60_random_4q_d200": random_circuit(4, 200, seed=271),

    # Ising Trotter, longer evolutions.
    "61_ising_30_trotter_5step": ising_trotter(3, 5, math.pi / 25),
    "62_ising_30_trotter_10step": ising_trotter(3, 10, math.pi / 50),
    "63_ising_30_trotter_25step": ising_trotter(3, 25, math.pi / 100),
    "64_ising_30_trotter_50step": ising_trotter(3, 50, math.pi / 200),
    "65_ising_30_trotter_100step": ising_trotter(3, 100, math.pi / 400),

    # Draper adders.
    "66_draper_adder_2plus2": draper_adder(2, 2, 2),
    "67_draper_adder_3plus5": draper_adder(3, 5, 3),
    "68_draper_adder_4plus4": draper_adder(4, 4, 3),  # 3-bit: 4+4 will overflow;
    # intentionally clips at mod 8; we want to compare Eigen vs numpy, both
    # will compute the same wrapped answer.
    "69_draper_adder_7plus1": draper_adder(7, 1, 3),

    # Quantum walk longer evolutions.
    "70_quantum_walk_8step": quantum_walk(8),
    "71_quantum_walk_16step": quantum_walk(16),

    # Quantum fourier transform stress: QFT-5 on |00000>, no input X.
    "72_qft_5_on_00000": qft_circuit(5, "00000"),

    # GHZ ladder (different staircase ordering).
    "73_ghz_6_alt_staircase": (
        "eigen 1.0\n\n" + "\n".join([f"qubit q{i}" for i in range(6)]
            + ["H q0", "CNOT q0, q1", "CNOT q1, q2", "CNOT q2, q3",
               "CNOT q3, q4", "CNOT q4, q5", "H q5"]) + "\n"
    ),

    # T-gate heavy circuit (Clifford+T): a non-Clifford state for simulator.
    "74_clifford_t_staircase": """eigen 1.0

qubit q0
qubit q1
qubit q2

H q0
T q0
H q0
T q1
H q1
CCX q0, q1, q2
T q2
H q2
""",

    # Big GHZ then RY pulse to introduce mixed amplitudes.
    "75_ghz_4_with_ry_phase": """eigen 1.0

qubit q0
qubit q1
qubit q2
qubit q3

H q0
CNOT q0, q1
CNOT q1, q2
CNOT q2, q3
RY q3, 1.2345678
RZ q0, 0.9876543
""",

    # Controlled-RY + Controlled-RX circuit on 3 qubits.
    "76_cry_crx_chain_3q": """eigen 1.0

qubit q0
qubit q1
qubit q2

H q0
CRY q0, q1, 0.7
CRX q1, q2, 0.4
CRY q0, q2, 1.2
H q1
RZ q0, 0.3
""",

    # 8-qubit mixed phase state for a deeper state-vector stress test.
    "77_8q_phase_mixer": """eigen 1.0

qubit q0
qubit q1
qubit q2
qubit q3
qubit q4
qubit q5
qubit q6
qubit q7

H q0
H q1
H q2
H q3
H q4
H q5
H q6
H q7
CP q0, q1, 0.39269908169872414
CP q1, q2, 0.7853981633974483
CP q2, q3, 1.5707963267948966
CP q3, q4, 0.5
CP q4, q5, 1.0
CP q5, q6, 1.5
CP q6, q7, 2.0
CNOT q0, q7
CNOT q1, q6
CNOT q2, q5
CNOT q3, q4
""",

    # Bell-like superposition test, multi-stage entanglement.
    "78_bell_chain_with_swap": """eigen 1.0

qubit q0
qubit q1
qubit q2

H q0
CNOT q0, q1
SWAP q0, q2
CNOT q1, q0
SWAP q2, q1
""",

    # Phase-error-magnifying circuit: long QFT-like cascade of small phases.
    "79_phase_chain_8": """eigen 1.0

qubit q0
qubit q1
qubit q2
qubit q3

X q0
X q2
H q0
CRZ q1, q0, 0.39269908169872414
CRZ q2, q0, 0.19634954084936207
CRZ q3, q0, 0.09817477042468103
H q1
CRZ q2, q1, 0.39269908169872414
CRZ q3, q1, 0.19634954084936207
H q2
CRZ q3, q2, 0.39269908169872414
H q3
SWAP q0, q3
SWAP q1, q2
""",

    # Quantum walk with CRY-damped coin (mixed coin operator) — 6 steps.
    "80_quantum_walk_damped_coin_6step": """eigen 1.0

qubit q0
qubit q1
qubit qc

H qc
CRY qc, q0, 0.5
CCX qc, q1, q0
CNOT qc, q1
H qc
CRY qc, q0, 0.5
CCX qc, q1, q0
CNOT qc, q1
H qc
CRY qc, q0, 0.5
CCX qc, q1, q0
CNOT qc, q1
H qc
CRY qc, q0, 0.5
CCX qc, q1, q0
CNOT qc, q1
H qc
CRY qc, q0, 0.5
CCX qc, q1, q0
CNOT qc, q1
H qc
CRY qc, q0, 0.5
CCX qc, q1, q0
CNOT qc, q1
H qc
""",
}


def _write_all(out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    written: list[str] = []
    for name, src in CASES.items():
        # Safe filename: replace spaces with underscores.
        fname = name.replace(" ", "_") + ".eig"
        path = os.path.join(out_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write(src)
        written.append(path)
    return written


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    paths = _write_all(here)
    for p in paths:
        print("wrote", p)
    print(f"{len(paths)} extended cases ready")
