"""Generates 20 hard quantum-mechanics Eigen benchmark programs.

Each program in ``cases/`` is a quantum circuit that prepares a specific
non-trivial pure state (Bell, GHZ, W-state, Grover, BV, QFT, phase
estimation, Ising Trotterization, Draper adder, quantum walk, superdense
coding ...). All end WITHOUT measurement so the runtime statevector is
deterministic and directly comparable to the numpy reference.

Run this script once to (re)materialize all 20 ``.eig`` files:

    python -m benchmarks.quantum_hard.cases._generate
"""

from __future__ import annotations

import os

CASES: dict[str, str] = {
    # 1. Bell |Φ+⟩
    "01_bell_phi_plus": """eigen 1.0

qubit q0
qubit q1

H q0
CNOT q0, q1
""",
    # 2. Bell |Ψ+⟩
    "02_bell_psi_plus": """eigen 1.0

qubit q0
qubit q1

X q0
H q0
CNOT q0, q1
""",
    # 3. Bell |Φ−⟩
    "03_bell_phi_minus": """eigen 1.0

qubit q0
qubit q1

H q0
CNOT q0, q1
Z q0
""",
    # 4. GHZ-3
    "04_ghz_3": """eigen 1.0

qubit q0
qubit q1
qubit q2

H q0
CNOT q0, q1
CNOT q1, q2
""",
    # 5. GHZ-5
    "05_ghz_5": """eigen 1.0

qubit q0
qubit q1
qubit q2
qubit q3
qubit q4

H q0
CNOT q0, q1
CNOT q1, q2
CNOT q2, q3
CNOT q3, q4
""",
    # 6. W-state:  (|001>+|010>+|100>)/sqrt(3)
    "06_w_state_3": """eigen 1.0

qubit q0
qubit q1
qubit q2

# Ry(arccos(1/sqrt(3))) on q0, splits population between |0> and |1>.
RY q0, 1.9106332362490186
# Now RY on q1 conditioned on q0=1: rotate q1 by arccos(1/sqrt(2)) when q0=1
CRY q0, q1, 0.7853981633974483
# Controlled CNOT q1 -> q2
CNOT q1, q2
CNOT q0, q1
X q0
""",
    # 7. Deutsch (balanced f, single-bit)
    "07_deutsch_balanced": """eigen 1.0

qubit q0  # decision bit
qubit q1  # ancilla

X q1
H q0
H q1
# Oracle for f(x)=x: CNOT q0 -> q1
CNOT q0, q1
# Final Hadamards
H q0
H q1
""",
    # 8. Deutsch-Jozsa balanced on 2 input bits (3 qubit total)
    "08_deutsch_jozsa_balanced_3": """eigen 1.0

qubit q0
qubit q1
qubit q2

X q2
H q0
H q1
H q2
# Oracle: balanced f(x0,x1) = x0 XOR x1
CNOT q0, q2
CNOT q1, q2
H q0
H q1
""",
    # 9. Bernstein-Vazirani hidden string s = 101 (3-bit)
    "09_bernstein_vazirani_101": """eigen 1.0

qubit q0
qubit q1
qubit q2
qubit qa

X qa
H q0
H q1
H q2
H qa
# Oracle for s = 0b101 (q0=1, q1=0, q2=1, MSB-first => q2 in position s_2)
CNOT q0, qa
CNOT q2, qa
H q0
H q1
H q2
""",
    # 10. Grover 2-qubit (marked = |11>)
    "10_grover_2q_marked_11": """eigen 1.0

qubit q0
qubit q1

H q0
H q1
# Oracle for |11>: CZ inverts |11> phase
CZ q0, q1
# Diffusion operator: H - X - CZ - X - H
H q0
H q1
X q0
X q1
CZ q0, q1
X q0
X q1
H q0
H q1
""",
    # 11. Grover 3-qubit marked = |101>
    "11_grover_3q_marked_101": """eigen 1.0

qubit q0  # MSB
qubit q1
qubit q2  # LSB

H q0
H q1
H q2
# Oracle for |101>: phase-flip. To flip ONLY |101> = q0=1, q1=0, q2=1
# Apply X to q1 (to make it |1> in oracle), then CCX/CCZ, then X q1 back.
X q1
# CCZ via H+CCX on q2 + H q2:
H q2
CCX q0, q1, q2
H q2
X q1
# Diffusion: H on all qubits, then -|0..0><0..0| + I, then H back
# Implement as H; X; multi-CZ; X; H. The multi-CZ is a CCZ on q0,q1,q2.
H q0
H q1
H q2
X q0
X q1
X q2
H q2
CCX q0, q1, q2
H q2
X q0
X q1
X q2
H q0
H q1
H q2
H q0
H q1
H q2
# A SECOND iteration for amplification
X q1
H q2
CCX q0, q1, q2
H q2
X q1
H q0
H q1
H q2
X q0
X q1
X q2
H q2
CCX q0, q1, q2
H q2
X q0
X q1
X q2
H q0
H q1
H q2
""",
    # 12. QFT-3 on |001>
    "12_qft_3_on_001": """eigen 1.0

qubit q0
qubit q1
qubit q2

# Prepare |001> = q2 in |1>
X q2

# QFT-3 manual decomposition:
# H on q0 (MSB), then controlled-R2 on q0<-q1, R3 on q0<-q2.
# H on q1 (middle), controlled-R2 on q1<-q2.
# H on q2 (LSB).
# Then SWAP q0 <-> q2 to put bits in conventional QFT order.
H q0
CRZ q1, q0, 1.5707963267948966   # pi/2
CRZ q2, q0, 0.7853981633974483   # pi/4
H q1
CRZ q2, q1, 1.5707963267948966   # pi/2
H q2
SWAP q0, q2
""",
    # 13. QFT-4 on |1101>
    "13_qft_4_on_1101": """eigen 1.0

qubit q0  # MSB
qubit q1
qubit q2
qubit q3  # LSB

# Prepare |1101>
X q0
X q1
X q3

H q0
CRZ q1, q0, 1.5707963267948966        # pi/2
CRZ q2, q0, 0.7853981633974483        # pi/4
CRZ q3, q0, 0.39269908169872414       # pi/8
H q1
CRZ q2, q1, 1.5707963267948966        # pi/2
CRZ q3, q1, 0.7853981633974483        # pi/4
H q2
CRZ q3, q2, 1.5707963267948966        # pi/2
H q3
SWAP q0, q3
SWAP q1, q2
""",
    # 14. Phase estimation for phase = 1/4 (3 ancilla + 1 eigenvector qubit)
    "14_phase_estimation_quarter": """eigen 1.0

qubit a0  # MSB ancilla
qubit a1
qubit a2  # LSB ancilla (gets the LEAST-significant bit of the phase)
qubit e   # eigenvector: |1> is eigenvector of Z with eigenvalue -1, but
         # we want a U with eigenvalue exp(2*pi*i*phase) where phase = 1/4

X e
H a0
H a1
H a2

# Controlled-U on e with control = a2 (LSB)  — U = RZ(pi/2) -> phase 1/4
CRZ a2, e, 1.5707963267948966
# Controlled-U^2 on e with control = a1   — U^2 = RZ(pi)  -> phase 1/2 (still exp(i*pi) - 1/4 = 0)
# WAIT — for phase estimation, U^j must produce eigenvalue exp(2*pi*i*j*phase).
# phase = 1/4 implies U = exp(2*pi*i*1/4) on the eigenstate = RZ(pi/2) (modulo global phase).
# U^2 = RZ(pi), so eigenvalue exp(2*pi*i/2) = exp(i*pi) = -1.
# U^4 = RZ(2*pi) = identity, eigenvalue 1.
CRZ a1, e, 3.141592653589793             # pi (U^2)
# Controlled-U^4 on e with control = a0  — U^4 = RZ(2pi) = I, eigenvalue 1.
# In a proper QPE we'd apply CU^4 but here it's identity so we omit.
# Inverse QFT on the ancilla register to extract the phase bit-pattern.
H a0
CRZ a1, a0, -1.5707963267948966          # -pi/2 (inverse R2)
CRZ a2, a0, -0.7853981633974483          # -pi/4 (inverse R3)
H a1
CRZ a2, a1, -1.5707963267948966          # -pi/2
H a2
SWAP a0, a2
""",
    # 15. Superdense coding: send "00" — Alice does nothing; Bob undoes Bell.
    "15_superdense_coding_00": """eigen 1.0

qubit q0  # Alice's half
qubit q1  # Bob's half

# Prepare Bell |Phi+>
H q0
CNOT q0, q1
# Alice encodes "00": identity, no operation.
# Bob decodes: CNOT q0 -> q1, then H q0.
CNOT q0, q1
H q0
""",
    # 16. Superdense coding: send "10" — Alice applies X; Bob undoes Bell.
    "16_superdense_coding_10": """eigen 1.0

qubit q0
qubit q1

H q0
CNOT q0, q1
# Alice encodes "10": apply X to her half of the entangled pair.
X q0
# Bob decodes:
CNOT q0, q1
H q0
""",
    # 17. Quantum walk on a 4-node cycle, 4 steps, with coin qubit.
    # Position register q0,q1 (4 positions). Coin register q2.
    "17_quantum_walk_cycle4_4step": """eigen 1.0

qubit q0  # position MSB
qubit q1  # position LSB
qubit qc  # coin

# Initialise coin in |+>
H qc

# One step = coin rotation + conditional shift.
# Conditional shift on (q0,q1) controlled by qc=0 vs qc=1:
# If qc=0, decrement position by 1; if qc=1, increment by 1.
# Decrement on 4-cycle: 00->11, 01->00, 10->01, 11->10
# Implement: if qc=0 then SWAP(q1,q0) - hmm we lack conditional-on-classical.

# Simpler model: 4 separate single-step operators; each is
#   TOFFOLI(qc, q1) -> q0; CNOT(qc, q1) — this is a (conditional) binary +1.
# We use CCX as TOFFOLI to flip q0 when both qc=1 and q1=1.
# And CNOT(qc,q1) flips q1 when qc=1. (This is +1 modulo 4 on the
# position when qc=1, identity when qc=0.)
# Step 1:
CCX qc, q1, q0
CNOT qc, q1
H qc

# Step 2:
CCX qc, q1, q0
CNOT qc, q1
H qc

# Step 3:
CCX qc, q1, q0
CNOT qc, q1
H qc

# Step 4:
CCX qc, q1, q0
CNOT qc, q1
H qc
""",
    # 18. Ising 3-spin Trotterized evolution over t = pi/4 (first-order, 4 steps).
    "18_ising_3_trotter": """eigen 1.0

qubit q0
qubit q1
qubit q2

# Prepare |+++> with q0=|+>, q1=|->, q2=|+> for non-trivial dynamics.
H q0
X q1
H q1
H q2

# Trotter step: e^{-i*ZZ*dt} ≈ RZ(2*dt) on each qubit applied as CNOT - RZ - CNOT
# Hamiltonian H = Z0Z1 + Z1Z2; dt = pi/16
# Apply e^{-i dt ZZ} on pair (q0, q1):
CNOT q0, q1
RZ q1, 0.19634954084936207      # pi/16
CNOT q0, q1
# Repeat for pair (q1, q2):
CNOT q1, q2
RZ q2, 0.19634954084936207
CNOT q1, q2

# Repeat the same 4 times for total t = pi/4.
CNOT q0, q1
RZ q1, 0.19634954084936207
CNOT q0, q1
CNOT q1, q2
RZ q2, 0.19634954084936207
CNOT q1, q2

CNOT q0, q1
RZ q1, 0.19634954084936207
CNOT q0, q1
CNOT q1, q2
RZ q2, 0.19634954084936207
CNOT q1, q2

CNOT q0, q1
RZ q1, 0.19634954084936207
CNOT q0, q1
CNOT q1, q2
RZ q2, 0.19634954084936207
CNOT q1, q2
""",
    # 19. Draper adder 1+1=2 (2-qubit input registers; minimal viable form).
    # Prepare |a>|b> = |01>|01>, apply QFT to a, then controlled-phase-gates
    # for b, then QFT^-1: result a = a + b = |10>.
    "19_draper_adder_1plus1": """eigen 1.0

qubit a0      # MSB of register a (= 0)
qubit a1      # LSB of register a (= 1)  -> a = 01
qubit b0      # MSB of register b (= 0)
qubit b1      # LSB of register b (= 1)  -> b = 01

# Prepare a = |01>, b = |01>
X a1
X b1

# QFT3-on-a using 2 qubits (QFT2):
H a0
CRZ a1, a0, 1.5707963267948966   # pi/2
H a1
SWAP a0, a1

# Add b1 (LSB of b) into register a: apply CP(pi/2^k) controlled by b1 at position k.
# For 2-qubit adder the SUM bits are stored in a; b is unchanged.
# j=1: b1 contributes phase pi/2^0 = pi to a's LSB.
CZ b1, a1
# j=0: b1 contributes phase pi/2^1 = pi/2 to a's MSB.
CRZ b1, a0, 1.5707963267948966
# b0 contributes only at MSB (phase pi/2^0*... but b0=0 anyway; skip CCZ).

# Inverse QFT2 on a:
SWAP a0, a1
H a1
CRZ a1, a0, -1.5707963267948966
H a0

# After this, a should hold |10> = 2.
""",
    # 20. Grover 4-qubit, marked = |1000>; uses 1 ancilla for the
    # multi-controlled-Z cascade. Optimal iteration count for N=16 marked states
    # is ~pi/4 * sqrt(16) = pi ≈ 3 iterations. We do 3.
    "20_grover_4q_marked_1000": """eigen 1.0

qubit q0
qubit q1
qubit q2
qubit q3
qubit anc  # ancilla for 4-controlled-Z cascade

H q0
H q1
H q2
H q3

# Oracle for |1000>: X on q1,q2,q3 to map to |1111>, apply 4-controlled-Z, then X back.
X q1
X q2
X q3
H q3
CCX q0, q1, anc
CCX anc, q2, q3
CCX q0, q1, anc
H q3
X q1
X q2
X q3

# Diffusion: H^⊗4 ; X^⊗4 ; multi-CZ(q0,q1,q2,q3) ; X^⊗4 ; H^⊗4.
H q0
H q1
H q2
H q3
X q0
X q1
X q2
X q3
H q3
CCX q0, q1, anc
CCX anc, q2, q3
CCX q0, q1, anc
H q3
X q0
X q1
X q2
X q3
H q0
H q1
H q2
H q3

# Iteration 2 — repeat oracle+diffusion.
X q1
X q2
X q3
H q3
CCX q0, q1, anc
CCX anc, q2, q3
CCX q0, q1, anc
H q3
X q1
X q2
X q3
H q0
H q1
H q2
H q3
X q0
X q1
X q2
X q3
H q3
CCX q0, q1, anc
CCX anc, q2, q3
CCX q0, q1, anc
H q3
X q0
X q1
X q2
X q3
H q0
H q1
H q2
H q3

# Iteration 3 — repeat.
X q1
X q2
X q3
H q3
CCX q0, q1, anc
CCX anc, q2, q3
CCX q0, q1, anc
H q3
X q1
X q2
X q3
H q0
H q1
H q2
H q3
X q0
X q1
X q2
X q3
H q3
CCX q0, q1, anc
CCX anc, q2, q3
CCX q0, q1, anc
H q3
X q0
X q1
X q2
X q3
H q0
H q1
H q2
H q3
""",
}


def _write_all(out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    written: list[str] = []
    for name, src in CASES.items():
        path = os.path.join(out_dir, name + ".eig")
        with open(path, "w", encoding="utf-8") as f:
            f.write(src)
        written.append(path)
    return written


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    paths = _write_all(here)
    for p in paths:
        print("wrote", p)
    print(f"{len(paths)} cases ready")
