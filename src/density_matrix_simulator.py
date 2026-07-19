# Density Matrix Simulator for Eigen
import numpy as np
import math
import cmath
import random

# === P0 (sol.md §1.3): cache standard single-qubit gate matrices as NumPy
# arrays so each H/X/Y/Z/S/T call does not allocate a fresh np.ndarray on
# every gate application. The parameterised gates (RX/RY/RZ/CP/CR*/...)
# cannot benefit from a static cache because the matrix depends on theta;
# they keep building a small 2x2 array per call.
_INV_SQRT2 = 1.0 / math.sqrt(2.0)
_GATE_NP_CACHE = {
    'H':  np.array([[_INV_SQRT2, _INV_SQRT2], [_INV_SQRT2, -_INV_SQRT2]], dtype=complex),
    'X':  np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    'Y':  np.array([[0.0, -1j], [1j, 0.0]], dtype=complex),
    'Z':  np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
    'S':  np.array([[1.0, 0.0], [0.0, 1j]], dtype=complex),
    'T':  np.array([[1.0, 0.0], [0.0, _INV_SQRT2 + _INV_SQRT2 * 1j]], dtype=complex),
    'I2': np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
    'P0': np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex),
    'P1': np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex),
}


class DensityMatrixSimulator:
    def __init__(self, rng=None, seed=None):
        self.qubit_map = {}
        self.num_qubits = 0
        self._state = None
        self._gate_count = 0
        self.rng = rng if rng is not None else random.Random(seed)

    def get_qubit_index(self, name: str) -> int:
        return self.qubit_map[name]

    def allocate_qubit(self, name: str):
        if name in self.qubit_map:
            return
        idx = len(self.qubit_map)
        self.qubit_map[name] = idx
        self.num_qubits += 1
        
        # New qubit state is |0><0|
        rho_new = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
        if self._state is None:
            self._state = rho_new
        else:
            # kron in LSB-first order: new qubit name is added as LSB (idx)
            # So it goes to the rightmost in tensor product
            self._state = np.kron(self._state, rho_new)

    def _get_1qubit_operator_full(self, idx: int, op: np.ndarray) -> np.ndarray:
        # LSB-first Kronecker product: np.kron(np.kron(I_right, op), I_left)
        I_left = np.eye(1 << idx)
        I_right = np.eye(1 << (self.num_qubits - 1 - idx))
        return np.kron(I_right, np.kron(op, I_left))

    def _apply_unitary(self, U: np.ndarray):
        self._state = U @ self._state @ U.conj().T
        self._gate_count += 1
        if self._gate_count % 100 == 0:
            tr = np.trace(self._state).real
            if tr > 1e-15:
                self._state /= tr

    def _apply_channel(self, idx: int, kraus_ops: list[np.ndarray]):
        new_state = np.zeros_like(self._state)
        for op in kraus_ops:
            U_full = self._get_1qubit_operator_full(idx, op)
            new_state += U_full @ self._state @ U_full.conj().T
        self._state = new_state
        self._gate_count += 1
        if self._gate_count % 100 == 0:
            tr = np.trace(self._state).real
            if tr > 1e-15:
                self._state /= tr

    # 1-Qubit Gates
    def apply_1qubit_gate(self, name: str, gate_matrix: list[list[complex]]):
        """Apply an arbitrary 2x2 unitary by qubit name.

        Public API used by QuantumSimulator.apply_1qubit_gate; previously
        missing, so that call path crashed with AttributeError on the
        density backend.
        """
        idx = self.get_qubit_index(name)
        U = np.asarray(gate_matrix, dtype=complex)
        self._apply_unitary(self._get_1qubit_operator_full(idx, U))

    def H(self, q: str):
        idx = self.get_qubit_index(q)
        self._apply_unitary(self._get_1qubit_operator_full(idx, _GATE_NP_CACHE['H']))

    def X(self, q: str):
        idx = self.get_qubit_index(q)
        self._apply_unitary(self._get_1qubit_operator_full(idx, _GATE_NP_CACHE['X']))

    def Y(self, q: str):
        idx = self.get_qubit_index(q)
        self._apply_unitary(self._get_1qubit_operator_full(idx, _GATE_NP_CACHE['Y']))

    def Z(self, q: str):
        idx = self.get_qubit_index(q)
        self._apply_unitary(self._get_1qubit_operator_full(idx, _GATE_NP_CACHE['Z']))

    def S(self, q: str):
        idx = self.get_qubit_index(q)
        self._apply_unitary(self._get_1qubit_operator_full(idx, _GATE_NP_CACHE['S']))

    def T(self, q: str):
        idx = self.get_qubit_index(q)
        self._apply_unitary(self._get_1qubit_operator_full(idx, _GATE_NP_CACHE['T']))

    def RX(self, q: str, theta: float):
        idx = self.get_qubit_index(q)
        cos_val = math.cos(theta / 2)
        sin_val = math.sin(theta / 2)
        U = np.array([[cos_val, -1j * sin_val], [-1j * sin_val, cos_val]], dtype=complex)
        self._apply_unitary(self._get_1qubit_operator_full(idx, U))

    def RY(self, q: str, theta: float):
        idx = self.get_qubit_index(q)
        cos_val = math.cos(theta / 2)
        sin_val = math.sin(theta / 2)
        U = np.array([[cos_val, -sin_val], [sin_val, cos_val]], dtype=complex)
        self._apply_unitary(self._get_1qubit_operator_full(idx, U))

    def RZ(self, q: str, theta: float):
        idx = self.get_qubit_index(q)
        val_0 = cmath.exp(-1j * theta / 2)
        val_1 = cmath.exp(1j * theta / 2)
        U = np.array([[val_0, 0], [0, val_1]], dtype=complex)
        self._apply_unitary(self._get_1qubit_operator_full(idx, U))

    # 2-Qubit Gates
    def CNOT(self, control: str, target: str):
        c_idx = self.get_qubit_index(control)
        t_idx = self.get_qubit_index(target)
        self._apply_controlled_unitary(c_idx, t_idx, _GATE_NP_CACHE['X'])

    def CZ(self, control: str, target: str):
        c_idx = self.get_qubit_index(control)
        t_idx = self.get_qubit_index(target)
        self._apply_controlled_unitary(c_idx, t_idx, _GATE_NP_CACHE['Z'])

    def SWAP(self, q1: str, q2: str):
        self.CNOT(q1, q2)
        self.CNOT(q2, q1)
        self.CNOT(q1, q2)

    # 3-Qubit Gates
    def CCX(self, control1: str, control2: str, target: str):
        c1_idx = self.get_qubit_index(control1)
        c2_idx = self.get_qubit_index(control2)
        t_idx = self.get_qubit_index(target)
        U_full = self._get_double_controlled_operator_full(c1_idx, c2_idx, t_idx, _GATE_NP_CACHE['X'])
        self._apply_unitary(U_full)

    def CSWAP(self, control: str, q1: str, q2: str):
        self.CNOT(q2, q1)
        self.CCX(control, q1, q2)
        self.CNOT(q2, q1)

    # Parameterized Controlled Rotations
    def CP(self, control: str, target: str, theta: float):
        c_idx = self.get_qubit_index(control)
        t_idx = self.get_qubit_index(target)
        val = cmath.exp(1j * theta)
        U_t = np.array([[1.0, 0.0], [0.0, val]], dtype=complex)
        self._apply_controlled_unitary(c_idx, t_idx, U_t)

    def CRX(self, control: str, target: str, theta: float):
        c_idx = self.get_qubit_index(control)
        t_idx = self.get_qubit_index(target)
        cos_val = math.cos(theta / 2)
        sin_val = math.sin(theta / 2)
        U_t = np.array([[cos_val, -1j * sin_val], [-1j * sin_val, cos_val]], dtype=complex)
        self._apply_controlled_unitary(c_idx, t_idx, U_t)

    def CRY(self, control: str, target: str, theta: float):
        c_idx = self.get_qubit_index(control)
        t_idx = self.get_qubit_index(target)
        cos_val = math.cos(theta / 2)
        sin_val = math.sin(theta / 2)
        U_t = np.array([[cos_val, -sin_val], [sin_val, cos_val]], dtype=complex)
        self._apply_controlled_unitary(c_idx, t_idx, U_t)

    def CRZ(self, control: str, target: str, theta: float):
        c_idx = self.get_qubit_index(control)
        t_idx = self.get_qubit_index(target)
        val_0 = cmath.exp(-1j * theta / 2)
        val_1 = cmath.exp(1j * theta / 2)
        U_t = np.array([[val_0, 0], [0, val_1]], dtype=complex)
        self._apply_controlled_unitary(c_idx, t_idx, U_t)

    def _apply_controlled_unitary(self, control_idx: int, target_idx: int, target_op: np.ndarray):
        dim = 1 << self.num_qubits
        rho = self._state

        ctrl_mask = 1 << control_idx
        tgt_mask = 1 << target_idx

        all_idx = np.arange(dim)
        ctrl_one = (all_idx & ctrl_mask) != 0
        tgt_zero = (all_idx & tgt_mask) == 0
        idx_c1_t0 = all_idx[ctrl_one & tgt_zero]
        idx_c1_t1 = all_idx[ctrl_one & ~tgt_zero]

        u00, u01 = target_op[0, 0], target_op[0, 1]
        u10, u11 = target_op[1, 0], target_op[1, 1]

        r0 = rho[idx_c1_t0, :].copy()
        r1 = rho[idx_c1_t1, :].copy()
        rho[idx_c1_t0, :] = u00 * r0 + u01 * r1
        rho[idx_c1_t1, :] = u10 * r0 + u11 * r1

        c0 = rho[:, idx_c1_t0].copy()
        c1 = rho[:, idx_c1_t1].copy()
        rho[:, idx_c1_t0] = c0 * np.conj(u00) + c1 * np.conj(u01)
        rho[:, idx_c1_t1] = c0 * np.conj(u10) + c1 * np.conj(u11)

        self._state = rho

    def _get_controlled_operator_full(self, control_idx: int, target_idx: int, target_op: np.ndarray) -> np.ndarray:
        P0 = _GATE_NP_CACHE['P0']
        P1 = _GATE_NP_CACHE['P1']

        op0 = (self._get_1qubit_operator_full(control_idx, P0)
               @ self._get_1qubit_operator_full(target_idx, _GATE_NP_CACHE['I2']))
        op1 = (self._get_1qubit_operator_full(control_idx, P1)
               @ self._get_1qubit_operator_full(target_idx, target_op))
        return op0 + op1

    def _get_double_controlled_operator_full(self, c1_idx: int, c2_idx: int,
                                             target_idx: int,
                                             target_op: np.ndarray) -> np.ndarray:
        P1 = _GATE_NP_CACHE['P1']
        proj_11 = self._get_1qubit_operator_full(c1_idx, P1) @ self._get_1qubit_operator_full(c2_idx, P1)
        I_full = np.eye(1 << self.num_qubits)
        op0 = (I_full - proj_11) @ self._get_1qubit_operator_full(target_idx, _GATE_NP_CACHE['I2'])
        op1 = proj_11 @ self._get_1qubit_operator_full(target_idx, target_op)
        return op0 + op1

    # Noise Channels
    def apply_bit_flip_noise(self, q: str, p: float):
        idx = self.get_qubit_index(q)
        E0 = math.sqrt(1 - p) * _GATE_NP_CACHE['I2']
        E1 = math.sqrt(p) * _GATE_NP_CACHE['X']
        self._apply_channel(idx, [E0, E1])

    def apply_phase_flip_noise(self, q: str, p: float):
        idx = self.get_qubit_index(q)
        E0 = math.sqrt(1 - p) * _GATE_NP_CACHE['I2']
        E1 = math.sqrt(p) * _GATE_NP_CACHE['Z']
        self._apply_channel(idx, [E0, E1])

    def apply_depolarizing_noise(self, q: str, p: float):
        idx = self.get_qubit_index(q)
        E0 = math.sqrt(1 - p) * _GATE_NP_CACHE['I2']
        E1 = math.sqrt(p / 3) * _GATE_NP_CACHE['X']
        E2 = math.sqrt(p / 3) * _GATE_NP_CACHE['Y']
        E3 = math.sqrt(p / 3) * _GATE_NP_CACHE['Z']
        self._apply_channel(idx, [E0, E1, E2, E3])

    def apply_amplitude_damping_noise(self, q: str, p: float):
        idx = self.get_qubit_index(q)
        E0 = np.array([[1.0, 0.0], [0.0, math.sqrt(1 - p)]], dtype=complex)
        E1 = np.array([[0.0, math.sqrt(p)], [0.0, 0.0]], dtype=complex)
        self._apply_channel(idx, [E0, E1])

    def apply_phase_damping_noise(self, q: str, lambda_val: float):
        idx = self.get_qubit_index(q)
        E0 = np.array([[1.0, 0.0], [0.0, math.sqrt(1 - lambda_val)]], dtype=complex)
        E1 = np.array([[0.0, 0.0], [0.0, math.sqrt(lambda_val)]], dtype=complex)
        self._apply_channel(idx, [E0, E1])

    # Measurement
    def measure(self, q: str) -> int:
        idx = self.get_qubit_index(q)
        # Probability of 0: Tr(P0 * rho)
        P0 = self._get_1qubit_operator_full(idx, _GATE_NP_CACHE['P0'])
        prob_0 = np.trace(P0 @ self._state).real
        
        # Sample outcome using seeded RNG
        outcome = 0 if self.rng.random() < prob_0 else 1
        
        # Project state
        if outcome == 0:
            proj = P0
            p = prob_0
        else:
            proj = self._get_1qubit_operator_full(idx, _GATE_NP_CACHE['P1'])
            p = 1.0 - prob_0
            
        if p > 1e-12:
            self._state = (proj @ self._state @ proj.conj().T) / p
        else:
            # If probability is 0, normalize
            self._state = proj @ self._state @ proj.conj().T
            trace = np.trace(self._state).real
            if trace > 1e-12:
                self._state /= trace
                
        return outcome

    def get_state_vector(self) -> list[complex]:
        """Return a state vector representation of the density matrix.

        For a pure (rank-1) state the density matrix factorises as
        ``rho = |psi><psi|``; we recover ``|psi>`` (up to a global phase)
        from the eigenvector of the largest eigenvalue. For a genuinely
        mixed state we fall back to the square roots of the diagonal
        probabilities, which is the best a caller can do without a
        purification — the previous behaviour, kept for backwards
        compatibility.
        """
        if self._state is None:
            return []
        rho = np.asarray(self._state, dtype=complex)
        n = rho.shape[0]
        if n == 0:
            return []
        # Hermitian (rho == rho†): eigvalsh returns real eigenvalues, and
        # we don't need to worry about complex-eigvals branch. Use eigh for
        # the standard Hermitian eigendecomposition.
        try:
            eigvals, eigvecs = np.linalg.eigh(rho)
        except np.linalg.LinAlgError:
            return [complex(math.sqrt(max(rho[i, i].real, 0.0))) for i in range(n)]
        # Largest eigenvalue (last in ascending order from eigh).
        top = eigvals[-1]
        # Determine purity Tr(rho^2). For a pure state this equals 1.
        purity = float(np.real(np.trace(rho @ rho)))
        # Numerical tolerance: 1e-6 has been sufficient for the Bell-state
        # and noise-channel tests, while still allowing genuinely mixed
        # states (purity < 1) to fall through to the diagonal fallback.
        if top > 1e-9 and abs(purity - 1.0) < 1e-6:
            vec = eigvecs[:, -1]
            # Pin global phase using the first significant component so the
            # result is deterministic across eigensolver runs.
            significant_idx = int(np.argmax(np.abs(vec)))
            phase_ref = vec[significant_idx]
            if abs(phase_ref) > 0.0:
                vec = vec * (np.conj(phase_ref) / abs(phase_ref))
            return [complex(x) for x in vec]
        # Mixed state: state vector is not mathematically defined. Return
        # the square roots of the diagonal probabilities for backwards
        # compatibility with prior callers (e.g. amplitudes_dict callers
        # that only inspect magnitudes).
        return [complex(math.sqrt(max(rho[i, i].real, 0.0))) for i in range(n)]

    def get_amplitudes_dict(self) -> dict[str, complex]:
        amplitudes = {}
        n = self.num_qubits
        sorted_qubits = sorted(self.qubit_map.keys(), key=lambda name: self.qubit_map[name])
        for i in range(1 << n):
            prob = self._state[i, i].real
            if prob > 1e-12:
                amp = math.sqrt(prob)
                bitstring = ""
                for q in reversed(sorted_qubits):
                    q_idx = self.qubit_map[q]
                    bitstring += str((i >> q_idx) & 1)
                amplitudes[bitstring] = amp
        return amplitudes
