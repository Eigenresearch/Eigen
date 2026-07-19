import numpy as np
import math
import random
import cmath
import logging

from src.numerical_stability import TruncationAccumulator

logger = logging.getLogger('eigen.mps')
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('[MPS] %(message)s'))
    logger.addHandler(_handler)
    logger.setLevel(logging.WARNING)

DEFAULT_MAX_BOND_DIM = 64

# === sol.md P0 §1.3 — cached standard 1-qubit gate matrices as NumPy
# arrays. Reused by all standard gate applications on the MPS path, avoiding
# a per-call rebuild of the 2x2 np.ndarray.
_INV_SQRT2_MPS = 1.0 / math.sqrt(2.0)
_MPS_GATE_NP_CACHE = {
    'H': np.array([[_INV_SQRT2_MPS, _INV_SQRT2_MPS], [_INV_SQRT2_MPS, -_INV_SQRT2_MPS]], dtype=complex),
    'X': np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    'Y': np.array([[0.0, -1j], [1j, 0.0]], dtype=complex),
    'Z': np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
    'S': np.array([[1.0, 0.0], [0.0, 1j]], dtype=complex),
    'T': np.array([[1.0, 0.0], [0.0, _INV_SQRT2_MPS + _INV_SQRT2_MPS * 1j]], dtype=complex),
}
DEFAULT_MAX_TRUNCATION_ERROR = 1e-4
AUTO_BOND_DIM_FACTOR = 2

def native_svd(matrix_2d):
    try:
        import eigen_native
        # Convert 2D complex numpy array/list to nested lists of (real, imag)
        raw_mat = [[(float(val.real), float(val.imag)) for val in row] for row in matrix_2d]
        u_raw, s_raw, vh_raw = eigen_native.compute_svd_native(raw_mat)
        U = np.array([[complex(r, i) for r, i in row] for row in u_raw], dtype=complex)
        S = np.array(s_raw, dtype=float)
        Vh = np.array([[complex(r, i) for r, i in row] for row in vh_raw], dtype=complex)
        return U, S, Vh
    except Exception:
        return np.linalg.svd(matrix_2d, full_matrices=False)

class MPSSimulator:
    def __init__(self, max_bond_dim=DEFAULT_MAX_BOND_DIM, seed=None,
                 auto_bond_dim=False, max_truncation_error=DEFAULT_MAX_TRUNCATION_ERROR):
        self.tensors = []       # List of np.ndarray of shape (left_bond, 2, right_bond)
        self.qubits = []        # List of qubit names in chain order
        self.qubit_map = {}     # qubit_name -> chain index
        self.max_bond_dim = max_bond_dim
        self.auto_bond_dim = auto_bond_dim
        self.max_truncation_error = max_truncation_error
        self.cumulative_truncation_error = 0.0
        self.last_entropy = 0.0
        self.last_discarded_weight = 0.0
        self._warned_degraded = False
        self._warned_auto_increase = False
        self.rng = random.Random(seed)
        self.created_qubits = [] # Qubits in original creation order
        self._truncation_accumulator = TruncationAccumulator()
        self._truncation_step = 0

    def allocate_qubit(self, name: str):
        if name in self.qubit_map:
            return
        idx = len(self.tensors)
        self.qubit_map[name] = idx
        self.qubits.append(name)
        self.created_qubits.append(name)
        
        # New qubit is $|0\rangle$, tensor shape (1, 2, 1)
        # A[0, 0, 0] = 1.0, A[0, 1, 0] = 0.0
        tensor = np.zeros((1, 2, 1), dtype=complex)
        tensor[0, 0, 0] = 1.0
        self.tensors.append(tensor)

    def get_qubit_index(self, name: str) -> int:
        if name not in self.qubit_map:
            raise KeyError(f"Qubit '{name}' is not allocated in the simulator")
        return self.qubit_map[name]

    def _truncate_svd(self, U, S, Vh):
        """Truncate SVD result with optional auto bond dimension.

        Returns truncated (U, S, Vh, chi) and updates tracking metrics.
        Handles auto_bond_dim increase and accuracy warnings.
        Uses entropy-adaptive chi selection: reduces chi when entanglement
        entropy is low, increases chi (up to chi_max=128) when high.
        """
        available = len(S)

        S_sum2 = np.sum(S**2)
        if S_sum2 > 1e-15:
            S_norm = S / np.sqrt(S_sum2)
        else:
            S_norm = S.copy()

        probs = S_norm**2
        full_entropy = -float(np.sum(probs * np.log2(probs + 1e-15)))

        if self.auto_bond_dim and available > self.max_bond_dim:
            for test_chi in range(self.max_bond_dim, available + 1):
                dw = np.sum(S_norm[test_chi:]**2) if test_chi < len(S_norm) else 0.0
                if dw <= self.max_truncation_error:
                    chi = test_chi
                    break
            else:
                chi = available
            if chi > self.max_bond_dim:
                self.max_bond_dim = chi
                if not self._warned_auto_increase:
                    logger.info(
                        "Auto-increased bond dimension to %d "
                        "(truncation error threshold: %g)",
                        chi, self.max_truncation_error
                    )
                    self._warned_auto_increase = True
        else:
            chi = min(available, self.max_bond_dim)

        if full_entropy < 2.0 and chi > 1:
            for test_chi in range(1, chi + 1):
                dw = np.sum(S_norm[test_chi:]**2) if test_chi < len(S_norm) else 0.0
                if dw <= self.max_truncation_error:
                    chi = test_chi
                    break
        elif full_entropy > 4.0 and chi < available:
            chi = min(chi * 2, available, 128)

        chi = min(chi, 128)

        discarded_weight = np.sum(S_norm[chi:]**2) if chi < len(S_norm) else 0.0
        self.cumulative_truncation_error += discarded_weight
        self.last_discarded_weight = discarded_weight

        schmidt = S_norm[:chi]
        schmidt_sum2 = np.sum(schmidt**2)
        if schmidt_sum2 > 1e-15:
            schmidt = schmidt / np.sqrt(schmidt_sum2)
            self.last_entropy = -float(np.sum(schmidt**2 * np.log2(schmidt**2 + 1e-15)))

        self._truncation_accumulator.record(
            step_index=self._truncation_step,
            bond_dimension=int(chi),
            truncation_error=float(discarded_weight),
            discarded_weight=float(discarded_weight),
        )
        self._truncation_step += 1
        if (discarded_weight > self.max_truncation_error
                and not self._warned_degraded):
            logger.warning(
                "Simulation accuracy may be degraded: "
                "discarded weight %g exceeds threshold %g "
                "(cumulative error: %g, bond dim: %d/%d available)",
                discarded_weight, self.max_truncation_error,
                self.cumulative_truncation_error, chi, available
            )
            self._warned_degraded = True

        U = U[:, :chi]
        S = S[:chi]
        Vh = Vh[:chi, :]
        return U, S, Vh, chi

    def apply_1qubit_gate(self, name: str, gate_matrix: list[list[complex]]):
        idx = self.get_qubit_index(name)
        U = np.array(gate_matrix, dtype=complex)
        # Tensors[idx] shape: (L, 2, R)
        # Contract physical index of Tensors[idx] with second index of U
        # U shape: (2_new, 2_old)
        # Result shape: (2_new, L, R)
        res = np.tensordot(U, self.tensors[idx], axes=(1, 1))
        # Transpose to (L, 2_new, R)
        self.tensors[idx] = np.transpose(res, (1, 0, 2))

    def _apply_named_1qubit_np(self, q: str, U_np: np.ndarray):
        """Apply a pre-built NumPy 2x2 gate (cached for one of H/X/Y/Z/S/T)
        directly, skipping the list -> np.array conversion. Used by the
        standard 1-qubit gate shortcuts below."""
        idx = self.get_qubit_index(q)
        res = np.tensordot(U_np, self.tensors[idx], axes=(1, 1))
        self.tensors[idx] = np.transpose(res, (1, 0, 2))

    def H(self, q: str):
        self._apply_named_1qubit_np(q, _MPS_GATE_NP_CACHE['H'])

    def X(self, q: str):
        self._apply_named_1qubit_np(q, _MPS_GATE_NP_CACHE['X'])

    def Y(self, q: str):
        self._apply_named_1qubit_np(q, _MPS_GATE_NP_CACHE['Y'])

    def Z(self, q: str):
        self._apply_named_1qubit_np(q, _MPS_GATE_NP_CACHE['Z'])

    def S(self, q: str):
        self._apply_named_1qubit_np(q, _MPS_GATE_NP_CACHE['S'])

    def T(self, q: str):
        self._apply_named_1qubit_np(q, _MPS_GATE_NP_CACHE['T'])

    def RX(self, q: str, theta: float):
        cos_val = math.cos(theta / 2)
        sin_val = math.sin(theta / 2)
        self.apply_1qubit_gate(q, [
            [cos_val, -1j * sin_val],
            [-1j * sin_val, cos_val]
        ])

    def RY(self, q: str, theta: float):
        cos_val = math.cos(theta / 2)
        sin_val = math.sin(theta / 2)
        self.apply_1qubit_gate(q, [
            [cos_val, -sin_val],
            [sin_val, cos_val]
        ])

    def RZ(self, q: str, theta: float):
        self.apply_1qubit_gate(q, [
            [cmath.exp(-1j * theta / 2), 0.0j],
            [0.0j, cmath.exp(1j * theta / 2)]
        ])

    def _swap_adjacent(self, i: int):
        """Swaps adjacent tensors at index i and i+1."""
        A = self.tensors[i]      # shape (L, 2, M)
        B = self.tensors[i+1]    # shape (M, 2, R)
        
        # Contract A and B over M
        # shape: (L, 2, 2, R)
        theta = np.tensordot(A, B, axes=(2, 0))
        
        # We want to swap the physical indices: transpose (L, 2_A, 2_B, R) -> (L, 2_B, 2_A, R)
        theta = np.transpose(theta, (0, 2, 1, 3))
        
        # Reshape to (L * 2_B, 2_A * R)
        L, d1, d2, R = theta.shape
        theta_mat = theta.reshape(L * d1, d2 * R)
        
        # SVD
        U, S, Vh = native_svd(theta_mat)
        
        # Truncate bond dimension with auto-increase support
        U, S, Vh, chi = self._truncate_svd(U, S, Vh)
        
        # New tensors
        # New A shape: (L, 2_B, chi)
        self.tensors[i] = U.reshape(L, d1, chi)
        # New B shape: (chi, 2_A, R)
        self.tensors[i+1] = (np.diag(S) @ Vh).reshape(chi, d2, R)
        
        # Swap names in chain order
        self.qubits[i], self.qubits[i+1] = self.qubits[i+1], self.qubits[i]
        self.qubit_map[self.qubits[i]] = i
        self.qubit_map[self.qubits[i+1]] = i+1

    def apply_2qubit_gate(self, control: str, target: str, gate_matrix_4x4):
        c_idx = self.get_qubit_index(control)
        t_idx = self.get_qubit_index(target)
        
        # 1. Bring control and target adjacent
        # Move c_idx to t_idx - 1
        if c_idx < t_idx:
            while c_idx < t_idx - 1:
                self._swap_adjacent(c_idx)
                c_idx += 1
            i = c_idx
            j = t_idx
            control_at_i = True
        else:
            while c_idx > t_idx + 1:
                self._swap_adjacent(c_idx - 1)
                c_idx -= 1
            i = t_idx
            j = c_idx
            control_at_i = False
            
        # Now control and target are at adjacent positions i and j=i+1
        # 2. Apply gate at i and j
        A = self.tensors[i]      # shape (L, 2, M)
        B = self.tensors[j]      # shape (M, 2, R)
        
        # shape: (L, 2, 2, R)
        theta = np.tensordot(A, B, axes=(2, 0))
        
        # Reshape gate matrix to (2, 2, 2, 2)
        # Default ordering: U[c_new, t_new, c_old, t_old]
        U = np.array(gate_matrix_4x4, dtype=complex).reshape(2, 2, 2, 2)
        
        if not control_at_i:
            # Target is at position i (left), control at j (right).
            # theta is (L, t_old, c_old, R). Transpose U to match
            # the (target, control) physical index ordering.
            U = U.transpose(1, 0, 3, 2)
        
        # Contract: U[new_d1, new_d2, old_d1, old_d2] with theta[L, old_d1, old_d2, R]
        # shape: (2, 2, L, R)
        res = np.tensordot(U, theta, axes=((2, 3), (1, 2)))
        # Transpose to (L, new_d1, new_d2, R)
        res = np.transpose(res, (2, 0, 1, 3))
        
        # SVD reshape
        L, d1, d2, R = res.shape
        res_mat = res.reshape(L * d1, d2 * R)
        U_svd, S, Vh = native_svd(res_mat)
        
        # Truncate with auto-increase support
        U_svd, S, Vh, chi = self._truncate_svd(U_svd, S, Vh)
        
        self.tensors[i] = U_svd.reshape(L, d1, chi)
        self.tensors[j] = (np.diag(S) @ Vh).reshape(chi, d2, R)

    def CNOT(self, control: str, target: str):
        # 4x4 CNOT matrix
        cnot_matrix = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ]
        self.apply_2qubit_gate(control, target, cnot_matrix)

    def CZ(self, control: str, target: str):
        cz_matrix = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ]
        self.apply_2qubit_gate(control, target, cz_matrix)

    def SWAP(self, q1: str, q2: str):
        swap_matrix = [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ]
        self.apply_2qubit_gate(q1, q2, swap_matrix)

    def CCX(self, c1: str, c2: str, t: str):
        self.H(t)
        self.CNOT(c2, t)
        self.apply_1qubit_gate(t, [[1.0, 0.0], [0.0, cmath.exp(-1j * math.pi / 4)]]) # T*
        self.CNOT(c1, t)
        self.apply_1qubit_gate(t, [[1.0, 0.0], [0.0, cmath.exp(1j * math.pi / 4)]]) # T
        self.CNOT(c2, t)
        self.apply_1qubit_gate(t, [[1.0, 0.0], [0.0, cmath.exp(-1j * math.pi / 4)]]) # T*
        self.CNOT(c1, t)
        self.apply_1qubit_gate(t, [[1.0, 0.0], [0.0, cmath.exp(1j * math.pi / 4)]]) # T
        self.apply_1qubit_gate(c2, [[1.0, 0.0], [0.0, cmath.exp(1j * math.pi / 4)]]) # T
        self.CNOT(c1, c2)
        self.apply_1qubit_gate(c1, [[1.0, 0.0], [0.0, cmath.exp(1j * math.pi / 4)]]) # T
        self.apply_1qubit_gate(c2, [[1.0, 0.0], [0.0, cmath.exp(-1j * math.pi / 4)]]) # T*
        self.CNOT(c1, c2)
        self.H(t)

    def CSWAP(self, c: str, t1: str, t2: str):
        self.CNOT(t2, t1)
        self.CCX(c, t1, t2)
        self.CNOT(t2, t1)

    def CP(self, control: str, target: str, theta: float):
        val = cmath.exp(1j * theta)
        cp_matrix = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, val]
        ]
        self.apply_2qubit_gate(control, target, cp_matrix)

    def CRX(self, control: str, target: str, theta: float):
        cos_val = math.cos(theta / 2)
        sin_val = math.sin(theta / 2)
        crx_matrix = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, cos_val, -1j * sin_val],
            [0.0, 0.0, -1j * sin_val, cos_val]
        ]
        self.apply_2qubit_gate(control, target, crx_matrix)

    def CRY(self, control: str, target: str, theta: float):
        cos_val = math.cos(theta / 2)
        sin_val = math.sin(theta / 2)
        cry_matrix = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, cos_val, -sin_val],
            [0.0, 0.0, sin_val, cos_val]
        ]
        self.apply_2qubit_gate(control, target, cry_matrix)

    def CRZ(self, control: str, target: str, theta: float):
        val_0 = cmath.exp(-1j * theta / 2)
        val_1 = cmath.exp(1j * theta / 2)
        crz_matrix = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, val_0, 0.0],
            [0.0, 0.0, 0.0, val_1]
        ]
        self.apply_2qubit_gate(control, target, crz_matrix)

    def measure(self, q: str) -> int:
        idx = self.get_qubit_index(q)
        
        # To measure, we calculate the probability of measuring 0.
        # We can project qubit q to state |0> (and |1>), contract the MPS to compute norm,
        # and sample outcome based on that.
        
        # Save state to restore
        tensors_backup = [t.copy() for t in self.tensors]
        
        # Project site to |0>
        # P0 matrix: [[1, 0], [0, 0]]
        self.apply_1qubit_gate(q, [[1, 0], [0, 0]])
        p0 = self.norm_squared()
        
        # Restore state
        self.tensors = tensors_backup
        
        r = self.rng.random()
        if r < p0:
            outcome = 0
            # Project to |0> and normalize
            self.apply_1qubit_gate(q, [[1, 0], [0, 0]])
            norm = math.sqrt(p0) if p0 > 1e-15 else 1.0
            self.tensors[idx] /= norm
        else:
            outcome = 1
            # Project to |1> and normalize
            self.apply_1qubit_gate(q, [[0, 0], [0, 1]])
            p1 = 1.0 - p0
            norm = math.sqrt(p1) if p1 > 1e-15 else 1.0
            self.tensors[idx] /= norm
            
        return outcome

    def norm_squared(self) -> float:
        if not self.tensors:
            return 1.0
        # Contract the chain from left to right to compute internal product <psi|psi>
        # We contract A^[i] and A^[i]* over left, physical, and right indices
        # We maintain a transfer matrix boundary T of shape (bond_psi, bond_psi_conj)
        # Initially T = np.array([[1.0]])
        T = np.array([[1.0]], dtype=complex)
        for A in self.tensors:
            # A shape: (L, 2, R)
            # Contract T(L, L*) with A(L, d, R) -> T_new(L*, d, R)
            T_new = np.tensordot(T, A, axes=(0, 0))
            # Contract T_new(L*, d, R) with A*(L*, d, R*) -> T_final(R, R*)
            T = np.tensordot(T_new, np.conj(A), axes=((0, 1), (0, 1)))
        return float(T[0, 0].real)

    def get_state_vector(self) -> list[complex]:
        # To get the full state vector, we contract all tensors in chain order
        if not self.tensors:
            return [1.0 + 0.0j]
            
        if len(self.tensors) > 24:
            raise RuntimeError(f"Cannot reconstruct full state vector for {len(self.tensors)} qubits.")
            
        # Contract from left to right
        curr = self.tensors[0]  # shape: (1, 2, R)
        for i in range(1, len(self.tensors)):
            # Contract curr(L, d_prevs..., R_prev) with tensors[i](R_prev, d_new, R_new)
            # shape: (L, d_prevs..., d_new, R_new)
            curr = np.tensordot(curr, self.tensors[i], axes=(-1, 0))
            
        # Squeeze boundary dimensions: (1, d0, d1, ..., 1) -> (d0, d1, ...)
        curr = np.squeeze(curr)
        # We need to map the output to index layout matching qubit_map names sorted order
        # Chain order is self.qubits
        # Targets order should be sorted(self.qubits)
        n = len(self.tensors)
        state_vec = np.zeros(1 << n, dtype=complex)
        
        # Flatten the multidimensional array matching chain order
        flat_curr = curr.flatten()
        # For each element, map its binary representation from chain order to original creation order
        for idx in range(1 << n):
            val = flat_curr[idx]
            if abs(val) > 1e-12:
                sorted_idx = 0
                for k, q in enumerate(self.qubits):
                    bit = (idx >> (n - 1 - k)) & 1
                    if bit:
                        created_idx = self.created_qubits.index(q)
                        sorted_idx |= (1 << created_idx)
                state_vec[sorted_idx] = val
        return list(state_vec)

    def get_amplitudes_dict(self) -> dict[str, complex]:
        if not self.tensors:
            return {"": 1.0 + 0.0j}
        # Get full state vector if small enough, else return sparse from mps contraction
        n = len(self.tensors)
        if n <= 16:
            vec = self.get_state_vector()
            sorted_qubits = sorted(self.qubit_map.keys(), key=lambda name: self.qubit_map[name])
            amplitudes = {}
            for i, amp in enumerate(vec):
                if abs(amp) > 1e-12:
                    bitstring = ""
                    for q in reversed(sorted_qubits):
                        q_idx = self.qubit_map[q]
                        bitstring += str((i >> q_idx) & 1)
                    amplitudes[bitstring] = amp
            return amplitudes
        else:
            # For large MPS, we can sample or return a message
            return {"Large MPS (details omitted)": 1.0}

    def get_last_entropy(self) -> float:
        return self.last_entropy

    def get_cumulative_truncation_error(self) -> float:
        return self.cumulative_truncation_error

    def get_truncation_accumulator(self) -> 'TruncationAccumulator':
        return self._truncation_accumulator

    def get_last_discarded_weight(self) -> float:
        return self.last_discarded_weight

    def get_max_bond_dim(self) -> int:
        return self.max_bond_dim
