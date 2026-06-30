import numpy as np
import math
import random
import cmath

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
    def __init__(self, max_bond_dim=32, seed=None):
        self.tensors = []       # List of np.ndarray of shape (left_bond, 2, right_bond)
        self.qubits = []        # List of qubit names in chain order
        self.qubit_map = {}     # qubit_name -> chain index
        self.max_bond_dim = max_bond_dim
        self.cumulative_truncation_error = 0.0
        self.last_entropy = 0.0
        self.rng = random.Random(seed)
        self.created_qubits = [] # Qubits in original creation order

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

    def H(self, q: str):
        inv_sqrt2 = 0.7071067811865475
        self.apply_1qubit_gate(q, [
            [inv_sqrt2, inv_sqrt2],
            [inv_sqrt2, -inv_sqrt2]
        ])

    def X(self, q: str):
        self.apply_1qubit_gate(q, [
            [0.0, 1.0],
            [1.0, 0.0]
        ])

    def Y(self, q: str):
        self.apply_1qubit_gate(q, [
            [0.0, -1j],
            [1j, 0.0]
        ])

    def Z(self, q: str):
        self.apply_1qubit_gate(q, [
            [1.0, 0.0],
            [0.0, -1.0]
        ])

    def S(self, q: str):
        self.apply_1qubit_gate(q, [
            [1.0, 0.0],
            [0.0, 1j]
        ])

    def T(self, q: str):
        val = 0.7071067811865475 + 0.7071067811865475j
        self.apply_1qubit_gate(q, [
            [1.0, 0.0],
            [0.0, val]
        ])

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
        
        # Truncate bond dimension
        chi = min(len(S), self.max_bond_dim)
        
        # Track metrics
        S_sum2 = np.sum(S**2)
        if S_sum2 > 1e-15:
            S_norm = S / np.sqrt(S_sum2)
            discarded_weight = np.sum(S_norm[chi:]**2) if chi < len(S_norm) else 0.0
            self.cumulative_truncation_error += discarded_weight
            
            schmidt = S_norm[:chi]
            schmidt_sum2 = np.sum(schmidt**2)
            if schmidt_sum2 > 1e-15:
                schmidt = schmidt / np.sqrt(schmidt_sum2)
                self.last_entropy = -float(np.sum(schmidt**2 * np.log2(schmidt**2 + 1e-15)))
        
        U = U[:, :chi]
        S = S[:chi]
        Vh = Vh[:chi, :]
        
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
        else:
            while c_idx > t_idx + 1:
                self._swap_adjacent(c_idx - 1)
                c_idx -= 1
            i = t_idx
            j = c_idx
            
        # Now control is at i, target is at j = i+1
        # 2. Apply gate at i and j
        A = self.tensors[i]      # shape (L, 2_c, M)
        B = self.tensors[j]      # shape (M, 2_t, R)
        
        # shape: (L, 2_c, 2_t, R)
        theta = np.tensordot(A, B, axes=(2, 0))
        
        # Reshape gate matrix to (2_c_new, 2_t_new, 2_c_old, 2_t_old)
        U = np.array(gate_matrix_4x4, dtype=complex).reshape(2, 2, 2, 2)
        
        # Contract: U[c_new, t_new, c_old, t_old] with theta[L, c_old, t_old, R]
        # shape: (2, 2, L, R)
        res = np.tensordot(U, theta, axes=((2, 3), (1, 2)))
        # Transpose to (L, 2_c_new, 2_t_new, R)
        res = np.transpose(res, (2, 0, 1, 3))
        
        # SVD reshape
        L, d1, d2, R = res.shape
        res_mat = res.reshape(L * d1, d2 * R)
        U_svd, S, Vh = native_svd(res_mat)
        
        # Truncate
        chi = min(len(S), self.max_bond_dim)
        
        # Track metrics
        S_sum2 = np.sum(S**2)
        if S_sum2 > 1e-15:
            S_norm = S / np.sqrt(S_sum2)
            discarded_weight = np.sum(S_norm[chi:]**2) if chi < len(S_norm) else 0.0
            self.cumulative_truncation_error += discarded_weight
            
            schmidt = S_norm[:chi]
            schmidt_sum2 = np.sum(schmidt**2)
            if schmidt_sum2 > 1e-15:
                schmidt = schmidt / np.sqrt(schmidt_sum2)
                self.last_entropy = -float(np.sum(schmidt**2 * np.log2(schmidt**2 + 1e-15)))
        
        U_svd = U_svd[:, :chi]
        S = S[:chi]
        Vh = Vh[:chi, :]
        
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
        return float(np.abs(T[0, 0]))

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
