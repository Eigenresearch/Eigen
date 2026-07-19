"""
Stabilizer Simulator for Clifford circuits.
Uses the CHP (Clifford-Hadamard-Pauli) algorithm for O(n^2) simulation.

Reference: Aaronson & Gottesman, "Improved Simulation of Stabilizer Circuits",
Phys. Rev. A 70, 052328 (2004).

The stabilizer simulator supports only the Clifford group:
  {H, S, X, Y, Z, CNOT, CZ, SWAP}

Non-Clifford gates (T, RX, RY, RZ, CCX, CSWAP, CP, CRX, CRY, CRZ) raise
NonCliffordGateError. Use check_circuit_compatibility() for pre-flight
circuit analysis, or use QuantumSimulator(sim_type='stabilizer') for
automatic fallback to the dense state-vector simulator.
"""
import random

CLIFFORD_GATES = frozenset({
    'H', 'S', 'X', 'Y', 'Z', 'CNOT', 'CZ', 'SWAP',
    'SDG', 'I', 'SX',
})

NON_CLIFFORD_GATES = frozenset({
    'T', 'TDG', 'RX', 'RY', 'RZ', 'CCX', 'CSWAP',
    'CP', 'CRX', 'CRY', 'CRZ', 'U1', 'U2', 'U3',
})


def _pauli_phase(x1, z1, x2, z2):
    if (x1, z1) == (0, 0) or (x2, z2) == (0, 0):
        return 0
    if (x1, z1) == (x2, z2):
        return 0
    p1 = (x1, z1)
    p2 = (x2, z2)
    if p1 == (1, 0) and p2 == (0, 1): return 3
    if p1 == (0, 1) and p2 == (1, 0): return 1
    if p1 == (1, 0) and p2 == (1, 1): return 1
    if p1 == (1, 1) and p2 == (1, 0): return 3
    if p1 == (1, 1) and p2 == (0, 1): return 1
    if p1 == (0, 1) and p2 == (1, 1): return 3
    return 0


class NonCliffordGateError(ValueError):
    """Raised when a non-Clifford gate is applied to a stabilizer simulator.

    Subclass of ValueError for backward compatibility with code that
    catches ValueError from the old implementation.
    """

    def __init__(self, gate_name: str, message: str = None):
        self.gate_name = gate_name
        if message is None:
            message = (
                f"{gate_name} gate is non-Clifford, not supported by stabilizer simulator. "
                f"Supported gates: {sorted(CLIFFORD_GATES)}. "
                f"Use a state-vector backend or QuantumSimulator with auto-fallback."
            )
        super().__init__(message)

class StabilizerSimulator:
    def __init__(self, rng=None, seed=None):
        self.n = 0
        self.rng = rng if rng is not None else random.Random(seed)
        self._init_state()

    def _init_state(self):
        n = self.n
        # destabilizer generators (rows 0..n-1) + stabilizer generators (rows n..2n-1)
        # Each row is [x_0..x_{n-1}, z_0..z_{n-1}, r]
        self.destab = [[0]*(2*n+1) for _ in range(n)]
        self.stab = [[0]*(2*n+1) for _ in range(n)]
        for i in range(n):
            self.destab[i][i] = 1         # X_i
            self.stab[i][n + i] = 1        # Z_i

    def allocate_qubit(self, name: str):
        if not hasattr(self, 'qubit_map'):
            self.qubit_map = {}
        if name in self.qubit_map:
            return
        idx = self.n
        self.qubit_map[name] = idx
        self.n += 1
        n = self.n
        for row in self.destab + self.stab:
            row.insert(idx, 0)
            row.insert(n + idx, 0)
        new_destab = [0] * (2 * n + 1)
        new_destab[idx] = 1
        self.destab.append(new_destab)
        new_stab = [0] * (2 * n + 1)
        new_stab[n + idx] = 1
        self.stab.append(new_stab)

    def get_qubit_index(self, name: str) -> int:
        return self.qubit_map[name]

    def _rowsum(self, h: int, i: int):
        """Row sum: add stabilizer generator i into row h."""
        if h < 0 or i < 0:
            return
        n = self.n
        r_h = self.destab[h][2*n] if h < n else self.stab[h - n][2*n]
        r_i = self.destab[i][2*n] if i < n else self.stab[i - n][2*n]

        row_h = self.destab[h] if h < n else self.stab[h - n]
        row_i = self.destab[i] if i < n else self.stab[i - n]

        phase = (r_h + r_i) % 4
        for j in range(n):
            x_h, z_h = row_h[j], row_h[n + j]
            x_i, z_i = row_i[j], row_i[n + j]
            phase = (phase + _pauli_phase(x_h, z_h, x_i, z_i)) % 4

        for j in range(2 * n):
            row_h[j] = (row_h[j] + row_i[j]) % 2

        if h < n:
            self.destab[h][2*n] = phase
        else:
            self.stab[h - n][2*n] = phase

    def _rowsum_simple(self, h_row, i_row):
        """Simplified rowsum for two full rows (each is [x0..xn-1, z0..zn-1, r])."""
        n = self.n
        phase = (h_row[2*n] + i_row[2*n]) % 4
        for j in range(n):
            x1, z1 = h_row[j], h_row[n + j]
            x2, z2 = i_row[j], i_row[n + j]
            phase = (phase + _pauli_phase(x1, z1, x2, z2)) % 4
        for j in range(2 * n):
            h_row[j] = (h_row[j] + i_row[j]) % 2
        h_row[2*n] = phase

    def _apply_h(self, q: int):
        """Apply H gate to qubit q."""
        n = self.n
        for row in self.destab + self.stab:
            x_q, z_q = row[q], row[n + q]
            if x_q == 1 and z_q == 1:
                row[2*n] = (row[2*n] + 2) % 4
            row[q], row[n + q] = z_q, x_q

    def _apply_s(self, q: int):
        """Apply S gate to qubit q: X -> Y, Y -> -X (Z unchanged)."""
        n = self.n
        for row in self.destab + self.stab:
            x_q, z_q = row[q], row[n + q]
            if x_q == 1 and z_q == 0:
                row[2*n] = (row[2*n] + 1) % 4
            elif x_q == 1 and z_q == 1:
                row[2*n] = (row[2*n] + 2) % 4
            row[n + q] = (row[q] + row[n + q]) % 2

    def _apply_cnot(self, control: int, target: int):
        """Apply CNOT(control, target)."""
        n = self.n
        for row in self.destab + self.stab:
            # X_c -> X_c X_t, Z_t -> Z_c Z_t
            row[target] = (row[target] + row[control]) % 2
            row[n + control] = (row[n + control] + row[n + target]) % 2

    def _apply_x(self, q: int):
        """Apply X gate: flip phase of Z_q stabilizers."""
        n = self.n
        for row in self.destab + self.stab:
            if row[n + q] == 1:
                row[2*n] = (row[2*n] + 2) % 4

    def _apply_z(self, q: int):
        """Apply Z gate: flip phase of X_q stabilizers."""
        n = self.n
        for row in self.destab + self.stab:
            if row[q] == 1:
                row[2*n] = (row[2*n] + 2) % 4

    def H(self, q):
        self._apply_h(self.get_qubit_index(q))

    def X(self, q):
        self._apply_x(self.get_qubit_index(q))

    def Y(self, q):
        self._apply_x(self.get_qubit_index(q))
        self._apply_z(self.get_qubit_index(q))

    def Z(self, q):
        self._apply_z(self.get_qubit_index(q))

    def S(self, q):
        self._apply_s(self.get_qubit_index(q))

    def T(self, q):
        raise NonCliffordGateError("T")

    def RX(self, q, theta):
        raise NonCliffordGateError("RX")

    def RY(self, q, theta):
        raise NonCliffordGateError("RY")

    def RZ(self, q, theta):
        raise NonCliffordGateError("RZ")

    def CNOT(self, control, target):
        self._apply_cnot(self.get_qubit_index(control), self.get_qubit_index(target))

    def CZ(self, control, target):
        c = self.get_qubit_index(control)
        t = self.get_qubit_index(target)
        self._apply_h(t)
        self._apply_cnot(c, t)
        self._apply_h(t)

    def SWAP(self, q1, q2):
        c = self.get_qubit_index(q1)
        t = self.get_qubit_index(q2)
        self._apply_cnot(c, t)
        self._apply_cnot(t, c)
        self._apply_cnot(c, t)

    def CCX(self, c1, c2, target):
        raise NonCliffordGateError("CCX")

    def CSWAP(self, control, q1, q2):
        raise NonCliffordGateError("CSWAP")

    def CP(self, control, target, theta):
        raise NonCliffordGateError("CP")

    def CRX(self, control, target, theta):
        raise NonCliffordGateError("CRX")

    def CRY(self, control, target, theta):
        raise NonCliffordGateError("CRY")

    def CRZ(self, control, target, theta):
        raise NonCliffordGateError("CRZ")

    def measure(self, q: str) -> int:
        """Measure qubit q in the computational basis (non-destructive)."""
        n = self.n
        q_idx = self.get_qubit_index(q)

        # Check for random measurement: stabilizer with X_q=1 (anti-commutes with Z_q)
        random_row = -1
        for i in range(n):
            if self.stab[i][q_idx] == 1:
                random_row = i
                break

        if random_row == -1:
            scratch = [0] * (2 * n + 1)
            for i in range(n):
                if self.destab[i][q_idx] == 1:
                    self._rowsum_simple(scratch, self.stab[i])
            return scratch[2 * n] // 2

        outcome = self.rng.randint(0, 1)

        if random_row != q_idx:
            self.stab[random_row], self.stab[q_idx] = self.stab[q_idx], self.stab[random_row]
            self.destab[random_row], self.destab[q_idx] = self.destab[q_idx], self.destab[random_row]

        for i in range(n):
            if i == q_idx:
                continue
            if self.destab[i][q_idx] == 1:
                self._rowsum_simple(self.destab[i], self.stab[q_idx])
            if self.stab[i][q_idx] == 1:
                self._rowsum_simple(self.stab[i], self.stab[q_idx])

        self.destab[q_idx] = self.stab[q_idx][:]

        self.stab[q_idx] = [0] * (2*n + 1)
        self.stab[q_idx][n + q_idx] = 1
        self.stab[q_idx][2*n] = outcome * 2

        return outcome

    def get_state_vector(self) -> list:
        """Compute the full state vector from the stabilizer tableau.
        
        O(2^n * n^2) complexity. Feasible only for small n (n <= 25).
        """
        n = self.n
        if n == 0:
            return [1.0 + 0.0j]
        if n > 25:
            raise MemoryError(f"Cannot compute state vector for {n} qubits (limit 25).")
            
        dim = 1 << n

        # Start with all-zero state and project onto the stabilizer subspace.
        # A simpler way: since we only have Clifford gates, we can just 
        # track the state vector if we really need to, but that defeats 
        # the purpose of the stabilizer simulator.
        # Alternatively: iterate through all basis states and check if they 
        # satisfy all stabilizers.
        
        # Proper way: Start with |0...0>, then for each stabilizer g, 
        # apply (I + g)/sqrt(2).
        import numpy as np
        vec = np.zeros(dim, dtype=complex)
        vec[0] = 1.0
        
        for i in range(n):
            # stabilizer g_i
            row = self.stab[i]
            # g_i = (-1)^r * (X^x * Z^z)
            phase = 1.0 if row[2*n] == 0 else (1j if row[2*n] == 1 else (-1.0 if row[2*n] == 2 else -1j))
            
            # Apply (I + g_i)
            # g_i |psi>
            g_vec = np.zeros(dim, dtype=complex)
            x_bits = 0
            for j in range(n):
                if row[j]: x_bits |= (1 << j)
            
            for idx in range(dim):
                if vec[idx] == 0: continue
                # X part
                new_idx = idx ^ x_bits
                # Z part: parity of (idx & z_bits)
                z_bits = 0
                for j in range(n):
                    if row[n+j]: z_bits |= (1 << j)
                
                z_phase = 1.0 if bin(idx & z_bits).count('1') % 2 == 0 else -1.0
                g_vec[new_idx] += vec[idx] * phase * z_phase
            
            vec = (vec + g_vec)
            norm = np.linalg.norm(vec)
            if norm > 1e-15:
                vec /= norm
                
        return vec.tolist()

    def get_amplitudes_dict(self) -> dict:
        return {}

    @staticmethod
    def check_circuit_compatibility(gates: list) -> list:
        """Pre-flight check: detect non-Clifford gates in a circuit.

        Args:
            gates: List of (gate_name, targets, args) tuples.

        Returns:
            List of gate names that are non-Clifford (empty if all Clifford).
        """
        incompatible = []
        for entry in gates:
            gate_name = entry[0] if isinstance(entry, (list, tuple)) else entry
            if isinstance(gate_name, str) and gate_name.upper() not in CLIFFORD_GATES:
                incompatible.append(gate_name)
        return incompatible

    @staticmethod
    def is_clifford_gate(gate_name: str) -> bool:
        """Check whether a gate name is in the Clifford group."""
        return gate_name.upper() in CLIFFORD_GATES
