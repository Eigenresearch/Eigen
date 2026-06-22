import math
import cmath
import random

class QuantumSimulator:
    def __init__(self):
        # State vector starts with 0 qubits: a single state |> with amplitude 1.0
        self.state_vector = [1.0 + 0.0j]
        # Maps qubit_name (str) -> internal qubit_index (int)
        self.qubit_map = {}
        self.num_qubits = 0

    def allocate_qubit(self, name: str):
        if name in self.qubit_map:
            return  # Already allocated
        self.qubit_map[name] = self.num_qubits
        self.num_qubits += 1
        
        # Double the size of the state vector: new qubit starts in state |0>
        # New amplitudes: state_vector[i] is for the state where the new qubit is 0,
        # and 0.0 for the state where the new qubit is 1.
        self.state_vector = self.state_vector + [0.0j] * len(self.state_vector)

    def get_qubit_index(self, name: str) -> int:
        if name not in self.qubit_map:
            raise KeyError(f"Qubit '{name}' is not allocated in the simulator")
        return self.qubit_map[name]

    def apply_1qubit_gate(self, name: str, gate_matrix: list[list[complex]]):
        k = self.get_qubit_index(name)
        n = len(self.state_vector)
        
        # Apply gate_matrix (2x2) to qubit k
        u00, u01 = gate_matrix[0][0], gate_matrix[0][1]
        u10, u11 = gate_matrix[1][0], gate_matrix[1][1]
        
        for i in range(n):
            if ((i >> k) & 1) == 0:
                i0 = i
                i1 = i | (1 << k)
                
                a0 = self.state_vector[i0]
                a1 = self.state_vector[i1]
                
                self.state_vector[i0] = u00 * a0 + u01 * a1
                self.state_vector[i1] = u10 * a0 + u11 * a1

    def H(self, q: str):
        inv_sqrt2 = 1.0 / math.sqrt(2)
        matrix = [
            [inv_sqrt2, inv_sqrt2],
            [inv_sqrt2, -inv_sqrt2]
        ]
        self.apply_1qubit_gate(q, matrix)

    def X(self, q: str):
        matrix = [
            [0.0j, 1.0 + 0.0j],
            [1.0 + 0.0j, 0.0j]
        ]
        self.apply_1qubit_gate(q, matrix)

    def Y(self, q: str):
        matrix = [
            [0.0j, -1j],
            [1j, 0.0j]
        ]
        self.apply_1qubit_gate(q, matrix)

    def Z(self, q: str):
        matrix = [
            [1.0 + 0.0j, 0.0j],
            [0.0j, -1.0 + 0.0j]
        ]
        self.apply_1qubit_gate(q, matrix)

    def S(self, q: str):
        matrix = [
            [1.0 + 0.0j, 0.0j],
            [0.0j, 1j]
        ]
        self.apply_1qubit_gate(q, matrix)

    def T(self, q: str):
        matrix = [
            [1.0 + 0.0j, 0.0j],
            [0.0j, cmath.exp(1j * math.pi / 4)]
        ]
        self.apply_1qubit_gate(q, matrix)

    def RX(self, q: str, theta: float):
        cos_val = math.cos(theta / 2)
        sin_val = math.sin(theta / 2)
        matrix = [
            [cos_val, -1j * sin_val],
            [-1j * sin_val, cos_val]
        ]
        self.apply_1qubit_gate(q, matrix)

    def RY(self, q: str, theta: float):
        cos_val = math.cos(theta / 2)
        sin_val = math.sin(theta / 2)
        matrix = [
            [cos_val, -sin_val],
            [sin_val, cos_val]
        ]
        self.apply_1qubit_gate(q, matrix)

    def RZ(self, q: str, theta: float):
        matrix = [
            [cmath.exp(-1j * theta / 2), 0.0j],
            [0.0j, cmath.exp(1j * theta / 2)]
        ]
        self.apply_1qubit_gate(q, matrix)

    def CNOT(self, control: str, target: str):
        c = self.get_qubit_index(control)
        t = self.get_qubit_index(target)
        n = len(self.state_vector)
        
        for i in range(n):
            # If control bit is 1 and target bit is 0, swap amplitude with state where target bit is 1
            if ((i >> c) & 1) == 1 and ((i >> t) & 1) == 0:
                i_target_1 = i | (1 << t)
                self.state_vector[i], self.state_vector[i_target_1] = (
                    self.state_vector[i_target_1],
                    self.state_vector[i]
                )

    def CZ(self, control: str, target: str):
        c = self.get_qubit_index(control)
        t = self.get_qubit_index(target)
        n = len(self.state_vector)
        
        for i in range(n):
            # If control is 1 and target is 1, multiply amplitude by -1
            if ((i >> c) & 1) == 1 and ((i >> t) & 1) == 1:
                self.state_vector[i] = -self.state_vector[i]

    def SWAP(self, q1: str, q2: str):
        idx1 = self.get_qubit_index(q1)
        idx2 = self.get_qubit_index(q2)
        n = len(self.state_vector)
        
        for i in range(n):
            # Swap if qubit1 is 1 and qubit2 is 0
            if ((i >> idx1) & 1) == 1 and ((i >> idx2) & 1) == 0:
                # Find index where qubit1 is 0 and qubit2 is 1
                j = (i & ~(1 << idx1)) | (1 << idx2)
                self.state_vector[i], self.state_vector[j] = (
                    self.state_vector[j],
                    self.state_vector[i]
                )

    def measure(self, q: str) -> int:
        k = self.get_qubit_index(q)
        n = len(self.state_vector)
        
        # Calculate probability of measuring 0
        p0 = sum(abs(amp)**2 for i, amp in enumerate(self.state_vector) if ((i >> k) & 1) == 0)
        
        # Roll random float in [0.0, 1.0)
        r = random.random()
        
        if r < p0:
            # Measure 0: collapse state vector to only states where qubit k is 0
            norm = math.sqrt(p0)
            for i in range(n):
                if ((i >> k) & 1) == 1:
                    self.state_vector[i] = 0.0j
                else:
                    self.state_vector[i] /= norm
            return 0
        else:
            # Measure 1: collapse state vector to only states where qubit k is 1
            p1 = 1.0 - p0
            norm = math.sqrt(p1) if p1 > 1e-15 else 1.0
            for i in range(n):
                if ((i >> k) & 1) == 0:
                    self.state_vector[i] = 0.0j
                else:
                    self.state_vector[i] /= norm
            return 1

    def get_state_vector(self) -> list[complex]:
        return self.state_vector

    def get_amplitudes_dict(self) -> dict[str, complex]:
        """Returns a dict of state (e.g. '01') to complex amplitude."""
        if self.num_qubits == 0:
            return {"": 1.0 + 0.0j}
            
        # Get sorted list of qubit names to construct consistent bitstrings
        sorted_qubits = sorted(self.qubit_map.keys(), key=lambda name: self.qubit_map[name])
        
        amplitudes = {}
        for i, amp in enumerate(self.state_vector):
            if abs(amp) > 1e-12:
                # Construct bitstring representation
                bitstring = ""
                for q in reversed(sorted_qubits):
                    idx = self.qubit_map[q]
                    bitstring += str((i >> idx) & 1)
                amplitudes[bitstring] = amp
        return amplitudes
