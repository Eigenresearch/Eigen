import math
import cmath
import random

class SparseQuantumSimulator:
    def __init__(self):
        self.state = {"": 1.0 + 0.0j}  # maps bitstrings (e.g. "01") -> complex amplitude
        self.qubit_map = {}            # qubit_name -> index (0-based)
        self.num_qubits = 0

    def allocate_qubit(self, name: str):
        if name in self.qubit_map:
            return
        self.qubit_map[name] = self.num_qubits
        self.num_qubits += 1
        
        # Append '0' to all existing state keys
        new_state = {}
        for key, amp in self.state.items():
            new_state[key + "0"] = amp
        self.state = new_state

    def get_qubit_index(self, name: str) -> int:
        if name not in self.qubit_map:
            raise KeyError(f"Qubit '{name}' is not allocated in the simulator")
        return self.qubit_map[name]

    def apply_1qubit_gate(self, name: str, gate_matrix: list[list[complex]]):
        k = self.get_qubit_index(name)
        u00, u01 = gate_matrix[0][0], gate_matrix[0][1]
        u10, u11 = gate_matrix[1][0], gate_matrix[1][1]
        
        groups = {}
        for key, amp in self.state.items():
            prefix = key[:k]
            suffix = key[k+1:]
            base = (prefix, suffix)
            if base not in groups:
                groups[base] = [0.0j, 0.0j]
            bit = key[k]
            if bit == '0':
                groups[base][0] = amp
            else:
                groups[base][1] = amp
                
        new_state = {}
        for (prefix, suffix), [a0, a1] in groups.items():
            v0 = u00 * a0 + u01 * a1
            v1 = u10 * a0 + u11 * a1
            
            key0 = prefix + '0' + suffix
            key1 = prefix + '1' + suffix
            
            if abs(v0) > 1e-12:
                new_state[key0] = v0
            if abs(v1) > 1e-12:
                new_state[key1] = v1
        self.state = new_state

    def H(self, q: str):
        k = self.get_qubit_index(q)
        inv_sqrt2 = 0.7071067811865475
        
        groups = {}
        for key, amp in self.state.items():
            prefix = key[:k]
            suffix = key[k+1:]
            base = (prefix, suffix)
            if base not in groups:
                groups[base] = [0.0j, 0.0j]
            if key[k] == '0':
                groups[base][0] = amp
            else:
                groups[base][1] = amp
                
        new_state = {}
        for (prefix, suffix), [a0, a1] in groups.items():
            v0 = (a0 + a1) * inv_sqrt2
            v1 = (a0 - a1) * inv_sqrt2
            
            key0 = prefix + '0' + suffix
            key1 = prefix + '1' + suffix
            
            if abs(v0) > 1e-12:
                new_state[key0] = v0
            if abs(v1) > 1e-12:
                new_state[key1] = v1
        self.state = new_state

    def X(self, q: str):
        k = self.get_qubit_index(q)
        new_state = {}
        for key, amp in self.state.items():
            prefix = key[:k]
            suffix = key[k+1:]
            new_bit = '1' if key[k] == '0' else '0'
            new_state[prefix + new_bit + suffix] = amp
        self.state = new_state

    def Y(self, q: str):
        k = self.get_qubit_index(q)
        new_state = {}
        for key, amp in self.state.items():
            prefix = key[:k]
            suffix = key[k+1:]
            if key[k] == '0':
                new_state[prefix + '1' + suffix] = 1j * amp
            else:
                new_state[prefix + '0' + suffix] = -1j * amp
        self.state = new_state

    def Z(self, q: str):
        k = self.get_qubit_index(q)
        new_state = {}
        for key, amp in self.state.items():
            if key[k] == '1':
                new_state[key] = -amp
            else:
                new_state[key] = amp
        self.state = new_state

    def S(self, q: str):
        k = self.get_qubit_index(q)
        new_state = {}
        for key, amp in self.state.items():
            if key[k] == '1':
                new_state[key] = 1j * amp
            else:
                new_state[key] = amp
        self.state = new_state

    def T(self, q: str):
        k = self.get_qubit_index(q)
        t_phase = 0.7071067811865475 + 0.7071067811865475j
        new_state = {}
        for key, amp in self.state.items():
            if key[k] == '1':
                new_state[key] = t_phase * amp
            else:
                new_state[key] = amp
        self.state = new_state

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

    def CNOT(self, control: str, target: str):
        c = self.get_qubit_index(control)
        t = self.get_qubit_index(target)
        new_state = {}
        for key, amp in self.state.items():
            if key[c] == '1':
                flipped_char = '1' if key[t] == '0' else '0'
                new_key = key[:t] + flipped_char + key[t+1:]
                new_state[new_key] = amp
            else:
                new_state[key] = amp
        self.state = new_state

    def CZ(self, control: str, target: str):
        c = self.get_qubit_index(control)
        t = self.get_qubit_index(target)
        new_state = {}
        for key, amp in self.state.items():
            if key[c] == '1' and key[t] == '1':
                new_state[key] = -amp
            else:
                new_state[key] = amp
        self.state = new_state

    def SWAP(self, q1: str, q2: str):
        idx1 = self.get_qubit_index(q1)
        idx2 = self.get_qubit_index(q2)
        new_state = {}
        for key, amp in self.state.items():
            chars = list(key)
            chars[idx1], chars[idx2] = chars[idx2], chars[idx1]
            new_key = "".join(chars)
            new_state[new_key] = amp
        self.state = new_state

    def measure(self, q: str) -> int:
        k = self.get_qubit_index(q)
        p0 = sum(abs(amp)**2 for key, amp in self.state.items() if key[k] == '0')
        r = random.random()
        
        if r < p0:
            outcome = 0
            norm = math.sqrt(p0) if p0 > 1e-15 else 1.0
            new_state = {}
            for key, amp in self.state.items():
                if key[k] == '0':
                    new_state[key] = amp / norm
            self.state = new_state
        else:
            outcome = 1
            p1 = 1.0 - p0
            norm = math.sqrt(p1) if p1 > 1e-15 else 1.0
            new_state = {}
            for key, amp in self.state.items():
                if key[k] == '1':
                    new_state[key] = amp / norm
            self.state = new_state
        return outcome

    def get_state_vector(self) -> list[complex]:
        if self.num_qubits > 24:
            raise RuntimeError(f"Cannot reconstruct full state vector for {self.num_qubits} qubits (memory limit).")
        n = 1 << self.num_qubits
        vec = [0.0j] * n
        sorted_qubits = sorted(self.qubit_map.keys(), key=lambda name: self.qubit_map[name])
        for key, amp in self.state.items():
            idx = 0
            for i, q in enumerate(sorted_qubits):
                q_idx = self.qubit_map[q]
                if key[q_idx] == '1':
                    idx |= (1 << q_idx)
            vec[idx] = amp
        return vec

    def get_amplitudes_dict(self) -> dict[str, complex]:
        if self.num_qubits == 0:
            return {"": 1.0 + 0.0j}
        sorted_qubits = sorted(self.qubit_map.keys(), key=lambda name: self.qubit_map[name])
        
        amplitudes = {}
        for key, amp in self.state.items():
            if abs(amp) > 1e-12:
                bitstring = ""
                for q in reversed(sorted_qubits):
                    q_idx = self.qubit_map[q]
                    bitstring += key[q_idx]
                amplitudes[bitstring] = amp
        return amplitudes
