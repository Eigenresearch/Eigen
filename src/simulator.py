import math
import cmath
import random
from src.sparse_simulator import SparseQuantumSimulator
from src.tensor_network.mps import MPSSimulator

try:
    import eigen_native as native
except ImportError:
    native = None

class QuantumSimulator:
    def __init__(self, sim_type='dense', gpu_platform='none'):
        # sim_type: 'dense', 'sparse', or 'mps'
        self.sim_type = sim_type
        self.gpu_platform = gpu_platform
        self.state_vector = [1.0 + 0.0j]
        self.qubit_map = {}
        self.num_qubits = 0
        self.is_sparse = False
        
        # Delegates
        self.sparse_sim = None
        self.mps_sim = None
        self.gpu_engine = None
        
        if self.gpu_platform != 'none' and self.sim_type == 'dense':
            from src.backend.gpu.gpu_engine import GPUEngine
            self.gpu_engine = GPUEngine(self.gpu_platform)
            if self.gpu_engine.platform == 'none':
                self.gpu_platform = 'none'
                self.gpu_engine = None
                
        if self.sim_type == 'sparse':
            self.is_sparse = True
            self.sparse_sim = SparseQuantumSimulator()
            self.state_vector = None
        elif self.sim_type == 'mps':
            self.mps_sim = MPSSimulator()
            self.state_vector = None

        # Precomputed gate matrices cache for dense path
        inv_sqrt2 = 0.7071067811865475
        self.gate_cache = {
            'H': [[inv_sqrt2, inv_sqrt2], [inv_sqrt2, -inv_sqrt2]],
            'X': [[0.0, 1.0], [1.0, 0.0]],
            'Y': [[0.0, -1j], [1j, 0.0]],
            'Z': [[1.0, 0.0], [0.0, -1.0]],
            'S': [[1.0, 0.0], [0.0, 1j]],
            'T': [[1.0, 0.0], [0.0, 0.7071067811865475 + 0.7071067811865475j]],
            'CNOT': [[1.0, 0.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0],
                     [0.0, 0.0, 1.0, 0.0]],
            'CZ': [[1.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, -1.0]],
            'SWAP': [[1.0, 0.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0],
                     [0.0, 1.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]]
        }

    def allocate_qubit(self, name: str):
        if name in self.qubit_map:
            return  # Already allocated
        self.qubit_map[name] = self.num_qubits
        self.num_qubits += 1
        
        if self.sim_type == 'mps':
            self.mps_sim.allocate_qubit(name)
            return
            
        if self.is_sparse:
            self.sparse_sim.allocate_qubit(name)
        elif self.gpu_engine:
            if self.num_qubits == 1:
                self.gpu_engine.initialize_state(1)
            else:
                if self.gpu_engine.platform in ('cuda', 'none'):
                    zeros = self.gpu_engine.xp.zeros_like(self.gpu_engine.device_state)
                    self.gpu_engine.device_state = self.gpu_engine.xp.concatenate([self.gpu_engine.device_state, zeros])
                else:
                    import torch
                    zeros = torch.zeros_like(self.gpu_engine.device_state)
                    self.gpu_engine.device_state = torch.cat([self.gpu_engine.device_state, zeros])
        elif self.num_qubits > 20:
            # Transition to sparse mode
            self.is_sparse = True
            self.sparse_sim = SparseQuantumSimulator()
            # Allocate all existing qubits in order
            sorted_qubits = sorted(self.qubit_map.keys(), key=lambda q: self.qubit_map[q])
            for q in sorted_qubits:
                self.sparse_sim.allocate_qubit(q)
            # Transfer dense state vector into sparse simulator
            # dense state vector index is converted to key mapping
            for i, amp in enumerate(self.state_vector):
                if abs(amp) > 1e-12:
                    # Construct the key
                    key = ""
                    for q in sorted_qubits:
                        q_idx = self.qubit_map[q]
                        key += '1' if (i & (1 << q_idx)) else '0'
                    self.sparse_sim.state[key] = amp
            self.state_vector = None
        else:
            self.state_vector = self.state_vector + [0.0j] * len(self.state_vector)

    def get_qubit_index(self, name: str) -> int:
        if name not in self.qubit_map:
            raise KeyError(f"Qubit '{name}' is not allocated in the simulator")
        return self.qubit_map[name]

    def apply_1qubit_gate(self, name: str, gate_matrix: list[list[complex]]):
        if self.sim_type == 'mps':
            self.mps_sim.apply_1qubit_gate(name, gate_matrix)
            return
        if self.is_sparse:
            self.sparse_sim.apply_1qubit_gate(name, gate_matrix)
            return
        if self.gpu_engine:
            self.gpu_engine.apply_gate([self.get_qubit_index(name)], gate_matrix)
            return
            
        k = self.get_qubit_index(name)
        n = len(self.state_vector)
        u00, u01 = gate_matrix[0][0], gate_matrix[0][1]
        u10, u11 = gate_matrix[1][0], gate_matrix[1][1]
        
        step = 1 << k
        for i in range(0, n, step * 2):
            for j in range(i, i + step):
                i0 = j
                i1 = j + step
                a0 = self.state_vector[i0]
                a1 = self.state_vector[i1]
                self.state_vector[i0] = u00 * a0 + u01 * a1
                self.state_vector[i1] = u10 * a0 + u11 * a1

    def _run_native_1qubit(self, q: str, fn_name: str) -> bool:
        if native is not None:
            fn = getattr(native, fn_name)
            k = self.get_qubit_index(q)
            state_tuples = [(c.real, c.imag) for c in self.state_vector]
            res_tuples = fn(state_tuples, k)
            self.state_vector = [complex(r, i) for r, i in res_tuples]
            return True
        return False

    def _run_native_cnot(self, control: str, target: str) -> bool:
        if native is not None:
            c = self.get_qubit_index(control)
            t = self.get_qubit_index(target)
            state_tuples = [(c.real, c.imag) for c in self.state_vector]
            res_tuples = native.apply_cnot(state_tuples, c, t)
            self.state_vector = [complex(r, i) for r, i in res_tuples]
            return True
        return False

    def H(self, q: str):
        if self.sim_type == 'mps':
            self.mps_sim.H(q)
            return
        if self.is_sparse:
            self.sparse_sim.H(q)
            return
        if self.gpu_engine:
            self.gpu_engine.apply_gate([self.get_qubit_index(q)], self.gate_cache['H'])
            return
        if self._run_native_1qubit(q, 'apply_h'):
            return
        k = self.get_qubit_index(q)
        n = len(self.state_vector)
        inv_sqrt2 = 0.7071067811865475
        step = 1 << k
        for i in range(0, n, step * 2):
            for j in range(i, i + step):
                i0 = j
                i1 = j + step
                a0 = self.state_vector[i0]
                a1 = self.state_vector[i1]
                self.state_vector[i0] = (a0 + a1) * inv_sqrt2
                self.state_vector[i1] = (a0 - a1) * inv_sqrt2

    def X(self, q: str):
        if self.sim_type == 'mps':
            self.mps_sim.X(q)
            return
        if self.is_sparse:
            self.sparse_sim.X(q)
            return
        if self.gpu_engine:
            self.gpu_engine.apply_gate([self.get_qubit_index(q)], self.gate_cache['X'])
            return
        if self._run_native_1qubit(q, 'apply_x'):
            return
        k = self.get_qubit_index(q)
        n = len(self.state_vector)
        step = 1 << k
        for i in range(0, n, step * 2):
            for j in range(i, i + step):
                i0 = j
                i1 = j + step
                self.state_vector[i0], self.state_vector[i1] = self.state_vector[i1], self.state_vector[i0]

    def Y(self, q: str):
        if self.sim_type == 'mps':
            self.mps_sim.Y(q)
            return
        if self.is_sparse:
            self.sparse_sim.Y(q)
            return
        if self.gpu_engine:
            self.gpu_engine.apply_gate([self.get_qubit_index(q)], self.gate_cache['Y'])
            return
        if self._run_native_1qubit(q, 'apply_y'):
            return
        k = self.get_qubit_index(q)
        n = len(self.state_vector)
        step = 1 << k
        for i in range(0, n, step * 2):
            for j in range(i, i + step):
                i0 = j
                i1 = j + step
                a0 = self.state_vector[i0]
                a1 = self.state_vector[i1]
                self.state_vector[i0] = -1j * a1
                self.state_vector[i1] = 1j * a0

    def Z(self, q: str):
        if self.sim_type == 'mps':
            self.mps_sim.Z(q)
            return
        if self.is_sparse:
            self.sparse_sim.Z(q)
            return
        if self.gpu_engine:
            self.gpu_engine.apply_gate([self.get_qubit_index(q)], self.gate_cache['Z'])
            return
        if self._run_native_1qubit(q, 'apply_z'):
            return
        k = self.get_qubit_index(q)
        n = len(self.state_vector)
        step = 1 << k
        for i in range(0, n, step * 2):
            for j in range(i, i + step):
                self.state_vector[j + step] = -self.state_vector[j + step]

    def S(self, q: str):
        if self.sim_type == 'mps':
            self.mps_sim.S(q)
            return
        if self.is_sparse:
            self.sparse_sim.S(q)
            return
        if self.gpu_engine:
            self.gpu_engine.apply_gate([self.get_qubit_index(q)], self.gate_cache['S'])
            return
        if self._run_native_1qubit(q, 'apply_s'):
            return
        k = self.get_qubit_index(q)
        n = len(self.state_vector)
        step = 1 << k
        for i in range(0, n, step * 2):
            for j in range(i, i + step):
                self.state_vector[j + step] *= 1j

    def T(self, q: str):
        if self.sim_type == 'mps':
            self.mps_sim.T(q)
            return
        if self.is_sparse:
            self.sparse_sim.T(q)
            return
        if self.gpu_engine:
            self.gpu_engine.apply_gate([self.get_qubit_index(q)], self.gate_cache['T'])
            return
        if self._run_native_1qubit(q, 'apply_t'):
            return
        k = self.get_qubit_index(q)
        n = len(self.state_vector)
        step = 1 << k
        t_phase = 0.7071067811865475 + 0.7071067811865475j
        for i in range(0, n, step * 2):
            for j in range(i, i + step):
                self.state_vector[j + step] *= t_phase

    def RX(self, q: str, theta: float):
        if self.sim_type == 'mps':
            self.mps_sim.RX(q, theta)
            return
        if self.is_sparse:
            self.sparse_sim.RX(q, theta)
            return
        cos_val = math.cos(theta / 2)
        sin_val = math.sin(theta / 2)
        self.apply_1qubit_gate(q, [
            [cos_val, -1j * sin_val],
            [-1j * sin_val, cos_val]
        ])

    def RY(self, q: str, theta: float):
        if self.sim_type == 'mps':
            self.mps_sim.RY(q, theta)
            return
        if self.is_sparse:
            self.sparse_sim.RY(q, theta)
            return
        cos_val = math.cos(theta / 2)
        sin_val = math.sin(theta / 2)
        self.apply_1qubit_gate(q, [
            [cos_val, -sin_val],
            [sin_val, cos_val]
        ])

    def RZ(self, q: str, theta: float):
        if self.sim_type == 'mps':
            self.mps_sim.RZ(q, theta)
            return
        if self.is_sparse:
            self.sparse_sim.RZ(q, theta)
            return
        self.apply_1qubit_gate(q, [
            [cmath.exp(-1j * theta / 2), 0.0j],
            [0.0j, cmath.exp(1j * theta / 2)]
        ])

    def CNOT(self, control: str, target: str):
        if self.sim_type == 'mps':
            self.mps_sim.CNOT(control, target)
            return
        if self.is_sparse:
            self.sparse_sim.CNOT(control, target)
            return
        if self.gpu_engine:
            c = self.get_qubit_index(control)
            t = self.get_qubit_index(target)
            self.gpu_engine.apply_gate([c, t], self.gate_cache['CNOT'])
            return
        if self._run_native_cnot(control, target):
            return
            
        c = self.get_qubit_index(control)
        t = self.get_qubit_index(target)
        n = len(self.state_vector)
        c_mask = 1 << c
        t_mask = 1 << t
        
        for i in range(n):
            if (i & c_mask) and not (i & t_mask):
                i_target_1 = i | t_mask
                self.state_vector[i], self.state_vector[i_target_1] = (
                    self.state_vector[i_target_1],
                    self.state_vector[i]
                )

    def CZ(self, control: str, target: str):
        if self.sim_type == 'mps':
            self.mps_sim.CZ(control, target)
            return
        if self.is_sparse:
            self.sparse_sim.CZ(control, target)
            return
        if self.gpu_engine:
            c = self.get_qubit_index(control)
            t = self.get_qubit_index(target)
            self.gpu_engine.apply_gate([c, t], self.gate_cache['CZ'])
            return
            
        c = self.get_qubit_index(control)
        t = self.get_qubit_index(target)
        n = len(self.state_vector)
        c_mask = 1 << c
        t_mask = 1 << t
        
        for i in range(n):
            if (i & c_mask) and (i & t_mask):
                self.state_vector[i] = -self.state_vector[i]

    def SWAP(self, q1: str, q2: str):
        if self.sim_type == 'mps':
            self.mps_sim.SWAP(q1, q2)
            return
        if self.is_sparse:
            self.sparse_sim.SWAP(q1, q2)
            return
        if self.gpu_engine:
            idx1 = self.get_qubit_index(q1)
            idx2 = self.get_qubit_index(q2)
            self.gpu_engine.apply_gate([idx1, idx2], self.gate_cache['SWAP'])
            return
            
        idx1 = self.get_qubit_index(q1)
        idx2 = self.get_qubit_index(q2)
        n = len(self.state_vector)
        idx1_mask = 1 << idx1
        idx2_mask = 1 << idx2
        
        for i in range(n):
            if (i & idx1_mask) and not (i & idx2_mask):
                j = (i & ~idx1_mask) | idx2_mask
                self.state_vector[i], self.state_vector[j] = (
                    self.state_vector[j],
                    self.state_vector[i]
                )

    def measure(self, q: str) -> int:
        if self.sim_type == 'mps':
            return self.mps_sim.measure(q)
        if self.is_sparse:
            return self.sparse_sim.measure(q)
        if self.gpu_engine:
            # Pull state to CPU, measure, push back
            self.state_vector = self.gpu_engine.get_state()
            k = self.get_qubit_index(q)
            n = len(self.state_vector)
            k_mask = 1 << k
            p0 = sum(abs(amp)**2 for i, amp in enumerate(self.state_vector) if not (i & k_mask))
            r = random.random()
            if r < p0:
                norm = math.sqrt(p0) if p0 > 1e-15 else 1.0
                for i in range(n):
                    if i & k_mask:
                        self.state_vector[i] = 0.0j
                    else:
                        self.state_vector[i] /= norm
                self.gpu_engine.set_state(self.state_vector)
                self.state_vector = None
                return 0
            else:
                p1 = 1.0 - p0
                norm = math.sqrt(p1) if p1 > 1e-15 else 1.0
                for i in range(n):
                    if not (i & k_mask):
                        self.state_vector[i] = 0.0j
                    else:
                        self.state_vector[i] /= norm
                self.gpu_engine.set_state(self.state_vector)
                self.state_vector = None
                return 1
            
        k = self.get_qubit_index(q)
        n = len(self.state_vector)
        k_mask = 1 << k
        
        p0 = sum(abs(amp)**2 for i, amp in enumerate(self.state_vector) if not (i & k_mask))
        r = random.random()
        
        if r < p0:
            norm = math.sqrt(p0) if p0 > 1e-15 else 1.0
            for i in range(n):
                if i & k_mask:
                    self.state_vector[i] = 0.0j
                else:
                    self.state_vector[i] /= norm
            return 0
        else:
            p1 = 1.0 - p0
            norm = math.sqrt(p1) if p1 > 1e-15 else 1.0
            for i in range(n):
                if not (i & k_mask):
                    self.state_vector[i] = 0.0j
                else:
                    self.state_vector[i] /= norm
            return 1

    def get_state_vector(self) -> list[complex]:
        if self.sim_type == 'mps':
            return self.mps_sim.get_state_vector()
        if self.is_sparse:
            return self.sparse_sim.get_state_vector()
        if self.gpu_engine:
            return self.gpu_engine.get_state()
        return self.state_vector

    def get_amplitudes_dict(self) -> dict[str, complex]:
        if self.sim_type == 'mps':
            return self.mps_sim.get_amplitudes_dict()
        if self.is_sparse:
            return self.sparse_sim.get_amplitudes_dict()
            
        if self.num_qubits == 0:
            return {"": 1.0 + 0.0j}
        
        state_vec = self.gpu_engine.get_state() if self.gpu_engine else self.state_vector
        sorted_qubits = sorted(self.qubit_map.keys(), key=lambda name: self.qubit_map[name])
        amplitudes = {}
        for i, amp in enumerate(state_vec):
            if abs(amp) > 1e-12:
                bitstring = ""
                for q in reversed(sorted_qubits):
                    q_idx = self.qubit_map[q]
                    bitstring += str((i >> q_idx) & 1)
                amplitudes[bitstring] = amp
        return amplitudes
