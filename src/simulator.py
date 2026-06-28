import math
import cmath
import random
from src.sparse_simulator import SparseQuantumSimulator
from src.tensor_network.mps import MPSSimulator

try:
    import eigen_native as native
except ImportError:
    native = None


class StateBackend:
    def allocate_qubit(self):
        raise NotImplementedError

    def H(self, k: int):
        raise NotImplementedError

    def X(self, k: int):
        raise NotImplementedError

    def Y(self, k: int):
        raise NotImplementedError

    def Z(self, k: int):
        raise NotImplementedError

    def S(self, k: int):
        raise NotImplementedError

    def T(self, k: int):
        raise NotImplementedError

    def RX(self, k: int, theta: float):
        raise NotImplementedError

    def RY(self, k: int, theta: float):
        raise NotImplementedError

    def RZ(self, k: int, theta: float):
        raise NotImplementedError

    def apply_1qubit_gate(self, k: int, gate_matrix: list[list[complex]]):
        raise NotImplementedError

    def CNOT(self, control: int, target: int):
        raise NotImplementedError

    def CZ(self, control: int, target: int):
        raise NotImplementedError

    def SWAP(self, q1: int, q2: int):
        raise NotImplementedError

    def measure(self, k: int, r: float) -> int:
        raise NotImplementedError

    def get_state_vector(self) -> list[complex]:
        raise NotImplementedError

    def set_state_vector(self, value: list[complex]):
        raise NotImplementedError


class RustStatevectorWrapper(StateBackend):
    def __init__(self):
        self.rust_sv = native.RustStatevector()

    def allocate_qubit(self):
        self.rust_sv.allocate_qubit()

    def H(self, k: int):
        self.rust_sv.apply_h(k)

    def X(self, k: int):
        self.rust_sv.apply_x(k)

    def Y(self, k: int):
        self.rust_sv.apply_y(k)

    def Z(self, k: int):
        self.rust_sv.apply_z(k)

    def S(self, k: int):
        self.rust_sv.apply_s(k)

    def T(self, k: int):
        self.rust_sv.apply_t(k)

    def RX(self, k: int, theta: float):
        self.rust_sv.apply_rx(k, float(theta))

    def RY(self, k: int, theta: float):
        self.rust_sv.apply_ry(k, float(theta))

    def RZ(self, k: int, theta: float):
        self.rust_sv.apply_rz(k, float(theta))

    def apply_1qubit_gate(self, k: int, gate_matrix: list[list[complex]]):
        u00 = gate_matrix[0][0]
        u01 = gate_matrix[0][1]
        u10 = gate_matrix[1][0]
        u11 = gate_matrix[1][1]
        self.rust_sv.apply_1qubit_gate(
            k,
            float(u00.real), float(u00.imag),
            float(u01.real), float(u01.imag),
            float(u10.real), float(u10.imag),
            float(u11.real), float(u11.imag)
        )

    def CNOT(self, control: int, target: int):
        self.rust_sv.apply_cnot(control, target)

    def CZ(self, control: int, target: int):
        self.rust_sv.apply_cz(control, target)

    def SWAP(self, q1: int, q2: int):
        self.rust_sv.apply_swap(q1, q2)

    def measure(self, k: int, r: float) -> int:
        return self.rust_sv.measure(k, r)

    def get_state_vector(self) -> list[complex]:
        raw = self.rust_sv.get_state()
        return [complex(r, i) for r, i in raw]

    def set_state_vector(self, value: list[complex]):
        raw = [(c.real, c.imag) for c in value]
        self.rust_sv.set_state(raw)


class PythonDenseStatevector(StateBackend):
    def __init__(self):
        self._state = [1.0 + 0.0j]

    def allocate_qubit(self):
        self._state = self._state + [0.0j] * len(self._state)

    def H(self, k: int):
        n = len(self._state)
        inv_sqrt2 = 0.7071067811865475
        step = 1 << k
        for i in range(0, n, step * 2):
            for j in range(i, i + step):
                i0 = j
                i1 = j + step
                a0 = self._state[i0]
                a1 = self._state[i1]
                self._state[i0] = (a0 + a1) * inv_sqrt2
                self._state[i1] = (a0 - a1) * inv_sqrt2

    def X(self, k: int):
        n = len(self._state)
        step = 1 << k
        for i in range(0, n, step * 2):
            for j in range(i, i + step):
                i0 = j
                i1 = j + step
                self._state[i0], self._state[i1] = self._state[i1], self._state[i0]

    def Y(self, k: int):
        n = len(self._state)
        step = 1 << k
        for i in range(0, n, step * 2):
            for j in range(i, i + step):
                i0 = j
                i1 = j + step
                a0 = self._state[i0]
                a1 = self._state[i1]
                self._state[i0] = -1j * a1
                self._state[i1] = 1j * a0

    def Z(self, k: int):
        n = len(self._state)
        step = 1 << k
        for i in range(0, n, step * 2):
            for j in range(i, i + step):
                self._state[j + step] = -self._state[j + step]

    def S(self, k: int):
        n = len(self._state)
        step = 1 << k
        for i in range(0, n, step * 2):
            for j in range(i, i + step):
                self._state[j + step] *= 1j

    def T(self, k: int):
        n = len(self._state)
        step = 1 << k
        t_phase = 0.7071067811865475 + 0.7071067811865475j
        for i in range(0, n, step * 2):
            for j in range(i, i + step):
                self._state[j + step] *= t_phase

    def RX(self, k: int, theta: float):
        cos_val = math.cos(theta / 2)
        node_sin = math.sin(theta / 2)
        self.apply_1qubit_gate(k, [
            [cos_val, -1j * node_sin],
            [-1j * node_sin, cos_val]
        ])

    def RY(self, k: int, theta: float):
        cos_val = math.cos(theta / 2)
        node_sin = math.sin(theta / 2)
        self.apply_1qubit_gate(k, [
            [cos_val, -node_sin],
            [node_sin, cos_val]
        ])

    def RZ(self, k: int, theta: float):
        self.apply_1qubit_gate(k, [
            [cmath.exp(-1j * theta / 2), 0.0j],
            [0.0j, cmath.exp(1j * theta / 2)]
        ])

    def apply_1qubit_gate(self, k: int, gate_matrix: list[list[complex]]):
        n = len(self._state)
        u00, u01 = gate_matrix[0][0], gate_matrix[0][1]
        u10, u11 = gate_matrix[1][0], gate_matrix[1][1]
        
        step = 1 << k
        for i in range(0, n, step * 2):
            for j in range(i, i + step):
                i0 = j
                i1 = j + step
                a0 = self._state[i0]
                a1 = self._state[i1]
                self._state[i0] = u00 * a0 + u01 * a1
                self._state[i1] = u10 * a0 + u11 * a1

    def CNOT(self, control: int, target: int):
        n = len(self._state)
        c_mask = 1 << control
        t_mask = 1 << target
        for i in range(n):
            if (i & c_mask) and not (i & t_mask):
                i_target_1 = i | t_mask
                self._state[i], self._state[i_target_1] = (
                    self._state[i_target_1],
                    self._state[i]
                )

    def CZ(self, control: int, target: int):
        n = len(self._state)
        c_mask = 1 << control
        t_mask = 1 << target
        for i in range(n):
            if (i & c_mask) and (i & t_mask):
                self._state[i] = -self._state[i]

    def SWAP(self, q1: int, q2: int):
        n = len(self._state)
        idx1_mask = 1 << q1
        idx2_mask = 1 << q2
        for i in range(n):
            if (i & idx1_mask) and not (i & idx2_mask):
                j = (i & ~idx1_mask) | idx2_mask
                self._state[i], self._state[j] = (
                    self._state[j],
                    self._state[i]
                )

    def measure(self, k: int, r: float) -> int:
        n = len(self._state)
        k_mask = 1 << k
        
        p0 = sum(abs(amp)**2 for i, amp in enumerate(self._state) if not (i & k_mask))
        
        if r < p0:
            norm = math.sqrt(p0) if p0 > 1e-15 else 1.0
            for i in range(n):
                if i & k_mask:
                    self._state[i] = 0.0j
                else:
                    self._state[i] /= norm
            return 0
        else:
            p1 = 1.0 - p0
            norm = math.sqrt(p1) if p1 > 1e-15 else 1.0
            for i in range(n):
                if not (i & k_mask):
                    self._state[i] = 0.0j
                else:
                    self._state[i] /= norm
            return 1

    def get_state_vector(self) -> list[complex]:
        return self._state

    def set_state_vector(self, value: list[complex]):
        self._state = list(value)


class StateVectorList(list):
    def __init__(self, iterable, on_update):
        super().__init__(iterable)
        self.on_update = on_update

    def __setitem__(self, index, value):
        super().__setitem__(index, value)
        self.on_update(self)

    def __delitem__(self, index):
        super().__delitem__(index)
        self.on_update(self)


class QuantumSimulator:
    def __init__(self, sim_type='dense', gpu_platform='none', seed=None):
        # sim_type: 'dense', 'sparse', or 'mps'
        self.sim_type = sim_type
        self.gpu_platform = gpu_platform
        self.rng = random.Random(seed)
        self._state_vector = [1.0 + 0.0j]
        self.qubit_map = {}
        self.num_qubits = 0
        self.is_sparse = False
        
        # Delegates
        self.sparse_sim = None
        self.mps_sim = None
        self.gpu_engine = None
        self.dense_backend = None
        
        if self.gpu_platform != 'none' and self.sim_type == 'dense':
            from src.backend.gpu.gpu_engine import GPUEngine
            self.gpu_engine = GPUEngine(self.gpu_platform)
            if self.gpu_engine.platform == 'none':
                self.gpu_platform = 'none'
                self.gpu_engine = None
                
        if self.sim_type == 'sparse':
            self.is_sparse = True
            self.sparse_sim = SparseQuantumSimulator()
            self.sparse_sim.rng = self.rng
        elif self.sim_type == 'mps':
            self.mps_sim = MPSSimulator()
            self.mps_sim.rng = self.rng
        elif self.sim_type == 'dense' and not self.gpu_engine:
            if native is not None and hasattr(native, 'RustStatevector'):
                self.dense_backend = RustStatevectorWrapper()
            else:
                self.dense_backend = PythonDenseStatevector()
        elif self.sim_type == 'auto':
            if native is not None and hasattr(native, 'RustStatevector'):
                self.dense_backend = RustStatevectorWrapper()
            else:
                self.dense_backend = PythonDenseStatevector()

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

    @property
    def state_vector(self) -> list[complex]:
        if self.is_sparse or self.sim_type in ('sparse', 'mps'):
            return None
        if self.dense_backend:
            raw_state = self.dense_backend.get_state_vector()
            def update_cb(new_list):
                self.dense_backend.set_state_vector(new_list)
            return StateVectorList(raw_state, update_cb)
        if hasattr(self, '_state_vector'):
            return self._state_vector
        return [1.0 + 0.0j]

    @state_vector.setter
    def state_vector(self, value: list[complex]):
        if self.is_sparse or self.sim_type in ('sparse', 'mps'):
            return
        if self.dense_backend:
            self.dense_backend.set_state_vector(value)
        else:
            self._state_vector = value

    def configure_backend(self, chosen_sim: str):
        if self.sim_type != 'auto':
            return
        self.sim_type = chosen_sim
        self.sparse_sim = None
        self.mps_sim = None
        self.dense_backend = None
        self.is_sparse = False
        
        if chosen_sim == 'sparse':
            self.is_sparse = True
            self.sparse_sim = SparseQuantumSimulator()
            self.sparse_sim.rng = self.rng
            sorted_qubits = sorted(self.qubit_map.keys(), key=lambda q: self.qubit_map[q])
            for q in sorted_qubits:
                self.sparse_sim.allocate_qubit(q)
        elif chosen_sim == 'mps':
            self.mps_sim = MPSSimulator()
            self.mps_sim.rng = self.rng
            sorted_qubits = sorted(self.qubit_map.keys(), key=lambda q: self.qubit_map[q])
            for q in sorted_qubits:
                self.mps_sim.allocate_qubit(q)
        else:
            if native is not None and hasattr(native, 'RustStatevector'):
                self.dense_backend = RustStatevectorWrapper()
            else:
                self.dense_backend = PythonDenseStatevector()
            for _ in range(self.num_qubits):
                self.dense_backend.allocate_qubit()

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
            self.sparse_sim.rng = self.rng
            sorted_qubits = sorted(self.qubit_map.keys(), key=lambda q: self.qubit_map[q])
            for q in sorted_qubits:
                self.sparse_sim.allocate_qubit(q)
            
            dense_state = self.dense_backend.get_state_vector()
            for i, amp in enumerate(dense_state):
                if abs(amp) > 1e-12:
                    key = ""
                    for q in sorted_qubits:
                        q_idx = self.qubit_map[q]
                        key += '1' if (i & (1 << q_idx)) else '0'
                    self.sparse_sim.state[key] = amp
            self.dense_backend = None
        else:
            self.dense_backend.allocate_qubit()

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
        self.dense_backend.apply_1qubit_gate(self.get_qubit_index(name), gate_matrix)

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
        self.dense_backend.H(self.get_qubit_index(q))

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
        self.dense_backend.X(self.get_qubit_index(q))

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
        self.dense_backend.Y(self.get_qubit_index(q))

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
        self.dense_backend.Z(self.get_qubit_index(q))

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
        self.dense_backend.S(self.get_qubit_index(q))

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
        self.dense_backend.T(self.get_qubit_index(q))

    def RX(self, q: str, theta: float):
        if self.sim_type == 'mps':
            self.mps_sim.RX(q, theta)
            return
        if self.is_sparse:
            self.sparse_sim.RX(q, theta)
            return
        self.dense_backend.RX(self.get_qubit_index(q), theta)

    def RY(self, q: str, theta: float):
        if self.sim_type == 'mps':
            self.mps_sim.RY(q, theta)
            return
        if self.is_sparse:
            self.sparse_sim.RY(q, theta)
            return
        self.dense_backend.RY(self.get_qubit_index(q), theta)

    def RZ(self, q: str, theta: float):
        if self.sim_type == 'mps':
            self.mps_sim.RZ(q, theta)
            return
        if self.is_sparse:
            self.sparse_sim.RZ(q, theta)
            return
        self.dense_backend.RZ(self.get_qubit_index(q), theta)

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
        self.dense_backend.CNOT(self.get_qubit_index(control), self.get_qubit_index(target))

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
        self.dense_backend.CZ(self.get_qubit_index(control), self.get_qubit_index(target))

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
        self.dense_backend.SWAP(self.get_qubit_index(q1), self.get_qubit_index(q2))

    def measure(self, q: str) -> int:
        if self.sim_type == 'mps':
            return self.mps_sim.measure(q)
        if self.is_sparse:
            return self.sparse_sim.measure(q)
        if self.gpu_engine:
            self.state_vector = self.gpu_engine.get_state()
            k = self.get_qubit_index(q)
            n = len(self.state_vector)
            k_mask = 1 << k
            p0 = sum(abs(amp)**2 for i, amp in enumerate(self.state_vector) if not (i & k_mask))
            r = self.rng.random()
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
        r = self.rng.random()
        return self.dense_backend.measure(k, r)

    def get_state_vector(self) -> list[complex]:
        if self.sim_type == 'mps':
            return self.mps_sim.get_state_vector()
        if self.is_sparse:
            return self.sparse_sim.get_state_vector()
        if self.gpu_engine:
            return self.gpu_engine.get_state()
        return self.dense_backend.get_state_vector()

    def get_amplitudes_dict(self) -> dict[str, complex]:
        if self.sim_type == 'mps':
            return self.mps_sim.get_amplitudes_dict()
        if self.is_sparse:
            return self.sparse_sim.get_amplitudes_dict()
            
        if self.num_qubits == 0:
            return {"": 1.0 + 0.0j}
        
        state_vec = self.gpu_engine.get_state() if self.gpu_engine else self.get_state_vector()
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
