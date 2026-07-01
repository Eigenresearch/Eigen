import math
import cmath
import random
import numpy as np
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

    def CCX(self, control1: int, control2: int, target: int):
        raise NotImplementedError

    def CSWAP(self, control: int, q1: int, q2: int):
        raise NotImplementedError

    def CP(self, control: int, target: int, theta: float):
        raise NotImplementedError

    def CRX(self, control: int, target: int, theta: float):
        raise NotImplementedError

    def CRY(self, control: int, target: int, theta: float):
        raise NotImplementedError

    def CRZ(self, control: int, target: int, theta: float):
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

    def CCX(self, control1: int, control2: int, target: int):
        self.rust_sv.apply_ccx(control1, control2, target)

    def CSWAP(self, control: int, q1: int, q2: int):
        self.rust_sv.apply_cswap(control, q1, q2)

    def CP(self, control: int, target: int, theta: float):
        self.rust_sv.apply_cp(control, target, float(theta))

    def CRX(self, control: int, target: int, theta: float):
        self.rust_sv.apply_crx(control, target, float(theta))

    def CRY(self, control: int, target: int, theta: float):
        self.rust_sv.apply_cry(control, target, float(theta))

    def CRZ(self, control: int, target: int, theta: float):
        self.rust_sv.apply_crz(control, target, float(theta))

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
        self._state = np.array([1.0 + 0.0j], dtype=complex)
        self._index_cache = {}
        self._index_cache_2q = {}
        self._index_cache_3q = {}

    def _get_indices(self, k: int):
        n = self._state.shape[0]
        cache_key = (n, k)
        if cache_key not in self._index_cache:
            i_low = np.arange(1 << k)
            i_high = np.arange(n >> (k + 1))
            idx0 = ((i_high[:, None] << (k + 1)) | i_low[None, :]).ravel()
            self._index_cache[cache_key] = (idx0, idx0 + (1 << k))
        return self._index_cache[cache_key]

    def _get_indices_2q(self, q1: int, q2: int, val1: int, val2: int):
        n = self._state.shape[0]
        cache_key = (n, q1, q2, val1, val2)
        if cache_key not in self._index_cache_2q:
            idx = np.arange(n)
            idx = idx[((idx & (1 << q1)) != 0) if val1 else ((idx & (1 << q1)) == 0)]
            idx = idx[((idx & (1 << q2)) != 0) if val2 else ((idx & (1 << q2)) == 0)]
            self._index_cache_2q[cache_key] = idx
        return self._index_cache_2q[cache_key]

    def _get_indices_3q(self, q1: int, q2: int, q3: int, val1: int, val2: int, val3: int):
        n = self._state.shape[0]
        cache_key = (n, q1, q2, q3, val1, val2, val3)
        if cache_key not in self._index_cache_3q:
            idx = np.arange(n)
            idx = idx[((idx & (1 << q1)) != 0) if val1 else ((idx & (1 << q1)) == 0)]
            idx = idx[((idx & (1 << q2)) != 0) if val2 else ((idx & (1 << q2)) == 0)]
            idx = idx[((idx & (1 << q3)) != 0) if val3 else ((idx & (1 << q3)) == 0)]
            self._index_cache_3q[cache_key] = idx
        return self._index_cache_3q[cache_key]

    def allocate_qubit(self):
        n = self._state.shape[0]
        if n >= (1 << 25):
            raise MemoryError("Dense simulation is limited to 25 qubits to prevent memory exhaustion.")
        self._state = np.concatenate([self._state, np.zeros_like(self._state)])
        self._index_cache.clear()
        self._index_cache_2q.clear()
        self._index_cache_3q.clear()

    def H(self, k: int):
        inv_sqrt2 = 0.7071067811865475
        idx0, idx1 = self._get_indices(k)
        a0 = self._state[idx0]
        a1 = self._state[idx1]
        self._state[idx0] = (a0 + a1) * inv_sqrt2
        self._state[idx1] = (a0 - a1) * inv_sqrt2

    def X(self, k: int):
        idx0, idx1 = self._get_indices(k)
        self._state[idx0], self._state[idx1] = self._state[idx1].copy(), self._state[idx0].copy()

    def Y(self, k: int):
        idx0, idx1 = self._get_indices(k)
        a0 = self._state[idx0]
        a1 = self._state[idx1]
        self._state[idx0] = -1j * a1
        self._state[idx1] = 1j * a0

    def Z(self, k: int):
        _, idx1 = self._get_indices(k)
        self._state[idx1] *= -1

    def S(self, k: int):
        _, idx1 = self._get_indices(k)
        self._state[idx1] *= 1j

    def T(self, k: int):
        _, idx1 = self._get_indices(k)
        self._state[idx1] *= (0.7071067811865475 + 0.7071067811865475j)

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
        u00, u01 = gate_matrix[0][0], gate_matrix[0][1]
        u10, u11 = gate_matrix[1][0], gate_matrix[1][1]
        idx0, idx1 = self._get_indices(k)
        a0 = self._state[idx0]
        a1 = self._state[idx1]
        self._state[idx0] = u00 * a0 + u01 * a1
        self._state[idx1] = u10 * a0 + u11 * a1

    def CNOT(self, control: int, target: int):
        idx0 = self._get_indices_2q(control, target, 1, 0)
        idx1 = idx0 + (1 << target)
        self._state[idx0], self._state[idx1] = self._state[idx1].copy(), self._state[idx0].copy()

    def CZ(self, control: int, target: int):
        idx = self._get_indices_2q(control, target, 1, 1)
        self._state[idx] *= -1

    def SWAP(self, q1: int, q2: int):
        idx0 = self._get_indices_2q(q1, q2, 1, 0)
        idx1 = (idx0 & ~(1 << q1)) | (1 << q2)
        self._state[idx0], self._state[idx1] = self._state[idx1].copy(), self._state[idx0].copy()

    def CCX(self, control1: int, control2: int, target: int):
        idx0 = self._get_indices_3q(control1, control2, target, 1, 1, 0)
        idx1 = idx0 + (1 << target)
        self._state[idx0], self._state[idx1] = self._state[idx1].copy(), self._state[idx0].copy()

    def CSWAP(self, control: int, q1: int, q2: int):
        idx0 = self._get_indices_3q(control, q1, q2, 1, 1, 0)
        idx1 = (idx0 & ~(1 << q1)) | (1 << q2)
        self._state[idx0], self._state[idx1] = self._state[idx1].copy(), self._state[idx0].copy()

    def CP(self, control: int, target: int, theta: float):
        idx = self._get_indices_2q(control, target, 1, 1)
        self._state[idx] *= cmath.exp(1j * theta)

    def CRX(self, control: int, target: int, theta: float):
        cos_val = math.cos(theta / 2)
        sin_val = math.sin(theta / 2)
        idx0 = self._get_indices_2q(control, target, 1, 0)
        idx1 = idx0 + (1 << target)
        a0 = self._state[idx0]
        a1 = self._state[idx1]
        self._state[idx0] = cos_val * a0 - 1j * sin_val * a1
        self._state[idx1] = -1j * sin_val * a0 + cos_val * a1

    def CRY(self, control: int, target: int, theta: float):
        cos_val = math.cos(theta / 2)
        sin_val = math.sin(theta / 2)
        idx0 = self._get_indices_2q(control, target, 1, 0)
        idx1 = idx0 + (1 << target)
        a0 = self._state[idx0]
        a1 = self._state[idx1]
        self._state[idx0] = cos_val * a0 - sin_val * a1
        self._state[idx1] = sin_val * a0 + cos_val * a1

    def CRZ(self, control: int, target: int, theta: float):
        val_0 = cmath.exp(-1j * theta / 2)
        val_1 = cmath.exp(1j * theta / 2)
        idx0 = self._get_indices_2q(control, target, 1, 0)
        idx1 = self._get_indices_2q(control, target, 1, 1)
        self._state[idx0] *= val_0
        self._state[idx1] *= val_1

    def measure(self, k: int, r: float) -> int:
        idx0, idx1 = self._get_indices(k)
        p0 = np.sum(np.abs(self._state[idx0])**2)
        if r < p0:
            norm = math.sqrt(p0) if p0 > 1e-15 else 1.0
            self._state[idx0] /= norm
            self._state[idx1] = 0.0
            return 0
        else:
            p1 = 1.0 - p0
            norm = math.sqrt(p1) if p1 > 1e-15 else 1.0
            self._state[idx1] /= norm
            self._state[idx0] = 0.0
            return 1

    def get_state_vector(self) -> list[complex]:
        return list(self._state)

    def set_state_vector(self, value: list[complex]):
        self._state = np.array(value, dtype=complex)


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
    def __init__(self, sim_type='dense', gpu_platform='none', seed=None,
                 mps_max_bond_dim=None, mps_auto_bond_dim=False, mps_max_truncation_error=None):
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
        self.density_sim = None
        self.stabilizer_sim = None

        # MPS configuration
        self.mps_max_bond_dim = mps_max_bond_dim
        self.mps_auto_bond_dim = mps_auto_bond_dim
        self.mps_max_truncation_error = mps_max_truncation_error
        
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
            kwargs = {}
            if self.mps_max_bond_dim is not None:
                kwargs['max_bond_dim'] = self.mps_max_bond_dim
            kwargs['auto_bond_dim'] = self.mps_auto_bond_dim
            if self.mps_max_truncation_error is not None:
                kwargs['max_truncation_error'] = self.mps_max_truncation_error
            self.mps_sim = MPSSimulator(**kwargs)
            self.mps_sim.rng = self.rng
        elif self.sim_type == 'stabilizer':
            from src.stabilizer_simulator import StabilizerSimulator
            self.stabilizer_sim = StabilizerSimulator(rng=self.rng)
        elif self.sim_type == 'density_matrix':
            from src.density_matrix_simulator import DensityMatrixSimulator
            self.density_sim = DensityMatrixSimulator(rng=self.rng)
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
            kwargs = {}
            if self.mps_max_bond_dim is not None:
                kwargs['max_bond_dim'] = self.mps_max_bond_dim
            kwargs['auto_bond_dim'] = self.mps_auto_bond_dim
            if self.mps_max_truncation_error is not None:
                kwargs['max_truncation_error'] = self.mps_max_truncation_error
            self.mps_sim = MPSSimulator(**kwargs)
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

    def _fallback_from_stabilizer(self, gate_name: str = None):
        """Auto-fallback from stabilizer to dense state-vector simulator.

        Called when a non-Clifford gate is applied to a stabilizer backend.
        Switches to the dense backend so the circuit can run without crashing.
        """
        if not self.stabilizer_sim:
            return
        import warnings
        gate_info = f" (gate: {gate_name})" if gate_name else ""
        warnings.warn(
            f"Non-Clifford gate{gate_info} detected on stabilizer backend. "
            f"Auto-falling back to dense state-vector simulator. "
            f"Stabilizer simulator only supports Clifford gates: "
            f"{{H, S, X, Y, Z, CNOT, CZ, SWAP}}.",
            stacklevel=3,
        )
        self.stabilizer_sim = None
        self.sim_type = 'dense'
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
        
        if self.stabilizer_sim:
            self.stabilizer_sim.allocate_qubit(name)
            return

        if self.mps_sim:
            self.mps_sim.allocate_qubit(name)
            return
            
        if self.density_sim:
            self.density_sim.allocate_qubit(name)
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
            
            if self.gpu_engine:
                dense_state = self.gpu_engine.get_state()
            else:
                dense_state = self.dense_backend.get_state_vector()
            for i, amp in enumerate(dense_state):
                if abs(amp) > 1e-12:
                    key = ""
                    for q in sorted_qubits:
                        q_idx = self.qubit_map[q]
                        key += '1' if (i & (1 << q_idx)) else '0'
                    self.sparse_sim.state[key] = amp
            self.dense_backend = None
            self.gpu_engine = None
        else:
            self.dense_backend.allocate_qubit()

    def get_qubit_index(self, name: str) -> int:
        if name not in self.qubit_map:
            raise KeyError(f"Qubit '{name}' is not allocated in the simulator")
        return self.qubit_map[name]

    def apply_1qubit_gate(self, name: str, gate_matrix: list[list[complex]]):
        if self.mps_sim:
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
        if self.stabilizer_sim:
            self.stabilizer_sim.H(q)
            return
        if self.is_sparse:
            self.sparse_sim.H(q)
            return
        if self.gpu_engine:
            self.gpu_engine.apply_gate([self.get_qubit_index(q)], self.gate_cache['H'])
            return
        if self.mps_sim:
            self.mps_sim.H(q)
            return
        if self.density_sim:
            self.density_sim.H(q)
            return
        self.dense_backend.H(self.get_qubit_index(q))

    def X(self, q: str):
        if self.stabilizer_sim:
            self.stabilizer_sim.X(q)
            return
        if self.is_sparse:
            self.sparse_sim.X(q)
            return
        if self.gpu_engine:
            self.gpu_engine.apply_gate([self.get_qubit_index(q)], self.gate_cache['X'])
            return
        if self.mps_sim:
            self.mps_sim.X(q)
            return
        if self.density_sim:
            self.density_sim.X(q)
            return
        self.dense_backend.X(self.get_qubit_index(q))

    def Y(self, q: str):
        if self.mps_sim:
            self.mps_sim.Y(q)
            return
        if self.density_sim:
            self.density_sim.Y(q)
            return
        if self.stabilizer_sim:
            self.stabilizer_sim.Y(q)
            return
        if self.is_sparse:
            self.sparse_sim.Y(q)
            return
        if self.gpu_engine:
            self.gpu_engine.apply_gate([self.get_qubit_index(q)], self.gate_cache['Y'])
            return
        self.dense_backend.Y(self.get_qubit_index(q))

    def Z(self, q: str):
        if self.mps_sim:
            self.mps_sim.Z(q)
            return
        if self.density_sim:
            self.density_sim.Z(q)
            return
        if self.stabilizer_sim:
            self.stabilizer_sim.Z(q)
            return
        if self.is_sparse:
            self.sparse_sim.Z(q)
            return
        if self.gpu_engine:
            self.gpu_engine.apply_gate([self.get_qubit_index(q)], self.gate_cache['Z'])
            return
        self.dense_backend.Z(self.get_qubit_index(q))

    def S(self, q: str):
        if self.mps_sim:
            self.mps_sim.S(q)
            return
        if self.density_sim:
            self.density_sim.S(q)
            return
        if self.stabilizer_sim:
            self.stabilizer_sim.S(q)
            return
        if self.is_sparse:
            self.sparse_sim.S(q)
            return
        if self.gpu_engine:
            self.gpu_engine.apply_gate([self.get_qubit_index(q)], self.gate_cache['S'])
            return
        self.dense_backend.S(self.get_qubit_index(q))

    def T(self, q: str):
        if self.stabilizer_sim:
            self._fallback_from_stabilizer('T')
        if self.mps_sim:
            self.mps_sim.T(q)
            return
        if self.density_sim:
            self.density_sim.T(q)
            return
        if self.is_sparse:
            self.sparse_sim.T(q)
            return
        if self.gpu_engine:
            self.gpu_engine.apply_gate([self.get_qubit_index(q)], self.gate_cache['T'])
            return
        self.dense_backend.T(self.get_qubit_index(q))

    def RX(self, q: str, theta: float):
        if self.stabilizer_sim:
            self._fallback_from_stabilizer('RX')
        if self.mps_sim:
            self.mps_sim.RX(q, theta)
            return
        if self.density_sim:
            self.density_sim.RX(q, theta)
            return
        if self.is_sparse:
            self.sparse_sim.RX(q, theta)
            return
        if self.gpu_engine:
            cos_val = math.cos(theta / 2)
            sin_val = math.sin(theta / 2)
            u = [[cos_val, -1j * sin_val], [-1j * sin_val, cos_val]]
            self.gpu_engine.apply_gate([self.get_qubit_index(q)], u)
            return
        self.dense_backend.RX(self.get_qubit_index(q), theta)

    def RY(self, q: str, theta: float):
        if self.stabilizer_sim:
            self._fallback_from_stabilizer('RY')
        if self.mps_sim:
            self.mps_sim.RY(q, theta)
            return
        if self.density_sim:
            self.density_sim.RY(q, theta)
            return
        if self.is_sparse:
            self.sparse_sim.RY(q, theta)
            return
        if self.gpu_engine:
            cos_val = math.cos(theta / 2)
            sin_val = math.sin(theta / 2)
            u = [[cos_val, -sin_val], [sin_val, cos_val]]
            self.gpu_engine.apply_gate([self.get_qubit_index(q)], u)
            return
        self.dense_backend.RY(self.get_qubit_index(q), theta)

    def RZ(self, q: str, theta: float):
        if self.stabilizer_sim:
            self._fallback_from_stabilizer('RZ')
        if self.mps_sim:
            self.mps_sim.RZ(q, theta)
            return
        if self.density_sim:
            self.density_sim.RZ(q, theta)
            return
        if self.is_sparse:
            self.sparse_sim.RZ(q, theta)
            return
        if self.gpu_engine:
            u = [[cmath.exp(-1j * theta / 2), 0.0j], [0.0j, cmath.exp(1j * theta / 2)]]
            self.gpu_engine.apply_gate([self.get_qubit_index(q)], u)
            return
        self.dense_backend.RZ(self.get_qubit_index(q), theta)

    def CNOT(self, control: str, target: str):
        if self.stabilizer_sim:
            self.stabilizer_sim.CNOT(control, target)
            return
        if self.mps_sim:
            self.mps_sim.CNOT(control, target)
            return
        if self.density_sim:
            self.density_sim.CNOT(control, target)
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
        if self.mps_sim:
            self.mps_sim.CZ(control, target)
            return
        if self.density_sim:
            self.density_sim.CZ(control, target)
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
        if self.mps_sim:
            self.mps_sim.SWAP(q1, q2)
            return
        if self.density_sim:
            self.density_sim.SWAP(q1, q2)
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

    def CCX(self, control1: str, control2: str, target: str):
        if self.stabilizer_sim:
            self._fallback_from_stabilizer('CCX')
        if self.mps_sim:
            self.mps_sim.CCX(control1, control2, target)
            return
        if self.density_sim:
            self.density_sim.CCX(control1, control2, target)
            return
        if self.is_sparse:
            self.sparse_sim.CCX(control1, control2, target)
            return
        if self.gpu_engine:
            c1 = self.get_qubit_index(control1)
            c2 = self.get_qubit_index(control2)
            t = self.get_qubit_index(target)
            u = [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            ]
            self.gpu_engine.apply_gate([c1, c2, t], u)
            return
        self.dense_backend.CCX(self.get_qubit_index(control1), self.get_qubit_index(control2), self.get_qubit_index(target))

    def CSWAP(self, control: str, q1: str, q2: str):
        if self.stabilizer_sim:
            self._fallback_from_stabilizer('CSWAP')
        if self.mps_sim:
            self.mps_sim.CSWAP(control, q1, q2)
            return
        if self.density_sim:
            self.density_sim.CSWAP(control, q1, q2)
            return
        if self.is_sparse:
            self.sparse_sim.CSWAP(control, q1, q2)
            return
        if self.gpu_engine:
            c = self.get_qubit_index(control)
            q1_idx = self.get_qubit_index(q1)
            q2_idx = self.get_qubit_index(q2)
            u = [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            ]
            self.gpu_engine.apply_gate([c, q1_idx, q2_idx], u)
            return
        self.dense_backend.CSWAP(self.get_qubit_index(control), self.get_qubit_index(q1), self.get_qubit_index(q2))

    def CP(self, control: str, target: str, theta: float):
        if self.stabilizer_sim:
            self._fallback_from_stabilizer('CP')
        if self.mps_sim:
            self.mps_sim.CP(control, target, theta)
            return
        if self.density_sim:
            self.density_sim.CP(control, target, theta)
            return
        if self.is_sparse:
            self.sparse_sim.CP(control, target, theta)
            return
        if self.gpu_engine:
            c = self.get_qubit_index(control)
            t = self.get_qubit_index(target)
            u = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, cmath.exp(1j * theta)]
            ]
            self.gpu_engine.apply_gate([c, t], u)
            return
        self.dense_backend.CP(self.get_qubit_index(control), self.get_qubit_index(target), theta)

    def CRX(self, control: str, target: str, theta: float):
        if self.stabilizer_sim:
            self._fallback_from_stabilizer('CRX')
        if self.mps_sim:
            self.mps_sim.CRX(control, target, theta)
            return
        if self.density_sim:
            self.density_sim.CRX(control, target, theta)
            return
        if self.is_sparse:
            self.sparse_sim.CRX(control, target, theta)
            return
        if self.gpu_engine:
            c = self.get_qubit_index(control)
            t = self.get_qubit_index(target)
            cos_val = math.cos(theta / 2)
            sin_val = math.sin(theta / 2)
            u = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, cos_val, -1j * sin_val],
                [0.0, 0.0, -1j * sin_val, cos_val]
            ]
            self.gpu_engine.apply_gate([c, t], u)
            return
        self.dense_backend.CRX(self.get_qubit_index(control), self.get_qubit_index(target), theta)

    def CRY(self, control: str, target: str, theta: float):
        if self.stabilizer_sim:
            self._fallback_from_stabilizer('CRY')
        if self.mps_sim:
            self.mps_sim.CRY(control, target, theta)
            return
        if self.density_sim:
            self.density_sim.CRY(control, target, theta)
            return
        if self.is_sparse:
            self.sparse_sim.CRY(control, target, theta)
            return
        if self.gpu_engine:
            c = self.get_qubit_index(control)
            t = self.get_qubit_index(target)
            cos_val = math.cos(theta / 2)
            sin_val = math.sin(theta / 2)
            u = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, cos_val, -sin_val],
                [0.0, 0.0, sin_val, cos_val]
            ]
            self.gpu_engine.apply_gate([c, t], u)
            return
        self.dense_backend.CRY(self.get_qubit_index(control), self.get_qubit_index(target), theta)

    def CRZ(self, control: str, target: str, theta: float):
        if self.stabilizer_sim:
            self._fallback_from_stabilizer('CRZ')
        if self.mps_sim:
            self.mps_sim.CRZ(control, target, theta)
            return
        if self.density_sim:
            self.density_sim.CRZ(control, target, theta)
            return
        if self.is_sparse:
            self.sparse_sim.CRZ(control, target, theta)
            return
        if self.gpu_engine:
            c = self.get_qubit_index(control)
            t = self.get_qubit_index(target)
            val_0 = cmath.exp(-1j * theta / 2)
            val_1 = cmath.exp(1j * theta / 2)
            u = [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, val_0, 0.0],
                [0.0, 0.0, 0.0, val_1]
            ]
            self.gpu_engine.apply_gate([c, t], u)
            return
        self.dense_backend.CRZ(self.get_qubit_index(control), self.get_qubit_index(target), theta)

    def measure(self, q: str) -> int:
        if self.stabilizer_sim:
            return self.stabilizer_sim.measure(q)
        if self.mps_sim:
            return self.mps_sim.measure(q)
        if self.density_sim:
            return self.density_sim.measure(q)
        if self.is_sparse:
            return self.sparse_sim.measure(q)
        if self.gpu_engine:
            state = self.gpu_engine.get_state()
            k = self.get_qubit_index(q)
            n = len(state)
            k_mask = 1 << k
            p0 = sum(abs(amp)**2 for i, amp in enumerate(state) if not (i & k_mask))
            r = self.rng.random()
            if r < p0:
                norm = math.sqrt(p0) if p0 > 1e-15 else 1.0
                for i in range(n):
                    if i & k_mask:
                        state[i] = 0.0j
                    else:
                        state[i] /= norm
                self.gpu_engine.set_state(state)
                self._state_vector = state
                return 0
            else:
                p1 = 1.0 - p0
                norm = math.sqrt(p1) if p1 > 1e-15 else 1.0
                for i in range(n):
                    if not (i & k_mask):
                        state[i] = 0.0j
                    else:
                        state[i] /= norm
                self.gpu_engine.set_state(state)
                self._state_vector = state
                return 1
            
        k = self.get_qubit_index(q)
        r = self.rng.random()
        return self.dense_backend.measure(k, r)

    def get_state_vector(self) -> list[complex]:
        if self.sim_type == 'mps':
            return self.mps_sim.get_state_vector()
        if self.sim_type == 'density_matrix':
            return self.density_sim.get_state_vector()
        if self.is_sparse:
            return self.sparse_sim.get_state_vector()
        if self.gpu_engine:
            return self.gpu_engine.get_state()
        return self.dense_backend.get_state_vector()

    def get_amplitudes_dict(self) -> dict[str, complex]:
        if self.sim_type == 'mps':
            return self.mps_sim.get_amplitudes_dict()
        if self.sim_type == 'density_matrix':
            return self.density_sim.get_amplitudes_dict()
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
