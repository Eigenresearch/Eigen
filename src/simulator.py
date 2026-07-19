import math
import cmath
import random
import numpy as np
from collections import OrderedDict
from typing import TYPE_CHECKING
from src.sparse_simulator import SparseQuantumSimulator
from src.tensor_network.mps import MPSSimulator
from src.simulator_optimizations import GPUAccelerationSurface, optimize_measurement_order

if TYPE_CHECKING:
    from src.pulse_control import PulseSchedule

try:
    import eigen_native as native
except ImportError:
    native = None

# §1.3 — Global GPU acceleration surface (lazy: only initialized on first
# GPU-eligible gate so that ``import src.simulator`` stays side-effect free
# on machines without a GPU stack — see _get_gpu_accel()).
_gpu_accel: GPUAccelerationSurface | None = None


def _get_gpu_accel() -> GPUAccelerationSurface:
    global _gpu_accel
    if _gpu_accel is None:
        _gpu_accel = GPUAccelerationSurface()
    return _gpu_accel


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


_INDEX_CACHE_MAX_ENTRIES = 1024


class PythonDenseStatevector(StateBackend):
    def __init__(self):
        self._state = np.array([1.0 + 0.0j], dtype=complex)
        self._index_cache = OrderedDict()
        self._index_cache_2q = OrderedDict()
        self._index_cache_3q = OrderedDict()
        self._buf0 = None
        self._buf1 = None
        self.rng = random.Random()
        # §1.3 (perf) — GPU-resident state. When the CuPy path is active
        # (see apply_1qubit_gate), the authoritative state lives on the GPU
        # and ``self._state`` is stale until ``_sync_from_gpu()`` runs.
        # This avoids the old CPU→list→GPU→CPU round-trip per gate.
        self._gpu_state = None

    def _sync_from_gpu(self):
        """Download the GPU-resident state back to ``self._state`` once."""
        if self._gpu_state is not None:
            self._state = _get_gpu_accel().from_gpu(self._gpu_state)
            self._gpu_state = None

    def _sync_to_gpu(self):
        """Upload ``self._state`` to the GPU (no-op if already resident)."""
        if self._gpu_state is None:
            self._gpu_state = _get_gpu_accel().to_gpu(self._state)
        return self._gpu_state

    def _cache_get(self, cache, key):
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        return None

    def _cache_put(self, cache, key, value):
        cache[key] = value
        if len(cache) > _INDEX_CACHE_MAX_ENTRIES:
            cache.popitem(last=False)

    def _ensure_buffers(self, half: int):
        if self._buf0 is None or self._buf0.shape[0] < half:
            self._buf0 = np.empty(half, dtype=complex)
            self._buf1 = np.empty(half, dtype=complex)
        return self._buf0[:half], self._buf1[:half]

    def _get_indices(self, k: int):
        n = self._state.shape[0]
        cache_key = (n, k)
        cached = self._cache_get(self._index_cache, cache_key)
        if cached is not None:
            return cached
        
        # O(2^(n-1)) construction
        num_qubits = int(math.log2(n))
        i_low = np.arange(1 << k)
        i_high = np.arange(1 << (num_qubits - k - 1))
        idx0 = ((i_high[:, None] << (k + 1)) | i_low[None, :]).ravel()
        value = (idx0, idx0 + (1 << k))
        self._cache_put(self._index_cache, cache_key, value)
        return value

    def _get_indices_2q(self, q1: int, q2: int, val1: int, val2: int):
        n = self._state.shape[0]
        cache_key = (n, q1, q2, val1, val2)
        cached = self._cache_get(self._index_cache_2q, cache_key)
        if cached is not None:
            return cached
        
        # O(2^(n-2)) construction
        num_qubits = int(math.log2(n))
        qubits = sorted([q1, q2])
        q_low, q_high = qubits
        
        i_low = np.arange(1 << q_low)
        i_mid = np.arange(1 << (q_high - q_low - 1))
        i_high = np.arange(1 << (num_qubits - q_high - 1))
        
        idx = (i_high[:, None, None] << (q_high + 1)) | \
              (i_mid[None, :, None] << (q_low + 1)) | \
              i_low[None, None, :]
        
        idx = idx.ravel()
        if val1: idx |= (1 << q1)
        if val2: idx |= (1 << q2)
        
        self._cache_put(self._index_cache_2q, cache_key, idx)
        return idx

    def _get_indices_3q(self, q1: int, q2: int, q3: int, val1: int, val2: int, val3: int):
        n = self._state.shape[0]
        cache_key = (n, q1, q2, q3, val1, val2, val3)
        cached = self._cache_get(self._index_cache_3q, cache_key)
        if cached is not None:
            return cached
            
        # O(2^(n-3)) construction
        num_qubits = int(math.log2(n))
        qubits = sorted([q1, q2, q3])
        q_low, q_mid, q_high = qubits
        
        i0 = np.arange(1 << q_low)
        i1 = np.arange(1 << (q_mid - q_low - 1))
        i2 = np.arange(1 << (q_high - q_mid - 1))
        i3 = np.arange(1 << (num_qubits - q_high - 1))
        
        idx = (i3[:, None, None, None] << (q_high + 1)) | \
              (i2[None, :, None, None] << (q_mid + 1)) | \
              (i1[None, None, :, None] << (q_low + 1)) | \
              i0[None, None, None, :]
              
        idx = idx.ravel()
        if val1: idx |= (1 << q1)
        if val2: idx |= (1 << q2)
        if val3: idx |= (1 << q3)
        
        self._cache_put(self._index_cache_3q, cache_key, idx)
        return idx

    def allocate_qubit(self):
        self._sync_from_gpu()
        n = self._state.shape[0]
        if n >= (1 << 25):
            raise MemoryError("Dense simulation is limited to 25 qubits to prevent memory exhaustion.")
        self._state = np.concatenate([self._state, np.zeros_like(self._state)])
        self._index_cache.clear()
        self._index_cache_2q.clear()
        self._index_cache_3q.clear()
        self._buf0 = None
        self._buf1 = None

    def H(self, k: int):
        self._sync_from_gpu()
        inv_sqrt2 = 0.7071067811865475
        idx0, idx1 = self._get_indices(k)
        a0 = self._state[idx0]
        a1 = self._state[idx1]
        b0, b1 = self._ensure_buffers(a0.shape[0])
        np.add(a0, a1, out=b0)
        np.subtract(a0, a1, out=b1)
        b0 *= inv_sqrt2
        b1 *= inv_sqrt2
        self._state[idx0] = b0
        self._state[idx1] = b1

    def X(self, k: int):
        self._sync_from_gpu()
        idx0, idx1 = self._get_indices(k)
        # Optimized swap using pre-allocated buffers to avoid large temp copies
        buf0, _ = self._ensure_buffers(idx0.shape[0])
        np.copyto(buf0, self._state[idx0])
        self._state[idx0] = self._state[idx1]
        self._state[idx1] = buf0

    def Y(self, k: int):
        self._sync_from_gpu()
        idx0, idx1 = self._get_indices(k)
        a0 = self._state[idx0]
        a1 = self._state[idx1]
        self._state[idx0] = -1j * a1
        self._state[idx1] = 1j * a0

    def Z(self, k: int):
        self._sync_from_gpu()
        _, idx1 = self._get_indices(k)
        self._state[idx1] *= -1

    def S(self, k: int):
        self._sync_from_gpu()
        _, idx1 = self._get_indices(k)
        self._state[idx1] *= 1j

    def T(self, k: int):
        self._sync_from_gpu()
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
        # §1.3 — GPU acceleration: keep the state resident on the GPU and
        # contract there; no host round-trip per gate. The state is only
        # downloaded when a CPU-side path (2q gates, measure, inspection)
        # needs it — see _sync_from_gpu().
        gpu = _get_gpu_accel()
        n = int(np.log2(self._state.shape[0])) if self._gpu_state is None \
            else int(np.log2(self._gpu_state.shape[0]))
        if gpu.available and gpu.backend == "cupy" and n > 8:
            gstate = self._sync_to_gpu()
            self._gpu_state = gpu.apply_gate_resident(
                gstate, gate_matrix, k, n)
            return
        self._sync_from_gpu()
        u00, u01 = gate_matrix[0][0], gate_matrix[0][1]
        u10, u11 = gate_matrix[1][0], gate_matrix[1][1]
        idx0, idx1 = self._get_indices(k)
        a0 = self._state[idx0]
        a1 = self._state[idx1]
        b0, b1 = self._ensure_buffers(a0.shape[0])
        np.multiply(u00, a0, out=b0)
        np.multiply(u01, a1, out=b1)
        b0 += b1
        np.multiply(u10, a0, out=b1)
        b1 += u11 * a1
        self._state[idx0] = b0
        self._state[idx1] = b1

    def CNOT(self, control: int, target: int):
        self._sync_from_gpu()
        idx0 = self._get_indices_2q(control, target, 1, 0)
        idx1 = idx0 + (1 << target)
        self._state[idx0], self._state[idx1] = self._state[idx1], self._state[idx0]

    def CZ(self, control: int, target: int):
        self._sync_from_gpu()
        idx = self._get_indices_2q(control, target, 1, 1)
        self._state[idx] *= -1

    def SWAP(self, q1: int, q2: int):
        self._sync_from_gpu()
        idx0 = self._get_indices_2q(q1, q2, 1, 0)
        idx1 = (idx0 & ~(1 << q1)) | (1 << q2)
        self._state[idx0], self._state[idx1] = self._state[idx1], self._state[idx0]

    def CCX(self, control1: int, control2: int, target: int):
        self._sync_from_gpu()
        idx0 = self._get_indices_3q(control1, control2, target, 1, 1, 0)
        idx1 = idx0 + (1 << target)
        self._state[idx0], self._state[idx1] = self._state[idx1], self._state[idx0]

    def CSWAP(self, control: int, q1: int, q2: int):
        self._sync_from_gpu()
        idx0 = self._get_indices_3q(control, q1, q2, 1, 1, 0)
        idx1 = (idx0 & ~(1 << q1)) | (1 << q2)
        self._state[idx0], self._state[idx1] = self._state[idx1], self._state[idx0]

    def CP(self, control: int, target: int, theta: float):
        self._sync_from_gpu()
        idx = self._get_indices_2q(control, target, 1, 1)
        self._state[idx] *= cmath.exp(1j * theta)

    def CRX(self, control: int, target: int, theta: float):
        self._sync_from_gpu()
        cos_val = math.cos(theta / 2)
        sin_val = math.sin(theta / 2)
        idx0 = self._get_indices_2q(control, target, 1, 0)
        idx1 = idx0 + (1 << target)
        a0 = self._state[idx0]
        a1 = self._state[idx1]
        self._state[idx0] = cos_val * a0 - 1j * sin_val * a1
        self._state[idx1] = -1j * sin_val * a0 + cos_val * a1

    def CRY(self, control: int, target: int, theta: float):
        self._sync_from_gpu()
        cos_val = math.cos(theta / 2)
        sin_val = math.sin(theta / 2)
        idx0 = self._get_indices_2q(control, target, 1, 0)
        idx1 = idx0 + (1 << target)
        a0 = self._state[idx0]
        a1 = self._state[idx1]
        self._state[idx0] = cos_val * a0 - sin_val * a1
        self._state[idx1] = sin_val * a0 + cos_val * a1

    def CRZ(self, control: int, target: int, theta: float):
        self._sync_from_gpu()
        val_0 = cmath.exp(-1j * theta / 2)
        val_1 = cmath.exp(1j * theta / 2)
        idx0 = self._get_indices_2q(control, target, 1, 0)
        idx1 = self._get_indices_2q(control, target, 1, 1)
        self._state[idx0] *= val_0
        self._state[idx1] *= val_1

    def measure(self, k: int, r: float) -> int:
        # §1.3 — Optimized measurement: uses cached indices for
        # efficient probability computation and in-place collapse.
        self._sync_from_gpu()
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

    def measure_multiple(self, qubits: list[int],
                          entanglement_graph: dict[int, set[int]] | None = None
                          ) -> list[int]:
        """Measure multiple qubits in optimized order.

        §1.3: "Оптимизировать пути измерения"
        Uses optimize_measurement_order to measure least-entangled
        qubits first, minimizing state collapse cascading.
        """
        if entanglement_graph is not None:
            ordered = optimize_measurement_order(
                qubits, entanglement_graph)
        else:
            ordered = list(qubits)
        results = []
        for q in ordered:
            r = self.rng.random()
            results.append(self.measure(q, r))
        return results

    def get_state_vector(self) -> list[complex]:
        self._sync_from_gpu()
        return list(self._state)

    def set_state_vector(self, value: list[complex]):
        self._gpu_state = None
        self._state = np.array(value, dtype=complex)


# Gates the stabilizer backend can execute natively (Clifford group).
# Everything else triggers the auto-fallback to the dense backend.
_CLIFFORD_GATES = frozenset({'H', 'X', 'Y', 'Z', 'S', 'CNOT', 'CZ', 'SWAP'})


def _validate_angle(theta: float, gate: str):
    """§6 (correctness): NaN/Inf rotation angles silently corrupt the whole
    state vector (cos(nan) = nan propagates everywhere). Reject early."""
    if not math.isfinite(theta):
        raise ValueError(f"{gate} angle must be finite, got {theta!r}")


# --- Parametrized gate matrices (GPU engine path) ------------------------
def _rx_matrix(theta):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return [[c, -1j * s], [-1j * s, c]]


def _ry_matrix(theta):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return [[c, -s], [s, c]]


def _rz_matrix(theta):
    return [[cmath.exp(-1j * theta / 2), 0.0j],
            [0.0j, cmath.exp(1j * theta / 2)]]


def _cp_matrix(theta):
    return [[1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, cmath.exp(1j * theta)]]


def _crx_matrix(theta):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return [[1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, c, -1j * s],
            [0.0, 0.0, -1j * s, c]]


def _cry_matrix(theta):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return [[1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, c, -s],
            [0.0, 0.0, s, c]]


def _crz_matrix(theta):
    return [[1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, cmath.exp(-1j * theta / 2), 0.0],
            [0.0, 0.0, 0.0, cmath.exp(1j * theta / 2)]]


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
        
        # §1.3: Lazy GPU initialization — only load backend if needed
        self._gpu_engine_lazy = None

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
            if hasattr(self.dense_backend, 'rng'):
                self.dense_backend.rng = self.rng
        elif self.sim_type == 'auto':
            if native is not None and hasattr(native, 'RustStatevector'):
                self.dense_backend = RustStatevectorWrapper()
            else:
                self.dense_backend = PythonDenseStatevector()
            if hasattr(self.dense_backend, 'rng'):
                self.dense_backend.rng = self.rng

    @property
    def gpu_engine(self):
        if self._gpu_engine_lazy is None and self.gpu_platform != 'none':
            from src.backend.gpu.gpu_engine import GPUEngine
            engine = GPUEngine(self.gpu_platform)
            if engine.platform == 'none':
                self.gpu_platform = 'none'
            else:
                self._gpu_engine_lazy = engine
        return self._gpu_engine_lazy

    @gpu_engine.setter
    def gpu_engine(self, value):
        self._gpu_engine_lazy = value

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
                     [0.0, 0.0, 0.0, 1.0]],
            'CCX': [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]],
            # CSWAP(c, q1, q2): control is the MSB of the 3-qubit block in
            # GPUEngine's convention; swaps |101> <-> |110>. (The pre-2.8
            # inline GPU matrix duplicated CCX — a latent GPU-path bug.)
            'CSWAP': [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
        }

    @property
    def state_vector(self) -> list[complex]:
        # §2 (perf): returns a plain copy. The old StateVectorList wrapper
        # re-uploaded the whole 2^n vector to the backend on EVERY element
        # assignment — an O(2^n) trap. Mutate via the property setter or
        # set_state_vector() instead.
        if self.is_sparse or self.sim_type in ('sparse', 'mps'):
            return None
        if self.dense_backend:
            return list(self.dense_backend.get_state_vector())
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
        # §1.3: Correct fallback — preserves the quantum state
        old_state = self.stabilizer_sim.get_state_vector()
        self.stabilizer_sim = None
        self.sim_type = 'dense'
        if native is not None and hasattr(native, 'RustStatevector'):
            self.dense_backend = RustStatevectorWrapper()
        else:
            self.dense_backend = PythonDenseStatevector()
        for _ in range(self.num_qubits):
            self.dense_backend.allocate_qubit()
        self.dense_backend.set_state_vector(old_state)

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

    def _dispatch_gate(self, gate: str, names: tuple, angles: tuple = (),
                       gpu_matrix=None):
        """Unified gate dispatch — the single source of truth for backend
        priority (audit §2 god-class fix: replaces ~114 hand-copied branch
        ladders whose per-gate ordering was inconsistent).

        Priority: stabilizer (Clifford only; anything else auto-falls back
        to dense) → mps → density → sparse → gpu → dense. At most one
        backend is active at any moment; the chain exists because *which*
        one is active changes at runtime (auto sparse switch at >20
        qubits, stabilizer fallback, configure_backend).
        """
        if self.stabilizer_sim:
            if gate in _CLIFFORD_GATES:
                return getattr(self.stabilizer_sim, gate)(*names, *angles)
            self._fallback_from_stabilizer(gate)
            # fall through: stabilizer_sim is now None, dense is active.
        if self.mps_sim:
            return getattr(self.mps_sim, gate)(*names, *angles)
        if self.density_sim:
            return getattr(self.density_sim, gate)(*names, *angles)
        if self.is_sparse:
            return getattr(self.sparse_sim, gate)(*names, *angles)
        if self.gpu_engine:
            indices = [self.get_qubit_index(n) for n in names]
            return self.gpu_engine.apply_gate(indices, gpu_matrix)
        return getattr(self.dense_backend, gate)(
            *(self.get_qubit_index(n) for n in names), *angles)

    def apply_1qubit_gate(self, name: str, gate_matrix: list[list[complex]]):
        if self.stabilizer_sim:
            # An arbitrary 2x2 unitary is generally non-Clifford.
            self._fallback_from_stabilizer('U')
        if self.mps_sim:
            self.mps_sim.apply_1qubit_gate(name, gate_matrix)
            return
        if self.density_sim:
            self.density_sim.apply_1qubit_gate(name, gate_matrix)
            return
        if self.is_sparse:
            self.sparse_sim.apply_1qubit_gate(name, gate_matrix)
            return
        if self.gpu_engine:
            self.gpu_engine.apply_gate([self.get_qubit_index(name)], gate_matrix)
            return
        self.dense_backend.apply_1qubit_gate(self.get_qubit_index(name), gate_matrix)

    def H(self, q: str):
        self._dispatch_gate('H', (q,), gpu_matrix=self.gate_cache['H'])

    def X(self, q: str):
        self._dispatch_gate('X', (q,), gpu_matrix=self.gate_cache['X'])

    def Y(self, q: str):
        self._dispatch_gate('Y', (q,), gpu_matrix=self.gate_cache['Y'])

    def Z(self, q: str):
        self._dispatch_gate('Z', (q,), gpu_matrix=self.gate_cache['Z'])

    def S(self, q: str):
        self._dispatch_gate('S', (q,), gpu_matrix=self.gate_cache['S'])

    def T(self, q: str):
        self._dispatch_gate('T', (q,), gpu_matrix=self.gate_cache['T'])

    def RX(self, q: str, theta: float):
        _validate_angle(theta, 'RX')
        self._dispatch_gate('RX', (q,), (theta,), _rx_matrix(theta))

    def RY(self, q: str, theta: float):
        _validate_angle(theta, 'RY')
        self._dispatch_gate('RY', (q,), (theta,), _ry_matrix(theta))

    def RZ(self, q: str, theta: float):
        _validate_angle(theta, 'RZ')
        self._dispatch_gate('RZ', (q,), (theta,), _rz_matrix(theta))

    def CNOT(self, control: str, target: str):
        self._dispatch_gate('CNOT', (control, target),
                            gpu_matrix=self.gate_cache['CNOT'])

    def CZ(self, control: str, target: str):
        self._dispatch_gate('CZ', (control, target),
                            gpu_matrix=self.gate_cache['CZ'])

    def SWAP(self, q1: str, q2: str):
        self._dispatch_gate('SWAP', (q1, q2),
                            gpu_matrix=self.gate_cache['SWAP'])

    def CCX(self, control1: str, control2: str, target: str):
        self._dispatch_gate('CCX', (control1, control2, target),
                            gpu_matrix=self.gate_cache['CCX'])

    def CSWAP(self, control: str, q1: str, q2: str):
        self._dispatch_gate('CSWAP', (control, q1, q2),
                            gpu_matrix=self.gate_cache['CSWAP'])

    def CP(self, control: str, target: str, theta: float):
        _validate_angle(theta, 'CP')
        self._dispatch_gate('CP', (control, target), (theta,),
                            _cp_matrix(theta))

    def CRX(self, control: str, target: str, theta: float):
        _validate_angle(theta, 'CRX')
        self._dispatch_gate('CRX', (control, target), (theta,),
                            _crx_matrix(theta))

    def CRY(self, control: str, target: str, theta: float):
        _validate_angle(theta, 'CRY')
        self._dispatch_gate('CRY', (control, target), (theta,),
                            _cry_matrix(theta))

    def CRZ(self, control: str, target: str, theta: float):
        _validate_angle(theta, 'CRZ')
        self._dispatch_gate('CRZ', (control, target), (theta,),
                            _crz_matrix(theta))

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
            # §1.3: GPU-resident measurement (vectorized, no CPU round-trip)
            k = self.get_qubit_index(q)
            r = self.rng.random()
            return self.gpu_engine.measure(k, r)

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

    # === §3.2 — Pulse-level control integration =========================
    def get_pulse_schedule(self, gate_sequence: list[tuple[str, list[str]]]
                            ) -> 'PulseSchedule':
        """Convert a gate sequence to a pulse-level schedule.

        Maps each gate to its canonical pulse shape using
        `gate_to_pulse()` from `src.pulse_control`. When a gate
        has no pulse mapping, it is silently skipped (the gate
        abstraction still applies via the normal simulator path).
        """
        from src.pulse_control import PulseSchedule, gate_to_pulse
        sched = PulseSchedule()
        t = 0.0
        for gate_name, targets in gate_sequence:
            pulse = gate_to_pulse(gate_name)
            if pulse is not None:
                channel = targets[0] if targets else "d0"
                sched.add(channel, pulse, start_time_ns=t)
                t += pulse.duration_ns
        return sched
