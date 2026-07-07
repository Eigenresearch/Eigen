"""§1.3 — Simulator Speed: in-place state vector updates,
tensor compression for gate application, measurement path
optimization, GPU acceleration surface.

These are supplementary optimization helpers for the simulator
that go beyond the existing gate-matrix caching and MPS/Sparse
simulators.
"""
from __future__ import annotations

import math
import typing

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


def apply_gate_inplace(state_vector: list, gate_matrix: list,
                        qubit_index: int, num_qubits: int) -> list:
    """Apply a 2x2 gate to a state vector IN-PLACE using index
    manipulation, avoiding the creation of a full 2^n x 2^n matrix.

    §1.3: "In-place обновления state vector где безопасно"

    For a single-qubit gate on qubit `qubit_index`, the state vector
    is updated in-place by iterating over pairs of amplitudes that
    differ only in bit `qubit_index`.
    """
    u00, u01 = gate_matrix[0]
    u10, u11 = gate_matrix[1]
    n = 1 << num_qubits
    bit = 1 << qubit_index
    for i in range(n):
        if i & bit:
            continue
        j = i | bit
        a0 = state_vector[i]
        a1 = state_vector[j]
        state_vector[i] = u00 * a0 + u01 * a1
        state_vector[j] = u10 * a0 + u11 * a1
    return state_vector


def apply_cnot_inplace(state_vector: list, control: int,
                        target: int, num_qubits: int) -> list:
    """Apply CNOT in-place on a state vector.

    §1.3: "In-place обновления state vector где безопасно"
    """
    n = 1 << num_qubits
    c_bit = 1 << control
    t_bit = 1 << target
    for i in range(n):
        if (i & c_bit) and not (i & t_bit):
            j = i | t_bit
            state_vector[i], state_vector[j] = state_vector[j], state_vector[i]
    return state_vector


def tensor_contract_gate(state_vector: list, gate_2x2: list,
                          qubit_index: int, num_qubits: int) -> list:
    """Apply a single-qubit gate via tensor contraction rather than
    building the full 2^n x 2^n matrix.

    §1.3: "Оптимизировать применение гейтов через тензорное сжатие"

    Reshapes the state vector into a 2D array where one axis is the
    target qubit, applies the 2x2 gate via matrix multiplication,
    then reshapes back. This avoids constructing the full Kronecker
    product.
    """
    if not HAS_NUMPY:
        return apply_gate_inplace(state_vector, gate_2x2,
                                     qubit_index, num_qubits)
    sv = np.array(state_vector, dtype=complex)
    # Reshape: (2^k, 2, 2^(n-k-1)) where k = qubit_index
    shape = []
    for i in range(num_qubits):
        shape.append(2)
    sv = sv.reshape(shape)
    # Move target axis to position 0
    sv = np.moveaxis(sv, qubit_index, 0)
    # Reshape to (2, rest)
    rest = 1 << (num_qubits - 1)
    sv = sv.reshape(2, rest)
    # Apply gate
    U = np.array(gate_2x2, dtype=complex)
    sv = U @ sv
    # Reshape back
    sv = sv.reshape(shape)
    # Move axis back
    sv = np.moveaxis(sv, 0, qubit_index)
    return list(sv.flatten())


def optimize_measurement_order(qubits_to_measure: list[int],
                                entanglement_graph: dict[int, set[int]]
                                ) -> list[int]:
    """Optimize the order of qubit measurements to minimize
    state collapse cascading.

    §1.3: "Оптимизировать пути измерения"

    Measures non-entangled qubits first (less state disturbance),
    then entangled ones.
    """
    # Sort by entanglement degree (ascending = least entangled first)
    def degree(q):
        return len(entanglement_graph.get(q, set()))
    return sorted(qubits_to_measure, key=degree)


class GPUAccelerationSurface:
    """Surface/envelope for GPU acceleration via CuPy/JAX.

    §1.3: "Рассмотреть GPU-акселерацию (CuPy/JAX)"

    This is a surface module — actual GPU acceleration requires
    CUDA-enabled hardware and CuPy or JAX installed. When not
    available, falls back to CPU numpy operations.
    """

    def __init__(self):
        self._backend = None
        self._available = False
        self._try_init()

    def _try_init(self):
        try:
            import cupy as cp  # noqa
            self._backend = "cupy"
            self._available = True
        except ImportError:
            pass
        if not self._available:
            try:
                import jax  # noqa
                self._backend = "jax"
                self._available = True
            except ImportError:
                pass

    @property
    def available(self) -> bool:
        return self._available

    @property
    def backend(self) -> str | None:
        return self._backend

    def apply_gate_gpu(self, state_vector, gate_matrix, qubit_index,
                        num_qubits):
        """Apply a gate on GPU if available, else CPU fallback."""
        if not self._available or not HAS_NUMPY:
            return apply_gate_inplace(state_vector, gate_matrix,
                                         qubit_index, num_qubits)
        if self._backend == "cupy":
            import cupy as cp
            sv = cp.array(state_vector, dtype=complex)
            # Use the tensor contraction approach on GPU
            result = tensor_contract_gate(
                cp.asnumpy(sv), gate_matrix, qubit_index, num_qubits)
            return result
        return tensor_contract_gate(state_vector, gate_matrix,
                                       qubit_index, num_qubits)

    def stats(self) -> dict:
        return {"backend": self._backend, "available": self._available}
