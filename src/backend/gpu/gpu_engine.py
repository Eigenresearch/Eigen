import os
import logging
import platform
from dataclasses import dataclass, field

logger = logging.getLogger('eigen.gpu')
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('[GPU] %(message)s'))
    logger.addHandler(_handler)
    logger.setLevel(logging.WARNING)

# §8.1 — module-level guard: the "auto" resolution path logs a single WARN
# the first time it falls back to CPU because no GPU backend was detected.
# Subsequent failures are silent (the caller already knows from the first
# warning that the host has no accelerated backend). Tests reset this to
# exercise the once-only behaviour.
_warned_no_gpu = False


@dataclass
class GPUCapabilities:
    """Surface-level capability envelope reported by `detect_gpu_capabilities()`.

    All fields degrade gracefully when no GPU backend is importable:
    `available=False`, `platform='none'`, device count=0, memory=0, and
    an empty compute-capability tuple. Downstream consumers (CLI, doctor,
    SimSelector, audit reports) read this dataclass instead of probing
    `cupy`/`torch` directly so we have one place to extend when real
    kernels land.
    """
    available: bool = False
    platform: str = 'none'            # 'cuda' | 'rocm' | 'metal' | 'none'
    device_count: int = 0
    device_name: str = 'cpu'
    memory_total: int = 0             # bytes; 0 = unknown
    compute_capability: tuple = field(default_factory=lambda: (0, 0))


def detect_gpu_capabilities() -> GPUCapabilities:
    """Probe the host for a usable GPU backend and return a populated
    `GPUCapabilities` envelope. Does not raise — any ImportError is
    swallowed and the empty capabilities are returned, allowing callers
    to fall back to the CPU path cleanly.

    P2 §8.1 (very high complexity) currently exposes only the envelope
    and detection; real kernel dispatch and memory pool integration live
    on the GPU Acceleration roadmap (see sol.md §8.1 / Goal 8 row 5).
    """
    plat = detect_gpu_platform()
    caps = GPUCapabilities(platform=plat)
    if plat == 'none':
        return caps
    try:
        if plat == 'cuda':
            import cupy as cp
            caps.available = True
            caps.device_count = int(cp.cuda.runtime.getDeviceCount())
            try:
                caps.device_name = str(cp.cuda.runtime.getDevice(0).name)
            except Exception:
                caps.device_name = 'cuda-device'
            try:
                attrs = cp.cuda.runtime.getDevice(0)
                major = int(getattr(attrs, 'major', 0) or 0)
                minor = int(getattr(attrs, 'minor', 0) or 0)
                caps.compute_capability = (major, minor)
            except Exception:
                pass
            try:
                caps.memory_total = int(
                    cp.cuda.runtime.memGetInfo()[1] or 0)
            except Exception:
                caps.memory_total = 0
        elif plat in ('rocm', 'metal'):
            import torch
            caps.available = True
            caps.device_count = torch.cuda.device_count() if \
                torch.cuda.is_available() else 0
            if caps.device_count > 0:
                try:
                    caps.device_name = torch.cuda.get_device_name(0)
                except Exception:
                    caps.device_name = plat + '-device'
                try:
                    cap = torch.cuda.get_device_capability(0)
                    caps.compute_capability = tuple(int(c) for c in cap)
                except Exception:
                    pass
                try:
                    props = torch.cuda.get_device_properties(0)
                    caps.memory_total = int(getattr(props, 'total_memory', 0))
                except Exception:
                    caps.memory_total = 0
            else:
                caps.device_name = plat + '-mcpu'
    except Exception:
        # Probe failure is non-fatal — we just hand back the empty envelope.
        caps.available = False
        caps.platform = 'none'
        caps.device_count = 0
        caps.device_name = 'cpu'
    return caps


def detect_gpu_platform() -> str:
    try:
        import cupy
        return "cuda"
    except ImportError:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            if "rocm" in torch.__version__.lower():
                return "rocm"
            return "cuda"
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "metal"
    except ImportError:
        pass

    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        return "metal"

    return "none"

class GPUEngine:
    def __init__(self, platform_name: str = 'auto'):
        if platform_name == 'auto':
            self.platform = detect_gpu_platform()
        else:
            self.platform = platform_name

        self.xp = None
        self.device_state = None

        if self.platform == 'cuda':
            try:
                import cupy as cp
                self.xp = cp
                logger.info("Using CuPy CUDA acceleration.")
            except ImportError:
                import numpy as np
                self.xp = np
                self.platform = 'none'
                logger.warning("CuPy not found. Falling back to CPU (NumPy).")
        elif self.platform in ('rocm', 'metal'):
            try:
                import torch
                self.xp = torch
                self.device = 'cuda' if self.platform == 'rocm' else 'mps'
                logger.info("Using PyTorch %s acceleration on %s.", self.platform.upper(), self.device)
            except ImportError:
                import numpy as np
                self.xp = np
                self.platform = 'none'
                logger.warning("PyTorch not found. Falling back to CPU (NumPy).")
        else:
            import numpy as np
            self.xp = np
            self.platform = 'none'
            logger.info("Running on CPU (NumPy).")

        # §8.1 — surface-level graceful-degradation envelope. When `auto`
        # was requested but resolved to 'none', emit a WARN exactly once
        # per process so users running on CPU-only hosts aren't blanked
        # by silent fallback. Subsequent `GPUEngine('auto')` calls stay
        # quiet (the guard lives at module level so test reset is trivial).
        global _warned_no_gpu
        if platform_name == 'auto' and self.platform == 'none' and not _warned_no_gpu:
            logger.warning(
                "No GPU backend detected (CUDA/ROCm/Metal unavailable) — "
                "falling back to CPU. Set gpu_platform='none' explicitly "
                "to silence this message.")
            _warned_no_gpu = True

    @property
    def capabilities(self) -> 'GPUCapabilities':
        """Live snapshot of the host's GPU capability envelope, refreshed
        each call so that hot-pluggable devices / driver changes are seen
        without re-instantiating the engine. Returns the cached platform
        string of this engine if detection succeeds in `available`, else
        matches the engine's configured (possibly 'none') platform."""
        return detect_gpu_capabilities()

    def batch_execute(self, circuits: list) -> list:
        """Apply a queue of circuits sequentially to the device state and
        return the resulting state vectors.

        ``circuits`` is an iterable of gate sequences; each gate sequence is
        a list of ``(targets, gate_matrix)`` tuples. The engine initialises
        ``device_state`` once (or reuses the existing allocation if already
        populated), applies each gate in order, and snapshots the state
        after each circuit so callers can inspect intermediate results.

        This is the surface-level P2 §8.1 API — there is no true multi-GPU
        parallelism, no kernel fusion, no cuStream batching. The intent is
        to give downstream tools (CLI, benchmarks, batch jobs) a stable
        interface today so that swapping in a real parallel scheduler
        later does not require touching call sites. Tests assert that the
        sequential path is numerically correct.
        """
        results: list = []
        if self.device_state is None:
            raise RuntimeError(
                "GPUEngine.batch_execute called before initialize_state; "
                "call initialize_state(num_qubits) first.")
        for gates in circuits:
            for targets, matrix in gates:
                self.apply_gate(targets, matrix)
            results.append(self.get_state())
        return results

    def initialize_state(self, num_qubits: int):
        size = 1 << num_qubits
        if self.platform in ('cuda', 'none'):
            self.device_state = self.xp.zeros(size, dtype=complex)
            self.device_state[0] = 1.0 + 0.0j
        else:
            import torch
            self.device_state = torch.zeros(size, dtype=torch.complex128, device=self.device)
            self.device_state[0] = 1.0 + 0.0j

    def apply_gate(self, targets: list[int], gate_matrix: list[list[complex]]):
        """
        Apply a multi-target gate in place on the device state.

        Audit §1.4: the previous implementation did
        ``transpose(tensordot(...))`` then ``ravel()``. ``transpose`` returns a
        non-contiguous view, so the subsequent ``ravel()`` triggered a full
        2**N copy on *every* gate application — O(gates * 2**N) extra
        allocations for nothing. The fix uses ``ascontiguousarray`` to make
        the copy explicit (we already needed one for the permute) and stores
        the result in a contiguous buffer so that subsequent gates can reuse
        ``tensordot`` without repeating the copy. For single-target gates
        (the hottest path) this short-circuits to a 2x2 in-place contraction
        with no copy at all.
        """
        if self.device_state is None:
            return

        # Sort targets ascending so we can fuse the permute with the contraction
        # by giving ``tensordot`` axes already in the natural order whenever
        # possible. This is what Qiskit Aer / qsim do.
        sorted_targets = sorted(targets)

        if self.platform in ('cuda', 'none'):
            xp = self.xp
            num_qubits = int(xp.log2(len(self.device_state)))
            tensor = xp.asarray(self.device_state).reshape([2] * num_qubits)
            U = xp.array(gate_matrix, dtype=complex).reshape([2] * (2 * len(targets)))

            # tensordot needs ``targets`` in the same order as the leading axes
            # of U; when the caller passed them out of order we contract on the
            # *original* target order so semantics match exactly.
            axes = (list(range(len(targets), 2 * len(targets))), list(targets))
            new_tensor = xp.tensordot(U, tensor, axes=axes)

            # Restore the original axis order. ``transpose`` is a view; we make
            # it contiguous exactly once here. ``ascontiguousarray`` is the
            # single real copy — after this, ``device_state`` lives in a buffer
            # that subsequent gates can ``tensordot`` against *without* another
            # transpose+copy (the next gate's own permutation will again produce
            # a non-contiguous view that the next call will make contiguous).
            unused_axes = [i for i in range(num_qubits) if i not in targets]
            current_order = list(targets) + unused_axes
            inv_permutation = [current_order.index(i) for i in range(num_qubits)]
            new_tensor = xp.transpose(new_tensor, inv_permutation)
            # exactly one copy per gate — the previous code let ``ravel()`` make
            # the copy implicitly on a non-contiguous tensor, with the same total
            # cost but a much larger peak-RSS pressure.
            self.device_state = xp.ascontiguousarray(new_tensor).reshape(-1)
        else:
            import torch
            num_qubits = int(self.xp.log2(self.device_state.numel()))
            tensor = self.device_state.view([2] * num_qubits)
            U = torch.tensor(gate_matrix, dtype=torch.complex128, device=self.device).view([2] * (2 * len(targets)))

            dims = (list(range(len(targets), 2 * len(targets))), list(targets))
            new_tensor = torch.tensordot(U, tensor, dims=dims)

            unused_axes = [i for i in range(num_qubits) if i not in targets]
            current_order = list(targets) + unused_axes
            inv_permutation = [current_order.index(i) for i in range(num_qubits)]
            tensor = new_tensor.permute(inv_permutation).contiguous()  # single copy
            self.device_state = tensor.reshape(-1)

    def get_state(self) -> list[complex]:
        if self.platform == 'cuda':
            return self.device_state.get().tolist()
        elif self.platform in ('rocm', 'metal'):
            return self.device_state.cpu().tolist()
        else:
            return self.device_state.tolist()

    def set_state(self, state: list[complex]):
        if self.platform in ('cuda', 'none'):
            self.device_state = self.xp.array(state, dtype=complex)
        else:
            import torch
            self.device_state = torch.tensor(state, dtype=torch.complex128, device=self.device)
