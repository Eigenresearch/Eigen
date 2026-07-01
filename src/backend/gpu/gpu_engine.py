import os
import logging
import platform

logger = logging.getLogger('eigen.gpu')
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('[GPU] %(message)s'))
    logger.addHandler(_handler)
    logger.setLevel(logging.WARNING)

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
        if self.device_state is None:
            return

        if self.platform in ('cuda', 'none'):
            num_qubits = int(self.xp.log2(len(self.device_state)))
            tensor = self.device_state.reshape([2] * num_qubits)
            U = self.xp.array(gate_matrix, dtype=complex).reshape([2] * (2 * len(targets)))

            axes = (list(range(len(targets), 2 * len(targets))), targets)
            new_tensor = self.xp.tensordot(U, tensor, axes=axes)

            unused_axes = [i for i in range(num_qubits) if i not in targets]
            current_order = targets + unused_axes
            inv_permutation = [current_order.index(i) for i in range(num_qubits)]

            tensor = self.xp.transpose(new_tensor, inv_permutation)
            self.device_state = tensor.ravel()
        else:
            import torch
            num_qubits = int(self.xp.log2(self.device_state.numel()))
            tensor = self.device_state.view([2] * num_qubits)
            U = torch.tensor(gate_matrix, dtype=torch.complex128, device=self.device).view([2] * (2 * len(targets)))

            dims = (list(range(len(targets), 2 * len(targets))), targets)
            new_tensor = torch.tensordot(U, tensor, dims=dims)

            unused_axes = [i for i in range(num_qubits) if i not in targets]
            current_order = targets + unused_axes
            inv_permutation = [current_order.index(i) for i in range(num_qubits)]

            tensor = new_tensor.permute(inv_permutation)
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
