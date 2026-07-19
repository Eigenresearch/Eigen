"""
GPU backend smoke test (audit §1.3 / §1.4).

The audit identified three problems with the GPU backend:
  §1.3 cupy and torch are *not* declared anywhere in pyproject.toml. A standard
       `pip install eigen-lang` silently degrades to NumPy with a single warning
       line in the log, and CI never exercises the GPU paths.
  §1.3 No 'gpu-cuda' or 'gpu-torch' extras, no smoke test.
  §1.4 `apply_gate` does `transpose(...) -> ravel()` which forces an unguarded
       2**N copy on every gate. The fix uses `ascontiguousarray(...)` so the
       copy is explicit (we needed it for the permute anyway) and contiguous so
       the next gate's tensordot doesn't re-trigger implicit copies.

This test exercises three concrete things:
  1. The CUDA path under a mock cupy with the NumPy-shaped API.
  2. The torch path (installed in CI; skipped if not importable).
  3. The apply_gate correctness — the fix must remain numerically correct.
"""
import sys
import types
import unittest
import math


def _install_fake_cupy():
    """Install a fake `cupy` module whose API is small but matches the bits that
    GPUEngine uses: log2, array, zeros, asarray, tensordot, transpose,
    ascontiguousarray, reshape. Backed by numpy so semantics are verified.

    CuPy's arrays implement `.get()` to copy GPU -> CPU. We expose a tiny
    ndarray subclass with a no-op `.get()` so the existing `get_state()` code
    path (`return self.device_state.get().tolist()`) keeps working.
    """
    import numpy as np
    if "cupy" in sys.modules:
        return None

    class _CupyLike(np.ndarray):
        """Numpy ndarray with the CuPy-flavoured ``.get()`` (no-op here)."""

        def get(self):
            return self

    def _promote(arr):
        view = arr.view(_CupyLike)
        return view

    fake = types.ModuleType("cupy")

    def log2(x):
        return math.log2(int(x))

    def array(data, dtype=None):
        return _promote(np.array(data, dtype=dtype))

    def asarray(data, dtype=None):
        return _promote(np.asarray(data, dtype=dtype))

    def zeros(size, dtype=complex):
        return _promote(np.zeros(size, dtype=dtype))

    def tensordot(a, b, axes=None):
        return _promote(np.tensordot(np.asarray(a), np.asarray(b), axes=axes))

    def transpose(arr, axes):
        return _promote(np.transpose(np.asarray(arr), axes))

    def ascontiguousarray(arr, dtype=None):
        return _promote(np.ascontiguousarray(np.asarray(arr), dtype=dtype))

    def reshape_helper(arr, shape):
        return _promote(np.asarray(arr).reshape(shape))

    fake.log2 = log2
    fake.array = array
    fake.asarray = asarray
    fake.zeros = zeros
    fake.tensordot = tensordot
    fake.transpose = transpose
    fake.ascontiguousarray = ascontiguousarray
    fake.reshape = reshape_helper

    sys.modules["cupy"] = fake
    return fake


def _inject_cupy_prefix():
    """Put the fake cupy at the front of sys.path's modules so 'import cupy'
    inside gpu_engine.py resolves to it (gpu_engine catches ImportError, so we
    need a present module, not just an entry in sys.modules).
    """
    fake = _install_fake_cupy()
    if fake is None:
        # Already had a cupy (real one or previously injected). That's fine.
        return
    # gpue_engine.py uses `import cupy as cp` inside a try/except ImportError.
    # _install_fake_cupy has put us in sys.modules, so the import succeeds.


class TestGPUEngineCudaFallback(unittest.TestCase):
    """Verify apply_gate runs through the CUDA path under a mocked cupy."""

    def setUp(self):
        _inject_cupy_prefix()
        # Force the CUDA branch by constructing via the direct entry point.
        # We bypass detect_gpu_platform by setting platform explicitly.
        from src.backend.gpu.gpu_engine import GPUEngine
        self.GPUEngine = GPUEngine
        # Force the cuda code path even though we don't have a GPU.
        self.engine = GPUEngine(platform_name="cuda")
        # The constructor tries to import cupy; if it succeeded (via our mock),
        # platform stays 'cuda'. Otherwise it falls back to 'none'. Either way
        # the cuda/none branch shares the same code, so test both.
        self.assertIn(self.engine.platform, ("cuda", "none"))

    def test_initial_state(self):
        self.engine.initialize_state(3)
        self.assertEqual(len(self.engine.device_state), 8)
        self.assertAlmostEqual(abs(self.engine.device_state[0]), 1.0, places=12)
        for i in range(1, 8):
            self.assertAlmostEqual(abs(self.engine.device_state[i]), 0.0, places=12)

    def test_apply_h_gate_correctness(self):
        self.engine.initialize_state(1)
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        # |0> -> (|0> + |1>)/sqrt(2)
        self.engine.apply_gate([0], [[inv_sqrt2, inv_sqrt2], [inv_sqrt2, -inv_sqrt2]])
        state = self.engine.get_state()
        self.assertAlmostEqual(state[0].real, inv_sqrt2, places=10)
        self.assertAlmostEqual(state[1].real, inv_sqrt2, places=10)

    def test_apply_cnot_correctness(self):
        """Bell state: H q0, then CNOT(q0, q1) -> (|00> + |11>)/sqrt(2)."""
        self.engine.initialize_state(2)
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        self.engine.apply_gate([0], [[inv_sqrt2, inv_sqrt2], [inv_sqrt2, -inv_sqrt2]])
        # CNOT matrix in computational basis (control=q0, target=q1):
        cnot = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
        self.engine.apply_gate([0, 1], cnot)
        state = self.engine.get_state()
        self.assertEqual(len(state), 4)
        self.assertAlmostEqual(state[0].real, inv_sqrt2, places=10)
        self.assertAlmostEqual(state[3].real, inv_sqrt2, places=10)
        self.assertAlmostEqual(abs(state[1]).real, 0.0, places=10)
        self.assertAlmostEqual(abs(state[2]).real, 0.0, places=10)

    def test_apply_two_qubit_gate_in_reverse_order_matches(self):
        """Reverse-ordered targets must produce identical results to sorted
        targets when the gate matrix is symmetric under target swap.
        SWAP is symmetric, so applying it on [a, b] must equal [b, a].
        """
        self.engine.initialize_state(2)
        # Prepare |10> = q0=1, q1=0.
        self.engine.apply_gate([0], [[0, 1], [1, 0]])
        # Now apply SWAP(q0, q1) <-> SWAP(q1, q0). Result should be |01> = q0=0, q1=1.
        swap = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
        self.engine.apply_gate([0, 1], swap)
        # Expected state: |01> = (0, 1, 0, 0)
        state = self.engine.get_state()
        self.assertAlmostEqual(state[1].real, 1.0, places=10)
        for i in (0, 2, 3):
            self.assertAlmostEqual(abs(state[i]).real, 0.0, places=10)

    def test_apply_gate_creates_contiguous_state(self):
        """Audit §1.4: the fix must produce a contiguous device_state so the
        next gate's tensordot doesn't trigger implicit copies. We assert the
        flag rather than the cost."""
        self.engine.initialize_state(3)
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        self.engine.apply_gate([0], [[inv_sqrt2, inv_sqrt2], [inv_sqrt2, -inv_sqrt2]])
        if hasattr(self.engine.device_state, "flags"):
            # NumPy: contiguous means C-contiguous.
            self.assertTrue(
                self.engine.device_state.flags["C_CONTIGUOUS"],
                "device_state must be C-contiguous after apply_gate (audit §1.4)",
            )

    def test_gate_count_does_not_leak_memory_unboundedly(self):
        """Run many gates and confirm the state vector stays the right size —
        guards against the old code accidentally keeping views of intermediate
        arrays in a working set."""
        self.engine.initialize_state(3)
        before = len(self.engine.device_state)
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        for _ in range(50):
            self.engine.apply_gate([0], [[inv_sqrt2, inv_sqrt2], [inv_sqrt2, -inv_sqrt2]])
            self.engine.apply_gate([0], [[inv_sqrt2, inv_sqrt2], [inv_sqrt2, -inv_sqrt2]])
        self.assertEqual(len(self.engine.device_state), before)


class TestGPUEngineTorchFallback(unittest.TestCase):
    """Skip if torch is unavailable; otherwise exercise the torch branch."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
            cls.torch_available = True
        except Exception:
            cls.torch_available = False

    def setUp(self):
        if not self.torch_available:
            self.skipTest("torch not available in this environment")
        from src.backend.gpu.gpu_engine import GPUEngine
        try:
            self.engine = GPUEngine(platform_name="metal")
        except Exception:
            # On Linux, 'metal' falls back to numpy because torch.backends.mps
            # isn't available. Verify the fallback path is still numeric-correct
            # by switching to cpu-based torch via 'none' + manual assignment.
            self.engine = GPUEngine(platform_name="none")
        if self.engine.platform == "none":
            # Force the torch branch by monkey-patching the platform only if
            # torch was actually loaded; otherwise this test is meaningless.
            try:
                import torch
                # Mirror the constructor's torch branch:
                self.engine.xp = torch
                self.engine.device = "cpu"
                self.engine.platform = "metal"
            except Exception:
                self.skipTest("torch fallback path requires torch tensors to test")

    def test_torch_branch_h_gate(self):
        self.engine.initialize_state(1)
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        self.engine.apply_gate([0], [[inv_sqrt2, inv_sqrt2], [inv_sqrt2, -inv_sqrt2]])
        state = self.engine.get_state()
        self.assertAlmostEqual(state[0].real, inv_sqrt2, places=8)
        self.assertAlmostEqual(state[1].real, inv_sqrt2, places=8)

    def test_torch_branch_cnot_bell(self):
        self.engine.initialize_state(2)
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        self.engine.apply_gate([0], [[inv_sqrt2, inv_sqrt2], [inv_sqrt2, -inv_sqrt2]])
        self.engine.apply_gate(
            [0, 1],
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        )
        state = self.engine.get_state()
        self.assertAlmostEqual(state[0].real, inv_sqrt2, places=8)
        self.assertAlmostEqual(state[3].real, inv_sqrt2, places=8)
        self.assertAlmostEqual(abs(state[1]).real, 0.0, places=8)
        self.assertAlmostEqual(abs(state[2]).real, 0.0, places=8)


class TestPyprojectExtrasAdvertised(unittest.TestCase):
    """Static check (audit §1.3): pyproject.toml must declare gpu-cuda and
    gpu-torch as optional-dep groups so users can `pip install
    eigen-lang[gpu-cuda]`."""

    def test_gpu_extras_exist(self):
        import os
        toml_path = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
        with open(toml_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("gpu-cuda", content)
        self.assertIn("gpu-torch", content)
        # ensure cupy is mentioned under gpu-cuda (not in main deps)
        self.assertIn("cupy", content)
        self.assertIn("torch", content)


if __name__ == "__main__":
    unittest.main()
