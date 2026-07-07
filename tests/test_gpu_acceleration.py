"""
P2 §8.1 — GPU Acceleration surface-level envelope.

The GPU Acceleration roadmap item is rated "very high complexity" and is
expected to span ~1 month of work touching cuStateVec kernels, Vulkan
Compute, multi-GPU distribution, and a CUDA-statevector rewrite. Most of
that work is not testable on a CPU-only CI host. What we *can* lock in
today (and what this file covers) is the graceful-degradation envelope:

  * `GPUCapabilities` dataclass exposes a stable, populated envelope so
    downstream tools (CLI doctor, SimSelector, audit reports, LSP hover)
    don't probe `cupy`/`torch` directly.
  * `detect_gpu_capabilities()` returns the envelope from any host; on
    CPU-only machines it returns `available=False, platform='none'`,
    `device_count=0`. No exception is raised on probe failure.
  * `GPUEngine('auto')` emits exactly ONE WARN log entry across the
    process lifetime when no GPU is detected; subsequent `'auto'`
    constructions stay silent (guards noisy loops/tests).
  * `GPUEngine.batch_execute(circuits)` is the surface-level batch API —
    sequential today, but its signature is the contract a future
    parallel scheduler will fill in. Correctness is verified against a
    small explicit circuit and a no-op pass-through.
  * `GPUEngine.capabilities` exposes a live snapshot via the same
    detector so per-engine self-description is one property access.

These tests run with mock `cupy` (mirrors `test_gpu_smoke.py`) plus the
real CPU path, so they exercise both the GPU-surface code and the
fallback envelope without requiring any GPU hardware.
"""
import logging
import math
import sys
import types
import unittest

import src.backend.gpu.gpu_engine as gpu_mod


class _CaptureHandler(logging.Handler):
    """Append every emitted record into a caller-supplied list so tests
    can assert on warning/info content without poking at the global log
    configuration. Used by the once-only-WARN tests."""

    def __init__(self, buf):
        super().__init__()
        self._records = buf

    def emit(self, record):
        self._records.append(record)


def _ensure_fake_cupy():
    """Install a tiny numpy-backed cupy in sys.modules. Idempotent.
    Returns the installed module so callers can poke attributes.

    Mirrors the approach in test_gpu_smoke.py but augmented with the
    extra attributes QuantumSimulator touches on the GPU branch:
    `zeros_like`, `concatenate`, and a `_CupyLike` ndarray view that
    adds a no-op `.get()` so `get_state()` keeps working.
    """
    import numpy as np
    if "cupy" in sys.modules and hasattr(sys.modules["cupy"], "_eigen_fake"):
        return sys.modules["cupy"]

    class _CupyLike(np.ndarray):
        """Numpy ndarray with the CuPy-flavoured `.get()` (no-op here)."""

        def get(self):
            return self

    def _promote(arr):
        return arr.view(_CupyLike)

    class _Runtime:
        def getDeviceCount(self):
            return 1

        def getDevice(self, idx):
            class _Dev:
                name = "MockCUDA-Device"
                major = 0
                minor = 0
            return _Dev()

        def memGetInfo(self):
            # (free, total) bytes
            return (1024 * 1024 * 256, 1024 * 1024 * 512)

    fake = types.ModuleType("cupy")
    fake._eigen_fake = True
    fake.cuda = types.SimpleNamespace(runtime=_Runtime())

    def log2(x):
        return math.log2(int(x))

    def array(data, dtype=None):
        return _promote(np.array(data, dtype=dtype))

    def asarray(data, dtype=None):
        return _promote(np.asarray(data, dtype=dtype))

    def zeros(size, dtype=complex):
        return _promote(np.zeros(size, dtype=dtype))

    def zeros_like(arr):
        return _promote(np.zeros_like(np.asarray(arr)))

    def concatenate(arrs):
        return _promote(np.concatenate([np.asarray(a) for a in arrs]))

    def tensordot(a, b, axes=None):
        return _promote(np.tensordot(np.asarray(a), np.asarray(b), axes=axes))

    def transpose(arr, axes):
        return _promote(np.transpose(np.asarray(arr), axes))

    def ascontiguousarray(arr, dtype=None):
        return _promote(np.ascontiguousarray(np.asarray(arr), dtype=dtype))

    fake.log2 = log2
    fake.array = array
    fake.asarray = asarray
    fake.zeros = zeros
    fake.zeros_like = zeros_like
    fake.concatenate = concatenate
    fake.tensordot = tensordot
    fake.transpose = transpose
    fake.ascontiguousarray = ascontiguousarray
    sys.modules["cupy"] = fake
    return fake


class TestGPUCapabilitiesDataclass(unittest.TestCase):
    """The dataclass shape is the stable contract that downstream code
    reads (CliDoctor / SimSelector / audit reports). Lock it down."""

    def test_default_fields_match_cpu_only_envelope(self):
        from src.backend.gpu.gpu_engine import GPUCapabilities
        caps = GPUCapabilities()
        self.assertFalse(caps.available)
        self.assertEqual(caps.platform, 'none')
        self.assertEqual(caps.device_count, 0)
        self.assertEqual(caps.device_name, 'cpu')
        self.assertEqual(caps.memory_total, 0)
        self.assertEqual(caps.compute_capability, (0, 0))

    def test_fields_are_mutable_and_type_preserved(self):
        from src.backend.gpu.gpu_engine import GPUCapabilities
        caps = GPUCapabilities(available=True, platform='cuda',
                               device_count=1, device_name='RTX',
                               memory_total=8_000_000_000,
                               compute_capability=(8, 6))
        self.assertTrue(caps.available)
        self.assertEqual(caps.compute_capability, (8, 6))
        self.assertIsInstance(caps.memory_total, int)

    def test_repr_does_not_raise(self):
        # Downstream logging calls repr(caps); ensure stability.
        from src.backend.gpu.gpu_engine import GPUCapabilities
        caps = GPUCapabilities()
        self.assertIn('GPUCapabilities', repr(caps))


class TestDetectGPUCapabilities(unittest.TestCase):

    def test_cpu_host_returns_unavailable_envelope(self):
        # If the host has a real GPU (cupy/torch installed & visible), this
        # test would still pass because `available` flips True; what we
        # actually assert is that the function returns a populated
        # dataclass without raising.
        caps = gpu_mod.detect_gpu_capabilities()
        self.assertEqual(caps.__class__.__name__, 'GPUCapabilities')
        self.assertIn(caps.platform, ('cuda', 'rocm', 'metal', 'none'))
        if caps.platform == 'none':
            self.assertFalse(caps.available)
            self.assertEqual(caps.device_count, 0)
        else:
            self.assertTrue(caps.available)
            self.assertGreaterEqual(caps.device_count, 1)

    def test_probe_failure_returns_none_envelope_not_exception(self):
        """If detect_gpu_platform returns 'cuda' but importing cupy fails
        inside detect_gpu_capabilities, we must NOT raise — the contract
        is that callers can use this function unconditionally."""
        # Force the 'cuda' branch with no cupy importable for the duration
        # of the probe. The try/except in detect_gpu_capabilities swallows
        # the ImportError and returns an unavailable envelope.
        orig_plat = gpu_mod.detect_gpu_platform
        saved_cupy = sys.modules.pop('cupy', None)

        class _BlockCupy:
            def find_spec(self, name, path=None, target=None):
                if name == 'cupy':
                    raise ImportError("test-blocked cupy")
                return None

        blocker = _BlockCupy()
        sys.meta_path.insert(0, blocker)
        gpu_mod.detect_gpu_platform = lambda: 'cuda'
        try:
            caps = gpu_mod.detect_gpu_capabilities()
            self.assertEqual(caps.__class__.__name__, 'GPUCapabilities')
            # Probe failed -> envelope marked unavailable.
            self.assertFalse(caps.available)
        finally:
            gpu_mod.detect_gpu_platform = orig_plat
            sys.meta_path.remove(blocker)
            if saved_cupy is not None:
                sys.modules['cupy'] = saved_cupy

    def test_real_cupy_path_populates_envelope(self):
        """With mock cupy installed and platform forced to 'cuda', the
        envelope reads device_count=1, device_name='MockCUDA-Device',
        and a populated memory_total. This is the surface contract that
        audit / doctor tools read; future kernel work must keep it."""
        _ensure_fake_cupy()
        orig_plat = gpu_mod.detect_gpu_platform
        gpu_mod.detect_gpu_platform = lambda: 'cuda'
        try:
            caps = gpu_mod.detect_gpu_capabilities()
            self.assertTrue(caps.available)
            self.assertEqual(caps.platform, 'cuda')
            self.assertEqual(caps.device_count, 1)
            self.assertEqual(caps.device_name, 'MockCUDA-Device')
            self.assertGreater(caps.memory_total, 0)
        finally:
            gpu_mod.detect_gpu_platform = orig_plat


class TestGPUEngineAutoFallbackWarnsOnce(unittest.TestCase):

    def setUp(self):
        # Reset the once-only guard so each test exercises the first-timer
        # path deterministically. We save & restore other state too.
        self._saved_warned = gpu_mod._warned_no_gpu
        gpu_mod._warned_no_gpu = False
        self._records = []
        self._handler = _CaptureHandler(self._records)
        self._handler.setLevel(logging.DEBUG)
        self._logger = logging.getLogger('eigen.gpu')
        self._logger.addHandler(self._handler)
        self._orig_level = self._logger.level
        self._logger.setLevel(logging.DEBUG)
        self._orig_plat = gpu_mod.detect_gpu_platform
        # Pop any installed mock cupy so the 'auto' resolution is
        # deterministic per test (each test that needs mock cupy installs
        # its own fresh copy via _ensure_fake_cupy).
        self._saved_cupy = sys.modules.pop('cupy', None)

    def tearDown(self):
        gpu_mod._warned_no_gpu = self._saved_warned
        self._logger.removeHandler(self._handler)
        self._logger.setLevel(self._orig_level)
        gpu_mod.detect_gpu_platform = self._orig_plat
        if self._saved_cupy is not None:
            sys.modules['cupy'] = self._saved_cupy
        else:
            sys.modules.pop('cupy', None)

    def _warnings_for_no_gpu(self):
        return [r for r in self._records
                if 'No GPU backend detected' in r.getMessage()]

    def test_auto_when_no_gpu_emits_single_warn(self):
        gpu_mod.detect_gpu_platform = lambda: 'none'
        gp = gpu_mod.GPUEngine('auto')
        self.assertEqual(gp.platform, 'none')
        gp2 = gpu_mod.GPUEngine('auto')
        self.assertEqual(gp2.platform, 'none')
        warns = self._warnings_for_no_gpu()
        self.assertEqual(len(warns), 1,
                         f"expected exactly 1 warn, got {len(warns)}: {warns}")

    def test_explicit_none_platform_does_not_warn(self):
        gpu_mod.GPUEngine('none')
        gpu_mod.GPUEngine('none')
        self.assertEqual(self._warnings_for_no_gpu(), [])

    def test_auto_with_real_gpu_available_does_not_warn(self):
        _ensure_fake_cupy()
        gpu_mod.detect_gpu_platform = lambda: 'cuda'
        engine = gpu_mod.GPUEngine('auto')
        self.assertEqual(engine.platform, 'cuda')
        self.assertEqual(self._warnings_for_no_gpu(), [])


class TestGPUEngineBatchExecute(unittest.TestCase):

    def setUp(self):
        _ensure_fake_cupy()
        self.engine = gpu_mod.GPUEngine('cuda')
        self.assertIn(self.engine.platform, ('cuda', 'none'))
        self.inv_sqrt2 = 1.0 / math.sqrt(2.0)
        self.h_mat = [[self.inv_sqrt2, self.inv_sqrt2],
                      [self.inv_sqrt2, -self.inv_sqrt2]]
        self.cnot = [[1, 0, 0, 0], [0, 1, 0, 0],
                     [0, 0, 0, 1], [0, 0, 1, 0]]

    def tearDown(self):
        # Remove mock cupy so it doesn't leak into later tests in the file
        # (detect_gpu_platform would otherwise resolve to 'cuda').
        sys.modules.pop('cupy', None)

    def test_empty_circuits_returns_empty_list(self):
        self.engine.initialize_state(2)
        self.assertEqual(self.engine.batch_execute([]), [])

    def test_no_op_circuit_returns_initial_state(self):
        self.engine.initialize_state(2)
        results = self.engine.batch_execute([[]])
        self.assertEqual(len(results), 1)
        state = results[0]
        self.assertAlmostEqual(abs(state[0]), 1.0, places=10)
        for i in range(1, 4):
            self.assertAlmostEqual(abs(state[i]), 0.0, places=10)

    def test_batch_sequential_correctness_h_cnot(self):
        """Three independent H+CNOT runs each produce Bell state. We
        reset device_state between runs by re-initializing before each
        batch_execute call — batch_execute itself is sequential and
        accumulates state, so the caller chooses whether the
        subsequent circuit starts fresh."""
        # batch_execute takes a list of circuits; each circuit is a list
        # of (targets, matrix) gates. So a single Bell-state circuit is
        # expressed as [[([0], h), ([0,1], cnot)]] — outer = 1 circuit,
        # inner = the 2 gates of that circuit.
        results = []
        for _ in range(3):
            self.engine.initialize_state(2)
            results.extend(self.engine.batch_execute(
                [[([0], self.h_mat), ([0, 1], self.cnot)]]))
        self.assertEqual(len(results), 3)
        for state in results:
            self.assertAlmostEqual(state[0].real, self.inv_sqrt2, places=8)
            self.assertAlmostEqual(state[3].real, self.inv_sqrt2, places=8)
            self.assertAlmostEqual(abs(state[1]), 0.0, places=8)
            self.assertAlmostEqual(abs(state[2]), 0.0, places=8)

    def test_batch_executes_in_caller_visible_order(self):
        """First circuit = X q0; second circuit = H q0 — verify the
        snapshot after circuit 0 is |1> (post-X) and after circuit 1
        is H|1> = (|0> - |1>)/sqrt2 (post-H applied on top of X).
        This locks in that batch_execute does NOT parallelise — order
        matters and states are dependent snapshots, not independent
        runs."""
        self.engine.initialize_state(1)
        x_mat = [[0, 1], [1, 0]]
        results = self.engine.batch_execute([
            [([0], x_mat)],
            [([0], self.h_mat)],
        ])
        self.assertEqual(len(results), 2)
        self.assertAlmostEqual(abs(results[0][0]), 0.0, places=10)
        self.assertAlmostEqual(abs(results[0][1]), 1.0, places=10)
        self.assertAlmostEqual(results[1][0].real, self.inv_sqrt2, places=10)
        self.assertAlmostEqual(results[1][1].real, -self.inv_sqrt2, places=10)

    def test_batch_without_initialize_raises(self):
        engine = gpu_mod.GPUEngine('none')
        with self.assertRaises(RuntimeError):
            engine.batch_execute([[([0], self.h_mat)]])


class TestGPUEngineCapabilitiesProperty(unittest.TestCase):

    def setUp(self):
        # Ensure detect_gpu_platform returns 'none' so the property snapshot
        # is the cpu-only envelope regardless of cupy contamination.
        self._saved_cupy = sys.modules.pop('cupy', None)

    def tearDown(self):
        if self._saved_cupy is not None:
            sys.modules['cupy'] = self._saved_cupy

    def test_capabilities_returns_dataclass(self):
        engine = gpu_mod.GPUEngine('none')
        caps = engine.capabilities
        self.assertEqual(caps.__class__.__name__, 'GPUCapabilities')
        self.assertEqual(caps.platform, 'none')

    def test_capabilities_recomputes_each_call(self):
        """The property must NOT cache; downstream tools call it to detect
        hot-plugged GPUs between calls. We assert by spying on the
        detector's call count."""
        engine = gpu_mod.GPUEngine('none')
        calls = []
        orig = gpu_mod.detect_gpu_capabilities

        def _spy():
            calls.append(1)
            return orig()

        gpu_mod.detect_gpu_capabilities = _spy
        try:
            _ = engine.capabilities
            _ = engine.capabilities
            _ = engine.capabilities
        finally:
            gpu_mod.detect_gpu_capabilities = orig
        self.assertEqual(len(calls), 3)


class TestGPUDispatchOnAllGateTypes(unittest.TestCase):
    """Regression guard for BUG-C13 and friends: every gate supported by
    QuantumSimulator (including CRX/CRY/CRZ/CCX/CSWAP/CP) must route
    through GPUEngine.apply_gate without AttributeError when the GPU
    branch is active. The simulator already has GPU-paths for all of
    these — we make sure they don't regress by forcing the GPU branch
    via mock cupy and exercising each gate through the high-level
    QuantumSimulator API. This catches the bug class where someone adds
    a new gate to QuantumSimulator but forgets the if-self.gpu_engine
    branch (the simulator would then fall through to dense_backend,
    which is None while GPU-active, raising AttributeError)."""

    def setUp(self):
        _ensure_fake_cupy()
        from src.simulator import QuantumSimulator
        self.sim = QuantumSimulator(sim_type='dense', gpu_platform='cuda')
        if self.sim.gpu_engine is None:
            self.skipTest("GPU branch not active in this environment "
                          "(GPUEngine fell back to 'none')")
        # Defensive check: dense_backend must be None when GPU is on,
        # since the simulator deliberately does not allocate it.
        self.assertIsNone(self.sim.dense_backend,
                          "GPU-active sim must NOT allocate dense_backend — "
                          "otherwise a forgotten GPU path on a new gate would "
                          "silently work via the dense fallback rather than "
                          "raise AttributeError as the audit warned.")

    def tearDown(self):
        sys.modules.pop('cupy', None)

    def test_all_named_gates_route_through_gpu_engine(self):
        import cmath
        # Allocate 3 qubits for CCX/CSWAP/CP/CRX/CRY/CRZ.
        for n in ('q0', 'q1', 'q2'):
            self.sim.allocate_qubit(n)

        # 1-qubit gates — should not raise through GPU branch.
        self.sim.H('q0')
        self.sim.X('q1')
        self.sim.Y('q0')
        self.sim.Z('q1')
        self.sim.S('q0')
        self.sim.T('q1')
        self.sim.RX('q0', 0.5)
        self.sim.RY('q1', 0.7)
        self.sim.RZ('q2', 0.9)

        # 2-qubit gates
        self.sim.CNOT('q0', 'q1')
        self.sim.CZ('q1', 'q2')
        self.sim.SWAP('q0', 'q2')
        self.sim.CP('q0', 'q1', 0.4)
        self.sim.CRX('q1', 'q2', 0.4)
        self.sim.CRY('q0', 'q1', 0.6)
        self.sim.CRZ('q0', 'q2', 0.8)

        # 3-qubit gates
        self.sim.CCX('q0', 'q1', 'q2')
        self.sim.CSWAP('q0', 'q1', 'q2')

        # State should still be 8-dim and norm-1 (unitary evolution).
        state = self.sim.get_state_vector()
        self.assertEqual(len(state), 8)
        norm = sum(abs(a) ** 2 for a in state)
        self.assertAlmostEqual(norm, 1.0, places=8)


if __name__ == "__main__":
    unittest.main()
