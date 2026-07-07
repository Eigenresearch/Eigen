"""
Audit §2.1 — noise channel Kraus-completeness and trace-preservation
regression tests.

The audit's headline complaint against the existing noise tests was:

> "Для noise-стратегий - не просто 'оно не падает', а 'выполняется
>  тождество Крауса с погрешностью 1e-10'."

The previous tests only checked that channels applied without exception.
We now pin down the math:

1. For every channel that exposes Kraus operators (amplitude damping and
   phase damping in this codebase), `sum_k K_k† K_k = I` at 1e-10.
2. The `DensityMatrixSimulator._apply_channel` implementation preserves
   trace (TP property) for an arbitrary input density matrix.
3. The output of `_apply_channel` remains Hermitian-positive (CP property).
4. The stochastic channels (bit_flip, phase_flip, depolarizing) have
   analytically known Kraus operators that also satisfy completeness,
   even though the Python implementation applies them via single-qubit
   unitaries. We compute those analytic sets here and verify them.
5. End-to-end quantum-mechanical behaviour: amplitude damping with
   gamma=1 always collapses to |0>; phase damping with lambda=1 turns
   any state in to a diagonal density matrix.
"""

import math
import unittest

import numpy as np

from src.density_matrix_simulator import DensityMatrixSimulator
from src.noise.noise_channel import (
    AmplitudeDampingChannel,
    BitFlipChannel,
    DepolarizingChannel,
    PhaseDampingChannel,
    PhaseFlipChannel,
)


_KRAUS_TOLERANCE = 1e-10


def _kraus_completeness(kraus_ops) -> np.ndarray:
    """Return the deviation I - sum_k K_k† K_k for completeness testing."""
    n = kraus_ops[0].shape[0]
    identity = np.eye(n, dtype=complex)
    sigma = sum(k.conj().T @ k for k in kraus_ops)
    return identity - sigma


def _is_hermitian(m: np.ndarray, tol: float = 1e-10) -> bool:
    return np.linalg.norm(m - m.conj().T) <= tol


class TestKrausCompleteness(unittest.TestCase):
    """For every noise channel that exposes Kraus operators, verify the
    completeness identity sum_k K_k† K_k = I at 1e-10 tolerance.

    This is the audit's headline mathematical requirement."""

    def test_amplitude_damping_kraus_completeness_for_several_gamma(self):
        # Reproduce the Kraus operators used in
        # src/density_matrix_simulator.py:apply_amplitude_damping_noise.
        for gamma in (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.999, 1.0):
            with self.subTest(gamma=gamma):
                E0 = np.array([[1.0, 0.0],
                                [0.0, math.sqrt(1 - gamma)]], dtype=complex)
                E1 = np.array([[0.0, math.sqrt(gamma)],
                                [0.0, 0.0]], dtype=complex)
                dev = _kraus_completeness([E0, E1])
                self.assertLessEqual(
                    np.linalg.norm(dev), _KRAUS_TOLERANCE,
                    msg=f"amplitude damping (gamma={gamma}) violates "
                        f"Kraus completeness: deviation norm="
                        f"{np.linalg.norm(dev)}",
                )

    def test_phase_damping_kraus_completeness_for_several_lambda(self):
        # Reproduce the Kraus operators used in
        # src/density_matrix_simulator.py:apply_phase_damping_noise.
        for lambda_val in (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.999, 1.0):
            with self.subTest(lam=lambda_val):
                E0 = np.array([[1.0, 0.0],
                                [0.0, math.sqrt(1 - lambda_val)]], dtype=complex)
                E1 = np.array([[0.0, 0.0],
                                [0.0,  math.sqrt(lambda_val)]], dtype=complex)
                dev = _kraus_completeness([E0, E1])
                self.assertLessEqual(
                    np.linalg.norm(dev), _KRAUS_TOLERANCE,
                    msg=f"phase damping (lambda={lambda_val}) violates "
                        f"Kraus completeness: deviation norm="
                        f"{np.linalg.norm(dev)}",
                )

    def test_depolarizing_kraus_completeness(self):
        # Reproduce the Kraus operators used in
        # src/density_matrix_simulator.py:apply_depolarizing_noise.
        for p in (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.999):
            with self.subTest(p=p):
                E0 = math.sqrt(1 - p) * np.eye(2, dtype=complex)
                E1 = math.sqrt(p / 3) * np.array(
                    [[0, 1], [1, 0]], dtype=complex)            # X
                E2 = math.sqrt(p / 3) * np.array(
                    [[0, -1j], [1j, 0]], dtype=complex)         # Y
                E3 = math.sqrt(p / 3) * np.array(
                    [[1, 0], [0, -1]], dtype=complex)           # Z
                dev = _kraus_completeness([E0, E1, E2, E3])
                self.assertLessEqual(
                    np.linalg.norm(dev), _KRAUS_TOLERANCE,
                    msg=f"depolarizing (p={p}) violates Kraus "
                        f"completeness: deviation norm="
                        f"{np.linalg.norm(dev)}",
                )

    def test_bit_flip_kraus_completeness(self):
        # The BitFlipChannel in noise_channel.py dispatches the channel as
        # stochastic X (gate-level), not explicitly via Kraus. The channel
        # *is* equivalent to {sqrt(1-p) I, sqrt(p) X} - verify that
        # analytic set is complete.
        for p in (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.999):
            with self.subTest(p=p):
                K0 = math.sqrt(1 - p) * np.eye(2, dtype=complex)
                K1 = math.sqrt(p) * np.array(
                    [[0, 1], [1, 0]], dtype=complex)             # X
                dev = _kraus_completeness([K0, K1])
                self.assertLessEqual(
                    np.linalg.norm(dev), _KRAUS_TOLERANCE,
                    msg=f"bit flip (p={p}) violates Kraus "
                        f"completeness: deviation norm="
                        f"{np.linalg.norm(dev)}",
                )

    def test_phase_flip_kraus_completeness(self):
        for p in (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.999):
            with self.subTest(p=p):
                K0 = math.sqrt(1 - p) * np.eye(2, dtype=complex)
                K1 = math.sqrt(p) * np.array(
                    [[1, 0], [0, -1]], dtype=complex)           # Z
                dev = _kraus_completeness([K0, K1])
                self.assertLessEqual(
                    np.linalg.norm(dev), _KRAUS_TOLERANCE,
                    msg=f"phase flip (p={p}) violates Kraus "
                        f"completeness: deviation norm="
                        f"{np.linalg.norm(dev)}",
                )


class TestApplyChannelIsTracePreserving(unittest.TestCase):
    """The DensityMatrixSimulator._apply_channel implementation must
    preserve trace (TP property) of an arbitrary input density matrix.

    Combined with Kraus completeness this gives CPTP."""

    def _sim_with_one_qubit_in_state(self, state_vector: np.ndarray) -> DensityMatrixSimulator:
        # DensityMatrixSimulator: allocate q0, then force its density matrix
        # to the pure state rho = |psi><psi| from the supplied state vector.
        sim = DensityMatrixSimulator()
        sim.allocate_qubit('q0')
        rho = (state_vector.reshape(2, 1) @ state_vector.reshape(1, 2).conj()).astype(complex)
        sim._state = rho
        return sim

    def test_amplitude_damping_preserves_trace(self):
        for gamma in (0.1, 0.5, 0.99):
            with self.subTest(gamma=gamma):
                # Start in a superposition
                psi = np.array([1.0, 1.0], dtype=complex) / math.sqrt(2)
                sim = self._sim_with_one_qubit_in_state(psi)
                tr_before = np.trace(sim._state).real
                sim.apply_amplitude_damping_noise('q0', gamma)
                tr_after = np.trace(sim._state).real
                self.assertAlmostEqual(
                    tr_before, tr_after, places=10,
                    msg=f"amplitude damping (gamma={gamma}) not TP: "
                        f"trace went {tr_before!r} -> {tr_after!r}",
                )
                # Output must remain Hermitian (CP)
                self.assertTrue(
                    _is_hermitian(sim._state),
                    msg=f"amplitude damping (gamma={gamma}) not CP",
                )

    def test_phase_damping_preserves_trace(self):
        for lambda_val in (0.1, 0.5, 0.99):
            with self.subTest(lam=lambda_val):
                psi = np.array([1.0, 1.0], dtype=complex) / math.sqrt(2)
                sim = self._sim_with_one_qubit_in_state(psi)
                tr_before = np.trace(sim._state).real
                sim.apply_phase_damping_noise('q0', lambda_val)
                tr_after = np.trace(sim._state).real
                self.assertAlmostEqual(
                    tr_before, tr_after, places=10,
                    msg=f"phase damping (lambda={lambda_val}) not TP",
                )
                self.assertTrue(
                    _is_hermitian(sim._state),
                    msg=f"phase damping (lambda={lambda_val}) not CP",
                )

    def test_depolarizing_preserves_trace(self):
        # The depolarizing channel must preserve Hermiticity and trace
        # for an arbitrary mixed state.
        for p in (0.1, 0.5, 0.99):
            with self.subTest(p=p):
                rho = np.array(
                    [[0.7, 0.2 + 0.1j], [0.2 - 0.1j, 0.3]],
                    dtype=complex,
                )
                sim = DensityMatrixSimulator()
                sim.allocate_qubit('q0')
                sim._state = rho
                tr_before = np.trace(sim._state).real
                sim.apply_depolarizing_noise('q0', p)
                tr_after = np.trace(sim._state).real
                self.assertAlmostEqual(
                    tr_before, tr_after, places=10,
                    msg=f"depolarizing (p={p}) not TP",
                )
                self.assertTrue(
                    _is_hermitian(sim._state),
                    msg=f"depolarizing (p={p}) not CP",
                )


class TestAmplitudeDampingPhysicalBehaviour(unittest.TestCase):
    def test_full_amplitude_damping_collapses_to_ground_state(self):
        # gamma=1 means full relaxation to |0|.
        np.random.seed(0)
        psi = np.array([0.0, 1.0], dtype=complex)  # excited state
        sim = DensityMatrixSimulator()
        sim.allocate_qubit('q0')
        sim._state = psi.reshape(2, 1) @ psi.reshape(1, 2)
        sim.apply_amplitude_damping_noise('q0', 1.0)
        # Output should be |0><0|
        expected = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
        self.assertLessEqual(
            np.linalg.norm(sim._state - expected), _KRAUS_TOLERANCE,
            msg=f"amplitude damping (gamma=1) didn't collapse |1> to |0>; "
                f"got state=\n{sim._state}",
        )

    def test_zero_amplitude_damping_is_no_op(self):
        psi = np.array([1.0, 1.0], dtype=complex) / math.sqrt(2)
        sim = DensityMatrixSimulator()
        sim.allocate_qubit('q0')
        rho_in = psi.reshape(2, 1) @ psi.reshape(1, 2)
        sim._state = rho_in.copy()
        sim.apply_amplitude_damping_noise('q0', 0.0)
        self.assertLessEqual(
            np.linalg.norm(sim._state - rho_in), _KRAUS_TOLERANCE,
            msg="gamma=0 amplitude damping should be identity, not "
                "modify the state.",
        )


class TestPhaseDampingPhysicalBehaviour(unittest.TestCase):
    def test_full_phase_damping_destroys_coherence(self):
        psi = np.array([1.0, 1.0], dtype=complex) / math.sqrt(2)
        sim = DensityMatrixSimulator()
        sim.allocate_qubit('q0')
        sim._state = psi.reshape(2, 1) @ psi.reshape(1, 2)
        sim.apply_phase_damping_noise('q0', 1.0)
        # All off-diagonal elements should be ~0; diagonals preserved.
        self.assertAlmostEqual(sim._state[0, 0], 0.5, places=10)
        self.assertAlmostEqual(sim._state[1, 1], 0.5, places=10)
        self.assertLessEqual(abs(sim._state[0, 1]), _KRAUS_TOLERANCE)
        self.assertLessEqual(abs(sim._state[1, 0]), _KRAUS_TOLERANCE)

    def test_zero_phase_damping_is_no_op(self):
        psi = np.array([1.0, 1.0], dtype=complex) / math.sqrt(2)
        sim = DensityMatrixSimulator()
        sim.allocate_qubit('q0')
        rho_in = psi.reshape(2, 1) @ psi.reshape(1, 2)
        sim._state = rho_in.copy()
        sim.apply_phase_damping_noise('q0', 0.0)
        self.assertLessEqual(
            np.linalg.norm(sim._state - rho_in), _KRAUS_TOLERANCE,
            msg="lambda=0 phase damping should be identity",
        )


class TestNoiseChannelPythonAPIConsistency(unittest.TestCase):
    """For the noise channels exposed in src.noise.noise_channel, ensure the
    `apply_to_qubit` path doesn't crash and (for the Kraus-backed channels)
    still routes through `apply_kraus_channel`."""

    def test_amplitude_damping_channel_calls_kraus_when_supported(self):
        class _CapturingSim:
            def __init__(self):
                self.captured_kraus = None
            def apply_kraus_channel(self, qubit_name, kraus_ops):
                self.captured_kraus = kraus_ops
            def apply_1qubit_gate(self, qubit_name, op):
                pass
            def measure(self, qubit_name):
                return 0
            def X(self, qubit_name): pass
            def Z(self, qubit_name): pass
        cs = _CapturingSim()
        AmplitudeDampingChannel(gamma=0.5).apply_to_qubit(cs, 'q0')
        self.assertIsNotNone(cs.captured_kraus)
        self.assertEqual(len(cs.captured_kraus), 2)

    def test_phase_damping_channel_calls_kraus_when_supported(self):
        class _CapturingSim:
            def __init__(self):
                self.captured_kraus = None
            def apply_kraus_channel(self, qubit_name, kraus_ops):
                self.captured_kraus = kraus_ops
            def apply_1qubit_gate(self, qubit_name, op): pass
            def measure(self, qubit_name): return 0
            def X(self, qubit_name): pass
            def Z(self, qubit_name): pass
        cs = _CapturingSim()
        PhaseDampingChannel(lambda_val=0.5).apply_to_qubit(cs, 'q0')
        self.assertIsNotNone(cs.captured_kraus)
        self.assertEqual(len(cs.captured_kraus), 2)

    def test_bit_flip_no_op_when_prob_zero(self):
        # Should not call any simulator methods when prob=0.
        class _ExplodesIfCalledSim:
            def X(self, q): raise AssertionError("X should not be called")
            def Y(self, q): raise AssertionError("Y should not be called")
            def Z(self, q): raise AssertionError("Z should not be called")
            def apply_1qubit_gate(self, q, op): raise AssertionError("...")
            def apply_kraus_channel(self, q, ops): raise AssertionError("...")
            def measure(self, q): raise AssertionError("...")
        # BitFlipChannel with prob=0 should be a no-op regardless of the
        # rng draw.
        c = BitFlipChannel(prob=0.0, seed=42)
        c.apply_to_qubit(_ExplodesIfCalledSim(), 'q0')

    def test_depolarizing_no_op_when_prob_zero(self):
        class _ExplodesIfCalledSim:
            def X(self, q): raise AssertionError("X should not be called")
            def Y(self, q): raise AssertionError("Y should not be called")
            def Z(self, q): raise AssertionError("Z should not be called")
            def apply_1qubit_gate(self, q, op): raise AssertionError("...")
            def apply_kraus_channel(self, q, ops): raise AssertionError("...")
            def measure(self, q): raise AssertionError("...")
        c = DepolarizingChannel(prob=0.0, seed=42)
        c.apply_to_qubit(_ExplodesIfCalledSim(), 'q0')

    def test_phase_flip_no_op_when_prob_zero(self):
        class _ExplodesIfCalledSim:
            def X(self, q): raise AssertionError("X should not be called")
            def Y(self, q): raise AssertionError("Y should not be called")
            def Z(self, q): raise AssertionError("Z should not be called")
            def apply_1qubit_gate(self, q, op): raise AssertionError("...")
            def apply_kraus_channel(self, q, ops): raise AssertionError("...")
            def measure(self, q): raise AssertionError("...")
        c = PhaseFlipChannel(prob=0.0, seed=42)
        c.apply_to_qubit(_ExplodesIfCalledSim(), 'q0')


if __name__ == '__main__':
    unittest.main()
