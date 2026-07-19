import math
import random
import unittest
import warnings

import numpy as np

from src.noise.noise_channel import (
    AmplitudeDampingChannel,
    BitFlipChannel,
    DepolarizingChannel,
    NoisePipeline,
    PhaseDampingChannel,
    PhaseFlipChannel,
    ReadoutError,
    ReadoutErrorChannel,
)
from src.noise.noise_model import NoiseModel


def _kraus_completeness(kraus_ops):
    n = kraus_ops[0].shape[0]
    identity = np.eye(n, dtype=complex)
    sigma = sum(k.conj().T @ k for k in kraus_ops)
    return identity - sigma


class _StubSimulator:
    def __init__(self):
        self.gate_log = []
        self.kraus_log = []

    def X(self, q):
        self.gate_log.append(("X", q))

    def Y(self, q):
        self.gate_log.append(("Y", q))

    def Z(self, q):
        self.gate_log.append(("Z", q))

    def apply_1qubit_gate(self, q, matrix):
        self.gate_log.append(("1Q", q, matrix))

    def apply_kraus_channel(self, q, kraus_ops):
        self.kraus_log.append((q, kraus_ops))


class TestAmplitudeDampingCorrectness(unittest.TestCase):
    def test_gamma_zero_does_not_modify_state(self):
        ch = AmplitudeDampingChannel(gamma=0.0)
        sim = _StubSimulator()
        ch.apply_to_qubit(sim, "q0")
        self.assertEqual(sim.gate_log, [])
        self.assertEqual(sim.kraus_log, [])

    def test_gamma_one_full_decay_uses_kraus(self):
        ch = AmplitudeDampingChannel(gamma=1.0)
        sim = _StubSimulator()
        ch.apply_to_qubit(sim, "q0")
        self.assertEqual(len(sim.kraus_log), 1)
        q, ops = sim.kraus_log[0]
        self.assertEqual(q, "q0")
        k0 = np.array(ops[0], dtype=complex)
        k1 = np.array(ops[1], dtype=complex)
        self.assertAlmostEqual(k0[1, 1], 0.0)
        self.assertAlmostEqual(k1[0, 1], 1.0)

    def test_kraus_completeness_for_several_gamma(self):
        for gamma in (0.0, 0.1, 0.25, 0.5, 0.9, 1.0):
            with self.subTest(gamma=gamma):
                k0 = np.array([[1.0, 0.0], [0.0, math.sqrt(1 - gamma)]], dtype=complex)
                k1 = np.array([[0.0, math.sqrt(gamma)], [0.0, 0.0]], dtype=complex)
                dev = _kraus_completeness([k0, k1])
                self.assertLessEqual(
                    np.linalg.norm(dev),
                    1e-10,
                    msg=f"amplitude damping gamma={gamma} violates Kraus completeness",
                )

    def test_kraus_operator_unitarity_off_diagonal(self):
        ch = AmplitudeDampingChannel(gamma=0.25)
        sim = _StubSimulator()
        ch.apply_to_qubit(sim, "q0")
        _, ops = sim.kraus_log[0]
        k1 = np.array(ops[1], dtype=complex)
        self.assertAlmostEqual(k1[0, 1].real, math.sqrt(0.25))
        self.assertAlmostEqual(k1[1, 0].real, 0.0)
        self.assertAlmostEqual(k1[1, 1].real, 0.0)
        self.assertAlmostEqual(k1[0, 0].real, 0.0)

    def test_name_property(self):
        self.assertEqual(AmplitudeDampingChannel(gamma=0.5).name, "amplitude_damping")

    def test_pair_applies_to_both_qubits(self):
        ch = AmplitudeDampingChannel(gamma=0.5)
        sim = _StubSimulator()
        ch.apply_to_pair(sim, "q0", "q1")
        self.assertEqual(len(sim.kraus_log), 2)
        self.assertEqual(sim.kraus_log[0][0], "q0")
        self.assertEqual(sim.kraus_log[1][0], "q1")


class TestPhaseDampingCorrectness(unittest.TestCase):
    def test_lambda_zero_no_op(self):
        ch = PhaseDampingChannel(lambda_val=0.0)
        sim = _StubSimulator()
        ch.apply_to_qubit(sim, "q0")
        self.assertEqual(sim.gate_log, [])
        self.assertEqual(sim.kraus_log, [])

    def test_lambda_one_dephases_completely(self):
        ch = PhaseDampingChannel(lambda_val=1.0)
        sim = _StubSimulator()
        ch.apply_to_qubit(sim, "q0")
        self.assertEqual(len(sim.kraus_log), 1)
        _, ops = sim.kraus_log[0]
        k0 = np.array(ops[0], dtype=complex)
        k1 = np.array(ops[1], dtype=complex)
        self.assertAlmostEqual(k0[1, 1], 0.0)
        self.assertAlmostEqual(k1[1, 1], 1.0)

    def test_kraus_completeness_for_several_lambda(self):
        for lam in (0.0, 0.1, 0.25, 0.5, 0.9, 1.0):
            with self.subTest(lam=lam):
                k0 = np.array([[1.0, 0.0], [0.0, math.sqrt(1 - lam)]], dtype=complex)
                k1 = np.array([[0.0, 0.0], [0.0, math.sqrt(lam)]], dtype=complex)
                dev = _kraus_completeness([k0, k1])
                self.assertLessEqual(
                    np.linalg.norm(dev),
                    1e-10,
                    msg=f"phase damping lambda={lam} violates Kraus completeness",
                )

    def test_kraus_operators_diagonal_structure(self):
        ch = PhaseDampingChannel(lambda_val=0.5)
        sim = _StubSimulator()
        ch.apply_to_qubit(sim, "q0")
        _, ops = sim.kraus_log[0]
        for k in ops:
            kn = np.array(k, dtype=complex)
            self.assertAlmostEqual(kn[0, 1], 0.0)
            self.assertAlmostEqual(kn[1, 0], 0.0)

    def test_name_property(self):
        self.assertEqual(PhaseDampingChannel(lambda_val=0.5).name, "phase_damping")


class TestT1T2Constraints(unittest.TestCase):
    def test_t2_exceeding_2_t1_triggers_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            NoiseModel(t1=10.0, t2=100.0)
            self.assertTrue(any("T2" in str(x.message) for x in w))

    def test_t2_below_2_t1_no_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            NoiseModel(t1=10.0, t2=15.0)
            self.assertFalse(any("T2" in str(x.message) for x in w))

    def test_t2_equal_2_t1_no_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            NoiseModel(t1=10.0, t2=20.0)
            self.assertFalse(any("T2" in str(x.message) for x in w))

    def test_t2_clamped_to_2_t1_when_exceeding(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nm = NoiseModel(t1=10.0, t2=100.0)
        self.assertAlmostEqual(nm.t2, 20.0)

    def test_t1_t2_must_be_positive(self):
        with self.assertRaises(ValueError):
            NoiseModel(t1=-1.0, t2=1.0)
        with self.assertRaises(ValueError):
            NoiseModel(t1=1.0, t2=-1.0)

    def test_t1_t2_stored_when_valid(self):
        nm = NoiseModel(t1=10.0, t2=18.0)
        self.assertEqual(nm.t1, 10.0)
        self.assertEqual(nm.t2, 18.0)


class TestReadoutErrorConfusion(unittest.TestCase):
    def test_identity_confusion_matrix_preserves_outcome(self):
        re_ch = ReadoutError([[1.0, 0.0], [0.0, 1.0]], rng=random.Random(0))
        outcomes = [re_ch.apply(o, rng=random.Random(7)) for o in [0, 1, 0, 1, 0]]
        self.assertEqual(outcomes, [0, 1, 0, 1, 0])

    def test_full_flip_confusion_matrix(self):
        re_ch = ReadoutError([[0.0, 1.0], [1.0, 0.0]], rng=random.Random(0))
        outcomes = [re_ch.apply(o, rng=random.Random(11)) for o in [0, 1, 0, 1]]
        self.assertEqual(outcomes, [1, 0, 1, 0])

    def test_confusion_matrix_must_be_stochastic(self):
        with self.assertRaises(ValueError):
            ReadoutError([[0.5, 0.5], [0.5, 0.4]])

    def test_confusion_matrix_rejects_non_2x2(self):
        with self.assertRaises(ValueError):
            ReadoutError([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        with self.assertRaises(ValueError):
            ReadoutError([[1.0], [0.0]])

    def test_confusion_matrix_rejects_negative_row_sum(self):
        with self.assertRaises(ValueError):
            ReadoutError([[-0.5, -0.5], [1.0, 0.0]])

    def test_statistical_distribution_with_biased_matrix(self):
        rng = random.Random(42)
        re_ch = ReadoutError([[0.9, 0.1], [0.4, 0.6]], rng=rng)
        counts = {0: 0, 1: 0}
        for _ in range(2000):
            out = re_ch.apply(0, rng=rng)
            counts[out] += 1
        p0_est = counts[0] / 2000
        self.assertLess(abs(p0_est - 0.9), 0.05)

    def test_readout_error_channel_random_flips(self):
        ch = ReadoutErrorChannel(prob=1.0)
        self.assertEqual(ch.apply_readout_noise(0), 1)
        self.assertEqual(ch.apply_readout_noise(1), 0)

    def test_readout_error_channel_zero_prob_preserves(self):
        ch = ReadoutErrorChannel(prob=0.0)
        self.assertEqual(ch.apply_readout_noise(0), 0)
        self.assertEqual(ch.apply_readout_noise(1), 1)


class TestNoiseModelMultipleChannels(unittest.TestCase):
    def test_pipeline_apply_gate_noise_runs_all_channels(self):
        sim = _StubSimulator()
        bf = BitFlipChannel(prob=1.0)
        pf = PhaseFlipChannel(prob=1.0)
        pipe = NoisePipeline(channels=[bf, pf])
        pipe.apply_gate_noise(sim, "q0")
        self.assertIn(("X", "q0"), sim.gate_log)
        self.assertIn(("Z", "q0"), sim.gate_log)

    def test_pipeline_channel_names(self):
        pipe = NoisePipeline(channels=[
            BitFlipChannel(prob=0.1),
            PhaseFlipChannel(prob=0.1),
            DepolarizingChannel(prob=0.1),
        ])
        self.assertEqual(pipe.channel_names, ["bit_flip", "phase_flip", "depolarizing"])

    def test_pipeline_apply_readout_noise_uses_readout_channel_only(self):
        bf = BitFlipChannel(prob=1.0)
        ro = ReadoutErrorChannel(prob=1.0)
        pipe = NoisePipeline(channels=[bf, ro])
        self.assertEqual(pipe.apply_readout_noise(0), 1)
        self.assertEqual(pipe.apply_readout_noise(1), 0)

    def test_pipeline_add_channel_returns_self(self):
        pipe = NoisePipeline()
        rv = pipe.add_channel(BitFlipChannel(prob=0.1))
        self.assertIs(rv, pipe)
        self.assertEqual(len(pipe.channels), 1)

    def test_pipeline_empty_does_nothing(self):
        sim = _StubSimulator()
        pipe = NoisePipeline()
        pipe.apply_gate_noise(sim, "q0")
        self.assertEqual(sim.gate_log, [])

    def test_pipeline_two_qubit_noise_applies_pair(self):
        sim = _StubSimulator()
        bf = BitFlipChannel(prob=1.0)
        pipe = NoisePipeline(channels=[bf])
        pipe.apply_two_qubit_noise(sim, "q0", "q1")
        gates = [g for g in sim.gate_log if g[0] == "X"]
        self.assertEqual(len(gates), 2)

    def test_depolarizing_channel_randomizes(self):
        sim = _StubSimulator()
        ch = DepolarizingChannel(prob=1.0, rng=random.Random(0))
        ch.apply_to_qubit(sim, "q0")
        ops = [g[0] for g in sim.gate_log]
        self.assertIn(ops[0], {"X", "Y", "Z"})

    def test_bit_flip_zero_prob_no_op(self):
        sim = _StubSimulator()
        ch = BitFlipChannel(prob=0.0)
        ch.apply_to_qubit(sim, "q0")
        self.assertEqual(sim.gate_log, [])

    def test_phase_flip_zero_prob_no_op(self):
        sim = _StubSimulator()
        ch = PhaseFlipChannel(prob=0.0)
        ch.apply_to_qubit(sim, "q0")
        self.assertEqual(sim.gate_log, [])


class TestNoiseStatisticalRuns(unittest.TestCase):
    def test_bit_flip_channel_statistics_1000_shots(self):
        zeros_when_one = 0
        for s in range(1000):
            sim = _StubSimulator()
            ch = BitFlipChannel(prob=0.5, rng=random.Random(s))
            ch.apply_to_qubit(sim, "q0")
            if ("X", "q0") in sim.gate_log:
                zeros_when_one += 1
        frac = zeros_when_one / 1000
        self.assertLess(abs(frac - 0.5), 0.06)

    def test_depolarizing_channel_distribution(self):
        counts = {"X": 0, "Y": 0, "Z": 0}
        for s in range(1500):
            sim = _StubSimulator()
            ch = DepolarizingChannel(prob=1.0, rng=random.Random(s))
            ch.apply_to_qubit(sim, "q0")
            if sim.gate_log:
                counts[sim.gate_log[0][0]] = counts.get(sim.gate_log[0][0], 0) + 1
        for op in ("X", "Y", "Z"):
            self.assertGreater(counts[op], 350)
        total = sum(counts.values())
        self.assertEqual(total, 1500)

    def test_readout_error_distribution_1000_shots(self):
        rng = random.Random(123)
        re_ch = ReadoutError([[0.8, 0.2], [0.3, 0.7]], rng=rng)
        zeros = sum(1 for _ in range(1000) if re_ch.apply(0, rng=rng) == 0)
        p0_est = zeros / 1000
        self.assertLess(abs(p0_est - 0.8), 0.05)

    def test_noise_model_seed_reproducibility(self):
        nm1 = NoiseModel(noise_type="bit_flip", noise_prob=0.3, rng=random.Random(99))
        nm2 = NoiseModel(noise_type="bit_flip", noise_prob=0.3, rng=random.Random(99))
        sim1 = _StubSimulator()
        sim2 = _StubSimulator()
        for _ in range(20):
            nm1.apply_gate_noise(sim1, "q0")
            nm2.apply_gate_noise(sim2, "q0")
        self.assertEqual(sim1.gate_log, sim2.gate_log)


if __name__ == "__main__":
    unittest.main()
