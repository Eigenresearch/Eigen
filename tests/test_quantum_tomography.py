"""§12.1 part 2 — Quantum tomography + error mitigation tests."""
import math
import types
import unittest

from src.quantum_tomography import (
    StateTomographyResult,
    state_tomography,
    ProcessTomographyResult,
    process_tomography,
    zero_noise_extrapolation,
    probabilistic_error_cancellation,
    m3_measurement_mitigation,
)


def _mock_sim(state_vector):
    return types.SimpleNamespace(state_vector=list(state_vector))


# ---------------------------------------------------------------------------
# State tomography
# ---------------------------------------------------------------------------

class TestStateTomographyResultDefaults(unittest.TestCase):
    def test_defaults(self):
        r = StateTomographyResult(rho=[[1 + 0j, 0j], [0j, 0j]])
        self.assertEqual(r.fidelity, -1.0)
        self.assertEqual(r.sample_count, 0)


class TestStateTomographySingleQubit(unittest.TestCase):
    def test_zero_state_diagonal_rho(self):
        r = state_tomography(_mock_sim([1 + 0j, 0 + 0j]))
        self.assertEqual(len(r.rho), 2)
        self.assertAlmostEqual(r.rho[0][0].real, 1.0, places=6)
        self.assertAlmostEqual(r.rho[1][1].real, 0.0, places=6)
        self.assertEqual(r.sample_count, 1000)

    def test_one_state_diagonal_rho(self):
        r = state_tomography(_mock_sim([0 + 0j, 1 + 0j]))
        self.assertAlmostEqual(r.rho[0][0].real, 0.0, places=6)
        self.assertAlmostEqual(r.rho[1][1].real, 1.0, places=6)

    def test_plus_state_off_diagonal(self):
        inv = 1 / math.sqrt(2)
        r = state_tomography(_mock_sim([inv + 0j, inv + 0j]))
        self.assertAlmostEqual(r.rho[0][0].real, 0.5, places=6)
        self.assertAlmostEqual(r.rho[1][1].real, 0.5, places=6)
        self.assertAlmostEqual(r.rho[0][1].real, 0.5, places=6)
        self.assertAlmostEqual(r.rho[1][0].real, 0.5, places=6)

    def test_minus_i_state_off_diagonal(self):
        # state = (|0> - i|1>) / sqrt(2)
        inv = 1 / math.sqrt(2)
        r = state_tomography(_mock_sim([inv + 0j, -1j * inv]))
        # rho should have rho[0][1] = 0.5*(<X> - i<Y>) where <X>=0, <Y>=-1
        # => rho[0][1] = 0.5*(0 - i*(-1)) = 0.5i
        self.assertAlmostEqual(r.rho[0][1].imag, 0.5, places=6)
        self.assertAlmostEqual(r.rho[1][0].imag, -0.5, places=6)

    def test_fidelity_matches_target(self):
        inv = 1 / math.sqrt(2)
        r = state_tomography(_mock_sim([inv + 0j, inv + 0j]),
                              target_state=[inv, inv])
        self.assertAlmostEqual(r.fidelity, 1.0, places=6)

    def test_fidelity_zero_for_orthogonal_target(self):
        r = state_tomography(_mock_sim([1 + 0j, 0j]),
                              target_state=[0j, 1 + 0j])
        self.assertAlmostEqual(r.fidelity, 0.0, places=6)

    def test_no_target_returns_minus_one_fidelity(self):
        r = state_tomography(_mock_sim([1 + 0j, 0j]))
        self.assertEqual(r.fidelity, -1.0)

    def test_custom_shots_recorded(self):
        r = state_tomography(_mock_sim([1 + 0j, 0j]), shots_per_basis=42)
        self.assertEqual(r.sample_count, 42)


class TestStateTomographyMultiQubit(unittest.TestCase):
    def test_two_qubit_diagonal_rho(self):
        sim = _mock_sim([1 + 0j, 0, 0, 0])  # |00>
        r = state_tomography(sim)
        self.assertEqual(len(r.rho), 4)
        self.assertAlmostEqual(r.rho[0][0].real, 1.0, places=6)
        for i in range(4):
            for j in range(4):
                if i != j:
                    self.assertAlmostEqual(r.rho[i][j], 0, places=6)

    def test_two_qubit_fidelity_with_target(self):
        sim = _mock_sim([1 + 0j, 0, 0, 0])
        target = [1 + 0j, 0, 0, 0]
        r = state_tomography(sim, target_state=target)
        self.assertAlmostEqual(r.fidelity, 1.0, places=6)

    def test_empty_state_returns_empty_rho(self):
        r = state_tomography(_mock_sim([]))
        self.assertEqual(r.rho, [])
        self.assertEqual(r.fidelity, -1.0)
        self.assertEqual(r.sample_count, 0)

    def test_non_power_of_two_dimension_returns_empty(self):
        # 3-element vector doesn't factor into 2^n
        inv = 1 / math.sqrt(3)
        r = state_tomography(_mock_sim([inv, inv, inv]))
        self.assertEqual(r.rho, [])
        self.assertEqual(r.fidelity, -1.0)


# ---------------------------------------------------------------------------
# Process tomography
# ---------------------------------------------------------------------------

class TestProcessTomographyResultDataclass(unittest.TestCase):
    def test_fields_preserved(self):
        r = ProcessTomographyResult(
            chi=[[1 + 0j]],
            process_matrix=[[1 + 0j]],
            dimension=2,
        )
        self.assertEqual(r.dimension, 2)
        self.assertEqual(r.chi, [[1 + 0j]])
        self.assertEqual(r.process_matrix, [[1 + 0j]])


class TestProcessTomographyChi(unittest.TestCase):
    def test_dimension_below_2_raises(self):
        with self.assertRaises(ValueError):
            process_tomography(1, lambda v: list(v))

    def _flatten_chi(self, chi):
        return [(m, n, chi[m][n]) for m in range(4) for n in range(4)]

    def test_identity_channel(self):
        r = process_tomography(2, lambda v: list(v))
        self.assertEqual(len(r.chi), 4)
        self.assertAlmostEqual(r.chi[0][0].real, 1.0, places=6)
        for m, n, val in self._flatten_chi(r.chi):
            if (m, n) != (0, 0):
                self.assertAlmostEqual(abs(val), 0, places=6)
        self.assertEqual(r.dimension, 2)

    def test_x_channel(self):
        r = process_tomography(2, lambda v: [v[1], v[0]])
        self.assertAlmostEqual(r.chi[1][1].real, 1.0, places=6)
        for m, n, val in self._flatten_chi(r.chi):
            if (m, n) != (1, 1):
                self.assertAlmostEqual(abs(val), 0, places=6)

    def test_y_channel(self):
        r = process_tomography(2, lambda v: [-1j * v[1], 1j * v[0]])
        self.assertAlmostEqual(r.chi[2][2].real, 1.0, places=6)
        for m, n, val in self._flatten_chi(r.chi):
            if (m, n) != (2, 2):
                self.assertAlmostEqual(abs(val), 0, places=6)

    def test_z_channel(self):
        r = process_tomography(2, lambda v: [v[0], -v[1]])
        self.assertAlmostEqual(r.chi[3][3].real, 1.0, places=6)
        for m, n, val in self._flatten_chi(r.chi):
            if (m, n) != (3, 3):
                self.assertAlmostEqual(abs(val), 0, places=6)

    def test_h_channel_chi_entries(self):
        inv = math.sqrt(2) / 2

        def h(v):
            return [inv * (v[0] + v[1]), inv * (v[0] - v[1])]

        r = process_tomography(2, h)
        for m, n, val in self._flatten_chi(r.chi):
            if (m, n) in {(1, 1), (3, 3), (1, 3), (3, 1)}:
                self.assertAlmostEqual(val.real, 0.5, places=6,
                                          msg=f"chi[{m}][{n}] should be 0.5")
            else:
                self.assertAlmostEqual(abs(val), 0, places=6,
                                          msg=f"chi[{m}][{n}] should be 0")

    def test_process_matrix_is_U_for_x_channel(self):
        r = process_tomography(2, lambda v: [v[1], v[0]])
        # X matrix = [[0, 1], [1, 0]]
        self.assertEqual(r.process_matrix, [[0 + 0j, 1 + 0j],
                                               [1 + 0j, 0 + 0j]])

    def test_process_matrix_is_U_for_identity(self):
        r = process_tomography(2, lambda v: list(v))
        self.assertEqual(r.process_matrix, [[1 + 0j, 0 + 0j],
                                               [0 + 0j, 1 + 0j]])

    def test_d_greater_than_two_returns_identity_stub(self):
        r = process_tomography(4, lambda v: list(v))
        self.assertEqual(len(r.chi), 16)
        self.assertEqual(len(r.chi[0]), 16)
        for i in range(16):
            self.assertAlmostEqual(r.chi[i][i].real, 1.0, places=6)


# ---------------------------------------------------------------------------
# Zero-noise extrapolation
# ---------------------------------------------------------------------------

class TestZeroNoiseExtrapolation(unittest.TestCase):
    def test_linear_fit_intercept(self):
        # f(s) = 3 + 2*s -> ZNE at s=0 is 3
        r = zero_noise_extrapolation(
            lambda s: 3 + 2 * s,
            noise_scales=[1.0, 2.0, 3.0],
            fit="linear",
        )
        self.assertAlmostEqual(r.mitigated_value, 3.0, places=6)
        self.assertEqual(r.fit_degree, 1)
        self.assertEqual(r.noisy_values, [5.0, 7.0, 9.0])

    def test_linear_fit_default_scales(self):
        r = zero_noise_extrapolation(lambda s: 1.5 - 0.5 * s)
        self.assertAlmostEqual(r.mitigated_value, 1.5, places=6)
        self.assertEqual(r.noise_scales, [1.0, 2.0, 3.0])

    def test_quadratic_fit_intercept(self):
        # f(s) = 1 + 2*s + 3*s^2 -> ZNE at s=0 is 1
        r = zero_noise_extrapolation(
            lambda s: 1 + 2 * s + 3 * s * s,
            noise_scales=[1.0, 2.0, 3.0],
            fit="quadratic",
        )
        self.assertAlmostEqual(r.mitigated_value, 1.0, places=6)
        self.assertEqual(r.fit_degree, 2)

    def test_exponential_fit_constant_offset(self):
        # f(s) = 2 + exp(-s) -> ZNE at s=0 should be ~2 since exp(0)=1, but
        # the fit assumes c=1, so we get the bias-corrected offset
        r = zero_noise_extrapolation(
            lambda s: 2 + math.exp(-s),
            noise_scales=[1.0, 2.0, 3.0],
            fit="exponential",
        )
        self.assertAlmostEqual(r.mitigated_value, 2.0, places=2)
        self.assertEqual(r.fit_degree, -1)

    def test_unknown_fit_kind_raises(self):
        with self.assertRaises(ValueError):
            zero_noise_extrapolation(lambda s: s, fit="cubic")

    def test_results_dataclass_fields(self):
        r = zero_noise_extrapolation(lambda s: s, noise_scales=[1.0, 2.0])
        self.assertEqual(len(r.noise_scales), 2)
        self.assertEqual(r.noisy_values, [1.0, 2.0])

    def test_too_few_points_for_quadratic_raises(self):
        with self.assertRaises(ValueError):
            zero_noise_extrapolation(
                lambda s: s,
                noise_scales=[1.0],
                fit="quadratic",
            )


# ---------------------------------------------------------------------------
# Probabilistic Error Cancellation (PEC)
# ---------------------------------------------------------------------------

class TestProbabilisticErrorCancellation(unittest.TestCase):
    def test_no_corrections_returns_noisy(self):
        r = probabilistic_error_cancellation(1.5, [])
        self.assertAlmostEqual(r.mitigated_value, 1.5, places=6)
        self.assertEqual(r.sample_count, 0)
        self.assertEqual(r.quasi_probabilities, [])

    def test_single_positive_correction(self):
        r = probabilistic_error_cancellation(1.0, [(0.1, "X", "X")])
        # 1.0 + 0.1 * 1.0 = 1.1
        self.assertAlmostEqual(r.mitigated_value, 1.1, places=6)
        self.assertEqual(r.sample_count, 1)

    def test_multiple_corrections(self):
        corrections = [
            (0.1, "X", "X"),
            (-0.2, "Z", "Z"),
            (0.05, "I", "X"),
        ]
        r = probabilistic_error_cancellation(1.0, corrections)
        # 1.0 * (1 + 0.1 - 0.2 + 0.05) = 0.95
        self.assertAlmostEqual(r.mitigated_value, 0.95, places=6)
        self.assertEqual(r.sample_count, 3)
        self.assertEqual(r.quasi_probabilities, corrections)

    def test_zero_correction_returns_noisy(self):
        r = probabilistic_error_cancellation(0.5, [(0.0, "I", "I")])
        self.assertAlmostEqual(r.mitigated_value, 0.5, places=6)

    def test_negative_noisy_value_propagated(self):
        r = probabilistic_error_cancellation(-2.0, [(0.5, "X", "X")])
        # -2.0 + 0.5*(-2.0) = -3.0
        self.assertAlmostEqual(r.mitigated_value, -3.0, places=6)


# ---------------------------------------------------------------------------
# M3 measurement mitigation
# ---------------------------------------------------------------------------

class TestM3MeasurementMitigation(unittest.TestCase):
    def test_identity_calibration_returns_noisy(self):
        cal = [[1.0, 0.0], [0.0, 1.0]]
        counts = {"0": 80, "1": 20}
        r = m3_measurement_mitigation(counts, cal)
        self.assertAlmostEqual(r.mitigated_counts["0"], 80.0, places=6)
        self.assertAlmostEqual(r.mitigated_counts["1"], 20.0, places=6)
        self.assertEqual(r.noisy_counts, counts)
        self.assertEqual(r.calibration_matrix, cal)

    def test_known_inverse_recovery(self):
        # M = [[0.9, 0.1], [0.1, 0.9]]
        # If true counts = [80, 20], noisy = [0.9*80+0.1*20, 0.1*80+0.9*20]
        # = [74, 26]
        cal = [[0.9, 0.1], [0.1, 0.9]]
        noisy = {"0": 74, "1": 26}
        r = m3_measurement_mitigation(noisy, cal)
        self.assertAlmostEqual(r.mitigated_counts["0"], 80.0, places=4)
        self.assertAlmostEqual(r.mitigated_counts["1"], 20.0, places=4)

    def test_shape_mismatch_returns_noisy_unchanged(self):
        cal = [[1.0, 0.0]]  # 1x2 not matching 2 keys
        noisy = {"0": 50, "1": 50}
        r = m3_measurement_mitigation(noisy, cal)
        self.assertEqual(r.mitigated_counts, {"0": 50.0, "1": 50.0})

    def test_three_outcome_identity(self):
        cal = [[1.0, 0.0, 0.0],
               [0.0, 1.0, 0.0],
               [0.0, 0.0, 1.0]]
        counts = {"0": 10, "1": 20, "2": 30}
        r = m3_measurement_mitigation(counts, cal)
        self.assertAlmostEqual(r.mitigated_counts["0"], 10.0, places=6)
        self.assertAlmostEqual(r.mitigated_counts["1"], 20.0, places=6)
        self.assertAlmostEqual(r.mitigated_counts["2"], 30.0, places=6)

    def test_negative_mitigated_clamped_to_zero(self):
        # noisy = [95, 5] with cal [[0.9, 0.1], [0.1, 0.9]] would give a
        # negative mitigated count for "1" - we clamp via max(0, v).
        cal = [[0.9, 0.1], [0.1, 0.9]]
        noisy = {"0": 95, "1": 5}
        r = m3_measurement_mitigation(noisy, cal)
        self.assertGreaterEqual(r.mitigated_counts["0"], 0.0)
        self.assertEqual(r.mitigated_counts["1"], 0.0)

    def test_round_trip_preserves_noisy(self):
        # If mitigated = M^(-1) * noisy, applying M back should give noisy
        cal = [[0.85, 0.15], [0.15, 0.85]]
        # True = [100, 30], noisy = M @ true
        true = [100, 30]
        noisy_vec = [sum(cal[i][j] * true[j] for j in range(2))
                      for i in range(2)]
        r = m3_measurement_mitigation(
            {"0": noisy_vec[0], "1": noisy_vec[1]},
            cal,
        )
        self.assertAlmostEqual(r.mitigated_counts["0"], 100.0, places=3)
        self.assertAlmostEqual(r.mitigated_counts["1"], 30.0, places=3)


if __name__ == "__main__":
    unittest.main()
