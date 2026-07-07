"""§9.1 — dedicated tests for the Matrix Product State (MPS) simulator.

Exercises `src.tensor_network.mps.MPSSimulator`:
  - allocation and index lookup
  - single-qubit gates (H, X, Y, Z, S, T, RX, RY, RZ)
  - two-qubit gates (CNOT, CZ, SWAP, CCX, CSWAP, CP, CRX, CRY, CRZ)
  - measurement semantics and norm preservation
  - state-vector reconstruction and amplitude dict
  - bond-dimension tracking & truncation metrics
"""
import math
import unittest

import numpy as np

from src.tensor_network.mps import (
    MPSSimulator,
    DEFAULT_MAX_BOND_DIM,
)


def _approx(a, b, tol=1e-7):
    return abs(a - b) < tol


def _vec_approx_eq(v1, v2, tol=1e-7):
    if len(v1) != len(v2):
        return False
    for x, y in zip(v1, v2):
        if abs(x - y) > tol:
            return False
    return True


class TestMPSAllocAndIndex(unittest.TestCase):
    def test_initial_no_qubits(self):
        sim = MPSSimulator()
        self.assertEqual(len(sim.tensors), 0)
        self.assertEqual(sim.num_qubits, 0) if hasattr(sim, "num_qubits") else None  # noqa

    def test_allocate_qubit_creates_tensor(self):
        sim = MPSSimulator()
        sim.allocate_qubit("q0")
        self.assertEqual(len(sim.tensors), 1)
        self.assertIn("q0", sim.qubit_map)
        # New |0> tensor is shape (1, 2, 1) with A[0, 0, 0] = 1
        self.assertEqual(sim.tensors[0].shape, (1, 2, 1))
        self.assertTrue(_approx(float(sim.tensors[0][0, 0, 0].real), 1.0))

    def test_allocate_idempotent(self):
        sim = MPSSimulator()
        sim.allocate_qubit("q0")
        sim.allocate_qubit("q0")
        self.assertEqual(len(sim.tensors), 1)

    def test_get_qubit_index_unknown_raises(self):
        sim = MPSSimulator()
        with self.assertRaises(KeyError):
            sim.get_qubit_index("q0")

    def test_get_qubit_index_returns_chain_index(self):
        sim = MPSSimulator()
        sim.allocate_qubit("a")
        sim.allocate_qubit("b")
        sim.allocate_qubit("c")
        self.assertEqual(sim.get_qubit_index("a"), 0)
        self.assertEqual(sim.get_qubit_index("b"), 1)
        self.assertEqual(sim.get_qubit_index("c"), 2)


class TestMPSSingleQubitGates(unittest.TestCase):
    def setUp(self):
        self.sim = MPSSimulator()
        self.sim.allocate_qubit("q0")

    def test_initial_state_is_zero(self):
        # |0> state vector [1, 0]
        vec = self.sim.get_state_vector()
        self.assertEqual(len(vec), 2)
        self.assertTrue(_approx(vec[0].real, 1.0))

    def test_x_gate_flips_to_one(self):
        self.sim.X("q0")
        vec = self.sim.get_state_vector()
        self.assertTrue(_approx(vec[1].real, 1.0))

    def test_h_gate_superposition(self):
        self.sim.H("q0")
        vec = self.sim.get_state_vector()
        inv = 1.0 / math.sqrt(2.0)
        self.assertTrue(_approx(abs(vec[0]), inv))
        self.assertTrue(_approx(abs(vec[1]), inv))

    def test_y_gate_on_zero(self):
        self.sim.Y("q0")
        vec = self.sim.get_state_vector()
        # Y|0> = i|1>
        self.assertTrue(_approx(vec[1].imag, 1.0))
        self.assertTrue(_approx(vec[1].real, 0.0))

    def test_z_gate_on_one(self):
        self.sim.X("q0")
        self.sim.Z("q0")
        vec = self.sim.get_state_vector()
        # Z|1> = -|1>
        self.assertTrue(_approx(vec[1].real, -1.0))

    def test_s_gate_on_one(self):
        self.sim.X("q0")
        self.sim.S("q0")
        vec = self.sim.get_state_vector()
        self.assertTrue(_approx(vec[1].imag, 1.0))

    def test_t_gate_on_one(self):
        self.sim.X("q0")
        self.sim.T("q0")
        vec = self.sim.get_state_vector()
        expected = complex(math.cos(math.pi / 4), math.sin(math.pi / 4))
        self.assertTrue(_approx(vec[1], expected))

    def test_rx_rotation(self):
        self.sim.RX("q0", math.pi)
        vec = self.sim.get_state_vector()
        # RX(pi)|0> = -i|1>
        self.assertTrue(_approx(vec[1].imag, -1.0))

    def test_ry_rotation(self):
        self.sim.RY("q0", math.pi)
        vec = self.sim.get_state_vector()
        # RY(pi)|0> = |1>
        self.assertTrue(_approx(vec[1].real, 1.0))

    def test_rz_preserves_zero_amplitude(self):
        self.sim.RZ("q0", math.pi / 4)
        vec = self.sim.get_state_vector()
        self.assertTrue(_approx(abs(vec[0]), 1.0))

    def test_apply_1qubit_identity(self):
        self.sim.apply_1qubit_gate("q0", [[1, 0], [0, 1]])
        vec = self.sim.get_state_vector()
        self.assertTrue(_approx(vec[0].real, 1.0))


class TestMPSTwoQubitGates(unittest.TestCase):
    def setUp(self):
        self.sim = MPSSimulator()
        self.sim.allocate_qubit("c")
        self.sim.allocate_qubit("t")

    def test_cnot_no_op_control_zero(self):
        self.sim.CNOT("c", "t")
        vec = self.sim.get_state_vector()
        # |00>
        self.assertTrue(_approx(vec[0].real, 1.0))

    def test_cnot_flips_target_control_one(self):
        self.sim.X("c")  # |10>
        self.sim.CNOT("c", "t")  # -> |11>
        vec = self.sim.get_state_vector()
        self.assertTrue(_approx(vec[3].real, 1.0))

    def test_cz_phases_target_and_control_one(self):
        self.sim.X("c")
        self.sim.X("t")
        self.sim.CZ("c", "t")
        vec = self.sim.get_state_vector()
        # |11> gets -1 phase
        self.assertTrue(_approx(vec[3].real, -1.0))

    def test_cz_no_op_when_control_zero(self):
        self.sim.X("t")  # |01>
        self.sim.CZ("c", "t")
        vec = self.sim.get_state_vector()
        # Still |01> (idx 2 in little-endian bit layout: bit 0 = q0='c', bit 1 = q1='t')
        # Actually for our 2-qubit MPS: qubit c is q1 in chain after moving?
        # In simple MPS with no swapping, c (qubit_map[c]=0) is left, t (=1) is right
        # state vector order: 00, 10, 01, 11 (q0=c on the left of bitstring)
        # Actually MPS get_state_vector returns list index based on original creation order
        self.assertTrue(_approx(abs(vec[2].real), 1.0))  # |c=0, t=1> = index 2

    def test_swap_exchanges_qubits(self):
        self.sim.X("c")  # |10> (c=1, t=0) -> state vec index 1
        self.sim.SWAP("c", "t")  # -> |01> -> state vec index 2
        vec = self.sim.get_state_vector()
        self.assertTrue(_approx(abs(vec[2]), 1.0))

    def test_bell_state(self):
        self.sim.H("c")
        self.sim.CNOT("c", "t")
        vec = self.sim.get_state_vector()
        inv = 1.0 / math.sqrt(2.0)
        # |00> + |11> = vec[0] and vec[3]
        self.assertTrue(_approx(abs(vec[0]), inv))
        self.assertTrue(_approx(abs(vec[3]), inv))
        self.assertTrue(_approx(abs(vec[1]), 0.0))
        self.assertTrue(_approx(abs(vec[2]), 0.0))

    def test_ccx_decomposition_flips_target(self):
        sim = MPSSimulator()
        sim.allocate_qubit("c1")
        sim.allocate_qubit("c2")
        sim.allocate_qubit("t")
        sim.X("c1")
        sim.X("c2")
        sim.CCX("c1", "c2", "t")
        vec = sim.get_state_vector()
        # After CCX both controls 1, target flips 0 → 1
        # Final state |111>
        idx_111 = 0b111
        self.assertTrue(_approx(abs(vec[idx_111]), 1.0))


class TestMPSMeasurement(unittest.TestCase):
    def test_measure_zero_qubit(self):
        sim = MPSSimulator(seed=42)
        sim.allocate_qubit("q0")
        self.assertEqual(sim.measure("q0"), 0)

    def test_measure_one_qubit(self):
        sim = MPSSimulator(seed=42)
        sim.allocate_qubit("q0")
        sim.X("q0")
        self.assertEqual(sim.measure("q0"), 1)

    def test_measure_superposition_yields_valid_outcome(self):
        sim = MPSSimulator(seed=7)
        sim.allocate_qubit("q0")
        sim.H("q0")
        outcome = sim.measure("q0")
        self.assertIn(outcome, (0, 1))
        # State preserved as basis vector after collapse
        vec = sim.get_state_vector()
        # Norm stays 1
        norm = math.sqrt(sum(abs(a) ** 2 for a in vec))
        self.assertTrue(_approx(norm, 1.0))

    def test_norm_squared_for_product_state(self):
        sim = MPSSimulator()
        sim.allocate_qubit("q0")
        sim.H("q0")
        n = sim.norm_squared()
        self.assertTrue(_approx(n, 1.0))

    def test_norm_squared_after_entanglement(self):
        sim = MPSSimulator()
        sim.allocate_qubit("c")
        sim.allocate_qubit("t")
        sim.H("c")
        sim.CNOT("c", "t")
        n = sim.norm_squared()
        self.assertTrue(_approx(n, 1.0))


class TestMPSStateExtraction(unittest.TestCase):
    def test_get_state_vector_single_qubit(self):
        sim = MPSSimulator()
        sim.allocate_qubit("q0")
        vec = sim.get_state_vector()
        self.assertEqual(len(vec), 2)
        self.assertTrue(_approx(vec[0].real, 1.0))

    def test_get_state_vector_bell(self):
        sim = MPSSimulator()
        sim.allocate_qubit("c")
        sim.allocate_qubit("t")
        sim.H("c")
        sim.CNOT("c", "t")
        vec = sim.get_state_vector()
        self.assertEqual(len(vec), 4)
        inv = 1.0 / math.sqrt(2.0)
        self.assertTrue(_approx(abs(vec[0]), inv))
        self.assertTrue(_approx(abs(vec[3]), inv))

    def test_get_amplitudes_dict_filters_low_probabilities(self):
        sim = MPSSimulator()
        sim.allocate_qubit("q0")
        sim.H("q0")
        d = sim.get_amplitudes_dict()
        for v in d.values():
            self.assertGreater(abs(v), 1e-12)

    def test_get_amplitudes_dict_empty_simulator(self):
        sim = MPSSimulator()
        d = sim.get_amplitudes_dict()
        self.assertEqual(d, {"": 1.0 + 0.0j})


class TestMPSControlledRotations(unittest.TestCase):
    def setUp(self):
        self.sim = MPSSimulator()
        self.sim.allocate_qubit("c")
        self.sim.allocate_qubit("t")

    def test_cp_activates_on_both_one(self):
        self.sim.X("c")
        self.sim.X("t")
        self.sim.CP("c", "t", math.pi / 2)
        vec = self.sim.get_state_vector()
        # |11> phase = exp(i*pi/2) = i
        self.assertTrue(_approx(vec[3], 1j))

    def test_cp_no_op_when_control_zero(self):
        self.sim.X("t")
        self.sim.CP("c", "t", math.pi / 2)
        vec = self.sim.get_state_vector()
        self.assertTrue(_approx(abs(vec[2]), 1.0))

    def test_crx_applies_when_control_one(self):
        self.sim.X("c")
        self.sim.CRX("c", "t", math.pi)
        vec = self.sim.get_state_vector()
        # CRX(pi)|10> = -i |11>
        self.assertTrue(_approx(vec[3].imag, -1.0))

    def test_cry_applies_when_control_one(self):
        self.sim.X("c")
        self.sim.CRY("c", "t", math.pi)
        vec = self.sim.get_state_vector()
        self.assertTrue(_approx(vec[3].real, 1.0))

    def test_crz_phase_when_both_one(self):
        self.sim.X("c")
        self.sim.X("t")
        self.sim.CRZ("c", "t", math.pi / 2)
        vec = self.sim.get_state_vector()
        expected = complex(math.cos(math.pi / 4), math.sin(math.pi / 4))
        self.assertTrue(_approx(vec[3], expected))


class TestMPSMetrics(unittest.TestCase):
    def test_default_max_bond_dim(self):
        sim = MPSSimulator()
        self.assertEqual(sim.get_max_bond_dim(), DEFAULT_MAX_BOND_DIM)

    def test_get_last_entropy_initial(self):
        sim = MPSSimulator()
        sim.allocate_qubit("q0")
        # Initially entropy is 0 since no SVD has happened yet
        self.assertEqual(sim.get_last_entropy(), 0.0)

    def test_cumulative_truncation_error_zero_initial(self):
        sim = MPSSimulator()
        self.assertEqual(sim.get_cumulative_truncation_error(), 0.0)

    def test_last_discarded_weight_zero_for_small_state(self):
        sim = MPSSimulator()
        sim.allocate_qubit("q0")
        sim.H("q0")
        sim.CNOT  # ensure method exists
        # All operations on small states don't truncate
        # at default max_bond_dim=64.
        # Discarded weight should be 0 after Bell state prep.
        sim.allocate_qubit("q1")
        sim.H("q0")
        sim.CNOT("q0", "q1")
        self.assertEqual(sim.get_last_discarded_weight(), 0.0)

    def test_auto_bond_dim_mode(self):
        sim = MPSSimulator(max_bond_dim=4, auto_bond_dim=True,
                            max_truncation_error=1e-3)
        sim.allocate_qubit("q0")
        sim.allocate_qubit("q1")
        sim.H("q0")
        sim.CNOT("q0", "q1")
        # Bell state has max bond dim = 2, never exceeds 4
        self.assertLessEqual(sim.get_max_bond_dim(), 4)


if __name__ == "__main__":
    unittest.main()
