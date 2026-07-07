"""§9.1 — dedicated tests for the Sparse Quantum Simulator.

Exercises `src.sparse_simulator.SparseQuantumSimulator`:
  - allocation and index lookup
  - single-qubit gates (H, X, Y, Z, S, T, RX, RY, RZ)
  - two-qubit gates (CNOT, CZ, SWAP, CCX, CSWAP, CP, CRX, CRY, CRZ)
  - measurement semantics and collapse
  - state-vector reconstruction and amplitude dict
"""
import math
import unittest

from src.sparse_simulator import SparseQuantumSimulator


def _approx_eq(a, b, tol=1e-10):
    return abs(a - b) < tol


class TestSparseAllocAndIndex(unittest.TestCase):
    def test_initial_state_is_empty_zero(self):
        sim = SparseQuantumSimulator()
        self.assertEqual(sim.num_qubits, 0)

    def test_allocate_increments_count(self):
        sim = SparseQuantumSimulator()
        sim.allocate_qubit("q0")
        sim.allocate_qubit("q1")
        self.assertEqual(sim.num_qubits, 2)
        self.assertEqual(sim.qubit_map, {"q0": 0, "q1": 1})

    def test_allocate_idempotent(self):
        sim = SparseQuantumSimulator()
        sim.allocate_qubit("q0")
        sim.allocate_qubit("q0")
        self.assertEqual(sim.num_qubits, 1)

    def test_get_qubit_index_unknown_raises(self):
        sim = SparseQuantumSimulator()
        with self.assertRaises(KeyError):
            sim.get_qubit_index("q0")

    def test_get_qubit_index_returns_allocation_index(self):
        sim = SparseQuantumSimulator()
        sim.allocate_qubit("a")
        sim.allocate_qubit("b")
        self.assertEqual(sim.get_qubit_index("a"), 0)
        self.assertEqual(sim.get_qubit_index("b"), 1)


class TestSparseSingleQubitGates(unittest.TestCase):
    def setUp(self):
        self.sim = SparseQuantumSimulator()
        self.sim.allocate_qubit("q0")

    def test_default_state_is_zero(self):
        st = self.sim.state
        # Bitstring "0" holds the |0> amplitude
        self.assertTrue(_approx_eq(abs(st["0"]), 1.0))

    def test_x_gate_flips_zero_to_one(self):
        self.sim.X("q0")
        st = self.sim.state
        # After X the |1> amplitude is populated and |0> zero
        self.assertIn("1", st)
        self.assertTrue(_approx_eq(abs(st["1"]), 1.0))

    def test_h_gate_creates_superposition(self):
        self.sim.H("q0")
        st = self.sim.state
        # Both amplitudes have magnitude 1/sqrt(2)
        inv = 1.0 / math.sqrt(2.0)
        self.assertTrue(_approx_eq(abs(st["0"]), inv))
        self.assertTrue(_approx_eq(abs(st["1"]), inv))

    def test_z_gate_phases_one(self):
        self.sim.X("q0")  # |1>
        self.sim.Z("q0")
        st = self.sim.state
        self.assertTrue(_approx_eq(st["1"], -1.0))

    def test_s_gate_phases_one_to_i(self):
        self.sim.X("q0")  # |1>
        self.sim.S("q0")
        st = self.sim.state
        self.assertTrue(_approx_eq(st["1"].imag, 1.0))
        self.assertTrue(_approx_eq(st["1"].real, 0.0))

    def test_t_gate_phases_one_to_pi_over_8(self):
        self.sim.X("q0")  # |1>
        self.sim.T("q0")
        st = self.sim.state
        expected = complex(math.cos(math.pi / 4), math.sin(math.pi / 4))
        self.assertTrue(_approx_eq(st["1"], expected))

    def test_y_gate_phases_correctly(self):
        self.sim.Y("q0")
        st = self.sim.state
        # Y|0> = i|1> => imaginary unit * |1>
        self.assertTrue(_approx_eq(st["1"].imag, 1.0))
        self.assertTrue(_approx_eq(st["1"].real, 0.0))

    def test_rx_gate_rotation(self):
        self.sim.RX("q0", math.pi)
        st = self.sim.state
        # RX(pi)|0> = -i|1>
        self.assertTrue(_approx_eq(st["1"].imag, -1.0))

    def test_ry_gate_rotation(self):
        self.sim.RY("q0", math.pi)
        st = self.sim.state
        # RY(pi)|0> = |1>
        self.assertTrue(_approx_eq(st["1"].real, 1.0))

    def test_rz_leaves_zero_state_unchanged(self):
        self.sim.RZ("q0", math.pi)
        st = self.sim.state
        # RZ(theta)|0> = exp(-i theta/2) |0>; magnitude unchanged
        self.assertTrue(_approx_eq(abs(st["0"]), 1.0))

    def test_apply_1qubit_gate_with_arbitrary_matrix(self):
        self.sim.apply_1qubit_gate("q0",
                                      [[1, 0], [0, 1]])  # identity
        st = self.sim.state
        self.assertTrue(_approx_eq(abs(st["0"]), 1.0))


class TestSparseTwoQubitGates(unittest.TestCase):
    def setUp(self):
        self.sim = SparseQuantumSimulator()
        self.sim.allocate_qubit("c")
        self.sim.allocate_qubit("t")

    def test_cnot_no_op_when_control_zero(self):
        self.sim.CNOT("c", "t")
        st = self.sim.state
        # |00> stays |00>
        self.assertTrue(_approx_eq(abs(sum(st.get(k, 0)
                                              for k in st)), 1.0))

    def test_cnot_flips_target_when_control_one(self):
        self.sim.X("c")  # |10>
        self.sim.CNOT("c", "t")  # -> |11>
        st = self.sim.state
        bitstring = "11"
        self.assertTrue(_approx_eq(abs(st[bitstring]), 1.0))

    def test_cz_phases_target_and_control_one(self):
        self.sim.X("c")
        self.sim.X("t")  # |11>
        self.sim.CZ("c", "t")
        st = self.sim.state
        self.assertTrue(_approx_eq(st["11"], -1.0))

    def test_cz_no_op_when_control_zero(self):
        self.sim.X("t")  # |01>
        self.sim.CZ("c", "t")
        st = self.sim.state
        self.assertTrue(_approx_eq(st["01"], 1.0))

    def test_swap_exchanges_two_qubits(self):
        self.sim.X("c")  # |10>
        self.sim.SWAP("c", "t")  # |01>
        st = self.sim.state
        self.assertTrue(_approx_eq(abs(st["01"]), 1.0))

    def test_bell_state_prep(self):
        self.sim.H("c")  # superposition
        self.sim.CNOT("c", "t")
        st = self.sim.state
        inv = 1.0 / math.sqrt(2.0)
        # Bell state: |00> + |11>, alpha = beta = 1/sqrt(2)
        keys = {k for k in st if abs(st[k]) > 1e-9}
        self.assertEqual(keys, {"00", "11"})
        self.assertTrue(_approx_eq(abs(st["00"]), inv))
        self.assertTrue(_approx_eq(abs(st["11"]), inv))

    def test_ccx_flips_target_on_two_controls(self):
        self.sim.X("c")
        # Allocate a third qubit 'u' to be the second control
        self.sim.allocate_qubit("u")
        self.sim.X("u")
        self.sim.allocate_qubit("t2")  # target initially |0>
        self.sim.CCX("c", "u", "t2")
        st = self.sim.state
        # Qubit layout in bitstring is by allocation index:
        #   idx 0 = c (1), idx 1 = t (0), idx 2 = u (1), idx 3 = t2 (1)
        # => final state bits "1011"
        self.assertTrue(_approx_eq(abs(st["1011"]), 1.0))


class TestSparseMeasurement(unittest.TestCase):
    def test_measure_zero_qubit_returns_zero(self):
        sim = SparseQuantumSimulator(seed=42)
        sim.allocate_qubit("q0")
        outcome = sim.measure("q0")
        self.assertEqual(outcome, 0)

    def test_measure_one_qubit_returns_one(self):
        sim = SparseQuantumSimulator(seed=42)
        sim.allocate_qubit("q0")
        sim.X("q0")
        outcome = sim.measure("q0")
        self.assertEqual(outcome, 1)

    def test_measurement_collapses_to_consistent_state(self):
        sim = SparseQuantumSimulator(seed=7)
        sim.allocate_qubit("q0")
        sim.H("q0")
        outcome = sim.measure("q0")
        # After measurement the state is collapsed to a basis vector
        st = sim.state
        valid_keys_with_unit_amp = all(
            _approx_eq(abs(v), 1.0) for v in st.values())
        self.assertTrue(st)  # non-empty
        self.assertTrue(valid_keys_with_unit_amp)
        # And the measured outcome matches the only remaining key bit
        only_key = next(iter(st.keys()))
        self.assertEqual(int(only_key), outcome)


class TestSparseStateExtraction(unittest.TestCase):
    def test_get_state_vector_single_qubit(self):
        sim = SparseQuantumSimulator()
        sim.allocate_qubit("q0")
        vec = sim.get_state_vector()
        self.assertEqual(len(vec), 2)
        self.assertTrue(_approx_eq(vec[0].real, 1.0))

    def test_get_state_vector_bell(self):
        sim = SparseQuantumSimulator()
        sim.allocate_qubit("q0")
        sim.allocate_qubit("q1")
        sim.H("q0")
        sim.CNOT("q0", "q1")
        vec = sim.get_state_vector()
        self.assertEqual(len(vec), 4)
        inv = 1.0 / math.sqrt(2.0)
        # little-endian: index 0 = |00>, index 3 = |11>
        self.assertTrue(_approx_eq(abs(vec[0]), inv))
        self.assertTrue(_approx_eq(abs(vec[3]), inv))
        self.assertTrue(_approx_eq(abs(vec[1]), 0.0))
        self.assertTrue(_approx_eq(abs(vec[2]), 0.0))

    def test_get_amplitudes_dict_filters_below_threshold(self):
        sim = SparseQuantumSimulator()
        sim.allocate_qubit("q0")
        sim.H("q0")
        d = sim.get_amplitudes_dict()
        for v in d.values():
            self.assertGreater(abs(v), 1e-12)

    def test_get_amplitudes_dict_empty_simulator(self):
        sim = SparseQuantumSimulator()
        d = sim.get_amplitudes_dict()
        self.assertEqual(d, {"": 1.0 + 0.0j})


class TestSparseControlledRotations(unittest.TestCase):
    def setUp(self):
        self.sim = SparseQuantumSimulator()
        self.sim.allocate_qubit("c")
        self.sim.allocate_qubit("t")

    def test_cp_activates_on_both_one(self):
        theta = math.pi / 2
        self.sim.X("c")
        self.sim.X("t")
        self.sim.CP("c", "t", theta)
        st = self.sim.state
        self.assertTrue(_approx_eq(st["11"], 1j))

    def test_cp_no_op_when_control_zero(self):
        theta = math.pi / 2
        self.sim.X("t")
        self.sim.CP("c", "t", theta)
        st = self.sim.state
        self.assertTrue(_approx_eq(abs(st["01"]), 1.0))

    def test_crx_applies_rotation_when_control_one(self):
        self.sim.X("c")
        self.sim.CRX("c", "t", math.pi)
        st = self.sim.state
        # CRX(pi)|10> = |1, -i|1>> = -i|11>
        self.assertTrue(_approx_eq(st["11"].imag, -1.0))

    def test_cry_applies_rotation_when_control_one(self):
        self.sim.X("c")
        self.sim.CRY("c", "t", math.pi)
        st = self.sim.state
        # CRY(pi)|10> = |11>
        self.assertTrue(_approx_eq(abs(st["11"]), 1.0))

    def test_crz_applies_phase_when_control_one(self):
        self.sim.X("c")
        self.sim.X("t")
        self.sim.CRZ("c", "t", math.pi / 2)
        st = self.sim.state
        # CRZ(pi/2)|11> = exp(i*pi/4) |11>
        expected = complex(math.cos(math.pi / 4), math.sin(math.pi / 4))
        self.assertTrue(_approx_eq(st["11"], expected))


if __name__ == "__main__":
    unittest.main()
