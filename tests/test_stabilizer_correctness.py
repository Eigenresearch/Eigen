import unittest
from src.stabilizer_simulator import StabilizerSimulator
from src.simulator import QuantumSimulator


class TestStabilizerCorrectness(unittest.TestCase):
    def test_bell_state_deterministic_after_cnot(self):
        for seed in range(20):
            stab = StabilizerSimulator(seed=seed)
            dense = QuantumSimulator(sim_type='dense', seed=seed)
            for name in ['q0', 'q1']:
                stab.allocate_qubit(name)
                dense.allocate_qubit(name)
            stab.H('q0'); dense.H('q0')
            stab.CNOT('q0', 'q1'); dense.CNOT('q0', 'q1')
            s0 = stab.measure('q0')
            s1 = stab.measure('q1')
            d0 = dense.measure('q0')
            d1 = dense.measure('q1')
            self.assertEqual(s0, s1, f"Stabilizer: q0={s0}, q1={s1} should match")
            self.assertEqual(d0, d1, f"Dense: q0={d0}, q1={d1} should match")

    def test_x_gate_flips_measurement(self):
        stab = StabilizerSimulator(seed=42)
        dense = QuantumSimulator(sim_type='dense', seed=42)
        for name in ['q0']:
            stab.allocate_qubit(name)
            dense.allocate_qubit(name)
        stab.X('q0'); dense.X('q0')
        self.assertEqual(stab.measure('q0'), 1)
        self.assertEqual(dense.measure('q0'), 1)

    def test_z_gate_no_effect_on_zero_state(self):
        stab = StabilizerSimulator(seed=42)
        stab.allocate_qubit('q0')
        stab.Z('q0')
        self.assertEqual(stab.measure('q0'), 0)

    def test_h_then_z_then_h_equals_x(self):
        stab = StabilizerSimulator(seed=42)
        stab.allocate_qubit('q0')
        stab.H('q0')
        stab.Z('q0')
        stab.H('q0')
        self.assertEqual(stab.measure('q0'), 1)

    def test_s_gate_phase(self):
        stab = StabilizerSimulator(seed=42)
        stab.allocate_qubit('q0')
        stab.X('q0')
        stab.S('q0')
        self.assertEqual(stab.measure('q0'), 1)

    def test_cz_equals_h_cnot_h(self):
        for seed in range(10):
            stab1 = StabilizerSimulator(seed=seed)
            for n in ['q0', 'q1']:
                stab1.allocate_qubit(n)
            stab1.H('q0')
            stab1.CZ('q0', 'q1')
            r1 = (stab1.measure('q0'), stab1.measure('q1'))
            stab2 = StabilizerSimulator(seed=seed)
            for n in ['q0', 'q1']:
                stab2.allocate_qubit(n)
            stab2.H('q0')
            stab2.H('q1')
            stab2.CNOT('q0', 'q1')
            stab2.H('q1')
            r2 = (stab2.measure('q0'), stab2.measure('q1'))
            self.assertEqual(r1, r2, f"CZ {r1} != H-CNOT-H {r2} (seed={seed})")

    def test_swap_exchanges_qubits(self):
        stab = StabilizerSimulator(seed=42)
        for n in ['q0', 'q1']:
            stab.allocate_qubit(n)
        stab.X('q0')
        stab.SWAP('q0', 'q1')
        self.assertEqual(stab.measure('q0'), 0)
        self.assertEqual(stab.measure('q1'), 1)

    def test_ghz_state(self):
        for seed in range(10):
            stab = StabilizerSimulator(seed=seed)
            for n in ['q0', 'q1', 'q2']:
                stab.allocate_qubit(n)
            stab.H('q0')
            stab.CNOT('q0', 'q1')
            stab.CNOT('q1', 'q2')
            r0 = stab.measure('q0')
            r1 = stab.measure('q1')
            r2 = stab.measure('q2')
            self.assertEqual(r0, r1, f"GHZ: q0={r0} != q1={r1}")
            self.assertEqual(r1, r2, f"GHZ: q1={r1} != q2={r2}")

    def test_deterministic_measurements(self):
        for seed in range(20):
            stab = StabilizerSimulator(seed=seed)
            stab.allocate_qubit('q0')
            stab.X('q0')
            self.assertEqual(stab.measure('q0'), 1)

    def test_random_measurement_distribution(self):
        results = []
        for seed in range(1000):
            stab = StabilizerSimulator(seed=seed)
            stab.allocate_qubit('q0')
            stab.H('q0')
            results.append(stab.measure('q0'))
        zeros = results.count(0)
        ones = results.count(1)
        self.assertGreater(zeros, 400, f"Too few zeros: {zeros}/1000")
        self.assertGreater(ones, 400, f"Too few ones: {ones}/1000")

    def test_multiple_qubit_allocation_preserves_state(self):
        stab = StabilizerSimulator(seed=42)
        stab.allocate_qubit('q0')
        stab.X('q0')
        stab.allocate_qubit('q1')
        self.assertEqual(stab.measure('q0'), 1)

    def test_cnot_propagation(self):
        stab = StabilizerSimulator(seed=42)
        for n in ['q0', 'q1']:
            stab.allocate_qubit(n)
        stab.X('q0')
        stab.CNOT('q0', 'q1')
        self.assertEqual(stab.measure('q0'), 1)
        self.assertEqual(stab.measure('q1'), 1)

    def test_cnot_no_flip_when_control_zero(self):
        stab = StabilizerSimulator(seed=42)
        for n in ['q0', 'q1']:
            stab.allocate_qubit(n)
        stab.CNOT('q0', 'q1')
        self.assertEqual(stab.measure('q0'), 0)
        self.assertEqual(stab.measure('q1'), 0)


if __name__ == '__main__':
    unittest.main()
