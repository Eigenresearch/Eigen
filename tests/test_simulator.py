import unittest
import math
from src.simulator import QuantumSimulator

class TestQuantumSimulator(unittest.TestCase):
    def test_single_qubit_hadamard(self):
        sim = QuantumSimulator()
        sim.allocate_qubit("q0")
        
        # Initially state is |0>
        self.assertAlmostEqual(abs(sim.state_vector[0])**2, 1.0)
        
        sim.H("q0")
        # Now state is 1/sqrt(2) (|0> + |1>)
        self.assertAlmostEqual(abs(sim.state_vector[0])**2, 0.5)
        self.assertAlmostEqual(abs(sim.state_vector[1])**2, 0.5)

    def test_bell_state_cnot(self):
        sim = QuantumSimulator()
        sim.allocate_qubit("q0")
        sim.allocate_qubit("q1")
        
        sim.H("q0")
        sim.CNOT("q0", "q1")
        
        # State vector should be 1/sqrt(2) (|00> + |11>)
        # qubit order in state vector index: q0 maps to index 0 (bit 0), q1 maps to index 1 (bit 1)
        # |00> is index 0
        # |01> is index 1 (q0=1, q1=0) -> amplitude 0
        # |10> is index 2 (q0=0, q1=1) -> amplitude 0
        # |11> is index 3 (q0=1, q1=1) -> amplitude 1/sqrt(2)
        
        self.assertAlmostEqual(abs(sim.state_vector[0])**2, 0.5)
        self.assertAlmostEqual(abs(sim.state_vector[1])**2, 0.0)
        self.assertAlmostEqual(abs(sim.state_vector[2])**2, 0.0)
        self.assertAlmostEqual(abs(sim.state_vector[3])**2, 0.5)

    def test_rotations(self):
        sim = QuantumSimulator()
        sim.allocate_qubit("q0")
        
        # Rotate RX by PI -> should flip state to |1> (up to phase)
        sim.RX("q0", math.pi)
        self.assertAlmostEqual(abs(sim.state_vector[0])**2, 0.0)
        self.assertAlmostEqual(abs(sim.state_vector[1])**2, 1.0)

    def test_measurement_collapse(self):
        sim = QuantumSimulator()
        sim.allocate_qubit("q0")
        sim.allocate_qubit("q1")
        
        sim.H("q0")
        sim.CNOT("q0", "q1")
        
        outcome = sim.measure("q0")
        
        # After measurement, the state collapses.
        # If outcome is 0, the state is |00>, so amplitude at index 0 must be 1.0.
        # If outcome is 1, the state is |11>, so amplitude at index 3 must be 1.0.
        if outcome == 0:
            self.assertAlmostEqual(abs(sim.state_vector[0])**2, 1.0)
            self.assertAlmostEqual(abs(sim.state_vector[3])**2, 0.0)
        else:
            self.assertAlmostEqual(abs(sim.state_vector[0])**2, 0.0)
            self.assertAlmostEqual(abs(sim.state_vector[3])**2, 1.0)

    def test_sparse_simulation(self):
        sim = QuantumSimulator()
        # Allocate 22 qubits. Qubits 0-19 will be dense, qubit 20 will trigger sparse mode transition.
        for i in range(22):
            sim.allocate_qubit(f"q{i}")
            
        self.assertTrue(sim.is_sparse)
        self.assertIsNone(sim.state_vector)
        
        # Test applying H and CNOT in sparse mode
        sim.H("q0")
        sim.CNOT("q0", "q1")
        
        # Check amplitudes dict contains the expected entangled states for q0 and q1.
        amps = sim.get_amplitudes_dict()
        self.assertEqual(len(amps), 2)
        
        # The states in the dictionary will be represented as bitstrings.
        # q0 maps to index 0, q1 maps to index 1.
        # In get_amplitudes_dict, we reverse sorted_qubits, so:
        # q21 is the first char (leftmost), q0 is the last char (rightmost).
        # So we expect bitstrings ending with '00' and '11'.
        states = list(amps.keys())
        self.assertTrue(all(state.endswith('00') or state.endswith('11') for state in states))
        
        # Test measurement in sparse mode
        outcome = sim.measure("q0")
        self.assertIn(outcome, (0, 1))
        
        # After measurement, the state should collapse to exactly 1 state in amps dict
        amps_collapsed = sim.get_amplitudes_dict()
        self.assertEqual(len(amps_collapsed), 1)
        collapsed_state = list(amps_collapsed.keys())[0]
        if outcome == 0:
            self.assertTrue(collapsed_state.endswith('00'))
        else:
            self.assertTrue(collapsed_state.endswith('11'))

if __name__ == "__main__":
    unittest.main()
