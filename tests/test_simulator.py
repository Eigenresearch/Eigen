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

if __name__ == "__main__":
    unittest.main()
