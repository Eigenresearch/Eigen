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

    def test_mps_simulator_gates(self):
        sim = QuantumSimulator(sim_type='mps')
        sim.allocate_qubit("q0")
        sim.allocate_qubit("q1")

        # Test single qubit gates on MPS
        sim.H("q0")
        sim.X("q0")
        sim.Y("q0")
        sim.Z("q0")
        sim.S("q0")
        sim.T("q0")
        sim.RX("q0", math.pi / 4)
        sim.RY("q0", math.pi / 2)
        sim.RZ("q0", math.pi / 3)  # Verified fix for NameError

        # Test two qubit gates on MPS
        sim.CNOT("q0", "q1")
        sim.CZ("q0", "q1")
        sim.SWAP("q0", "q1")

        # Get amplitudes and verify that we can measure
        amps = sim.get_amplitudes_dict()
        self.assertGreater(len(amps), 0)
        outcome = sim.measure("q0")
        self.assertIn(outcome, (0, 1))

    def test_mps_rz_gate(self):
        # Specific regression test for RZ NameError crash in MPS
        sim = QuantumSimulator(sim_type='mps')
        sim.allocate_qubit("q0")
        
        # Applying RZ gate (uses cmath.exp)
        sim.RZ("q0", math.pi / 2)
        
        # Verify it succeeds and we can get state vector
        vec = sim.get_state_vector()
        self.assertEqual(len(vec), 2)
        self.assertAlmostEqual(abs(vec[0]), 1.0)
        self.assertAlmostEqual(abs(vec[1]), 0.0)

    def test_mps_entropy_and_error_metrics(self):
        # Verify that MPS tracks entanglement entropy and truncation error correctly
        sim = QuantumSimulator(sim_type='mps')
        sim.allocate_qubit("q0")
        sim.allocate_qubit("q1")
        
        # Create an entangled state (Bell state): H q0, CNOT q0 q1
        sim.H("q0")
        sim.CNOT("q0", "q1")
        
        mps = sim.mps_sim
        self.assertIsNotNone(mps)
        
        # Entanglement entropy of a perfect Bell state is exactly 1.0
        entropy = mps.get_last_entropy()
        self.assertAlmostEqual(entropy, 1.0, places=5)
        
        # Since bond dimension 32 is larger than 2, truncation error should be 0.0
        trunc_err = mps.get_cumulative_truncation_error()
        self.assertAlmostEqual(trunc_err, 0.0, places=5)

    def test_auto_backend_routing(self):
        from src.backend.ebc_compiler import EBCCompiler
        from src.backend.vm import EigenVM
        from src.frontend.lexer import Lexer
        from src.frontend.parser import Parser
        
        # 1. 2 Qubits: should route to dense
        code_dense = """
        eigen 1.0
        qubit q0
        qubit q1
        H q0
        CNOT q0, q1
        """
        lexer = Lexer(code_dense)
        parser = Parser(lexer.tokenize())
        ast = parser.parse()
        
        compiler = EBCCompiler()
        instructions = compiler.compile_ast(ast)
        
        vm = EigenVM(sim_type='auto')
        vm.execute(instructions)
        # Should be dense backend!
        self.assertEqual(vm.simulator.sim_type, 'dense')
        
        # 2. 20 Qubits with few entangling gates (MPS)
        code_mps = ["eigen 1.0"]
        for i in range(20):
            code_mps.append(f"qubit q{i}")
        for i in range(20):
            code_mps.append(f"H q{i}")
        # Add 5 CNOT gates (ratio = 5/20 = 0.25 < 1.5)
        for i in range(5):
            code_mps.append(f"CNOT q{i}, q{i+1}")
        
        lexer = Lexer("\n".join(code_mps))
        parser = Parser(lexer.tokenize())
        ast = parser.parse()
        instructions = compiler.compile_ast(ast)
        
        vm = EigenVM(sim_type='auto')
        vm.execute(instructions)
        # Should be mps backend!
        self.assertEqual(vm.simulator.sim_type, 'mps')
        
        # 3. 20 Qubits with many entangling gates (Sparse)
        code_sparse = ["eigen 1.0"]
        for i in range(20):
            code_sparse.append(f"qubit q{i}")
        code_sparse.append("H q0")
        for i in range(1, 20):
            code_sparse.append(f"X q{i}")
        # Add 35 CNOT gates (ratio = 35/20 = 1.75 >= 1.5)
        for i in range(35):
            code_sparse.append(f"CNOT q{i % 19}, q{(i + 1) % 19}")
            
        lexer = Lexer("\n".join(code_sparse))
        parser = Parser(lexer.tokenize())
        ast = parser.parse()
        instructions = compiler.compile_ast(ast)
        
        vm = EigenVM(sim_type='auto')
        vm.execute(instructions)
        # Should be sparse backend!
        self.assertEqual(vm.simulator.sim_type, 'sparse')

if __name__ == "__main__":
    unittest.main()
