import unittest
import math
import cmath
from src.frontend.lexer import Lexer
from src.frontend.parser import Parser
from src.simulator import QuantumSimulator

class TestGates2_5(unittest.TestCase):
    def test_toffoli_dense(self):
        # Truth table of CCX (Toffoli):
        # Control qubits: q0 (index 0) and q1 (index 1). Target: q2 (index 2).
        # States format: q2 q1 q0
        # 011 (value 3) -> 111 (value 7)
        # 111 (value 7) -> 011 (value 3)
        # All other states remain unchanged.
        for input_state, expected_state in [
            (0, 0), (1, 1), (2, 2), (3, 7), (4, 4), (5, 5), (6, 6), (7, 3)
        ]:
            sim = QuantumSimulator(sim_type='dense')
            sim.allocate_qubit('q0')
            sim.allocate_qubit('q1')
            sim.allocate_qubit('q2')
            
            # Setup input state
            if input_state & 1:
                sim.X('q0')
            if input_state & 2:
                sim.X('q1')
            if input_state & 4:
                sim.X('q2')
                
            sim.CCX('q0', 'q1', 'q2')
            
            amps = sim.get_amplitudes_dict()
            expected_bitstring = f"{expected_state:03b}"
            self.assertIn(expected_bitstring, amps)
            self.assertAlmostEqual(abs(amps[expected_bitstring]), 1.0)

    def test_toffoli_sparse(self):
        for input_state, expected_state in [
            (0, 0), (1, 1), (2, 2), (3, 7), (4, 4), (5, 5), (6, 6), (7, 3)
        ]:
            sim = QuantumSimulator(sim_type='sparse')
            sim.allocate_qubit('q0')
            sim.allocate_qubit('q1')
            sim.allocate_qubit('q2')
            
            if input_state & 1:
                sim.X('q0')
            if input_state & 2:
                sim.X('q1')
            if input_state & 4:
                sim.X('q2')
                
            sim.CCX('q0', 'q1', 'q2')
            
            amps = sim.get_amplitudes_dict()
            expected_bitstring = f"{expected_state:03b}"
            self.assertIn(expected_bitstring, amps)
            self.assertAlmostEqual(abs(amps[expected_bitstring]), 1.0)

    def test_toffoli_mps(self):
        for input_state, expected_state in [
            (0, 0), (1, 1), (2, 2), (3, 7), (4, 4), (5, 5), (6, 6), (7, 3)
        ]:
            sim = QuantumSimulator(sim_type='mps')
            sim.allocate_qubit('q0')
            sim.allocate_qubit('q1')
            sim.allocate_qubit('q2')
            
            if input_state & 1:
                sim.X('q0')
            if input_state & 2:
                sim.X('q1')
            if input_state & 4:
                sim.X('q2')
                
            sim.CCX('q0', 'q1', 'q2')
            
            amps = sim.get_amplitudes_dict()
            expected_bitstring = f"{expected_state:03b}"
            self.assertIn(expected_bitstring, amps)
            self.assertAlmostEqual(abs(amps[expected_bitstring]), 1.0)

    def test_fredkin_dense(self):
        # Truth table of CSWAP (Fredkin):
        # Control qubit: q0 (index 0). Target swap qubits: q1 (index 1) and q2 (index 2).
        # If q0 = 1, swap q1 and q2.
        # States format: q2 q1 q0
        # 011 (value 3) -> 101 (value 5)
        # 101 (value 5) -> 011 (value 3)
        # All other states remain unchanged.
        for input_state, expected_state in [
            (0, 0), (1, 1), (2, 2), (3, 5), (4, 4), (5, 3), (6, 6), (7, 7)
        ]:
            sim = QuantumSimulator(sim_type='dense')
            sim.allocate_qubit('q0')
            sim.allocate_qubit('q1')
            sim.allocate_qubit('q2')
            
            if input_state & 1:
                sim.X('q0')
            if input_state & 2:
                sim.X('q1')
            if input_state & 4:
                sim.X('q2')
                
            sim.CSWAP('q0', 'q1', 'q2')
            
            amps = sim.get_amplitudes_dict()
            expected_bitstring = f"{expected_state:03b}"
            self.assertIn(expected_bitstring, amps)
            self.assertAlmostEqual(abs(amps[expected_bitstring]), 1.0)

    def test_controlled_phase(self):
        theta = math.pi / 3
        val = cmath.exp(1j * theta)
        
        for sim_type in ['dense', 'sparse', 'mps']:
            sim = QuantumSimulator(sim_type=sim_type)
            sim.allocate_qubit('q0')
            sim.allocate_qubit('q1')
            
            sim.X('q0')
            sim.X('q1')
            sim.CP('q0', 'q1', theta)
            
            amps = sim.get_amplitudes_dict()
            self.assertAlmostEqual(amps['11'], val)

    def test_controlled_rotations(self):
        theta = math.pi / 2
        
        for sim_type in ['dense', 'sparse', 'mps']:
            # CRX
            sim = QuantumSimulator(sim_type=sim_type)
            sim.allocate_qubit('q0')
            sim.allocate_qubit('q1')
            sim.X('q0')
            sim.CRX('q0', 'q1', theta)
            amps = sim.get_amplitudes_dict()
            # q0=1 (index 0), q1 (index 1) rotated. State format: q1 q0
            # RX(pi/2)|0> = cos(pi/4)|0> - i*sin(pi/4)|1>
            # So state is cos(pi/4)|01> - i*sin(pi/4)|11>
            self.assertAlmostEqual(amps['01'].real, math.cos(math.pi/4))
            self.assertAlmostEqual(amps['11'].imag, -math.sin(math.pi/4))

            # CRY
            sim = QuantumSimulator(sim_type=sim_type)
            sim.allocate_qubit('q0')
            sim.allocate_qubit('q1')
            sim.X('q0')
            sim.CRY('q0', 'q1', theta)
            amps = sim.get_amplitudes_dict()
            # RY(pi/2)|0> = cos(pi/4)|0> + sin(pi/4)|1>
            # So state is cos(pi/4)|01> + sin(pi/4)|11>
            self.assertAlmostEqual(amps['01'].real, math.cos(math.pi/4))
            self.assertAlmostEqual(amps['11'].real, math.sin(math.pi/4))

            # CRZ
            sim = QuantumSimulator(sim_type=sim_type)
            sim.allocate_qubit('q0')
            sim.allocate_qubit('q1')
            sim.X('q0')
            sim.H('q1')
            sim.CRZ('q0', 'q1', theta)
            amps = sim.get_amplitudes_dict()
            # CRZ(pi/2) on q1 when q0=1:
            # (|01> + |11>)/sqrt(2) -> (exp(-i*pi/4)|01> + exp(i*pi/4)|11>)/sqrt(2)
            self.assertAlmostEqual(amps['01'], cmath.exp(-1j * math.pi / 4) / math.sqrt(2))
            self.assertAlmostEqual(amps['11'], cmath.exp(1j * math.pi / 4) / math.sqrt(2))

    def test_parser_gates(self):
        source = """
        eigen 2.5
        qubit q0
        qubit q1
        qubit q2
        Toffoli q0, q1, q2
        Fredkin q0, q1, q2
        CCX q0, q1, q2
        CSWAP q0, q1, q2
        CP q0, q1, 0.5
        CRX q0, q1, 1.0
        CRY q0, q1, 1.5
        CRZ q0, q1, 2.0
        """
        tokens = Lexer(source).tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        self.assertIsNotNone(ast)

if __name__ == '__main__':
    unittest.main()
