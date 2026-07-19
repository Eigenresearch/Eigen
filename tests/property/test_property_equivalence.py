"""Property-based tests for quantum circuit equivalence.

Generates random equivalent circuits using known identities and verifies
that the equivalence checker correctly identifies them.
"""
import unittest
import random
import math
from src.simulator import QuantumSimulator


def apply_circuit(sim: QuantumSimulator, qubit: str, gates: list):
    """Apply a sequence of gates to a single qubit."""
    for gate in gates:
        if gate == 'H':
            sim.H(qubit)
        elif gate == 'X':
            sim.X(qubit)
        elif gate == 'Y':
            sim.Y(qubit)
        elif gate == 'Z':
            sim.Z(qubit)
        elif gate == 'S':
            sim.S(qubit)
        elif gate == 'T':
            sim.T(qubit)
        elif isinstance(gate, tuple) and gate[0] == 'RX':
            sim.RX(qubit, gate[1])
        elif isinstance(gate, tuple) and gate[0] == 'RY':
            sim.RY(qubit, gate[1])
        elif isinstance(gate, tuple) and gate[0] == 'RZ':
            sim.RZ(qubit, gate[1])


def states_equivalent(sv1, sv2, tol=1e-6) -> bool:
    """Check if two state vectors are equivalent up to global phase."""
    if len(sv1) != len(sv2):
        return False
    # Find first non-zero element to determine global phase
    phase = None
    for a, b in zip(sv1, sv2, strict=False):
        if abs(a) > tol and abs(b) > tol:
            phase = b / a
            break
    if phase is None:
        # Both are zero vectors (shouldn't happen for valid states)
        return all(abs(a) < tol and abs(b) < tol for a, b in zip(sv1, sv2, strict=False))
    # Check all elements match up to the global phase
    for a, b in zip(sv1, sv2, strict=False):
        if abs(a * phase - b) > tol:
            return False
    return True


class TestPropertyEquivalence(unittest.TestCase):

    # Known single-qubit identities
    IDENTITIES = [
        # HH = I
        (['H', 'H'], []),
        # XX = I
        (['X', 'X'], []),
        # YY = I
        (['Y', 'Y'], []),
        # ZZ = I
        (['Z', 'Z'], []),
        # HXH = Z
        (['H', 'X', 'H'], ['Z']),
        # HZH = X
        (['H', 'Z', 'H'], ['X']),
        # SS = Z
        (['S', 'S'], ['Z']),
        # XZ = -Y (equivalent up to global phase, i.e. iY)
        # Actually XZ = iY, so state vectors differ by global phase
        (['X', 'Z'], ['Y']),
    ]

    def _run_circuit(self, gates: list) -> list:
        """Run a sequence of gates on |0> and return the state vector."""
        sim = QuantumSimulator()
        sim.allocate_qubit('q')
        apply_circuit(sim, 'q', gates)
        return list(sim.state_vector)

    def test_known_identities(self):
        """Verify that known gate identities produce equivalent states."""
        for circuit_a, circuit_b in self.IDENTITIES:
            sv_a = self._run_circuit(circuit_a)
            sv_b = self._run_circuit(circuit_b)
            self.assertTrue(
                states_equivalent(sv_a, sv_b),
                f"Identity failed: {circuit_a} != {circuit_b}\n"
                f"  sv_a = {sv_a}\n  sv_b = {sv_b}"
            )

    def test_double_gate_identity(self):
        """Any self-inverse gate applied twice should return to |0>."""
        self_inverse_gates = ['H', 'X', 'Y', 'Z']
        for gate in self_inverse_gates:
            sv = self._run_circuit([gate, gate])
            self.assertAlmostEqual(abs(sv[0])**2, 1.0,
                msg=f"{gate}{gate} did not return to |0>")

    def test_random_rotation_cancellation(self):
        """RX(theta) followed by RX(-theta) should be identity."""
        rng = random.Random(42)
        for _ in range(20):
            theta = rng.uniform(-2 * math.pi, 2 * math.pi)
            sv = self._run_circuit([('RX', theta), ('RX', -theta)])
            self.assertAlmostEqual(abs(sv[0])**2, 1.0, places=6,
                msg=f"RX({theta}) RX({-theta}) not identity")

    def test_random_ry_cancellation(self):
        rng = random.Random(43)
        for _ in range(20):
            theta = rng.uniform(-2 * math.pi, 2 * math.pi)
            sv = self._run_circuit([('RY', theta), ('RY', -theta)])
            self.assertAlmostEqual(abs(sv[0])**2, 1.0, places=6,
                msg=f"RY({theta}) RY({-theta}) not identity")

    def test_random_rz_cancellation(self):
        rng = random.Random(44)
        for _ in range(20):
            theta = rng.uniform(-2 * math.pi, 2 * math.pi)
            sv = self._run_circuit([('RZ', theta), ('RZ', -theta)])
            self.assertAlmostEqual(abs(sv[0])**2, 1.0, places=6,
                msg=f"RZ({theta}) RZ({-theta}) not identity")

    def test_bell_state_symmetry(self):
        """Bell state created with H-CNOT should give equal probabilities for |00> and |11>."""
        sim = QuantumSimulator()
        sim.allocate_qubit('q0')
        sim.allocate_qubit('q1')
        sim.H('q0')
        sim.CNOT('q0', 'q1')
        sv = sim.state_vector
        self.assertAlmostEqual(abs(sv[0])**2, 0.5)
        self.assertAlmostEqual(abs(sv[3])**2, 0.5)
        self.assertAlmostEqual(abs(sv[1])**2, 0.0)
        self.assertAlmostEqual(abs(sv[2])**2, 0.0)

    def test_ghz_state_3_qubits(self):
        """GHZ state on 3 qubits: |000> + |111>."""
        sim = QuantumSimulator()
        sim.allocate_qubit('q0')
        sim.allocate_qubit('q1')
        sim.allocate_qubit('q2')
        sim.H('q0')
        sim.CNOT('q0', 'q1')
        sim.CNOT('q1', 'q2')
        sv = sim.state_vector
        # |000> = index 0, |111> = index 7
        self.assertAlmostEqual(abs(sv[0])**2, 0.5)
        self.assertAlmostEqual(abs(sv[7])**2, 0.5)
        for i in [1, 2, 3, 4, 5, 6]:
            self.assertAlmostEqual(abs(sv[i])**2, 0.0)

    def test_rx_pi_equals_x(self):
        """RX(pi) should be equivalent to X (up to global phase)."""
        sv_rx = self._run_circuit([('RX', math.pi)])
        sv_x = self._run_circuit(['X'])
        self.assertTrue(states_equivalent(sv_rx, sv_x),
            "RX(pi) should be equivalent to X up to global phase")

    def test_ry_pi_equals_y(self):
        """RY(pi) should be equivalent to Y (up to global phase)."""
        sv_ry = self._run_circuit([('RY', math.pi)])
        sv_y = self._run_circuit(['Y'])
        self.assertTrue(states_equivalent(sv_ry, sv_y),
            "RY(pi) should be equivalent to Y up to global phase")

    def test_random_gate_sequence_reproducibility(self):
        """Same random gate sequence should produce identical states."""
        rng = random.Random(99)
        gates_pool = ['H', 'X', 'Y', 'Z', 'S', 'T']
        for _ in range(10):
            seq = [rng.choice(gates_pool) for _ in range(rng.randint(1, 10))]
            sv1 = self._run_circuit(seq)
            sv2 = self._run_circuit(seq)
            for a, b in zip(sv1, sv2, strict=False):
                self.assertAlmostEqual(abs(a - b), 0.0)


if __name__ == "__main__":
    unittest.main()
