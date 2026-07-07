"""§9.1 — Property-based tests using random data generation.

These tests exercise invariants over randomized inputs without
requiring the hypothesis library.  Each test runs the same property
across many random samples and asserts the invariant holds.
"""
import math
import random
import unittest

from src.frontend.lexer import Lexer, TokenType
from src.frontend.parser import Parser
from src.frontend.ast import (
    LetNode,
    LiteralNode,
    GateNode,
)
from src.sparse_simulator import SparseQuantumSimulator
from src.tensor_network.mps import MPSSimulator
from src.backend.bytecode import (
    BytecodeVersion,
    parse_bytecode_version,
    check_bytecode_compatibility,
    is_bytecode_compatible,
    CompatibilityStatus,
    SUPPORTED_BYTECODE_VERSION,
)


def _approx(a, b, tol=1e-9):
    return abs(a - b) < tol


# ---------------------------------------------------------------------------
# Lexer property: any valid int literal round-trips
# ---------------------------------------------------------------------------

class TestLexerIntLiteralProperty(unittest.TestCase):
    def test_int_literals_round_trip(self):
        rng = random.Random(12345)
        for _ in range(50):
            n = rng.randint(0, 10**6)
            src = f"eigen 1.0\nlet x: int = {n}"
            tokens = Lexer(src).tokenize()
            # find the int literal token
            int_toks = [t for t in tokens if t.type == TokenType.INT_LIT]
            self.assertEqual(len(int_toks), 1)
            self.assertEqual(int(int_toks[0].value), n)

    def test_negative_int_literals(self):
        rng = random.Random(54321)
        for _ in range(20):
            n = -rng.randint(1, 10**6)
            src = f"eigen 1.0\nlet x: int = {n}"
            tokens = Lexer(src).tokenize()
            # Either INT_LIT or MINUS INT_LIT depending on lexer handling
            int_toks = [t for t in tokens if t.type == TokenType.INT_LIT]
            self.assertGreaterEqual(len(int_toks), 1)


class TestLexerFloatLiteralProperty(unittest.TestCase):
    def test_float_literals_round_trip(self):
        rng = random.Random(99)
        for _ in range(30):
            # Use positive-only floats to avoid MINUS sign tokenization
            f = round(rng.uniform(0.001, 1000), 6)
            src = f"eigen 1.0\nlet x: float = {f}"
            tokens = Lexer(src).tokenize()
            # `eigen 1.0` produces one FLOAT_LIT (the version),
            # and the let value produces another FLOAT_LIT.
            float_toks = [t for t in tokens if t.type == TokenType.FLOAT_LIT]
            self.assertGreaterEqual(len(float_toks), 2)
            # The LAST float token is the let value; verify round-trip
            self.assertAlmostEqual(float(float_toks[-1].value), f, places=5)


# ---------------------------------------------------------------------------
# Parser property: parse(let X: int = N) preserves N
# ---------------------------------------------------------------------------

class TestParserLetIntProperty(unittest.TestCase):
    def test_let_binding_preserves_int_value(self):
        rng = random.Random(777)
        for _ in range(30):
            n = rng.randint(0, 99999)
            src = f"eigen 1.0\nlet x: int = {n}"
            ast = Parser(Lexer(src).tokenize()).parse()
            lets = [s for s in ast.body if isinstance(s, LetNode)]
            self.assertEqual(len(lets), 1)
            self.assertEqual(lets[0].name, "x")
            self.assertEqual(lets[0].type_name, "int")
            # value should be a LiteralNode wrapping int n
            val_node = lets[0].value
            self.assertIsInstance(val_node, LiteralNode)
            self.assertEqual(int(val_node.value), n)


# ---------------------------------------------------------------------------
# Simulator property: applying identity matrix keeps state unchanged
# ---------------------------------------------------------------------------

class TestSimulatorIdentityProperty(unittest.TestCase):
    def test_sparse_identity_preserves_state(self):
        rng = random.Random(2024)
        for _ in range(10):
            sim = SparseQuantumSimulator(seed=rng.randint(0, 10**9))
            sim.allocate_qubit("q0")
            # Apply identity matrix
            sim.apply_1qubit_gate("q0", [[1, 0], [0, 1]])
            vec = sim.get_state_vector()
            # |0> stays |0>
            self.assertTrue(_approx(abs(vec[0]), 1.0))

    def test_mps_identity_preserves_state(self):
        sim = MPSSimulator()
        sim.allocate_qubit("q0")
        sim.apply_1qubit_gate("q0", [[1, 0], [0, 1]])
        vec = sim.get_state_vector()
        self.assertTrue(_approx(vec[0].real, 1.0))


class TestSimulatorXThenXProperty(unittest.TestCase):
    """X is its own inverse: applying X twice should leave |0> unchanged."""

    def test_sparse_x_then_x_returns_to_zero(self):
        sim = SparseQuantumSimulator(seed=42)
        sim.allocate_qubit("q0")
        sim.X("q0")
        sim.X("q0")
        vec = sim.get_state_vector()
        self.assertTrue(_approx(abs(vec[0]), 1.0))

    def test_mps_x_then_x_returns_to_zero(self):
        sim = MPSSimulator()
        sim.allocate_qubit("q0")
        sim.X("q0")
        sim.X("q0")
        vec = sim.get_state_vector()
        self.assertTrue(_approx(abs(vec[0]), 1.0))


class TestSimulatorHThenHProperty(unittest.TestCase):
    """H is its own inverse: H @ H = I."""

    def test_sparse_h_then_h_returns_to_zero(self):
        sim = SparseQuantumSimulator(seed=42)
        sim.allocate_qubit("q0")
        sim.H("q0")
        sim.H("q0")
        vec = sim.get_state_vector()
        # State should be |0> up to global phase
        self.assertTrue(_approx(abs(vec[0]), 1.0))

    def test_mps_h_then_h_returns_to_zero(self):
        sim = MPSSimulator()
        sim.allocate_qubit("q0")
        sim.H("q0")
        sim.H("q0")
        vec = sim.get_state_vector()
        self.assertTrue(_approx(abs(vec[0]), 1.0))


class TestSimulatorProbabilityConservation(unittest.TestCase):
    """Norm is always 1 for any sequence of unitary gates."""

    def test_sparse_norm_preserved_after_random_cliffords(self):
        rng = random.Random(13)
        gates = ["H", "X", "Y", "Z", "S", "T"]
        for trial in range(5):
            sim = SparseQuantumSimulator(seed=rng.randint(0, 10**9))
            sim.allocate_qubit("q0")
            sim.allocate_qubit("q1")
            sim.H("q0")
            sim.CNOT("q0", "q1")
            # Apply random single-qubit Clifford on q0
            for _ in range(5):
                g = rng.choice(gates)
                getattr(sim, g)("q0")
            vec = sim.get_state_vector()
            norm = math.sqrt(sum(abs(a) ** 2 for a in vec))
            self.assertTrue(_approx(norm, 1.0, tol=1e-6),
                              f"trial {trial}: norm={norm}")

    def test_mps_norm_preserved_after_random_cliffords(self):
        rng = random.Random(31)
        gates = ["H", "X", "Y", "Z", "S"]
        for trial in range(5):
            sim = MPSSimulator(seed=rng.randint(0, 10**9))
            sim.allocate_qubit("q0")
            sim.allocate_qubit("q1")
            sim.H("q0")
            sim.CNOT("q0", "q1")
            for _ in range(5):
                g = rng.choice(gates)
                getattr(sim, g)("q0")
            n = sim.norm_squared()
            self.assertTrue(_approx(n, 1.0, tol=1e-6),
                              f"trial {trial}: norm^2={n}")


# ---------------------------------------------------------------------------
# Bytecode Version property: parsing is idempotent and consistent
# ---------------------------------------------------------------------------

class TestBytecodeVersionParseIdempotent(unittest.TestCase):
    def test_parse_twice_gives_equal_versions(self):
        rng = random.Random(2025)
        for _ in range(50):
            major = rng.randint(0, 20)
            minor = rng.randint(0, 20)
            v_str = f"{major}.{minor}"
            v1 = parse_bytecode_version(v_str)
            v2 = parse_bytecode_version(v_str)
            self.assertEqual(v1, v2)
            self.assertEqual(v1.major, major)
            self.assertEqual(v1.minor, minor)

    def test_parse_int_equals_parse_tuple(self):
        rng = random.Random(7)
        for _ in range(30):
            n = rng.randint(0, 99)
            v_int = parse_bytecode_version(n)
            v_tuple = parse_bytecode_version((n, 0))
            self.assertEqual(v_int, v_tuple)

    def test_parse_str_v_prefix_equals_no_prefix(self):
        rng = random.Random(3)
        for _ in range(30):
            major = rng.randint(1, 9)
            minor = rng.randint(0, 9)
            v1 = parse_bytecode_version(f"v{major}.{minor}")
            v2 = parse_bytecode_version(f"{major}.{minor}")
            self.assertEqual(v1, v2)


class TestBytecodeVersionComparisonProperty(unittest.TestCase):
    def test_consistency_with_int_comparison_when_minor_zero(self):
        rng = random.Random(42)
        for _ in range(50):
            a = rng.randint(0, 99)
            b = rng.randint(0, 99)
            va = BytecodeVersion(a, 0)
            vb = BytecodeVersion(b, 0)
            # When minors are 0, comparison should match int comparison
            self.assertEqual(va < vb, a < b)
            self.assertEqual(va > vb, a > b)
            self.assertEqual(va == vb, a == b)


class TestBytecodeVersionCompatibilityProperty(unittest.TestCase):
    def test_forward_minor_always_compatible(self):
        rng = random.Random(99)
        for _ in range(50):
            # Same major as supported, strictly higher minor
            major = SUPPORTED_BYTECODE_VERSION.major
            minor = SUPPORTED_BYTECODE_VERSION.minor + rng.randint(1, 20)
            v = BytecodeVersion(major, minor)
            self.assertTrue(is_bytecode_compatible(v))
            self.assertEqual(
                check_bytecode_compatibility(v),
                CompatibilityStatus.FORWARD_MINOR
            )

    def test_future_major_always_incompatible(self):
        rng = random.Random(2024)
        for _ in range(30):
            # Strictly higher major than supported
            major = SUPPORTED_BYTECODE_VERSION.major + rng.randint(1, 50)
            minor = rng.randint(0, 99)
            v = BytecodeVersion(major, minor)
            self.assertFalse(is_bytecode_compatible(v))


# ---------------------------------------------------------------------------
# Cross-simulator consistency: Sparse and MPS agree on Bell state
# ---------------------------------------------------------------------------

class TestSimulatorCrossConsistency(unittest.TestCase):
    def test_sparse_and_mps_agree_on_bell_state(self):
        # Both simulators should produce the same Bell state amplitudes
        # up to a global phase.
        sparse = SparseQuantumSimulator(seed=42)
        sparse.allocate_qubit("q0")
        sparse.allocate_qubit("q1")
        sparse.H("q0")
        sparse.CNOT("q0", "q1")
        s_vec = sparse.get_state_vector()

        mps = MPSSimulator()
        mps.allocate_qubit("q0")
        mps.allocate_qubit("q1")
        mps.H("q0")
        mps.CNOT("q0", "q1")
        m_vec = mps.get_state_vector()

        inv = 1.0 / math.sqrt(2.0)
        # Both should have |00> and |11> populated
        self.assertTrue(_approx(abs(s_vec[0]), inv, tol=1e-6))
        self.assertTrue(_approx(abs(s_vec[3]), inv, tol=1e-6))
        self.assertTrue(_approx(abs(m_vec[0]), inv, tol=1e-6))
        self.assertTrue(_approx(abs(m_vec[3]), inv, tol=1e-6))
        # And |01>, |10> are zero
        self.assertTrue(_approx(abs(s_vec[1]), 0.0, tol=1e-6))
        self.assertTrue(_approx(abs(s_vec[2]), 0.0, tol=1e-6))
        self.assertTrue(_approx(abs(m_vec[1]), 0.0, tol=1e-6))
        self.assertTrue(_approx(abs(m_vec[2]), 0.0, tol=1e-6))


if __name__ == "__main__":
    unittest.main()
