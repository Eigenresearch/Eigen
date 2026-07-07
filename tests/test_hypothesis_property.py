"""§9.1 — Property-based tests using the Hypothesis library.

These tests exercise invariants over randomized inputs using
`hypothesis` for automatic case generation and shrinking.
"""
import math
import unittest

from hypothesis import given, strategies as st, assume, settings, HealthCheck

from src.frontend.lexer import Lexer, TokenType
from src.frontend.parser import Parser
from src.frontend.ast import LetNode, LiteralNode
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
# Lexer: int literal round-trip
# ---------------------------------------------------------------------------

class TestLexerIntPropertyHyp(unittest.TestCase):
    @given(st.integers(min_value=0, max_value=10**6))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_int_literals_round_trip(self, n):
        src = f"eigen 1.0\nlet x: int = {n}"
        tokens = Lexer(src).tokenize()
        int_toks = [t for t in tokens if t.type == TokenType.INT_LIT]
        self.assertGreaterEqual(len(int_toks), 1)
        self.assertEqual(int(int_toks[-1].value), n)


# ---------------------------------------------------------------------------
# Lexer: float literal round-trip (positive only to avoid MINUS tokenization)
# ---------------------------------------------------------------------------

class TestLexerFloatPropertyHyp(unittest.TestCase):
    @given(st.floats(min_value=0.001, max_value=1000,
                      allow_nan=False, allow_infinity=False))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_float_literals_round_trip(self, f):
        src = f"eigen 1.0\nlet x: float = {f}"
        tokens = Lexer(src).tokenize()
        float_toks = [t for t in tokens if t.type == TokenType.FLOAT_LIT]
        self.assertGreaterEqual(len(float_toks), 2)
        self.assertAlmostEqual(float(float_toks[-1].value), f, places=5)


# ---------------------------------------------------------------------------
# Parser: let binding preserves int value
# ---------------------------------------------------------------------------

class TestParserLetIntPropertyHyp(unittest.TestCase):
    @given(st.integers(min_value=0, max_value=99999))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_let_binding_preserves_int(self, n):
        src = f"eigen 1.0\nlet x: int = {n}"
        ast = Parser(Lexer(src).tokenize()).parse()
        lets = [s for s in ast.body if isinstance(s, LetNode)]
        self.assertEqual(len(lets), 1)
        self.assertEqual(lets[0].name, "x")
        val_node = lets[0].value
        self.assertIsInstance(val_node, LiteralNode)
        self.assertEqual(int(val_node.value), n)


# ---------------------------------------------------------------------------
# Simulator: identity preserves state
# ---------------------------------------------------------------------------

class TestSimulatorIdentityPropertyHyp(unittest.TestCase):
    @given(st.integers(min_value=1, max_value=100))
    @settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow])
    def test_sparse_identity_preserves_state(self, seed):
        sim = SparseQuantumSimulator(seed=seed)
        sim.allocate_qubit("q0")
        sim.apply_1qubit_gate("q0", [[1, 0], [0, 1]])
        vec = sim.get_state_vector()
        self.assertTrue(_approx(abs(vec[0]), 1.0))


# ---------------------------------------------------------------------------
# Simulator: X is its own inverse
# ---------------------------------------------------------------------------

class TestSimulatorXInverseHyp(unittest.TestCase):
    @given(st.integers(min_value=1, max_value=100))
    @settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow])
    def test_sparse_x_then_x_returns_to_zero(self, seed):
        sim = SparseQuantumSimulator(seed=seed)
        sim.allocate_qubit("q0")
        sim.X("q0")
        sim.X("q0")
        vec = sim.get_state_vector()
        self.assertTrue(_approx(abs(vec[0]), 1.0))


# ---------------------------------------------------------------------------
# Simulator: H is its own inverse
# ---------------------------------------------------------------------------

class TestSimulatorHInverseHyp(unittest.TestCase):
    @given(st.integers(min_value=1, max_value=100))
    @settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow])
    def test_sparse_h_then_h_returns_to_zero(self, seed):
        sim = SparseQuantumSimulator(seed=seed)
        sim.allocate_qubit("q0")
        sim.H("q0")
        sim.H("q0")
        vec = sim.get_state_vector()
        self.assertTrue(_approx(abs(vec[0]), 1.0))


# ---------------------------------------------------------------------------
# Simulator: norm preservation under random Clifford sequences
# ---------------------------------------------------------------------------

class TestSimulatorNormHyp(unittest.TestCase):
    @given(
        st.lists(
            st.sampled_from(["H", "X", "Y", "Z", "S"]),
            min_size=1, max_size=10
        ),
        st.integers(min_value=1, max_value=10000),
    )
    @settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow])
    def test_sparse_norm_preserved(self, gates, seed):
        sim = SparseQuantumSimulator(seed=seed)
        sim.allocate_qubit("q0")
        sim.allocate_qubit("q1")
        sim.H("q0")
        sim.CNOT("q0", "q1")
        for g in gates:
            getattr(sim, g)("q0")
        vec = sim.get_state_vector()
        norm = math.sqrt(sum(abs(a) ** 2 for a in vec))
        self.assertTrue(_approx(norm, 1.0, tol=1e-6))


# ---------------------------------------------------------------------------
# Bytecode version: parse idempotency
# ---------------------------------------------------------------------------

class TestBytecodeVersionParseHyp(unittest.TestCase):
    @given(
        st.integers(min_value=0, max_value=20),
        st.integers(min_value=0, max_value=20),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_parse_twice_equal(self, major, minor):
        v_str = f"{major}.{minor}"
        v1 = parse_bytecode_version(v_str)
        v2 = parse_bytecode_version(v_str)
        self.assertEqual(v1, v2)
        self.assertEqual(v1.major, major)
        self.assertEqual(v1.minor, minor)

    @given(st.integers(min_value=0, max_value=99))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_parse_int_equals_tuple(self, n):
        v_int = parse_bytecode_version(n)
        v_tuple = parse_bytecode_version((n, 0))
        self.assertEqual(v_int, v_tuple)


# ---------------------------------------------------------------------------
# Bytecode version: comparison consistency
# ---------------------------------------------------------------------------

class TestBytecodeVersionComparisonHyp(unittest.TestCase):
    @given(
        st.integers(min_value=0, max_value=99),
        st.integers(min_value=0, max_value=99),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_comparison_consistency(self, a, b):
        va = BytecodeVersion(a, 0)
        vb = BytecodeVersion(b, 0)
        self.assertEqual(va < vb, a < b)
        self.assertEqual(va > vb, a > vb)
        self.assertEqual(va == vb, a == b)


# ---------------------------------------------------------------------------
# Bytecode version: forward-minor always compatible
# ---------------------------------------------------------------------------

class TestBytecodeVersionForwardCompatHyp(unittest.TestCase):
    @given(st.integers(min_value=1, max_value=20))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_forward_minor_compatible(self, minor_offset):
        major = SUPPORTED_BYTECODE_VERSION.major
        minor = SUPPORTED_BYTECODE_VERSION.minor + minor_offset
        v = BytecodeVersion(major, minor)
        self.assertTrue(is_bytecode_compatible(v))
        self.assertEqual(
            check_bytecode_compatibility(v),
            CompatibilityStatus.FORWARD_MINOR
        )

    @given(st.integers(min_value=1, max_value=50))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_future_major_incompatible(self, major_offset):
        major = SUPPORTED_BYTECODE_VERSION.major + major_offset
        v = BytecodeVersion(major, 0)
        self.assertFalse(is_bytecode_compatible(v))


if __name__ == "__main__":
    unittest.main()
