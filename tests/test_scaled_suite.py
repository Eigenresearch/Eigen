"""Scaled parameterized test suite for Eigen 2.3 — Helios.

Provides broad coverage across all major subsystems using parameterized
test cases to ensure production-grade reliability (700+ total assertions).
"""
import unittest
import math
import random
from src.frontend.lexer import Lexer, TokenType
from src.frontend.parser import Parser
from src.semantic.type_checker import TypeChecker, TypeErrorException
from src.backend.ebc_compiler import EBCCompiler
from src.backend.vm import EigenVM, UndefinedVariableError
from src.simulator import QuantumSimulator
from src.ir.ir_converter import EQIRConverter
from src.ir.ir_graph import EQIRGraph
from src.resource_estimator.estimator import ResourceEstimator
from src.noise.noise_model import NoiseModel
from src.equivalence import EquivalenceChecker
from src.frontend.ast import (
    ProgramNode, LetNode, LiteralNode, VarRefNode, BinaryOpNode,
    VarDeclNode, GateNode, MeasureNode, IfNode, ReturnNode,
    FuncDeclNode, TraceNode, PrintNode, AssertNode
)


# ---------------------------------------------------------------------------
# Lexer tests (parameterized)
# ---------------------------------------------------------------------------
class TestLexerParameterized(unittest.TestCase):

    KEYWORD_MAP = {
        "eigen": TokenType.EIGEN,
        "qubit": TokenType.QUBIT,
        "cbit": TokenType.CBIT,
        "let": TokenType.LET,
        "func": TokenType.FUNC,
        "return": TokenType.RETURN,
        "if": TokenType.IF,
        "for": TokenType.FOR,
        "while": TokenType.WHILE,
        "struct": TokenType.STRUCT,
        "import": TokenType.IMPORT,
        "measure": TokenType.MEASURE,
        "trace": TokenType.TRACE,
        "assert": TokenType.ASSERT,
        "print": TokenType.PRINT,
    }

    def test_keyword_tokens(self):
        """Each keyword should produce the correct token type."""
        for keyword, expected_type in self.KEYWORD_MAP.items():
            with self.subTest(keyword=keyword):
                lexer = Lexer(keyword)
                tokens = lexer.tokenize()
                self.assertEqual(tokens[0].type, expected_type)

    def test_gate_tokens(self):
        gate_map = {
            "H": TokenType.GATE_H,
            "X": TokenType.GATE_X,
            "Y": TokenType.GATE_Y,
            "Z": TokenType.GATE_Z,
            "CNOT": TokenType.GATE_CNOT,
            "RX": TokenType.GATE_RX,
            "RY": TokenType.GATE_RY,
            "RZ": TokenType.GATE_RZ,
            "S": TokenType.GATE_S,
            "T": TokenType.GATE_T,
        }
        for gate, expected_type in gate_map.items():
            with self.subTest(gate=gate):
                lexer = Lexer(gate)
                tokens = lexer.tokenize()
                self.assertEqual(tokens[0].type, expected_type)

    def test_integer_literals(self):
        values = ["0", "1", "42", "100", "9999", "1000000"]
        for val in values:
            with self.subTest(value=val):
                lexer = Lexer(val)
                tokens = lexer.tokenize()
                self.assertEqual(tokens[0].type, TokenType.INT_LIT)
                self.assertEqual(tokens[0].value, val)

    def test_float_literals(self):
        values = ["0.0", "1.0", "3.14", "2.71828", "0.001", "99.99"]
        for val in values:
            with self.subTest(value=val):
                lexer = Lexer(val)
                tokens = lexer.tokenize()
                self.assertEqual(tokens[0].type, TokenType.FLOAT_LIT)

    def test_string_literals(self):
        values = ['"hello"', '"world"', '"test string"', '""', '"123"']
        for val in values:
            with self.subTest(value=val):
                lexer = Lexer(val)
                tokens = lexer.tokenize()
                self.assertEqual(tokens[0].type, TokenType.STRING_LIT)

    def test_operator_tokens(self):
        ops = {
            "+": TokenType.PLUS, "-": TokenType.MINUS,
            "*": TokenType.MUL, "/": TokenType.DIV,
            "==": TokenType.EQ, "!=": TokenType.NE,
            "<": TokenType.LT, ">": TokenType.GT,
            "<=": TokenType.LE, ">=": TokenType.GE,
            "->": TokenType.ARROW,
        }
        for op, expected_type in ops.items():
            with self.subTest(op=op):
                lexer = Lexer(op)
                tokens = lexer.tokenize()
                self.assertEqual(tokens[0].type, expected_type)

    def test_delimiter_tokens(self):
        delims = {
            "(": TokenType.LPAREN, ")": TokenType.RPAREN,
            "{": TokenType.LBRACE, "}": TokenType.RBRACE,
            "[": TokenType.LBRACK, "]": TokenType.RBRACK,
            ":": TokenType.COLON, ",": TokenType.COMMA,
            "=": TokenType.EQUALS,
        }
        for delim, expected_type in delims.items():
            with self.subTest(delim=delim):
                lexer = Lexer(delim)
                tokens = lexer.tokenize()
                self.assertEqual(tokens[0].type, expected_type)

    def test_identifiers(self):
        ids = ["x", "y", "myVar", "var_name", "q0", "c1", "abc123", "_test"]
        for ident in ids:
            with self.subTest(ident=ident):
                lexer = Lexer(ident)
                tokens = lexer.tokenize()
                self.assertEqual(tokens[0].type, TokenType.IDENTIFIER)

    def test_comments_are_ignored(self):
        sources = [
            "# comment\nlet x: int = 1",
            "qubit q0 # inline comment",
            "# full line comment",
        ]
        for source in sources:
            with self.subTest(source=source[:30]):
                lexer = Lexer(source)
                tokens = lexer.tokenize()
                for tok in tokens:
                    self.assertNotEqual(tok.type, "COMMENT")

    def test_invalid_characters(self):
        invalids = ["@", "$", "`", "\\"]
        for ch in invalids:
            with self.subTest(char=ch):
                lexer = Lexer(ch)
                with self.assertRaises(SyntaxError):
                    lexer.tokenize()

    def test_eof_always_present(self):
        sources = ["", "x", "let x: int = 1", "eigen 1.0\nqubit q0"]
        for source in sources:
            with self.subTest(source=source[:20]):
                lexer = Lexer(source)
                tokens = lexer.tokenize()
                self.assertEqual(tokens[-1].type, TokenType.EOF)


# ---------------------------------------------------------------------------
# Parser tests (parameterized)
# ---------------------------------------------------------------------------
class TestParserParameterized(unittest.TestCase):

    def _parse(self, source: str):
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        return parser.parse()

    def test_valid_programs_parse(self):
        programs = [
            "eigen 1.0\nqubit q0",
            "eigen 1.0\ncbit c0",
            "eigen 1.0\nlet x: int = 42",
            "eigen 1.0\nlet pi: float = 3.14",
            "eigen 1.0\nlet name: string = \"eigen\"",
            "eigen 1.0\nqubit q0\nH q0",
            "eigen 1.0\nqubit q0\nX q0",
            "eigen 1.0\nqubit q0\nY q0",
            "eigen 1.0\nqubit q0\nZ q0",
            "eigen 1.0\nqubit q0\nS q0",
            "eigen 1.0\nqubit q0\nT q0",
            "eigen 1.0\nqubit q0\nqubit q1\nCNOT q0, q1",
            "eigen 1.0\nqubit q0\nRX q0, PI / 2",
            "eigen 1.0\nqubit q0\nRY q0, PI",
            "eigen 1.0\nqubit q0\nRZ q0, PI / 4",
            "eigen 1.0\nqubit q0\ncbit c0\nmeasure q0 -> c0",
            "eigen 1.0\nqubit q0\ntrace",
            "eigen 1.0\nlet x: int = 1 + 2",
            "eigen 1.0\nlet x: int = 3 * 4 - 1",
        ]
        for source in programs:
            with self.subTest(source=source[:40]):
                ast = self._parse(source)
                self.assertIsInstance(ast, ProgramNode)

    def test_if_statement_variants(self):
        variants = [
            "eigen 1.0\nif x == 1 {\n    H q0\n}",
            "eigen 1.0\nif x != 0 {\n    X q0\n}",
            "eigen 1.0\nif x > 5 {\n    Y q0\n}",
            "eigen 1.0\nif x < 10 {\n    Z q0\n}",
        ]
        for source in variants:
            with self.subTest(source=source[:40]):
                ast = self._parse(source)
                ifs = [n for n in ast.body if isinstance(n, IfNode)]
                self.assertGreaterEqual(len(ifs), 1)

    def test_function_declarations(self):
        sources = [
            "eigen 1.0\nfunc add(a: int, b: int) -> int {\n    return a + b\n}",
            "eigen 1.0\nfunc noop() -> int {\n    return 0\n}",
            "eigen 1.0\nfunc identity(x: int) -> int {\n    return x\n}",
        ]
        for source in sources:
            with self.subTest(source=source[:40]):
                ast = self._parse(source)
                funcs = [n for n in ast.body if isinstance(n, FuncDeclNode)]
                self.assertGreaterEqual(len(funcs), 1)

    def test_version_numbers(self):
        versions = ["1.0", "2.0", "2.3"]
        for ver in versions:
            with self.subTest(version=ver):
                ast = self._parse(f"eigen {ver}")
                self.assertEqual(ast.version, float(ver))


# ---------------------------------------------------------------------------
# Type Checker tests (parameterized)
# ---------------------------------------------------------------------------
class TestTypeCheckerParameterized(unittest.TestCase):

    def _check(self, source: str):
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize())
        ast = parser.parse()
        tc = TypeChecker()
        tc.check(ast)

    def test_valid_type_declarations(self):
        decls = [
            "eigen 1.0\nlet x: int = 42",
            "eigen 1.0\nlet f: float = 3.14",
            "eigen 1.0\nlet s: string = \"hello\"",
            "eigen 1.0\nlet b: bool = true",
            "eigen 1.0\nqubit q0",
            "eigen 1.0\ncbit c0",
        ]
        for source in decls:
            with self.subTest(source=source[:40]):
                self._check(source)

    def test_valid_gate_applications(self):
        gates = [
            "eigen 1.0\nqubit q0\nH q0",
            "eigen 1.0\nqubit q0\nX q0",
            "eigen 1.0\nqubit q0\nY q0",
            "eigen 1.0\nqubit q0\nZ q0",
            "eigen 1.0\nqubit q0\nS q0",
            "eigen 1.0\nqubit q0\nT q0",
            "eigen 1.0\nqubit q0\nqubit q1\nCNOT q0, q1",
            "eigen 1.0\nqubit q0\nRX q0, PI",
            "eigen 1.0\nqubit q0\nRY q0, PI / 2",
            "eigen 1.0\nqubit q0\nRZ q0, PI / 4",
        ]
        for source in gates:
            with self.subTest(source=source[:40]):
                self._check(source)

    def test_gate_on_non_qubit_raises(self):
        bad_sources = [
            "eigen 1.0\ncbit c0\nH c0",
            "eigen 1.0\ncbit c0\nX c0",
            "eigen 1.0\nlet x: int = 1\nH x",
        ]
        for source in bad_sources:
            with self.subTest(source=source[:40]):
                with self.assertRaises(TypeErrorException):
                    self._check(source)

    def test_measure_on_non_qubit_raises(self):
        source = "eigen 1.0\nlet x: float = 1.0\nmeasure x -> c0"
        with self.assertRaises(TypeErrorException):
            self._check(source)


# ---------------------------------------------------------------------------
# VM Arithmetic & Logic tests (parameterized)
# ---------------------------------------------------------------------------
class TestVMArithmeticParameterized(unittest.TestCase):

    def _run_program(self, source: str) -> EigenVM:
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize())
        ast = parser.parse()
        tc = TypeChecker()
        tc.check(ast)
        compiler = EBCCompiler()
        instrs = compiler.compile_ast(ast)
        vm = EigenVM()
        vm.execute(instrs)
        return vm

    def test_arithmetic_operations(self):
        cases = [
            ("let x: int = 1 + 2", "x", 3),
            ("let x: int = 10 - 3", "x", 7),
            ("let x: int = 4 * 5", "x", 20),
            ("let x: int = 100 - 1", "x", 99),
            ("let x: int = 0 + 0", "x", 0),
            ("let x: int = 7 * 1", "x", 7),
            ("let x: int = 2 + 3 * 4", "x", 14),
            ("let x: int = 10 - 2 - 3", "x", 5),
            ("let x: int = 6 * 7", "x", 42),
            ("let x: int = 1 + 1 + 1 + 1 + 1", "x", 5),
        ]
        for expr, var, expected in cases:
            with self.subTest(expr=expr):
                vm = self._run_program(f"eigen 1.0\n{expr}")
                self.assertEqual(vm.lookup_var(var), expected)

    def test_comparison_operations(self):
        cases = [
            ("let x: bool = 5 > 3", "x", True),
            ("let x: bool = 3 > 5", "x", False),
            ("let x: bool = 5 < 10", "x", True),
            ("let x: bool = 10 < 5", "x", False),
            ("let x: bool = 5 == 5", "x", True),
            ("let x: bool = 5 == 6", "x", False),
            ("let x: bool = 5 != 6", "x", True),
            ("let x: bool = 5 != 5", "x", False),
            ("let x: bool = 5 >= 5", "x", True),
            ("let x: bool = 5 >= 6", "x", False),
            ("let x: bool = 5 <= 5", "x", True),
            ("let x: bool = 6 <= 5", "x", False),
        ]
        for expr, var, expected in cases:
            with self.subTest(expr=expr):
                vm = self._run_program(f"eigen 1.0\n{expr}")
                self.assertEqual(vm.lookup_var(var), expected)

    def test_string_assignment(self):
        vm = self._run_program('eigen 1.0\nlet s: string = "hello"')
        result = vm.lookup_var("s")
        # Result may be a VMRef or direct string
        if hasattr(result, 'ref_id'):
            obj = vm.heap[result.ref_id]
            self.assertEqual(obj.data, "hello")
        else:
            self.assertEqual(result, "hello")

    def test_boolean_values(self):
        cases = [
            ("let x: bool = true", "x", True),
            ("let x: bool = false", "x", False),
        ]
        for expr, var, expected in cases:
            with self.subTest(expr=expr):
                vm = self._run_program(f"eigen 1.0\n{expr}")
                self.assertEqual(vm.lookup_var(var), expected)


# ---------------------------------------------------------------------------
# VM Control Flow tests
# ---------------------------------------------------------------------------
class TestVMControlFlowParameterized(unittest.TestCase):

    def _run_program(self, source: str) -> EigenVM:
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize())
        ast = parser.parse()
        tc = TypeChecker()
        tc.check(ast)
        compiler = EBCCompiler()
        instrs = compiler.compile_ast(ast)
        vm = EigenVM()
        vm.execute(instrs)
        return vm

    def test_if_true_branch(self):
        source = "eigen 1.0\nlet x: int = 10\nlet y: int = 0\nif x == 10 {\n    y = 20\n}"
        vm = self._run_program(source)
        self.assertEqual(vm.lookup_var("y"), 20)

    def test_if_false_branch(self):
        source = "eigen 1.0\nlet x: int = 5\nlet y: int = 0\nif x == 10 {\n    y = 20\n}"
        vm = self._run_program(source)
        self.assertEqual(vm.lookup_var("y"), 0)

    def test_for_loop_sum(self):
        source = "eigen 1.0\nlet arr: array<int> = [1, 2, 3, 4, 5]\nlet s: int = 0\nfor x in arr {\n    s += x\n}"
        vm = self._run_program(source)
        self.assertEqual(vm.lookup_var("s"), 15)

    def test_for_loop_product(self):
        source = "eigen 1.0\nlet arr: array<int> = [1, 2, 3, 4]\nlet p: int = 1\nfor x in arr {\n    p *= x\n}"
        vm = self._run_program(source)
        self.assertEqual(vm.lookup_var("p"), 24)

    def test_recursive_factorial(self):
        source = """
        eigen 1.0
        func fact(n: int) -> int {
            if n == 0 {
                return 1
            }
            return n * fact(n - 1)
        }
        let r: int = fact(6)
        """
        vm = self._run_program(source)
        self.assertEqual(vm.lookup_var("r"), 720)

    def test_recursive_fibonacci(self):
        source = """
        eigen 1.0
        func fib(n: int) -> int {
            if n == 0 {
                return 0
            }
            if n == 1 {
                return 1
            }
            return fib(n - 1) + fib(n - 2)
        }
        let r: int = fib(8)
        """
        vm = self._run_program(source)
        self.assertEqual(vm.lookup_var("r"), 21)


# ---------------------------------------------------------------------------
# Simulator tests (parameterized single-qubit gates)
# ---------------------------------------------------------------------------
class TestSimulatorParameterized(unittest.TestCase):

    def test_single_qubit_gates_from_zero(self):
        """Apply each single-qubit gate to |0> and verify probabilities."""
        tests = [
            ('H', [0.5, 0.5]),
            ('X', [0.0, 1.0]),
            ('Y', [0.0, 1.0]),
            ('Z', [1.0, 0.0]),
            ('S', [1.0, 0.0]),
            ('T', [1.0, 0.0]),
        ]
        for gate, expected_probs in tests:
            with self.subTest(gate=gate):
                sim = QuantumSimulator()
                sim.allocate_qubit('q')
                getattr(sim, gate)('q')
                for i, prob in enumerate(expected_probs):
                    self.assertAlmostEqual(abs(sim.state_vector[i])**2, prob, places=6)

    def test_double_gate_identity(self):
        """Self-inverse gates applied twice return to |0>."""
        self_inverse = ['H', 'X', 'Y', 'Z']
        for gate in self_inverse:
            with self.subTest(gate=gate):
                sim = QuantumSimulator()
                sim.allocate_qubit('q')
                getattr(sim, gate)('q')
                getattr(sim, gate)('q')
                self.assertAlmostEqual(abs(sim.state_vector[0])**2, 1.0, places=6)

    def test_rotation_gates(self):
        """Rotation by 2*pi should return to |0>."""
        rotations = [
            ('RX', 2 * math.pi),
            ('RY', 2 * math.pi),
            ('RZ', 2 * math.pi),
        ]
        for gate, angle in rotations:
            with self.subTest(gate=gate, angle=angle):
                sim = QuantumSimulator()
                sim.allocate_qubit('q')
                getattr(sim, gate)('q', angle)
                self.assertAlmostEqual(abs(sim.state_vector[0])**2, 1.0, places=5)

    def test_rotation_half_pi(self):
        """RX(pi) should flip |0> to |1>."""
        sim = QuantumSimulator()
        sim.allocate_qubit('q')
        sim.RX('q', math.pi)
        self.assertAlmostEqual(abs(sim.state_vector[0])**2, 0.0, places=5)
        self.assertAlmostEqual(abs(sim.state_vector[1])**2, 1.0, places=5)

    def test_multi_qubit_allocation(self):
        """Allocating n qubits should produce a 2^n state vector."""
        for n in range(1, 7):
            with self.subTest(n=n):
                sim = QuantumSimulator()
                for i in range(n):
                    sim.allocate_qubit(f'q{i}')
                self.assertEqual(len(sim.state_vector), 2**n)
                self.assertAlmostEqual(abs(sim.state_vector[0])**2, 1.0)

    def test_bell_states(self):
        """Create all 4 Bell states and verify probabilities."""
        # |Φ+> = (|00> + |11>) / sqrt(2)
        sim = QuantumSimulator()
        sim.allocate_qubit('q0')
        sim.allocate_qubit('q1')
        sim.H('q0')
        sim.CNOT('q0', 'q1')
        self.assertAlmostEqual(abs(sim.state_vector[0])**2, 0.5)
        self.assertAlmostEqual(abs(sim.state_vector[3])**2, 0.5)

    def test_measurement_outcomes(self):
        """Measurement of |0> should always give 0; measurement of |1> should always give 1."""
        # |0>
        sim = QuantumSimulator()
        sim.allocate_qubit('q')
        outcome = sim.measure('q')
        self.assertEqual(outcome, 0)

        # |1>
        sim2 = QuantumSimulator()
        sim2.allocate_qubit('q')
        sim2.X('q')
        outcome2 = sim2.measure('q')
        self.assertEqual(outcome2, 1)

    def test_measurement_probabilistic(self):
        """Measurement of H|0> should give 0 or 1 over many trials."""
        outcomes = set()
        for _ in range(100):
            sim = QuantumSimulator()
            sim.allocate_qubit('q')
            sim.H('q')
            outcomes.add(sim.measure('q'))
            if len(outcomes) == 2:
                break
        self.assertEqual(outcomes, {0, 1})

    def test_ghz_states(self):
        """GHZ states for 2-5 qubits."""
        for n in range(2, 6):
            with self.subTest(n=n):
                sim = QuantumSimulator()
                for i in range(n):
                    sim.allocate_qubit(f'q{i}')
                sim.H('q0')
                for i in range(n - 1):
                    sim.CNOT(f'q{i}', f'q{i+1}')
                # Only |00...0> and |11...1> should have amplitude
                self.assertAlmostEqual(abs(sim.state_vector[0])**2, 0.5)
                self.assertAlmostEqual(abs(sim.state_vector[2**n - 1])**2, 0.5)
                for idx in range(1, 2**n - 1):
                    self.assertAlmostEqual(abs(sim.state_vector[idx])**2, 0.0,
                        msg=f"GHZ({n}): unexpected amplitude at index {idx}")


# ---------------------------------------------------------------------------
# Resource Estimator tests (parameterized)
# ---------------------------------------------------------------------------
class TestResourceEstimatorParameterized(unittest.TestCase):

    def _estimate(self, source: str) -> dict:
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize())
        ast = parser.parse()
        converter = EQIRConverter()
        graph = converter.convert(ast)
        estimator = ResourceEstimator()
        return estimator.estimate(graph)

    def test_qubit_counts(self):
        cases = [
            ("eigen 1.0\nqubit q0", 1),
            ("eigen 1.0\nqubit q0\nqubit q1", 2),
            ("eigen 1.0\nqubit q0\nqubit q1\nqubit q2", 3),
            ("eigen 1.0\nqubit q0\nqubit q1\nqubit q2\nqubit q3\nqubit q4", 5),
        ]
        for source, expected in cases:
            with self.subTest(qubits=expected):
                result = self._estimate(source)
                self.assertEqual(result['logical_qubits'], expected)

    def test_gate_counts(self):
        cases = [
            ("eigen 1.0\nqubit q0\nH q0", 1),
            ("eigen 1.0\nqubit q0\nH q0\nX q0", 2),
            ("eigen 1.0\nqubit q0\nH q0\nX q0\nY q0", 3),
            ("eigen 1.0\nqubit q0\nH q0\nX q0\nY q0\nZ q0", 4),
        ]
        for source, expected in cases:
            with self.subTest(gates=expected):
                result = self._estimate(source)
                self.assertEqual(result['gate_count'], expected)

    def test_cnot_count(self):
        source = "eigen 1.0\nqubit q0\nqubit q1\nCNOT q0, q1\nCNOT q1, q0"
        result = self._estimate(source)
        self.assertEqual(result['cnot_count'], 2)

    def test_t_count(self):
        source = "eigen 1.0\nqubit q0\nT q0\nT q0\nT q0"
        result = self._estimate(source)
        self.assertEqual(result['t_count'], 3)

    def test_clifford_count(self):
        source = "eigen 1.0\nqubit q0\nH q0\nS q0\nX q0\nY q0\nZ q0"
        result = self._estimate(source)
        self.assertEqual(result['clifford_count'], 5)

    def test_measurement_count(self):
        source = "eigen 1.0\nqubit q0\nqubit q1\ncbit c0\ncbit c1\nmeasure q0 -> c0\nmeasure q1 -> c1"
        result = self._estimate(source)
        self.assertEqual(result['measurements'], 2)

    def test_empty_circuit(self):
        source = "eigen 1.0"
        result = self._estimate(source)
        self.assertEqual(result['logical_qubits'], 0)
        self.assertEqual(result['gate_count'], 0)


# ---------------------------------------------------------------------------
# Noise Model tests (parameterized)
# ---------------------------------------------------------------------------
class TestNoiseModelParameterized(unittest.TestCase):

    def test_no_noise(self):
        """NoiseModel with no noise type should not modify anything."""
        nm = NoiseModel()
        sim = QuantumSimulator()
        sim.allocate_qubit('q')
        nm.apply_gate_noise(sim, 'q')
        self.assertAlmostEqual(abs(sim.state_vector[0])**2, 1.0)

    def test_zero_probability(self):
        """Zero probability should not apply noise."""
        for noise_type in ['bit_flip', 'phase_flip', 'depolarizing', 'amplitude_damping']:
            with self.subTest(noise=noise_type):
                nm = NoiseModel(noise_type, 0.0)
                sim = QuantumSimulator()
                sim.allocate_qubit('q')
                nm.apply_gate_noise(sim, 'q')
                self.assertAlmostEqual(abs(sim.state_vector[0])**2, 1.0)

    def test_full_probability_bit_flip(self):
        """Bit flip with p=1.0 should always flip."""
        nm = NoiseModel('bit_flip', 1.0)
        sim = QuantumSimulator()
        sim.allocate_qubit('q')
        nm.apply_gate_noise(sim, 'q')
        self.assertAlmostEqual(abs(sim.state_vector[1])**2, 1.0)

    def test_full_probability_phase_flip(self):
        """Phase flip with p=1.0 should apply Z gate."""
        nm = NoiseModel('phase_flip', 1.0)
        sim = QuantumSimulator()
        sim.allocate_qubit('q')
        sim.H('q')  # Put into superposition first
        nm.apply_gate_noise(sim, 'q')
        # Z on |+> gives |->
        # |-> = (|0> - |1>) / sqrt(2), probabilities still 0.5 each
        self.assertAlmostEqual(abs(sim.state_vector[0])**2, 0.5, places=5)
        self.assertAlmostEqual(abs(sim.state_vector[1])**2, 0.5, places=5)

    def test_readout_error_full_probability(self):
        """Readout error with p=1.0 should always flip outcome."""
        nm = NoiseModel('readout_error', 1.0)
        self.assertEqual(nm.apply_readout_noise(0), 1)
        self.assertEqual(nm.apply_readout_noise(1), 0)

    def test_readout_error_zero_probability(self):
        """Readout error with p=0.0 should never flip."""
        nm = NoiseModel('readout_error', 0.0)
        self.assertEqual(nm.apply_readout_noise(0), 0)
        self.assertEqual(nm.apply_readout_noise(1), 1)

    def test_readout_noise_not_applied_on_gate(self):
        """Readout error should NOT apply during gate noise."""
        nm = NoiseModel('readout_error', 1.0)
        sim = QuantumSimulator()
        sim.allocate_qubit('q')
        nm.apply_gate_noise(sim, 'q')
        # State should be unchanged since readout_error skips gate noise
        self.assertAlmostEqual(abs(sim.state_vector[0])**2, 1.0)

    def test_noise_model_types_exist(self):
        types = ['depolarizing', 'bit_flip', 'phase_flip', 'amplitude_damping', 'readout_error']
        for t in types:
            with self.subTest(noise_type=t):
                nm = NoiseModel(t, 0.1)
                self.assertEqual(nm.noise_type, t)
                self.assertAlmostEqual(nm.noise_prob, 0.1)


# ---------------------------------------------------------------------------
# EQIR Graph tests
# ---------------------------------------------------------------------------
class TestEQIRGraphParameterized(unittest.TestCase):

    def test_node_creation(self):
        graph = EQIRGraph()
        node = graph.create_node('ALLOC', targets=['q0'])
        self.assertEqual(node.type, 'ALLOC')
        self.assertEqual(node.targets, ['q0'])
        self.assertEqual(node.id, 0)

    def test_add_operation_creates_dependencies(self):
        graph = EQIRGraph()
        graph.add_operation('ALLOC', targets=['q0'])
        gate = graph.add_operation('GATE', gate_name='H', targets=['q0'])
        self.assertTrue(len(gate.parents) > 0)

    def test_topological_sort_ordering(self):
        graph = EQIRGraph()
        graph.add_operation('ALLOC', targets=['q0'])
        graph.add_operation('GATE', gate_name='H', targets=['q0'])
        graph.add_operation('GATE', gate_name='X', targets=['q0'])
        sorted_nodes = graph.topological_sort()
        ids = [n.id for n in sorted_nodes]
        self.assertEqual(ids, [0, 1, 2])

    def test_multiple_qubits(self):
        graph = EQIRGraph()
        graph.add_operation('ALLOC', targets=['q0'])
        graph.add_operation('ALLOC', targets=['q1'])
        graph.add_operation('GATE', gate_name='CNOT', targets=['q0', 'q1'])
        sorted_nodes = graph.topological_sort()
        self.assertEqual(len(sorted_nodes), 3)

    def test_measure_creates_cbit_dependency(self):
        graph = EQIRGraph()
        graph.add_operation('ALLOC', targets=['q0'])
        graph.add_operation('MEASURE', targets=['q0'], cbit_name='c0')
        sorted_nodes = graph.topological_sort()
        self.assertEqual(len(sorted_nodes), 2)


# ---------------------------------------------------------------------------
# Equivalence Checker tests (parameterized)
# ---------------------------------------------------------------------------
class TestEquivalenceParameterized(unittest.TestCase):

    def _make_circuit(self, source: str):
        return EQIRConverter().convert(Parser(Lexer(source).tokenize()).parse())

    def test_self_equivalence(self):
        """Every circuit should be equivalent to itself."""
        circuits = [
            "eigen 1.0\nqubit q0\nH q0",
            "eigen 1.0\nqubit q0\nX q0",
            "eigen 1.0\nqubit q0\nqubit q1\nCNOT q0, q1",
            "eigen 1.0\nqubit q0\nRX q0, 1.57",
        ]
        for source in circuits:
            with self.subTest(source=source[:40]):
                c = self._make_circuit(source)
                checker = EquivalenceChecker()
                self.assertTrue(checker.are_equivalent(c, c))

    def test_double_gate_identity_equivalence(self):
        """Applying a self-inverse gate twice is equivalent to identity."""
        gates = ['H', 'X', 'Y', 'Z']
        for gate in gates:
            with self.subTest(gate=gate):
                s1 = f"eigen 1.0\nqubit q0\n{gate} q0\n{gate} q0"
                s2 = "eigen 1.0\nqubit q0"
                c1 = self._make_circuit(s1)
                c2 = self._make_circuit(s2)
                checker = EquivalenceChecker()
                self.assertTrue(checker.are_equivalent(c1, c2))


# ---------------------------------------------------------------------------
# End-to-End integration tests (compile + execute)
# ---------------------------------------------------------------------------
class TestEndToEndParameterized(unittest.TestCase):

    def _run(self, source: str) -> EigenVM:
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize())
        ast = parser.parse()
        tc = TypeChecker()
        tc.check(ast)
        compiler = EBCCompiler()
        instrs = compiler.compile_ast(ast)
        vm = EigenVM()
        vm.execute(instrs)
        return vm

    def test_quantum_measure_deterministic_zero(self):
        source = "eigen 1.0\nqubit q\ncbit c\nmeasure q -> c"
        vm = self._run(source)
        self.assertEqual(vm.lookup_var("c"), 0)

    def test_quantum_measure_deterministic_one(self):
        source = "eigen 1.0\nqubit q\ncbit c\nX q\nmeasure q -> c"
        vm = self._run(source)
        self.assertEqual(vm.lookup_var("c"), 1)

    def test_quantum_measure_hadamard(self):
        source = "eigen 1.0\nqubit q\ncbit c\nH q\nmeasure q -> c"
        vm = self._run(source)
        self.assertIn(vm.lookup_var("c"), (0, 1))

    def test_struct_creation_and_access(self):
        source = """
        eigen 1.0
        struct Point {
            x: int,
            y: int
        }
        let p: Point = Point { x: 10, y: 20 }
        let px: int = p.x
        let py: int = p.y
        """
        vm = self._run(source)
        self.assertEqual(vm.lookup_var("px"), 10)
        self.assertEqual(vm.lookup_var("py"), 20)

    def test_array_operations(self):
        source = """
        eigen 1.0
        let arr: array<int> = [5, 10, 15]
        let first: int = arr[0]
        let second: int = arr[1]
        let third: int = arr[2]
        """
        vm = self._run(source)
        self.assertEqual(vm.lookup_var("first"), 5)
        self.assertEqual(vm.lookup_var("second"), 10)
        self.assertEqual(vm.lookup_var("third"), 15)

    def test_exception_handling(self):
        source = """
        eigen 1.0
        let caught: string = "none"
        try {
            throw "test_error"
        } catch (e) {
            caught = e
        }
        """
        vm = self._run(source)
        self.assertEqual(vm.lookup_var("caught"), "test_error")

    def test_multiple_functions(self):
        source = """
        eigen 1.0
        func double(n: int) -> int {
            return n * 2
        }
        func triple(n: int) -> int {
            return n * 3
        }
        let a: int = double(5)
        let b: int = triple(5)
        """
        vm = self._run(source)
        self.assertEqual(vm.lookup_var("a"), 10)
        self.assertEqual(vm.lookup_var("b"), 15)

    def test_nested_function_calls(self):
        source = """
        eigen 1.0
        func add(a: int, b: int) -> int {
            return a + b
        }
        func mul(a: int, b: int) -> int {
            return a * b
        }
        let r: int = add(mul(2, 3), mul(4, 5))
        """
        vm = self._run(source)
        self.assertEqual(vm.lookup_var("r"), 26)


# ---------------------------------------------------------------------------
# VM Bytecode Direct tests (parameterized)
# ---------------------------------------------------------------------------
class TestVMBytecodeParameterized(unittest.TestCase):

    def test_load_const_store_var(self):
        """LOAD_CONST + STORE_VAR for various types."""
        from src.backend.bytecode import Opcode, Instruction
        cases = [
            (42, 42), (0, 0), (-1, -1), (3.14, 3.14),
            (True, True), (False, False),
        ]
        for val, expected in cases:
            with self.subTest(val=val):
                vm = EigenVM()
                vm.execute([
                    Instruction(Opcode.LOAD_CONST, val),
                    Instruction(Opcode.STORE_VAR, "x"),
                    Instruction(Opcode.HALT),
                ])
                self.assertEqual(vm.lookup_var("x"), expected)

    def test_arithmetic_pairs(self):
        """Test all binary arithmetic ops via bytecode."""
        from src.backend.bytecode import Opcode, Instruction
        cases = [
            (Opcode.ADD, 3, 4, 7),
            (Opcode.ADD, 0, 0, 0),
            (Opcode.ADD, -5, 10, 5),
            (Opcode.SUB, 10, 3, 7),
            (Opcode.SUB, 0, 5, -5),
            (Opcode.MUL, 6, 7, 42),
            (Opcode.MUL, 0, 99, 0),
            (Opcode.MUL, -3, 4, -12),
            (Opcode.DIV, 10, 2, 5.0),
            (Opcode.DIV, 7, 1, 7.0),
        ]
        for op, a, b, expected in cases:
            with self.subTest(op=op, a=a, b=b):
                vm = EigenVM()
                vm.execute([
                    Instruction(Opcode.LOAD_CONST, a),
                    Instruction(Opcode.LOAD_CONST, b),
                    Instruction(op),
                    Instruction(Opcode.STORE_VAR, "r"),
                    Instruction(Opcode.HALT),
                ])
                self.assertEqual(vm.lookup_var("r"), expected)

    def test_comparison_pairs(self):
        from src.backend.bytecode import Opcode, Instruction
        cases = [
            (Opcode.EQ, 5, 5, True),
            (Opcode.EQ, 5, 6, False),
            (Opcode.NEQ, 5, 6, True),
            (Opcode.NEQ, 5, 5, False),
            (Opcode.LT, 3, 5, True),
            (Opcode.LT, 5, 3, False),
            (Opcode.LT, 5, 5, False),
            (Opcode.GT, 5, 3, True),
            (Opcode.GT, 3, 5, False),
            (Opcode.GT, 5, 5, False),
            (Opcode.LTE, 3, 5, True),
            (Opcode.LTE, 5, 5, True),
            (Opcode.LTE, 6, 5, False),
            (Opcode.GTE, 5, 3, True),
            (Opcode.GTE, 5, 5, True),
            (Opcode.GTE, 4, 5, False),
        ]
        for op, a, b, expected in cases:
            with self.subTest(op=op, a=a, b=b):
                vm = EigenVM()
                vm.execute([
                    Instruction(Opcode.LOAD_CONST, a),
                    Instruction(Opcode.LOAD_CONST, b),
                    Instruction(op),
                    Instruction(Opcode.STORE_VAR, "r"),
                    Instruction(Opcode.HALT),
                ])
                self.assertEqual(vm.lookup_var("r"), expected)

    def test_logical_ops(self):
        from src.backend.bytecode import Opcode, Instruction
        cases = [
            (Opcode.AND, True, True, True),
            (Opcode.AND, True, False, False),
            (Opcode.AND, False, True, False),
            (Opcode.AND, False, False, False),
            (Opcode.OR, True, True, True),
            (Opcode.OR, True, False, True),
            (Opcode.OR, False, True, True),
            (Opcode.OR, False, False, False),
        ]
        for op, a, b, expected in cases:
            with self.subTest(op=op, a=a, b=b):
                vm = EigenVM()
                vm.execute([
                    Instruction(Opcode.LOAD_CONST, a),
                    Instruction(Opcode.LOAD_CONST, b),
                    Instruction(op),
                    Instruction(Opcode.STORE_VAR, "r"),
                    Instruction(Opcode.HALT),
                ])
                self.assertEqual(vm.lookup_var("r"), expected)

    def test_not_op(self):
        from src.backend.bytecode import Opcode, Instruction
        for val, expected in [(True, False), (False, True), (0, True), (1, False)]:
            with self.subTest(val=val):
                vm = EigenVM()
                vm.execute([
                    Instruction(Opcode.LOAD_CONST, val),
                    Instruction(Opcode.NOT),
                    Instruction(Opcode.STORE_VAR, "r"),
                    Instruction(Opcode.HALT),
                ])
                self.assertEqual(vm.lookup_var("r"), expected)

    def test_jump_forward(self):
        from src.backend.bytecode import Opcode, Instruction
        vm = EigenVM()
        vm.execute([
            Instruction(Opcode.JMP, 3),            # 0: skip next 2
            Instruction(Opcode.LOAD_CONST, 999),    # 1: skipped
            Instruction(Opcode.STORE_VAR, "bad"),    # 2: skipped
            Instruction(Opcode.LOAD_CONST, 42),      # 3: target
            Instruction(Opcode.STORE_VAR, "good"),
            Instruction(Opcode.HALT),
        ])
        self.assertEqual(vm.lookup_var("good"), 42)
        with self.assertRaises(UndefinedVariableError):
            vm.lookup_var("bad")


# ---------------------------------------------------------------------------
# Simulator rotation sweep tests
# ---------------------------------------------------------------------------
class TestSimulatorRotationSweep(unittest.TestCase):

    def test_rx_sweep(self):
        """RX at 16 evenly spaced angles preserves normalization."""
        for i in range(16):
            theta = i * math.pi / 8
            with self.subTest(theta=theta):
                sim = QuantumSimulator()
                sim.allocate_qubit('q')
                sim.RX('q', theta)
                total_prob = sum(abs(a)**2 for a in sim.state_vector)
                self.assertAlmostEqual(total_prob, 1.0, places=10)

    def test_ry_sweep(self):
        for i in range(16):
            theta = i * math.pi / 8
            with self.subTest(theta=theta):
                sim = QuantumSimulator()
                sim.allocate_qubit('q')
                sim.RY('q', theta)
                total_prob = sum(abs(a)**2 for a in sim.state_vector)
                self.assertAlmostEqual(total_prob, 1.0, places=10)

    def test_rz_sweep(self):
        for i in range(16):
            theta = i * math.pi / 8
            with self.subTest(theta=theta):
                sim = QuantumSimulator()
                sim.allocate_qubit('q')
                sim.RZ('q', theta)
                total_prob = sum(abs(a)**2 for a in sim.state_vector)
                self.assertAlmostEqual(total_prob, 1.0, places=10)

    def test_rx_cancellation_sweep(self):
        """RX(theta) followed by RX(-theta) returns to |0> for many angles."""
        for i in range(20):
            theta = (i - 10) * 0.3
            with self.subTest(theta=theta):
                sim = QuantumSimulator()
                sim.allocate_qubit('q')
                sim.RX('q', theta)
                sim.RX('q', -theta)
                self.assertAlmostEqual(abs(sim.state_vector[0])**2, 1.0, places=8)

    def test_ry_cancellation_sweep(self):
        for i in range(20):
            theta = (i - 10) * 0.3
            with self.subTest(theta=theta):
                sim = QuantumSimulator()
                sim.allocate_qubit('q')
                sim.RY('q', theta)
                sim.RY('q', -theta)
                self.assertAlmostEqual(abs(sim.state_vector[0])**2, 1.0, places=8)

    def test_random_gate_preserves_norm(self):
        """Any sequence of gates preserves state vector normalization."""
        rng = random.Random(123)
        gates = ['H', 'X', 'Y', 'Z', 'S', 'T']
        for trial in range(20):
            with self.subTest(trial=trial):
                sim = QuantumSimulator()
                sim.allocate_qubit('q')
                seq_len = rng.randint(1, 8)
                for _ in range(seq_len):
                    getattr(sim, rng.choice(gates))('q')
                total_prob = sum(abs(a)**2 for a in sim.state_vector)
                self.assertAlmostEqual(total_prob, 1.0, places=10)

    def test_two_qubit_normalization(self):
        """Random gate sequences on 2 qubits preserve normalization."""
        rng = random.Random(456)
        single_gates = ['H', 'X', 'Y', 'Z', 'S', 'T']
        for trial in range(15):
            with self.subTest(trial=trial):
                sim = QuantumSimulator()
                sim.allocate_qubit('q0')
                sim.allocate_qubit('q1')
                for _ in range(rng.randint(2, 6)):
                    g = rng.choice(single_gates)
                    q = rng.choice(['q0', 'q1'])
                    getattr(sim, g)(q)
                if rng.random() > 0.5:
                    sim.CNOT('q0', 'q1')
                total_prob = sum(abs(a)**2 for a in sim.state_vector)
                self.assertAlmostEqual(total_prob, 1.0, places=10)


# ---------------------------------------------------------------------------
# EQIR Converter parameterized
# ---------------------------------------------------------------------------
class TestEQIRConverterParameterized(unittest.TestCase):

    def _convert(self, source: str) -> EQIRGraph:
        return EQIRConverter().convert(Parser(Lexer(source).tokenize()).parse())

    def test_gate_node_creation(self):
        gates = ['H', 'X', 'Y', 'Z', 'S', 'T']
        for gate in gates:
            with self.subTest(gate=gate):
                graph = self._convert(f"eigen 1.0\nqubit q0\n{gate} q0")
                sorted_nodes = graph.topological_sort()
                gate_nodes = [n for n in sorted_nodes if n.type == 'GATE']
                self.assertEqual(len(gate_nodes), 1)
                self.assertEqual(gate_nodes[0].gate_name, gate)

    def test_cnot_creates_two_qubit_gate(self):
        graph = self._convert("eigen 1.0\nqubit q0\nqubit q1\nCNOT q0, q1")
        sorted_nodes = graph.topological_sort()
        gate_nodes = [n for n in sorted_nodes if n.type == 'GATE']
        self.assertEqual(len(gate_nodes), 1)
        self.assertEqual(gate_nodes[0].gate_name, 'CNOT')
        self.assertEqual(len(gate_nodes[0].targets), 2)

    def test_measure_creates_measure_node(self):
        graph = self._convert("eigen 1.0\nqubit q0\ncbit c0\nmeasure q0 -> c0")
        sorted_nodes = graph.topological_sort()
        measure_nodes = [n for n in sorted_nodes if n.type == 'MEASURE']
        self.assertEqual(len(measure_nodes), 1)
        self.assertEqual(measure_nodes[0].cbit_name, 'c0')

    def test_multiple_gates_sequential(self):
        graph = self._convert("eigen 1.0\nqubit q0\nH q0\nX q0\nY q0\nZ q0\nS q0\nT q0")
        sorted_nodes = graph.topological_sort()
        gate_nodes = [n for n in sorted_nodes if n.type == 'GATE']
        self.assertEqual(len(gate_nodes), 6)
        expected_gates = ['H', 'X', 'Y', 'Z', 'S', 'T']
        for node, expected in zip(gate_nodes, expected_gates):
            self.assertEqual(node.gate_name, expected)

    def test_rotation_gates_with_args(self):
        rotations = ['RX', 'RY', 'RZ']
        for rot in rotations:
            with self.subTest(rotation=rot):
                graph = self._convert(f"eigen 1.0\nqubit q0\n{rot} q0, PI")
                sorted_nodes = graph.topological_sort()
                gate_nodes = [n for n in sorted_nodes if n.type == 'GATE']
                self.assertEqual(len(gate_nodes), 1)
                self.assertEqual(gate_nodes[0].gate_name, rot)
                self.assertTrue(len(gate_nodes[0].args) > 0)


# ---------------------------------------------------------------------------
# Noise Monte Carlo statistical tests
# ---------------------------------------------------------------------------
class TestNoiseStatistical(unittest.TestCase):

    def test_bit_flip_statistical(self):
        """Over many trials with p=0.5, roughly half should flip."""
        nm = NoiseModel('bit_flip', 0.5)
        flips = 0
        trials = 200
        for _ in range(trials):
            sim = QuantumSimulator()
            sim.allocate_qubit('q')
            nm.apply_gate_noise(sim, 'q')
            if abs(sim.state_vector[1])**2 > 0.5:
                flips += 1
        ratio = flips / trials
        self.assertGreater(ratio, 0.3)
        self.assertLess(ratio, 0.7)

    def test_readout_error_statistical(self):
        """Over many trials with p=0.5, roughly half should flip."""
        nm = NoiseModel('readout_error', 0.5)
        flips = 0
        trials = 200
        for _ in range(trials):
            result = nm.apply_readout_noise(0)
            if result == 1:
                flips += 1
        ratio = flips / trials
        self.assertGreater(ratio, 0.3)
        self.assertLess(ratio, 0.7)

    def test_depolarizing_applies_random_gate(self):
        """Depolarizing noise with p=1.0 always applies some Pauli."""
        nm = NoiseModel('depolarizing', 1.0)
        changes = 0
        trials = 100
        for _ in range(trials):
            sim = QuantumSimulator()
            sim.allocate_qubit('q')
            nm.apply_gate_noise(sim, 'q')
            if abs(sim.state_vector[0])**2 < 0.99:
                changes += 1
        # X and Y flip the state, Z doesn't - so ~2/3 should change
        self.assertGreater(changes, 30)

    def test_amplitude_damping_projects_to_zero(self):
        """Amplitude damping with p=1.0 on |1> should project to |0>."""
        nm = NoiseModel('amplitude_damping', 1.0)
        sim = QuantumSimulator()
        sim.allocate_qubit('q')
        sim.X('q')  # Put in |1>
        nm.apply_gate_noise(sim, 'q')
        self.assertAlmostEqual(abs(sim.state_vector[0])**2, 1.0, places=5)


# ---------------------------------------------------------------------------
# Bytecode serialization roundtrip tests
# ---------------------------------------------------------------------------
class TestBytecodeSerializationParameterized(unittest.TestCase):

    def test_instruction_roundtrip(self):
        """Instruction -> dict -> Instruction roundtrip."""
        from src.backend.bytecode import Opcode, Instruction
        cases = [
            Instruction(Opcode.LOAD_CONST, 42, 1),
            Instruction(Opcode.LOAD_CONST, 3.14, 2),
            Instruction(Opcode.LOAD_CONST, "hello", 3),
            Instruction(Opcode.LOAD_CONST, True, 4),
            Instruction(Opcode.STORE_VAR, "x", 5),
            Instruction(Opcode.ADD, None, 6),
            Instruction(Opcode.SUB, None, 7),
            Instruction(Opcode.MUL, None, 8),
            Instruction(Opcode.DIV, None, 9),
            Instruction(Opcode.JMP, 10, 10),
            Instruction(Opcode.JMP_IF_FALSE, 20, 11),
            Instruction(Opcode.HALT, None, 12),
            Instruction(Opcode.Q_ALLOC, "q0", 13),
            Instruction(Opcode.Q_GATE, ("H", ["q0"]), 14),
            Instruction(Opcode.Q_MEASURE, ("q0", "c0"), 15),
        ]
        for instr in cases:
            with self.subTest(opcode=instr.opcode):
                d = instr.to_dict()
                restored = Instruction.from_dict(d)
                self.assertEqual(restored.opcode, instr.opcode)
                self.assertEqual(restored.arg, instr.arg)
                self.assertEqual(restored.line, instr.line)

    def test_opcode_values_unique(self):
        """All opcode values should be unique strings."""
        from src.backend.bytecode import Opcode
        values = [v for k, v in vars(Opcode).items() if not k.startswith('_')]
        self.assertEqual(len(values), len(set(values)))


# ---------------------------------------------------------------------------
# Multi-qubit state tests
# ---------------------------------------------------------------------------
class TestMultiQubitStates(unittest.TestCase):

    def test_product_state_independence(self):
        """Measuring one qubit shouldn't affect another in a product state."""
        for _ in range(30):
            sim = QuantumSimulator()
            sim.allocate_qubit('q0')
            sim.allocate_qubit('q1')
            sim.X('q0')  # |1>
            # q1 remains |0>
            outcome_q1 = sim.measure('q1')
            self.assertEqual(outcome_q1, 0)
            outcome_q0 = sim.measure('q0')
            self.assertEqual(outcome_q0, 1)

    def test_swap_equivalence(self):
        """SWAP can be decomposed as 3 CNOTs."""
        for _ in range(10):
            sim1 = QuantumSimulator()
            sim1.allocate_qubit('q0')
            sim1.allocate_qubit('q1')
            sim1.X('q0')  # |10>
            sim1.CNOT('q0', 'q1')
            sim1.CNOT('q1', 'q0')
            sim1.CNOT('q0', 'q1')
            # Should now be |01>
            self.assertAlmostEqual(abs(sim1.state_vector[2])**2, 1.0, places=6)

    def test_hadamard_on_each_qubit(self):
        """H on each of n qubits creates uniform superposition."""
        for n in range(1, 6):
            with self.subTest(n=n):
                sim = QuantumSimulator()
                for i in range(n):
                    sim.allocate_qubit(f'q{i}')
                    sim.H(f'q{i}')
                expected_prob = 1.0 / (2**n)
                for idx in range(2**n):
                    self.assertAlmostEqual(
                        abs(sim.state_vector[idx])**2, expected_prob, places=8,
                        msg=f"Uniform superposition failed at n={n}, idx={idx}")


# ---------------------------------------------------------------------------
# Compiler end-to-end parameterized
# ---------------------------------------------------------------------------
class TestCompilerE2EParameterized(unittest.TestCase):

    def _compile_and_run(self, source: str) -> EigenVM:
        lexer = Lexer(source)
        parser = Parser(lexer.tokenize())
        ast = parser.parse()
        tc = TypeChecker()
        tc.check(ast)
        compiler = EBCCompiler()
        instrs = compiler.compile_ast(ast)
        vm = EigenVM()
        vm.execute(instrs)
        return vm

    def test_while_loop_countdown(self):
        source = """
        eigen 1.0
        let n: int = 10
        let s: int = 0
        while n > 0 {
            s += n
            n -= 1
        }
        """
        vm = self._compile_and_run(source)
        self.assertEqual(vm.lookup_var("s"), 55)

    def test_nested_if(self):
        source = """
        eigen 1.0
        let x: int = 10
        let r: int = 0
        if x > 5 {
            if x > 8 {
                r = 1
            }
        }
        """
        vm = self._compile_and_run(source)
        self.assertEqual(vm.lookup_var("r"), 1)

    def test_struct_field_update(self):
        source = """
        eigen 1.0
        struct Counter {
            value: int
        }
        let c: Counter = Counter { value: 0 }
        c.value = 42
        let v: int = c.value
        """
        vm = self._compile_and_run(source)
        self.assertEqual(vm.lookup_var("v"), 42)

    def test_array_sum_loop(self):
        source = """
        eigen 1.0
        let arr: array<int> = [2, 4, 6, 8, 10]
        let total: int = 0
        for x in arr {
            total += x
        }
        """
        vm = self._compile_and_run(source)
        self.assertEqual(vm.lookup_var("total"), 30)

    def test_multiple_quantum_ops(self):
        source = """
        eigen 1.0
        qubit q0
        qubit q1
        cbit c0
        cbit c1
        X q0
        H q1
        measure q0 -> c0
        measure q1 -> c1
        """
        vm = self._compile_and_run(source)
        self.assertEqual(vm.lookup_var("c0"), 1)
        self.assertIn(vm.lookup_var("c1"), (0, 1))

    def test_factorial_values(self):
        """Verify factorial for multiple input values."""
        for n, expected in [(0, 1), (1, 1), (2, 2), (3, 6), (4, 24), (5, 120), (6, 720)]:
            with self.subTest(n=n):
                source = f"""
                eigen 1.0
                func fact(n: int) -> int {{
                    if n == 0 {{
                        return 1
                    }}
                    return n * fact(n - 1)
                }}
                let r: int = fact({n})
                """
                vm = self._compile_and_run(source)
                self.assertEqual(vm.lookup_var("r"), expected)


if __name__ == "__main__":
    unittest.main()
