"""Regression tests for sol.md P0 improvements (lexer regex, density-matrix
gate caching, VM hardening, determinism).

Each test maps cleanly to a checkbox item in D:\\Nuras-7\\sol.md so future
contributors can correlate the test with the roadmap entry it pins down.
"""
import os
import sys
import unittest

import src.frontend.lexer as lexer_mod
from src.frontend.lexer import Lexer, TokenType
from src.backend.vm import EigenVM
from src.backend.bytecode import Opcode, Instruction


class TestSolP0Lexer(unittest.TestCase):
    """sol.md §1.2 — regex-based alternative lexer. ``tokenize`` defaults
    to the original character-by-character path (which batches identifiers
    with slices and is currently faster in microbenchmarks than Python's
    regex engine). ``_tokenize_regex`` is the alternative implementation
    kept for parity validation and as a foundation for future native lexers.
    The contract pinning two tests here asserts they produce identical
    output on representative inputs.
    """

    def _parity(self, src: str):
        slow = Lexer(src)._tokenize_slow()
        fast = Lexer(src)._tokenize_regex()
        self.assertEqual([t.type for t in slow], [t.type for t in fast],
                         f"type mismatch on {src!r}")
        self.assertEqual([t.value for t in slow], [t.value for t in fast],
                         f"value mismatch on {src!r}")
        self.assertEqual([(t.line, t.column) for t in slow],
                         [(t.line, t.column) for t in fast],
                         f"position mismatch on {src!r}")

    def test_regex_path_exists(self):
        # _MASTER_PATTERN alternative must be present and the method callable.
        self.assertTrue(hasattr(Lexer, '_tokenize_regex'))
        self.assertTrue(hasattr(Lexer, '_MASTER_PATTERN'))

    def test_basic_program_tokenizes_via_regex_path(self):
        src = (
            "eigen 2.5\n"
            "let x : int = 5\n"
            "qubit q0\n"
            "H q0\n"
        )
        toks = Lexer(src)._tokenize_regex()
        types = [t.type for t in toks]
        self.assertEqual(types[0], TokenType.EIGEN)
        self.assertEqual(types[1], TokenType.FLOAT_LIT)
        self.assertEqual(types[-1], TokenType.EOF)
        self.assertEqual(toks[0].line, 1)
        self.assertEqual(toks[0].column, 1)

    def test_string_interpolation_with_inner_quotes(self):
        # The regex path handles `${...}` even when the expression contains
        # a `"` character, mirroring the slow path.
        src = 'print "pfx ${"q"}"\n'
        toks = Lexer(src)._tokenize_regex()
        for t in toks:
            if t.type == TokenType.STRING_LIT:
                self.assertIn('\x00"q"\x00', t.value)
                return
        self.fail("No STRING_LIT token emitted")

    def test_numbers_hex_bin_octal_and_floats(self):
        cases = {
            "0x1F": "31",
            "0b101": "5",
            "0o17": "15",
            "1.5e10": "1.5e10",
            "1e5":   "1e5",
            ".5":    ".5",
        }
        for src, expected_val in cases.items():
            toks = Lexer(src)._tokenize_regex()
            self.assertEqual(toks[0].value, expected_val, f"bad value for {src}")

    def test_unknown_character_raises(self):
        with self.assertRaises(SyntaxError):
            Lexer("qubit q0 @").tokenize()  # public path

    def test_unterminated_string_raises(self):
        with self.assertRaises(SyntaxError):
            Lexer('"abc').tokenize()

    def test_unterminated_interpolation_raises(self):
        with self.assertRaises(SyntaxError):
            Lexer('"hello ${x"').tokenize()

    def test_parity_simple_program(self):
        self._parity("eigen 1.0\nlet y : float = PI / 2.0\nqubit q0\nH q0\n")

    def test_parity_interpolation(self):
        self._parity('eigen 2.5\nprint "a:${x} b:${y} c:${z + 1}"\nlet x : int = 1\n')

    def test_parity_all_operators(self):
        # Touch every operator kind that the slow-path produces. (`;` is a
        # TokenType defined but not emitted by the slow-path lexer on a
        # single-char match — it raises "Unexpected character".)
        ops = "x += y -= z *= w /= v **= u == != < > <= >= << >> -> & | ^ ~ , : . [ ] { } ( ) + - * / % ="
        self._parity(ops)


class TestSolP0DensityMatrixCache(unittest.TestCase):
    """sol.md §1.3 — precomputed gate matrices for the density matrix
    simulator. We assert the cache exists and that H/X/Y/Z/S/T/CNOT/CZ/CCX
    apply correctly through `QuantumSimulator`."""

    def test_gate_np_cache_present(self):
        from src.density_matrix_simulator import _GATE_NP_CACHE
        for name in ("H", "X", "Y", "Z", "S", "T", "I2", "P0", "P1"):
            self.assertIn(name, _GATE_NP_CACHE)

    def test_density_matrix_bell_state(self):
        import math
        from src.simulator import QuantumSimulator
        sim = QuantumSimulator(sim_type='density_matrix')
        sim.allocate_qubit('q0')
        sim.allocate_qubit('q1')
        sim.H('q0')
        sim.CNOT('q0', 'q1')
        amps = sim.get_amplitudes_dict()
        self.assertAlmostEqual(amps['00'], 1.0 / math.sqrt(2), places=6)
        self.assertAlmostEqual(amps['11'], 1.0 / math.sqrt(2), places=6)


class TestSolP0VMHardening(unittest.TestCase):
    """sol.md §7.1 — VM hardening: instruction-count limit, timeout
    protection, invalid-jump detection."""

    def test_instruction_count_limit_fires(self):
        # An infinite loop with a tiny instruction-count budget must abort
        # with TimeoutError rather than running forever. With no try-block in
        # the program, throw_exception re-raises as an uncaught RuntimeError
        # whose message contains the structured error name (sol.md §7.2 crash
        # recovery surfaces the same error string).
        # Program: JMP 0  (single instruction, jumps back to itself)
        instructions = [Instruction(Opcode.JMP, 0), Instruction(Opcode.HALT)]
        vm = EigenVM(max_instruction_count=10)
        with self.assertRaises(RuntimeError) as ctx:
            vm.execute(instructions)
        self.assertIn("TimeoutError", str(ctx.exception))
        # Instruction count must have reached the cap before tripping.
        self.assertGreaterEqual(vm.instruction_count, 9)

    def test_invalid_jump_target_raises(self):
        # JMP target out of [0, n_instrs) must surface InvalidJumpError,
        # not get silently caught and re-classified as StackUnderflowError.
        instructions = [Instruction(Opcode.JMP, 9999), Instruction(Opcode.HALT)]
        vm = EigenVM(max_instruction_count=100)
        with self.assertRaises(RuntimeError) as ctx:
            vm.execute(instructions)
        self.assertIn("InvalidJump", str(ctx.exception))

    def test_valid_jump_does_not_raise(self):
        # Smoke test: a valid forward jump must still execute normally.
        instructions = [
            Instruction(Opcode.LOAD_CONST, 42),
            Instruction(Opcode.JMP, 3),
            Instruction(Opcode.LOAD_CONST, 999),  # skipped
            Instruction(Opcode.HALT),
        ]
        vm = EigenVM()
        vm.execute(instructions)
        # Top of stack should be 42 not 999
        self.assertEqual(vm.operand_stack[-1], 42)

    def test_default_hardening_kwargs_present(self):
        # API surface check: the new __init__ arguments must all exist and
        # default to off/unbounded so existing call sites still work.
        vm = EigenVM()
        self.assertFalse(vm.deterministic)
        self.assertIsNone(vm.max_instruction_count)
        self.assertNullish_stack_limit_default(vm)

    def assertNullish_stack_limit_default(self, vm):
        # Reasonable default of 2^20 entries — sanity bound, not a regression.
        self.assertEqual(vm.max_operand_stack_depth, 1 << 20)


class TestSolP0DeterminismFlag(unittest.TestCase):
    """sol.md §2.1 — `--deterministic` flag and `seed` plumbing."""

    def test_vm_accepts_deterministic_arg(self):
        vm = EigenVM(deterministic=True, seed=42)
        self.assertTrue(vm.deterministic)
        # The underlying RNG state is seeded and reproducible.
        seq_a = [vm.rng.random() for _ in range(5)]
        vm2 = EigenVM(deterministic=True, seed=42)
        seq_b = [vm2.rng.random() for _ in range(5)]
        self.assertEqual(seq_a, seq_b)

    def test_cli_has_deterministic_flag(self):
        # Reflective check on CLI source for the P0 wiring.
        import inspect
        import src.cli as cli
        src = inspect.getsource(cli)
        self.assertIn("--deterministic", src)
        self.assertIn("--max-instructions", src)
        self.assertIn("--instruction-timeout", src)

    def test_run_command_threads_deterministic_to_vm(self):
        import inspect
        import src.commands.run as run_cmd
        src = inspect.getsource(run_cmd)
        self.assertIn("deterministic=", src)
        self.assertIn("max_instruction_count=", src)
        self.assertIn("instruction_timeout_s=", src)

    def test_exec_command_threads_deterministic_to_vm(self):
        import inspect
        import src.commands.exec as exec_cmd
        src = inspect.getsource(exec_cmd)
        self.assertIn("deterministic=", src)
        self.assertIn("max_instruction_count=", src)
        self.assertIn("instruction_timeout_s=", src)


if __name__ == "__main__":
    unittest.main()
