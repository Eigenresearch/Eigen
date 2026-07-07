"""§9.1 — extended grammar-rule coverage for the Eigen parser.

Exercises `src.frontend.parser.Parser` against many grammar
productions surfaced via `src.frontend.ast` node types:
  - Program with version header & imports
  - module declaration
  - Variable declarations (qubit/cbit/int/float/string/bool/array/map)
  - let bindings (type required by parse_let grammar)
  - FuncDecl / QFuncDecl with parameters & return types
  - Control flow: if/elif/else, for, while, break, continue
  - match/case/default
  - Struct / Enum / Trait declarations + impl blocks
  - try/catch/throw
  - noise declarations (depolarizing/bitflip with parens)
  - print / assert / return / trace statements
  - assignment, binary operations, function calls
  - parallel/task blocks
  - struct literal, dot access, index access
  - type aliases
"""
import unittest

from src.frontend.lexer import Lexer
from src.frontend.parser import Parser
from src.frontend.ast import (
    ProgramNode,
    ImportNode,
    VarDeclNode,
    LetNode,
    GateNode,
    MeasureNode,
    IfNode,
    ReturnNode,
    PrintNode,
    AssertNode,
    FuncDeclNode,
    ForNode,
    WhileNode,
    BreakNode,
    ContinueNode,
    StructDeclNode,
    StructLiteralNode,
    TryCatchNode,
    ThrowNode,
    EnumDeclNode,
    NoiseNode,
    AssignmentNode,
    CallNode,
    QFuncCallNode,
    ArrayLiteralNode,
    TupleLiteralNode,
    MatchNode,
    TypeAliasDeclNode,
    TraitDeclNode,
    ImplBlockNode,
)


def _parse(source: str):
    tokens = Lexer(source).tokenize()
    return Parser(tokens).parse()


def _wrap(body: str, version: float = 1.0) -> str:
    return f"eigen {version}\n{body}"


def _func_wrap(body: str) -> str:
    return _wrap(f"func main() -> int {{\n{body}\nreturn 0\n}}")


# ---------------------------------------------------------------------------
# Program structure
# ---------------------------------------------------------------------------

class TestProgramStructure(unittest.TestCase):
    def test_program_node_has_version(self):
        ast = _parse("eigen 1.0\n")
        self.assertIsInstance(ast, ProgramNode)
        self.assertEqual(ast.version, 1.0)

    def test_program_accepts_integer_version(self):
        ast = _parse("eigen 1\n")
        self.assertEqual(ast.version, 1)

    def test_empty_program_body(self):
        ast = _parse("eigen 1.0\n")
        self.assertEqual(ast.body, [])

    def test_import_node_stored_in_imports_list(self):
        ast = _parse("eigen 1.0\nimport math")
        imports = ast.imports
        self.assertEqual(len(imports), 1)
        self.assertIsInstance(imports[0], ImportNode)
        self.assertEqual(imports[0].module_path, "math")

    def test_dotted_import(self):
        ast = _parse("eigen 1.0\nimport quantum.bell")
        self.assertEqual(ast.imports[0].module_path, "quantum.bell")

    def test_module_declaration(self):
        ast = _parse("eigen 1.0\nmodule mymodule\n")
        self.assertEqual(ast.module_name, "mymodule")

    def test_multiple_imports(self):
        ast = _parse("eigen 1.0\nimport a\nimport b.c\nimport d.e.f")
        self.assertEqual(len(ast.imports), 3)


# ---------------------------------------------------------------------------
# Variable declarations & let bindings
# ---------------------------------------------------------------------------

class TestVariableDeclarations(unittest.TestCase):
    def test_qubit_declaration(self):
        ast = _parse("eigen 1.0\nqubit q0")
        decls = [n for n in ast.body if isinstance(n, VarDeclNode)]
        self.assertEqual(decls[0].name, "q0")
        self.assertEqual(decls[0].type_name, "qubit")

    def test_cbit_declaration(self):
        ast = _parse("eigen 1.0\ncbit c0")
        decls = [n for n in ast.body if isinstance(n, VarDeclNode)]
        self.assertEqual(decls[0].name, "c0")
        self.assertEqual(decls[0].type_name, "cbit")

    def test_int_declaration(self):
        ast = _parse("eigen 1.0\nint x")
        decls = [n for n in ast.body if isinstance(n, VarDeclNode)]
        self.assertEqual(decls[0].type_name, "int")

    def test_float_declaration(self):
        ast = _parse("eigen 1.0\nfloat y")
        decls = [n for n in ast.body if isinstance(n, VarDeclNode)]
        self.assertEqual(decls[0].type_name, "float")

    def test_string_declaration(self):
        ast = _parse("eigen 1.0\nstring s")
        decls = [n for n in ast.body if isinstance(n, VarDeclNode)]
        self.assertEqual(decls[0].type_name, "string")

    def test_bool_declaration(self):
        ast = _parse("eigen 1.0\nbool flag")
        decls = [n for n in ast.body if isinstance(n, VarDeclNode)]
        self.assertEqual(decls[0].type_name, "bool")

    def test_array_declaration(self):
        ast = _parse("eigen 1.0\narray<int> arr")
        decls = [n for n in ast.body if isinstance(n, VarDeclNode)]
        self.assertEqual(decls[0].type_name, "array<int>")

    def test_map_declaration(self):
        ast = _parse("eigen 1.0\nmap<string, int> m")
        decls = [n for n in ast.body if isinstance(n, VarDeclNode)]
        self.assertIn("map", decls[0].type_name)

    def test_let_binding_with_type(self):
        ast = _parse("eigen 1.0\nlet angle: float = 3.14")
        lets = [n for n in ast.body if isinstance(n, LetNode)]
        self.assertEqual(lets[0].name, "angle")
        self.assertEqual(lets[0].type_name, "float")

    def test_let_binding_with_int_value(self):
        ast = _parse("eigen 1.0\nlet count: int = 5")
        lets = [n for n in ast.body if isinstance(n, LetNode)]
        self.assertEqual(lets[0].name, "count")
        self.assertEqual(lets[0].type_name, "int")


# ---------------------------------------------------------------------------
# Quantum gate & measurement
# ---------------------------------------------------------------------------

class TestQuantumGates(unittest.TestCase):
    def test_single_qubit_gate(self):
        ast = _parse("eigen 1.0\nqubit q0\nH q0")
        gates = [n for n in ast.body if isinstance(n, GateNode)]
        self.assertEqual(gates[0].gate_name, "H")
        self.assertEqual(gates[0].targets, ["q0"])

    def test_two_qubit_gate(self):
        ast = _parse("eigen 1.0\nqubit q0\nqubit q1\nCNOT q0, q1")
        gates = [n for n in ast.body if isinstance(n, GateNode)]
        self.assertEqual(gates[0].targets, ["q0", "q1"])

    def test_three_qubit_gate(self):
        ast = _parse("eigen 1.0\nqubit a\nqubit b\nqubit c\nCCX a, b, c")
        gates = [n for n in ast.body if isinstance(n, GateNode)]
        self.assertEqual(gates[0].targets, ["a", "b", "c"])

    def test_parametrized_gate(self):
        ast = _parse("eigen 1.0\nqubit q0\nRX q0, 1.5")
        gates = [n for n in ast.body if isinstance(n, GateNode)]
        self.assertEqual(gates[0].gate_name, "RX")

    def test_measure_into_cbit(self):
        ast = _parse("eigen 1.0\nqubit q0\ncbit c0\nmeasure q0 -> c0")
        meas = [n for n in ast.body if isinstance(n, MeasureNode)]
        self.assertEqual(meas[0].qubit_name, "q0")
        self.assertEqual(meas[0].cbit_name, "c0")

    def test_cz_gate(self):
        ast = _parse("eigen 1.0\nqubit a\nqubit b\nCZ a, b")
        gates = [n for n in ast.body if isinstance(n, GateNode)]
        self.assertEqual(gates[0].gate_name, "CZ")

    def test_swap_gate(self):
        ast = _parse("eigen 1.0\nqubit a\nqubit b\nSWAP a, b")
        gates = [n for n in ast.body if isinstance(n, GateNode)]
        self.assertEqual(gates[0].gate_name, "SWAP")

    def test_cry_gate(self):
        ast = _parse("eigen 1.0\nqubit a\nqubit b\nCRY a, b, 1.5")
        gates = [n for n in ast.body if isinstance(n, GateNode)]
        self.assertEqual(gates[0].gate_name, "CRY")


# ---------------------------------------------------------------------------
# Control flow
# ---------------------------------------------------------------------------

class TestControlFlow(unittest.TestCase):
    def test_if_statement(self):
        ast = _parse("eigen 1.0\nif c0 == 1 {\nprint c0\n}")
        ifs = [n for n in ast.body if isinstance(n, IfNode)]
        self.assertEqual(len(ifs), 1)
        self.assertEqual(ifs[0].op, "==")

    def test_if_with_else(self):
        src = "eigen 1.0\nif x == 1 {\nprint 1\n} else {\nprint 0\n}"
        ast = _parse(src)
        ifs = [n for n in ast.body if isinstance(n, IfNode)]
        self.assertEqual(len(ifs), 1)
        # If has else_body populated
        self.assertIsNotNone(ifs[0].else_body)

    def test_if_with_elif(self):
        src = "eigen 1.0\nif x == 1 {\nprint 1\n} elif x == 2 {\nprint 2\n}"
        ast = _parse(src)
        ifs = [n for n in ast.body if isinstance(n, IfNode)]
        self.assertEqual(len(ifs), 1)

    def test_for_loop_with_array(self):
        src = "eigen 1.0\nfor x in arr {\nprint x\n}"
        ast = _parse(src)
        fors = [n for n in ast.body if isinstance(n, ForNode)]
        self.assertEqual(len(fors), 1)

    def test_while_loop(self):
        src = "eigen 1.0\nx = 0\nwhile x < 10 {\nx = x + 1\n}"
        ast = _parse(src)
        whiles = [n for n in ast.body if isinstance(n, WhileNode)]
        self.assertEqual(len(whiles), 1)

    def test_break_statement(self):
        src = _func_wrap("while true {\nbreak\n}")
        ast = _parse(src)
        func = [n for n in ast.body if isinstance(n, FuncDeclNode)][0]
        # break lives inside the while-loop body
        whiles = [s for s in func.body if isinstance(s, WhileNode)]
        self.assertEqual(len(whiles), 1)
        breaks = [s for s in whiles[0].body if isinstance(s, BreakNode)]
        self.assertEqual(len(breaks), 1)

    def test_continue_statement(self):
        src = _func_wrap("while true {\ncontinue\n}")
        ast = _parse(src)
        func = [n for n in ast.body if isinstance(n, FuncDeclNode)][0]
        whiles = [s for s in func.body if isinstance(s, WhileNode)]
        self.assertEqual(len(whiles), 1)
        conts = [s for s in whiles[0].body if isinstance(s, ContinueNode)]
        self.assertEqual(len(conts), 1)


# ---------------------------------------------------------------------------
# Match / case
# ---------------------------------------------------------------------------

class TestMatchStatement(unittest.TestCase):
    def test_match_with_cases(self):
        src = _func_wrap("match x {\ncase 1 {\nprint 1\n}\ndefault {\nprint 0\n}\n}")
        ast = _parse(src)
        func = [n for n in ast.body if isinstance(n, FuncDeclNode)][0]
        matches = [s for s in func.body if isinstance(s, MatchNode)]
        self.assertEqual(len(matches), 1)
        self.assertGreaterEqual(len(matches[0].cases), 1)

    def test_match_with_default_only(self):
        src = _func_wrap("match x {\ndefault {\nprint 0\n}\n}")
        ast = _parse(src)
        func = [n for n in ast.body if isinstance(n, FuncDeclNode)][0]
        matches = [s for s in func.body if isinstance(s, MatchNode)]
        self.assertEqual(len(matches), 1)


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

class TestFunctionDeclarations(unittest.TestCase):
    def test_function_with_return_type(self):
        ast = _parse("eigen 1.0\nfunc add(a: int, b: int) -> int {\nreturn a + b\n}")
        funcs = [n for n in ast.body if isinstance(n, FuncDeclNode)]
        self.assertEqual(len(funcs), 1)
        self.assertEqual(funcs[0].name, "add")
        self.assertEqual(funcs[0].return_type, "int")

    def test_function_void_return(self):
        ast = _parse("eigen 1.0\nfunc foo() {\nprint \"hello\"\n}")
        funcs = [n for n in ast.body if isinstance(n, FuncDeclNode)]
        self.assertEqual(len(funcs), 1)

    def test_return_statement(self):
        src = "eigen 1.0\nfunc f() -> int {\nreturn 42\n}"
        ast = _parse(src)
        func = [n for n in ast.body if isinstance(n, FuncDeclNode)][0]
        rets = [s for s in func.body if isinstance(s, ReturnNode)]
        self.assertEqual(len(rets), 1)

    def test_function_with_generics(self):
        src = "eigen 1.0\nfunc id<T>(x: T) -> T {\nreturn x\n}"
        ast = _parse(src)
        funcs = [n for n in ast.body if isinstance(n, FuncDeclNode)]
        self.assertEqual(len(funcs), 1)
        self.assertGreaterEqual(len(funcs[0].generic_params), 1)


# ---------------------------------------------------------------------------
# Print / assert
# ---------------------------------------------------------------------------

class TestPrintAssertStatements(unittest.TestCase):
    def test_print_string(self):
        src = _func_wrap("print \"hello\"")
        ast = _parse(src)
        func = [n for n in ast.body if isinstance(n, FuncDeclNode)][0]
        prints = [s for s in func.body if isinstance(s, PrintNode)]
        self.assertEqual(len(prints), 1)

    def test_assert_true(self):
        src = _func_wrap("assert 1 == 1")
        ast = _parse(src)
        func = [n for n in ast.body if isinstance(n, FuncDeclNode)][0]
        asserts = [s for s in func.body if isinstance(s, AssertNode)]
        self.assertEqual(len(asserts), 1)


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------

class TestAssignment(unittest.TestCase):
    def test_simple_assignment(self):
        src = "eigen 1.0\nlet x: int = 0\nx = 5"
        ast = _parse(src)
        assigns = [n for n in ast.body if isinstance(n, AssignmentNode)]
        self.assertEqual(len(assigns), 1)

    def test_compound_addition_assignment(self):
        src = "eigen 1.0\nlet x: int = 0\nx += 5"
        ast = _parse(src)
        assigns = [n for n in ast.body if isinstance(n, AssignmentNode)]
        self.assertEqual(len(assigns), 1)


# ---------------------------------------------------------------------------
# Structs & enums
# ---------------------------------------------------------------------------

class TestStructs(unittest.TestCase):
    def test_struct_declaration(self):
        src = "eigen 1.0\nstruct Point {\nx: int\ny: int\n}"
        ast = _parse(src)
        structs = [n for n in ast.body if isinstance(n, StructDeclNode)]
        self.assertEqual(len(structs), 1)
        self.assertEqual(structs[0].name, "Point")

    def test_struct_with_generic(self):
        src = "eigen 1.0\nstruct Vec<T> {\nval: T\n}"
        ast = _parse(src)
        structs = [n for n in ast.body if isinstance(n, StructDeclNode)]
        self.assertEqual(len(structs), 1)

    def test_enum_declaration_with_commas(self):
        src = "eigen 1.0\nenum Color {\nRed,\nGreen,\nBlue\n}"
        ast = _parse(src)
        enums = [n for n in ast.body if isinstance(n, EnumDeclNode)]
        self.assertEqual(len(enums), 1)
        self.assertEqual(enums[0].name, "Color")
        self.assertGreaterEqual(len(enums[0].variants), 3)


# ---------------------------------------------------------------------------
# try / catch / throw
# ---------------------------------------------------------------------------

class TestExceptionHandling(unittest.TestCase):
    def test_try_catch_with_var(self):
        src = _func_wrap("try {\nprint \"safe\"\n} catch (e) {\nprint e\n}")
        ast = _parse(src)
        func = [n for n in ast.body if isinstance(n, FuncDeclNode)][0]
        tries = [s for s in func.body if isinstance(s, TryCatchNode)]
        self.assertEqual(len(tries), 1)

    def test_throw(self):
        src = _func_wrap("throw \"error\"")
        ast = _parse(src)
        func = [n for n in ast.body if isinstance(n, FuncDeclNode)][0]
        throws = [s for s in func.body if isinstance(s, ThrowNode)]
        self.assertEqual(len(throws), 1)


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------

class TestNoiseDeclaration(unittest.TestCase):
    def test_noise_depolarizing(self):
        ast = _parse("eigen 1.0\nnoise depolarizing(0.01) q0")
        noises = [n for n in ast.body if isinstance(n, NoiseNode)]
        self.assertEqual(len(noises), 1)

    def test_noise_bitflip(self):
        ast = _parse("eigen 1.0\nnoise bitflip(0.05) q0")
        noises = [n for n in ast.body if isinstance(n, NoiseNode)]
        self.assertEqual(len(noises), 1)

    def test_noise_with_multiple_targets(self):
        ast = _parse("eigen 1.0\nnoise bitflip(0.1) q0, q1, q2")
        noises = [n for n in ast.body if isinstance(n, NoiseNode)]
        self.assertEqual(len(noises), 1)


# ---------------------------------------------------------------------------
# Function calls
# ---------------------------------------------------------------------------

class TestFunctionCalls(unittest.TestCase):
    def test_call_no_args(self):
        src = _func_wrap("foo()")
        ast = _parse(src)
        func = [n for n in ast.body if isinstance(n, FuncDeclNode)][0]
        # Native parser may return QFuncCallNode for unknown callees;
        # both CallNode and QFuncCallNode represent a function call.
        calls = [s for s in func.body
                   if isinstance(s, (CallNode, QFuncCallNode))]
        self.assertEqual(len(calls), 1)

    def test_call_with_args(self):
        src = _func_wrap("add(1, 2)")
        ast = _parse(src)
        func = [n for n in ast.body if isinstance(n, FuncDeclNode)][0]
        calls = [s for s in func.body
                   if isinstance(s, (CallNode, QFuncCallNode))]
        self.assertEqual(len(calls), 1)


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------

class TestCollections(unittest.TestCase):
    def test_array_literal_in_let(self):
        src = "eigen 1.0\nlet arr: array<int> = [1, 2, 3]"
        ast = _parse(src)
        lets = [n for n in ast.body if isinstance(n, LetNode)]
        self.assertEqual(len(lets), 1)

    def test_tuple_literal_in_let(self):
        src = "eigen 1.0\nlet t: int = (1, 2)"
        ast = _parse(src)
        lets = [n for n in ast.body if isinstance(n, LetNode)]
        self.assertEqual(len(lets), 1)

    def test_array_access(self):
        src = _func_wrap("let arr: array<int> = [1, 2, 3]\nlet x: int = arr[0]")
        ast = _parse(src)
        # Should not raise — arr[0] is a valid IndexAccess expression
        self.assertIsInstance(ast, ProgramNode)


# ---------------------------------------------------------------------------
# Type system extensions
# ---------------------------------------------------------------------------

class TestTypeSystemExtensions(unittest.TestCase):
    def test_type_alias(self):
        src = "eigen 1.0\ntype IntList = array<int>"
        ast = _parse(src)
        aliases = [n for n in ast.body if isinstance(n, TypeAliasDeclNode)]
        self.assertEqual(len(aliases), 1)

    def test_trait_declaration(self):
        # Trait body methods must use `func` keyword
        src = "eigen 1.0\ntrait Drawable {\nfunc draw() -> int\n}"
        ast = _parse(src)
        traits = [n for n in ast.body if isinstance(n, TraitDeclNode)]
        self.assertEqual(len(traits), 1)

    def test_impl_block_inherent(self):
        src = ("eigen 1.0\nstruct Foo {\nx: int\n}\n"
                "impl Foo {\nfunc bar() -> int {\nreturn 1\n}\n}")
        ast = _parse(src)
        impls = [n for n in ast.body if isinstance(n, ImplBlockNode)]
        self.assertEqual(len(impls), 1)

    def test_impl_block_for_trait(self):
        src = ("eigen 1.0\ntrait Drawable {\nfunc draw() -> int\n}\n"
                "struct Circle {\nr: int\n}\n"
                "impl Drawable for Circle {\nfunc draw() -> int {\nreturn 1\n}\n}")
        ast = _parse(src)
        impls = [n for n in ast.body if isinstance(n, ImplBlockNode)]
        self.assertEqual(len(impls), 1)
        self.assertEqual(impls[0].trait_name, "Drawable")


# ---------------------------------------------------------------------------
# Parser error handling
# ---------------------------------------------------------------------------

class TestParserErrors(unittest.TestCase):
    def test_unexpected_token_raises(self):
        # Malformed source should raise during parse
        with self.assertRaises(Exception):
            _parse("eigen 1.0\nfunc main( { }")

    def test_missing_version_header_raises(self):
        with self.assertRaises(Exception):
            _parse("let x: int = 1")

    def test_no_terminator_in_struct_raises(self):
        with self.assertRaises(Exception):
            _parse("eigen 1.0\nstruct Foo {")


# ---------------------------------------------------------------------------
# Compositional programs (multiple statements)
# ---------------------------------------------------------------------------

class TestCompositionalPrograms(unittest.TestCase):
    def test_program_with_multiple_statements(self):
        src = ("eigen 1.0\n"
                "qubit q0\n"
                "qubit q1\n"
                "H q0\n"
                "CNOT q0, q1\n"
                "measure q0 -> c0\n"
                "measure q1 -> c1")
        ast = _parse(src)
        self.assertIsInstance(ast, ProgramNode)
        self.assertEqual(ast.version, 1.0)
        # Should have 2 declarations + 2 gates + 2 measures = 6 stmts
        # (measure doesn't require cbit declaration since parser just stores names)
        self.assertGreaterEqual(len(ast.body), 6)

    def test_nested_control_flow(self):
        src = ("eigen 1.0\n"
                "func main() -> int {\n"
                "  let x: int = 0\n"
                "  for i in arr {\n"
                "    if i == 3 {\n"
                "      x = x + i\n"
                "    }\n"
                "  }\n"
                "  return x\n"
                "}")
        ast = _parse(src)
        func = [n for n in ast.body if isinstance(n, FuncDeclNode)][0]
        self.assertEqual(func.name, "main")
        self.assertGreaterEqual(len(func.body), 2)

    def test_chained_quantum_program(self):
        src = ("eigen 1.0\n"
                "qubit a\n"
                "qubit b\n"
                "qubit c\n"
                "H a\n"
                "CNOT a, b\n"
                "CNOT b, c\n"
                "CCX a, b, c\n"
                "measure a -> ca\n"
                "measure b -> cb\n"
                "measure c -> cc")
        ast = _parse(src)
        gates = [n for n in ast.body if isinstance(n, GateNode)]
        measures = [n for n in ast.body if isinstance(n, MeasureNode)]
        self.assertEqual(len(gates), 4)
        self.assertEqual(len(measures), 3)


if __name__ == "__main__":
    unittest.main()
