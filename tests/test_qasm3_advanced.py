"""§4.2 — Extended OpenQASM 3.0 tests, organised by the four
roadmap checkboxes."""
import math
import unittest

from src.qasm3_advanced import (
    Calibration,
    ConditionalBlock,
    Qasm3ExportOptions,
    Qasm3Importer,
    Qasm3Program,
    Subroutine,
    eqir_to_extended_qasm3,
    qasm3_to_eqir,
    tokenize,
)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class TestTokenizer(unittest.TestCase):
    def test_basic_tokens(self):
        toks = tokenize("OPENQASM 3.0;")
        self.assertEqual([t.text for t in toks],
                          ["OPENQASM", "3.0", ";"])

    def test_skip_whitespace_and_comments(self):
        toks = tokenize("// comment\nh q[0]; /* block */")
        kinds = [t.kind for t in toks]
        self.assertNotIn("LCOMMENT", kinds)
        self.assertNotIn("BCOMMENT", kinds)
        self.assertNotIn("WS", kinds)

    def test_strings(self):
        toks = tokenize('include "stdgates.inc";')
        # Tokens: include, stdgates.inc, ;
        self.assertEqual(toks[0].text, "include")
        self.assertEqual(toks[1].text, '"stdgates.inc"')
        self.assertEqual(toks[2].text, ";")

    def test_string_takes_quotes(self):
        toks = tokenize('include "stdlib";')
        self.assertTrue(toks[1].text.startswith('"'))
        self.assertTrue(toks[1].text.endswith('"'))

    def test_multi_char_operators(self):
        toks = tokenize("a == b != c <= d >= e < f > g")
        ops = [t.text for t in toks if t.kind == "OP"]
        self.assertEqual(ops, ["==", "!=", "<=", ">=", "<", ">"])

    def test_dot_token(self):
        toks = tokenize("3.0")
        # `3.0` parses as a single NUMBER token including `.`.
        self.assertEqual(toks[0].kind, "NUMBER")
        self.assertEqual(toks[0].text, "3.0")


# ---------------------------------------------------------------------------
# Subroutine envelope
# ---------------------------------------------------------------------------

class TestSubroutine(unittest.TestCase):
    def test_render_no_params(self):
        s = Subroutine(name="my_gate", qubits=["q"],
                        body_lines=["x q;"])
        rendered = s.render()
        self.assertIn("gate my_gate q {", rendered)
        self.assertIn("x q;", rendered)

    def test_render_with_params(self):
        s = Subroutine(name="mygate", params=["theta"],
                        qubits=["q"], body_lines=["rz(theta) q;"])
        rendered = s.render()
        self.assertIn("gate mygate(theta) q", rendered)
        self.assertIn("rz(theta) q;", rendered)


class TestCalibration(unittest.TestCase):
    def test_render_no_params(self):
        c = Calibration(name="x_pulse", qubits=["q"],
                          body_text="// pulse def")
        rendered = c.render()
        self.assertIn("defcal x_pulse q {", rendered)
        self.assertIn("// pulse def", rendered)

    def test_render_with_params(self):
        c = Calibration(name="xpulse", params=["duration"],
                          qubits=["q"], body_text="pulse ...")
        rendered = c.render()
        self.assertIn("defcal xpulse(duration) q", rendered)


class TestConditionalBlock(unittest.TestCase):
    def test_render_if_only(self):
        cb = ConditionalBlock(
            condition_left="c[0]", condition_op="==", condition_right="1",
            then_body=["x q[0];"])
        rendered = cb.render()
        self.assertIn("if (c[0] == 1) {", rendered)
        self.assertIn("x q[0];", rendered)
        self.assertNotIn("else", rendered)

    def test_render_if_else(self):
        cb = ConditionalBlock(
            condition_left="c[0]", condition_op="==", condition_right="1",
            then_body=["x q[0];"], else_body=["z q[0];"])
        rendered = cb.render()
        self.assertIn("if (c[0] == 1) {", rendered)
        self.assertIn("} else {", rendered)
        self.assertIn("z q[0];", rendered)


# ---------------------------------------------------------------------------
# QASM3 Importer
# ---------------------------------------------------------------------------

class TestQasm3Importer(unittest.TestCase):
    def setUp(self):
        self.i = Qasm3Importer()

    def test_parse_simple_h(self):
        program = self.i.parse(
            "OPENQASM 3.0;\ninclude \"stdgates.inc\";\n"
            "qubit[2] q;\nh q[0];\n")
        # "3.0" lexes as a single NUMBER token.
        self.assertEqual(program.version, "3.0")
        self.assertIn("stdgates.inc", program.includes)
        self.assertEqual(program.qubit_count, 2)
        self.assertEqual(len(program.gates), 1)
        self.assertEqual(program.gates[0]["name"], "h")
        self.assertEqual(program.gates[0]["targets"], [0])

    def test_parse_multiple_gates(self):
        program = self.i.parse(
            "qubit[3] q;\n"
            "h q[0];\n"
            "cx q[0], q[1];\n"
            "cx q[1], q[2];\n")
        self.assertEqual(program.qubit_count, 3)
        self.assertEqual(len(program.gates), 3)
        self.assertEqual(program.gates[0]["name"], "h")
        self.assertEqual(program.gates[1]["name"], "cx")
        self.assertEqual(program.gates[1]["targets"], [0, 1])
        self.assertEqual(program.gates[2]["targets"], [1, 2])

    def test_parse_rotation_with_arg(self):
        program = self.i.parse(
            "qubit[1] q;\n"
            "rx(1.0) q[0];")
        self.assertEqual(program.gates[0]["name"], "rx")
        self.assertEqual(program.gates[0]["args"], [1.0])

    def test_parse_measurements(self):
        program = self.i.parse(
            "qubit[2] q;\n"
            "bit[2] c;\n"
            "c[0] = measure q[0];\n"
            "c[1] = measure q[1];\n")
        self.assertEqual(program.qubit_count, 2)
        self.assertEqual(program.bit_count, 2)
        self.assertEqual(len(program.measures), 2)
        self.assertEqual(program.measures[0]["qubit_idx"], 0)
        self.assertEqual(program.measures[1]["qubit_idx"], 1)

    def test_parse_subroutine(self):
        program = self.i.parse(
            "gate mygate q {\n"
            "  h q;\n"
            "  x q;\n"
            "}\n"
            "qubit[1] q;\n"
            "mygate q[0];\n")
        self.assertEqual(len(program.subroutines), 1)
        self.assertEqual(program.subroutines[0].name, "mygate")
        self.assertEqual(program.subroutines[0].qubits, ["q"])
        self.assertEqual(len(program.subroutines[0].body_lines), 2)

    def test_subroutine_inline_expanded(self):
        program = self.i.parse(
            "gate bell q0, q1 {\n"
            "  h q0;\n"
            "  cx q0, q1;\n"
            "}\n"
            "qubit[2] q;\n"
            "bell q[0], q[1];\n")
        self.assertEqual(len(program.gates), 2)
        self.assertEqual(program.gates[0]["name"], "h")
        self.assertEqual(program.gates[0]["targets"], [0])
        self.assertEqual(program.gates[1]["name"], "cx")
        self.assertEqual(program.gates[1]["targets"], [0, 1])

    def test_parse_calibration(self):
        program = self.i.parse(
            "defcal x_pulse(40ns) q {\n"
            "  pulse DRAG q\n"
            "}\n")
        self.assertEqual(len(program.calibrations), 1)
        self.assertEqual(program.calibrations[0].name, "x_pulse")
        self.assertIn("DRAG", program.calibrations[0].body_text)

    def test_parse_conditional_block(self):
        program = self.i.parse(
            "qubit[1] q;\n"
            "bit[1] c;\n"
            "if (c[0] == 1) {\n"
            "  x q[0];\n"
            "}\n")
        self.assertEqual(len(program.conditional_blocks), 1)
        cb = program.conditional_blocks[0]
        self.assertEqual(cb.condition_left, "c[0]")
        self.assertEqual(cb.condition_op, "==")
        self.assertEqual(cb.condition_right, "1")
        # The expansion should add a gate with a condition.
        self.assertEqual(len(program.gates), 1)
        self.assertEqual(program.gates[0]["condition"],
                         ("c[0] == 1",))


# ---------------------------------------------------------------------------
# qasm3_to_eqir round-trip
# ---------------------------------------------------------------------------

class TestQasm3ToEqir(unittest.TestCase):
    def test_simple_h_round_trip(self):
        g = qasm3_to_eqir(
            "qubit[2] q;\n"
            "h q[0];\n"
            "cx q[0], q[1];\n")
        gate_count = sum(1 for n in g.nodes.values()
                          if n.type == "GATE")
        self.assertEqual(gate_count, 2)

    def test_measure_round_trip(self):
        g = qasm3_to_eqir(
            "qubit[2] q;\n"
            "bit[2] c;\n"
            "h q[0];\n"
            "c[0] = measure q[0];\n")
        measure_count = sum(1 for n in g.nodes.values()
                              if n.type == "MEASURE")
        self.assertEqual(measure_count, 1)

    def test_allocation_in_eqir(self):
        g = qasm3_to_eqir("qubit[2] q;")
        alloc_count = sum(1 for n in g.nodes.values()
                          if n.type == "ALLOC")
        self.assertEqual(alloc_count, 2)

    def test_subroutine_expansion_in_graph(self):
        g = qasm3_to_eqir(
            "gate mygate q {\n"
            "  h q;\n"
            "  x q;\n"
            "}\n"
            "qubit[1] q;\n"
            "mygate q[0];\n")
        gate_count = sum(1 for n in g.nodes.values()
                          if n.type == "GATE")
        # Subroutine body has 2 gates → 2 gate nodes.
        self.assertEqual(gate_count, 2)


# ---------------------------------------------------------------------------
# Extended exporter
# ---------------------------------------------------------------------------

class TestExtendedQasm3Exporter(unittest.TestCase):
    def _make_simple_graph(self, gate="H"):
        from src.ir.ir_graph import EQIRGraph
        # Build a tiny graph with explicit ALLOC so the existing
        # `Qasm3Exporter` can map qubit names → indices.
        g = EQIRGraph()
        g.add_operation("ALLOC", targets=["q0"])
        g.add_operation("GATE", gate_name=gate, args=[], targets=["q0"])
        return g

    def test_export_with_subroutine(self):
        g = self._make_simple_graph("H")
        g.add_operation("GATE", gate_name="X", args=[], targets=["q0"])
        sub = Subroutine(name="mygate", qubits=["q"], body_lines=["h q;"])

        out = eqir_to_extended_qasm3(g, subroutines=[sub])
        self.assertIn("OPENQASM 3.0;", out)
        self.assertIn("gate mygate q {", out)
        # The graph itself appears after the subroutine
        # declaration
        self.assertIn("h", out)

    def test_export_with_calibration(self):
        g = self._make_simple_graph("X")

        cal = Calibration(name="x_pulse", qubits=["q"],
                            body_text="pulse DRAG")
        out = eqir_to_extended_qasm3(g, calibrations=[cal])
        self.assertIn("defcal x_pulse q {", out)
        self.assertIn("pulse DRAG", out)

    def test_export_no_header(self):
        g = self._make_simple_graph("H")
        opts = Qasm3ExportOptions(include_header=False,
                                      include_stdgates=False)
        out = eqir_to_extended_qasm3(g, options=opts)
        self.assertNotIn("OPENQASM", out)
        self.assertNotIn("include", out)


if __name__ == "__main__":
    unittest.main()
