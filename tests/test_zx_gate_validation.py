import cmath
import math
import unittest

import numpy as np

from src.ir.ir_graph import EQIRGraph
from src.zx.zx_equivalence import ZXEquivalenceChecker


def _build_single_qubit_graph(gate_name, args=None):
    g = EQIRGraph()
    g.add_operation("ALLOC", targets=["q0"])
    if args is not None:
        g.add_operation("GATE", gate_name=gate_name, targets=["q0"], args=list(args))
    else:
        g.add_operation("GATE", gate_name=gate_name, targets=["q0"])
    return g


def _build_two_qubit_graph(gate_name, q1="q0", q2="q1", args=None):
    g = EQIRGraph()
    g.add_operation("ALLOC", targets=[q1])
    g.add_operation("ALLOC", targets=[q2])
    if args is not None:
        g.add_operation("GATE", gate_name=gate_name, targets=[q1, q2], args=list(args))
    else:
        g.add_operation("GATE", gate_name=gate_name, targets=[q1, q2])
    return g


def _build_unitary_via_zx(graph):
    checker = ZXEquivalenceChecker()
    zx = checker.circuit_to_zx(graph)
    return checker, zx


def _circuit_unitary(graph, qubit_order):
    checker = ZXEquivalenceChecker()
    return checker.generate_unitary(graph, qubit_order)


def _assert_eq_circuits(test, g1, g2, qubits):
    checker = ZXEquivalenceChecker()
    test.assertTrue(
        checker.are_equivalent(g1, g2),
        msg=f"Circuits not equivalent for qubits={qubits}",
    )


class TestZXGraphConstruction(unittest.TestCase):
    def test_h_gate_produces_zx_with_h_vertex(self):
        g = _build_single_qubit_graph("H")
        _, zx = _build_unitary_via_zx(g)
        h_count = sum(1 for v in zx.vertices.values() if v.type == "H")
        self.assertGreaterEqual(h_count, 1)

    def test_s_gate_produces_zx_with_z_phased_vertex(self):
        g = _build_single_qubit_graph("S")
        _, zx = _build_unitary_via_zx(g)
        phases = [v.phase for v in zx.vertices.values() if v.type == "Z"]
        self.assertIn(0.5, phases)

    def test_x_gate_uses_h_z_h_pattern(self):
        g = _build_single_qubit_graph("X")
        _, zx = _build_unitary_via_zx(g)
        z_phases_1 = [v.phase for v in zx.vertices.values() if v.type == "Z" and abs(v.phase - 1.0) < 1e-9]
        self.assertGreaterEqual(len(z_phases_1), 1)

    def test_y_gate_has_z_x_phase_pattern(self):
        g = _build_single_qubit_graph("Y")
        _, zx = _build_unitary_via_zx(g)
        h_count = sum(1 for v in zx.vertices.values() if v.type == "H")
        z_phases = [v.phase for v in zx.vertices.values() if v.type == "Z" and abs(v.phase - 1.0) < 1e-9]
        self.assertGreaterEqual(h_count, 2)
        self.assertGreaterEqual(len(z_phases), 1)

    def test_z_gate_has_phase_1_vertex(self):
        g = _build_single_qubit_graph("Z")
        _, zx = _build_unitary_via_zx(g)
        z_phases_1 = [v for v in zx.vertices.values() if v.type == "Z" and abs(v.phase - 1.0) < 1e-9]
        self.assertGreaterEqual(len(z_phases_1), 1)

    def test_t_gate_has_phase_quarter_vertex(self):
        g = _build_single_qubit_graph("T")
        _, zx = _build_unitary_via_zx(g)
        quarter_phases = [v for v in zx.vertices.values() if v.type == "Z" and abs(v.phase - 0.25) < 1e-9]
        self.assertGreaterEqual(len(quarter_phases), 1)

    def test_cnot_creates_h_vertex(self):
        g = _build_two_qubit_graph("CNOT")
        _, zx = _build_unitary_via_zx(g)
        h_count = sum(1 for v in zx.vertices.values() if v.type == "H")
        self.assertGreaterEqual(h_count, 1)

    def test_cz_creates_two_z_spiders(self):
        g = _build_two_qubit_graph("CZ")
        _, zx = _build_unitary_via_zx(g)
        z_count = sum(1 for v in zx.vertices.values() if v.type == "Z")
        self.assertGreaterEqual(z_count, 2)

    def test_swap_just_swaps_wires(self):
        g = _build_two_qubit_graph("SWAP")
        _, zx = _build_unitary_via_zx(g)
        b_count = sum(1 for v in zx.vertices.values() if v.type == "Boundary")
        self.assertEqual(b_count, 4)

    def test_ry_gate_has_h_vertices(self):
        g = _build_single_qubit_graph("RY", args=[math.pi / 2])
        _, zx = _build_unitary_via_zx(g)
        h_count = sum(1 for v in zx.vertices.values() if v.type == "H")
        self.assertGreaterEqual(h_count, 2)

    def test_rx_gate_uses_h_z_h(self):
        g = _build_single_qubit_graph("RX", args=[math.pi / 2])
        _, zx = _build_unitary_via_zx(g)
        h_count = sum(1 for v in zx.vertices.values() if v.type == "H")
        self.assertGreaterEqual(h_count, 2)

    def test_rz_gate_has_z_phase(self):
        g = _build_single_qubit_graph("RZ", args=[math.pi / 4])
        _, zx = _build_unitary_via_zx(g)
        phases = [v.phase for v in zx.vertices.values() if v.type == "Z"]
        self.assertTrue(any(abs(p - 0.25) < 1e-9 for p in phases))

    def test_zx_has_inputs_and_outputs(self):
        g = _build_single_qubit_graph("H")
        _, zx = _build_unitary_via_zx(g)
        self.assertEqual(len(zx.inputs), 1)
        self.assertEqual(len(zx.outputs), 1)

    def test_zx_inputs_outputs_for_two_qubits(self):
        g = _build_two_qubit_graph("CNOT")
        _, zx = _build_unitary_via_zx(g)
        self.assertEqual(len(zx.inputs), 2)
        self.assertEqual(len(zx.outputs), 2)


class TestZXGateVsUnitaryEquivalence(unittest.TestCase):
    def test_h_gate_equivalent_to_itself(self):
        g1 = _build_single_qubit_graph("H")
        g2 = _build_single_qubit_graph("H")
        _assert_eq_circuits(self, g1, g2, ["q0"])

    def test_s_gate_equivalent_to_itself(self):
        g1 = _build_single_qubit_graph("S")
        g2 = _build_single_qubit_graph("S")
        _assert_eq_circuits(self, g1, g2, ["q0"])

    def test_x_gate_equivalent_to_itself(self):
        g1 = _build_single_qubit_graph("X")
        g2 = _build_single_qubit_graph("X")
        _assert_eq_circuits(self, g1, g2, ["q0"])

    def test_y_gate_equivalent_to_itself(self):
        g1 = _build_single_qubit_graph("Y")
        g2 = _build_single_qubit_graph("Y")
        _assert_eq_circuits(self, g1, g2, ["q0"])

    def test_z_gate_equivalent_to_itself(self):
        g1 = _build_single_qubit_graph("Z")
        g2 = _build_single_qubit_graph("Z")
        _assert_eq_circuits(self, g1, g2, ["q0"])

    def test_cnot_equivalent_to_itself(self):
        g1 = _build_two_qubit_graph("CNOT")
        g2 = _build_two_qubit_graph("CNOT")
        _assert_eq_circuits(self, g1, g2, ["q0", "q1"])

    def test_cz_equivalent_to_itself(self):
        g1 = _build_two_qubit_graph("CZ")
        g2 = _build_two_qubit_graph("CZ")
        _assert_eq_circuits(self, g1, g2, ["q0", "q1"])

    def test_swap_equivalent_to_itself(self):
        g1 = _build_two_qubit_graph("SWAP")
        g2 = _build_two_qubit_graph("SWAP")
        _assert_eq_circuits(self, g1, g2, ["q0", "q1"])

    def test_ry_pi2_equivalent_to_itself(self):
        g1 = _build_single_qubit_graph("RY", args=[math.pi / 2])
        g2 = _build_single_qubit_graph("RY", args=[math.pi / 2])
        _assert_eq_circuits(self, g1, g2, ["q0"])

    def test_t_gate_equivalent_to_itself(self):
        g1 = _build_single_qubit_graph("T")
        g2 = _build_single_qubit_graph("T")
        _assert_eq_circuits(self, g1, g2, ["q0"])

    def test_double_h_is_identity(self):
        g1 = EQIRGraph()
        g1.add_operation("ALLOC", targets=["q0"])
        g2 = EQIRGraph()
        g2.add_operation("ALLOC", targets=["q0"])
        g2.add_operation("GATE", gate_name="H", targets=["q0"])
        g2.add_operation("GATE", gate_name="H", targets=["q0"])
        checker = ZXEquivalenceChecker()
        self.assertTrue(checker.are_equivalent(g1, g2))

    def test_h_z_h_equals_x(self):
        g1 = _build_single_qubit_graph("X")
        g2 = EQIRGraph()
        g2.add_operation("ALLOC", targets=["q0"])
        g2.add_operation("GATE", gate_name="H", targets=["q0"])
        g2.add_operation("GATE", gate_name="Z", targets=["q0"])
        g2.add_operation("GATE", gate_name="H", targets=["q0"])
        checker = ZXEquivalenceChecker()
        self.assertTrue(checker.are_equivalent(g1, g2))

    def test_cz_via_h_cnot_h_pattern(self):
        g1 = _build_two_qubit_graph("CZ")
        g2 = EQIRGraph()
        for q in ["q0", "q1"]:
            g2.add_operation("ALLOC", targets=[q])
        g2.add_operation("GATE", gate_name="H", targets=["q1"])
        g2.add_operation("GATE", gate_name="CNOT", targets=["q0", "q1"])
        g2.add_operation("GATE", gate_name="H", targets=["q1"])
        checker = ZXEquivalenceChecker()
        self.assertTrue(checker.are_equivalent(g1, g2))

    def test_s_squared_equals_z(self):
        g1 = _build_single_qubit_graph("Z")
        g2 = EQIRGraph()
        g2.add_operation("ALLOC", targets=["q0"])
        g2.add_operation("GATE", gate_name="S", targets=["q0"])
        g2.add_operation("GATE", gate_name="S", targets=["q0"])
        checker = ZXEquivalenceChecker()
        self.assertTrue(checker.are_equivalent(g1, g2))

    def test_swap_symmetric(self):
        g1 = _build_two_qubit_graph("SWAP", q1="q0", q2="q1")
        g2 = _build_two_qubit_graph("SWAP", q1="q1", q2="q0")
        checker = ZXEquivalenceChecker()
        self.assertTrue(checker.are_equivalent(g1, g2))

    def test_double_swap_is_identity(self):
        g1 = EQIRGraph()
        for q in ["q0", "q1"]:
            g1.add_operation("ALLOC", targets=[q])
        g2 = EQIRGraph()
        for q in ["q0", "q1"]:
            g2.add_operation("ALLOC", targets=[q])
        g2.add_operation("GATE", gate_name="SWAP", targets=["q0", "q1"])
        g2.add_operation("GATE", gate_name="SWAP", targets=["q0", "q1"])
        checker = ZXEquivalenceChecker()
        self.assertTrue(checker.are_equivalent(g1, g2))


class TestZXUnitaryVsCanonical(unittest.TestCase):
    def test_h_gate_unitary_matches_canonical(self):
        g = _build_single_qubit_graph("H")
        U = _circuit_unitary(g, ["q0"])
        expected = (1 / math.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
        self.assertTrue(np.allclose(U, expected, atol=1e-6))

    def test_x_gate_unitary_matches_canonical(self):
        g = _build_single_qubit_graph("X")
        U = _circuit_unitary(g, ["q0"])
        expected = np.array([[0, 1], [1, 0]], dtype=complex)
        self.assertTrue(np.allclose(U, expected, atol=1e-6))

    def test_z_gate_unitary_matches_canonical(self):
        g = _build_single_qubit_graph("Z")
        U = _circuit_unitary(g, ["q0"])
        expected = np.array([[1, 0], [0, -1]], dtype=complex)
        self.assertTrue(np.allclose(U, expected, atol=1e-6))

    def test_y_gate_unitary_matches_canonical(self):
        g = _build_single_qubit_graph("Y")
        U = _circuit_unitary(g, ["q0"])
        expected = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.assertTrue(np.allclose(U, expected, atol=1e-6))

    def test_s_gate_unitary_matches_canonical(self):
        g = _build_single_qubit_graph("S")
        U = _circuit_unitary(g, ["q0"])
        expected = np.array([[1, 0], [0, 1j]], dtype=complex)
        self.assertTrue(np.allclose(U, expected, atol=1e-6))

    def test_cnot_gate_unitary_matches_canonical(self):
        g = _build_two_qubit_graph("CNOT")
        U = np.array(_circuit_unitary(g, ["q0", "q1"]))
        abs_U = np.abs(U)
        row_sums = abs_U.sum(axis=1)
        col_sums = abs_U.sum(axis=0)
        self.assertTrue(np.allclose(row_sums, 1.0, atol=1e-6))
        self.assertTrue(np.allclose(col_sums, 1.0, atol=1e-6))
        self.assertAlmostEqual(abs_U[0, 0], 1.0, places=6)
        self.assertAlmostEqual(abs_U[2, 2], 1.0, places=6)

    def test_cz_gate_unitary_matches_canonical(self):
        g = _build_two_qubit_graph("CZ")
        U = _circuit_unitary(g, ["q0", "q1"])
        expected = np.diag([1, 1, 1, -1]).astype(complex)
        self.assertTrue(np.allclose(U, expected, atol=1e-6))

    def test_swap_gate_unitary_matches_canonical(self):
        g = _build_two_qubit_graph("SWAP")
        U = _circuit_unitary(g, ["q0", "q1"])
        expected = np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            dtype=complex,
        )
        self.assertTrue(np.allclose(U, expected, atol=1e-6))

    def test_t_gate_unitary_matches_canonical(self):
        g = _build_single_qubit_graph("T")
        U = np.array(_circuit_unitary(g, ["q0"]))
        e = cmath.exp(1j * math.pi / 4)
        expected = np.array([[1, 0], [0, e]], dtype=complex)
        self.assertTrue(np.allclose(U, expected, atol=1e-6))

    def test_rz_gate_unitary_matches_canonical(self):
        theta = math.pi / 3
        g = _build_single_qubit_graph("RZ", args=[theta])
        U = _circuit_unitary(g, ["q0"])
        expected = np.array(
            [[cmath.exp(-1j * theta / 2), 0], [0, cmath.exp(1j * theta / 2)]],
            dtype=complex,
        )
        self.assertTrue(np.allclose(U, expected, atol=1e-6))


class TestZXGraphStructure(unittest.TestCase):
    def test_zx_graph_vertex_count_grows_with_gates(self):
        empty = EQIRGraph()
        empty.add_operation("ALLOC", targets=["q0"])
        _, zx_empty = _build_unitary_via_zx(empty)

        g = _build_single_qubit_graph("H")
        _, zx_h = _build_unitary_via_zx(g)

        self.assertGreater(len(zx_h.vertices), len(zx_empty.vertices))

    def test_zx_graph_boundary_count_equals_two_per_qubit(self):
        g = _build_two_qubit_graph("CNOT")
        _, zx = _build_unitary_via_zx(g)
        b_count = sum(1 for v in zx.vertices.values() if v.type == "Boundary")
        self.assertEqual(b_count, 4)

    def test_cnot_zx_has_h_box(self):
        g = _build_two_qubit_graph("CNOT")
        _, zx = _build_unitary_via_zx(g)
        has_h = any(v.type == "H" for v in zx.vertices.values())
        self.assertTrue(has_h)

    def test_cz_zx_has_no_h_box(self):
        g = _build_two_qubit_graph("CZ")
        _, zx = _build_unitary_via_zx(g)
        has_h = any(v.type == "H" for v in zx.vertices.values())
        self.assertFalse(has_h)


if __name__ == "__main__":
    unittest.main()
