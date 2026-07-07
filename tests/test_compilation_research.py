"""§12.2 — Compilation research algorithm tests."""
import math
import cmath
import itertools
import unittest

from src.compilation_research import (
    PhaseTerm,
    PhasePolynomial,
    optimize_phase_polynomial,
    ZXSimplificationReport,
    ZXSimplifier,
    SolovayKitaevResult,
    solovay_kitaev,
    gauss_jordan_gf2,
    synthesize_cnot_circuit,
    apply_cnot_sequence,
    LayoutChoice,
    best_layout,
    PlacementResult,
    best_placement,
    ScheduledGate,
    schedule_circuit,
    circuit_depth,
)


# ---------------------------------------------------------------------------
# Phase term / phase polynomial
# ---------------------------------------------------------------------------

class TestPhaseTerm(unittest.TestCase):
    def test_construction_and_fields(self):
        t = PhaseTerm(target=2, angle=0.25, gate_name="T")
        self.assertEqual(t.target, 2)
        self.assertAlmostEqual(t.angle, 0.25, places=8)
        self.assertEqual(t.gate_name, "T")

    def test_default_gate_name_empty(self):
        t = PhaseTerm(target=0, angle=0.5)
        self.assertEqual(t.gate_name, "")

    def test_conjugate_inverts_angle_and_appends_INV(self):
        t = PhaseTerm(target=3, angle=0.25, gate_name="T")
        tc = t.conjugate()
        self.assertEqual(tc.target, 3)
        self.assertAlmostEqual(tc.angle, (-0.25) % 2, places=8)
        self.assertEqual(tc.gate_name, "T_INV")

    def test_conjugate_removes_INV_suffix(self):
        t = PhaseTerm(target=0, angle=1.75, gate_name="T_INV")
        tc = t.conjugate()
        self.assertEqual(tc.gate_name, "T")


class TestPhasePolynomial(unittest.TestCase):
    def test_construct_with_termlist(self):
        poly = PhasePolynomial([
            PhaseTerm(target=0, angle=0.25, gate_name="T"),
            PhaseTerm(target=0, angle=0.25, gate_name="T"),
        ])
        self.assertEqual(len(poly), 2)
        n = 0
        for _ in poly:
            n += 1
        self.assertEqual(n, 2)

    def test_append_increases_length(self):
        poly = PhasePolynomial()
        poly.append(PhaseTerm(target=1, angle=0.5, gate_name="S"))
        self.assertEqual(len(poly), 1)
        poly.append(PhaseTerm(target=2, angle=1.0, gate_name="Z"))
        self.assertEqual(len(poly), 2)

    def test_total_angle_per_qubit(self):
        poly = PhasePolynomial([
            PhaseTerm(target=0, angle=0.25, gate_name="T"),
            PhaseTerm(target=0, angle=0.5, gate_name="S"),
            PhaseTerm(target=1, angle=1.0, gate_name="Z"),
        ])
        self.assertAlmostEqual(poly.total_angle(0), 0.75, places=8)
        self.assertAlmostEqual(poly.total_angle(1), 1.0, places=8)
        self.assertAlmostEqual(poly.total_angle(2), 0.0, places=8)

    def test_simplify_cancels_opposite_angles(self):
        # T + T† = 0 (cancels)
        poly = PhasePolynomial([
            PhaseTerm(target=0, angle=0.25, gate_name="T"),
            PhaseTerm(target=0, angle=1.75, gate_name="T_INV"),
        ])
        s = poly.simplify()
        self.assertEqual(len(s), 0)

    def test_simplify_fuses_consecutive_same_qubit(self):
        poly = PhasePolynomial([
            PhaseTerm(target=0, angle=0.25, gate_name="T"),
            PhaseTerm(target=0, angle=0.25, gate_name="T"),
        ])
        s = poly.simplify()
        self.assertEqual(len(s), 1)
        self.assertEqual(s.terms[0].target, 0)
        self.assertAlmostEqual(s.terms[0].angle, 0.5, places=6)
        self.assertEqual(s.terms[0].gate_name, "S")  # 0.5 PI = S gate

    def test_simplify_drops_zero_terms_and_keeps_others(self):
        poly = PhasePolynomial([
            PhaseTerm(target=0, angle=0.25, gate_name="T"),
            PhaseTerm(target=0, angle=1.75, gate_name="T_INV"),
            PhaseTerm(target=1, angle=0.5, gate_name="S"),
        ])
        s = poly.simplify()
        self.assertEqual(len(s), 1)
        self.assertEqual(s.terms[0].target, 1)
        self.assertAlmostEqual(s.terms[0].angle, 0.5, places=6)


class TestOptimizePhasePolynomial(unittest.TestCase):
    def test_unknown_gates_ignored(self):
        poly = optimize_phase_polynomial([("UNICORN", 0)])
        self.assertEqual(len(poly), 0)

    def test_full_cancellation_returns_empty(self):
        poly = optimize_phase_polynomial([("T", 0), ("T_INV", 0)])
        self.assertEqual(len(poly), 0)

    def test_fusion_to_canonical_S(self):
        poly = optimize_phase_polynomial([("T", 0), ("T", 0)])
        self.assertEqual(len(poly), 1)
        # T @ T = S phase-wise
        self.assertEqual(poly.terms[0].gate_name, "S")


# ---------------------------------------------------------------------------
# ZX simplification report
# ---------------------------------------------------------------------------

class TestZXSimplificationReport(unittest.TestCase):
    def test_defaults_are_zero(self):
        r = ZXSimplificationReport()
        self.assertEqual(r.spider_fusions, 0)
        self.assertEqual(r.pivots, 0)
        self.assertEqual(r.local_complementations, 0)

    def test_total_sums_three_components(self):
        r = ZXSimplificationReport(spider_fusions=2, pivots=1,
                                       local_complementations=3)
        self.assertEqual(r.total(), 6)


class TestZXSimplifier(unittest.TestCase):
    """Verify the simplifier applies spider fusion on a tiny graph."""

    def test_run_with_nothing_changed_on_empty_graph(self):
        from src.zx.zx_graph import ZXGraph
        g = ZXGraph()
        simplifier = ZXSimplifier()
        report = simplifier.simplify(g)
        self.assertEqual(report.total(), 0)
        self.assertEqual(len(g.vertices), 0)

    def test_spider_fusion_reduces_vertex_count(self):
        from src.zx.zx_graph import ZXGraph
        g = ZXGraph()
        z0 = g.add_vertex('Z', 0.0)
        z1 = g.add_vertex('Z', 0.25)
        g.add_edge(z0.id, z1.id)
        self.assertEqual(len(g.vertices), 2)

        simplifier = ZXSimplifier()
        report = simplifier.simplify(g)

        self.assertGreater(report.spider_fusions, 0)
        self.assertEqual(len(g.vertices), 1)


# ---------------------------------------------------------------------------
# Solovay-Kitaev
# ---------------------------------------------------------------------------

class TestSolovayKitaevResult(unittest.TestCase):
    def test_defaults(self):
        r = SolovayKitaevResult(sequence=[], depth=0,
                                    precision=float("inf"), found=False)
        self.assertEqual(r.sequence, [])
        self.assertEqual(r.depth, 0)
        self.assertFalse(r.found)


class TestSolovayKitaevKnownTargets(unittest.TestCase):
    def _U(self, name):
        basis = {
            "H": [[ math.sqrt(2) / 2,  math.sqrt(2) / 2],
                  [ math.sqrt(2) / 2, -math.sqrt(2) / 2]],
            "T": [[1, 0], [0, cmath.exp(1j * math.pi / 4)]],
            "S": [[1, 0], [0, 1j]],
            "X": [[0, 1], [1, 0]],
            "Y": [[0, -1j], [1j, 0]],
            "Z": [[1, 0], [0, -1]],
        }
        return basis[name]

    def test_finds_H(self):
        r = solovay_kitaev(self._U("H"), max_depth=2, precision_tol=1e-6)
        self.assertTrue(r.found)
        self.assertEqual(r.sequence, ["H"])
        self.assertEqual(r.depth, 1)

    def test_finds_T(self):
        r = solovay_kitaev(self._U("T"), max_depth=2, precision_tol=1e-6)
        self.assertTrue(r.found)
        self.assertEqual(r.sequence, ["T"])

    def test_finds_S(self):
        r = solovay_kitaev(self._U("S"), max_depth=2, precision_tol=1e-6)
        self.assertTrue(r.found)
        self.assertEqual(r.sequence, ["S"])

    def test_finds_X(self):
        r = solovay_kitaev(self._U("X"), max_depth=2, precision_tol=1e-6)
        self.assertTrue(r.found)
        self.assertEqual(r.sequence, ["X"])

    def test_finds_Z(self):
        r = solovay_kitaev(self._U("Z"), max_depth=2, precision_tol=1e-6)
        self.assertTrue(r.found)
        self.assertEqual(r.sequence, ["Z"])

    def test_invalid_target_raises(self):
        with self.assertRaises(ValueError):
            solovay_kitaev([[1, 0, 0], [0, 1, 0]])

    def test_non_basis_target_can_return_found_false(self):
        # Small rotation; basis won't have an exact match, but a depth-1
        # search won't find one within 1e-6 tolerance.
        U = [[1, 0], [0, cmath.exp(1j * math.pi / 8)]]
        r = solovay_kitaev(U, max_depth=1, precision_tol=1e-6)
        self.assertFalse(r.found)
        # But it should still return a "best" candidate
        self.assertEqual(len(r.sequence), 1)

    def test_two_step_target_finds_word(self):
        # H @ T can be exactly represented as the 2-gate word ["H", "T"]
        # (applied right-to-left, so the last applied gate is "T").
        Ht = self._U("H")
        Tt = self._U("T")
        prod = [[sum(Ht[i][k] * Tt[k][j] for k in range(2))
                 for j in range(2)] for i in range(2)]
        r = solovay_kitaev(prod, max_depth=3, precision_tol=1e-6)
        self.assertTrue(r.found, msg=f"sequence={r.sequence}, "
                                         f"precision={r.precision}")
        self.assertEqual(len(r.sequence), 2)


# ---------------------------------------------------------------------------
# Gauss-Jordan over GF(2)
# ---------------------------------------------------------------------------

class TestGaussJordanGF2(unittest.TestCase):
    def test_identity_returns_identity_ops(self):
        work, ops = gauss_jordan_gf2([[1, 0], [0, 1]])
        self.assertEqual(work, [[1, 0], [0, 1]])
        self.assertEqual(ops, [])

    def test_single_cnot_3x3_identity(self):
        work, ops = gauss_jordan_gf2([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertEqual(work, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertEqual(ops, [])

    def test_singular_raises(self):
        with self.assertRaises(ValueError):
            gauss_jordan_gf2([[0, 0], [0, 0]])

    def test_recorded_ops_reduce_to_identity(self):
        # CNOT(0,1) matrix [[1,0],[1,1]]: gauss_jordan should still
        # reduce to identity in (at most) one row-op
        work, _ = gauss_jordan_gf2([[1, 0], [1, 1]])
        self.assertEqual(work, [[1, 0], [0, 1]])


# ---------------------------------------------------------------------------
# synthesize_cnot_circuit / apply_cnot_sequence
# ---------------------------------------------------------------------------

class TestSynthesizeCNOTCircuit(unittest.TestCase):
    def test_identity_matrix_returns_empty_ops(self):
        self.assertEqual(synthesize_cnot_circuit([[1, 0], [0, 1]]), [])

    def test_3bit_identity_returns_empty_ops(self):
        self.assertEqual(synthesize_cnot_circuit(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]]), [])

    def test_singular_matrix_raises(self):
        with self.assertRaises(ValueError):
            synthesize_cnot_circuit([[1, 1], [1, 1]])

    def test_apply_to_all_basis_vectors(self):
        """Verify M @ v == apply_cnot_sequence(v, ops) for all 2^n
        basis vectors and a target M."""
        for n in (2, 3):
            M = [[1] * n for _ in range(n)]  # all-ones matrix: simple check
            for i in range(n):
                M[i][i] = 0  # invertible permutation of complement
            M = [[1 if i != j else 0 for j in range(n)] for i in range(n)]
            # Actually, let me build a simpler non-singular matrix:
            # CNOT(0,1): M = [[1, 0], [1, 1]] (n=2) — but for n=3 build a
            # product of two single-CNOTs:
            if n == 2:
                M = [[1, 0], [1, 1]]
            elif n == 3:
                # M = CNOT(0,1) @ CNOT(1,2):
                M_01 = [[1, 0, 0], [1, 1, 0], [0, 0, 1]]
                M_12 = [[1, 0, 0], [0, 1, 0], [0, 1, 1]]
                M = [[sum(M_01[i][k] * M_12[k][j] for k in range(n))
                      for j in range(n)] for i in range(n)]
            ops = synthesize_cnot_circuit(M)
            for v in itertools.product((0, 1), repeat=n):
                v_list = list(v)
                expected = [sum(int(M[i][k]) * v_list[k] % 2
                                for k in range(n)) % 2 for i in range(n)]
                # XOR instead of standard sum (over GF(2))
                expected = []
                for i in range(n):
                    s = 0
                    for k in range(n):
                        if M[i][k] == 1 and v_list[k] == 1:
                            s ^= 1
                    expected.append(s)
                v_copy = list(v_list)
                apply_cnot_sequence(v_copy, ops)
                self.assertEqual(v_copy, expected,
                                   msg=f"M={M}, v={list(v)}, got={v_copy},"
                                        f" expected={expected}")


class TestApplyCNOTSequence(unittest.TestCase):
    def test_empty_ops_no_change(self):
        v = [1, 0, 1]
        apply_cnot_sequence(v, [])
        self.assertEqual(v, [1, 0, 1])

    def test_single_op_XORs_target_with_source(self):
        v = [1, 0, 1]
        # (m=0, s=2): v[0] ^= v[2]
        apply_cnot_sequence(v, [(0, 2)])
        self.assertEqual(v, [0, 0, 1])

    def test_chain_of_two_ops(self):
        v = [1, 0, 0]
        apply_cnot_sequence(v, [(0, 2), (1, 0)])
        # (0, 2): v[0] ^= v[2] = 1 -> [1, 0, 0]
        # (1, 0): v[1] ^= v[0] = 1 -> [1, 1, 0]
        self.assertEqual(v, [1, 1, 0])

    def test_returns_same_list(self):
        v = [1, 0]
        res = apply_cnot_sequence(v, [(1, 0)])
        self.assertIs(res, v)


# ---------------------------------------------------------------------------
# Layout optimization
# ---------------------------------------------------------------------------

class TestLayoutChoice(unittest.TestCase):
    def test_dataclass_fields(self):
        lc = LayoutChoice(mapping={0: 1, 1: 0}, cost=42)
        self.assertEqual(lc.mapping, {0: 1, 1: 0})
        self.assertEqual(lc.cost, 42)


class TestBestLayout(unittest.TestCase):
    def test_empty_cnots_returns_empty_mapping(self):
        lc = best_layout([], [0, 1, 2], {(0, 1), (1, 2)})
        self.assertEqual(lc.mapping, {})
        self.assertEqual(lc.cost, 0)

    def test_perfect_layout_zero_cost(self):
        # Two CNOTs on linear chain: (0,1), (1,2) → physical mapping
        # 0->0, 1->1, 2->2 satisfies all coupling edges
        coupling = {(0, 1), (1, 2), (2, 3)}
        lc = best_layout([(0, 1), (1, 2)], [0, 1, 2, 3], coupling)
        self.assertEqual(lc.cost, 0)

    def test_layout_finds_minimum_violations(self):
        # If the device is a linear chain 0-1-2 and CNOTs are (0, 2)
        # (only coupling edge directly accessible is 0-1, 1-2, not 0-2),
        # any assignment must place q_0 and q_2 adjacent. Test that the
        # best mapping has 0 violations.
        coupling = {(0, 1), (1, 2)}
        lc = best_layout([(0, 2)], [0, 1, 2], coupling)
        self.assertEqual(lc.cost, 0)

    def test_too_few_physicals_raises(self):
        with self.assertRaises(ValueError):
            best_layout([(0, 1)], [0], {(0, 1)})


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------

class TestPlacementResult(unittest.TestCase):
    def test_dataclass_fields(self):
        pr = PlacementResult(chosen_qubits=[0, 1, 2],
                                mapping={0: 0, 1: 1, 2: 2})
        self.assertEqual(pr.chosen_qubits, [0, 1, 2])
        self.assertEqual(pr.mapping, {0: 0, 1: 1, 2: 2})


class TestBestPlacement(unittest.TestCase):
    def test_selects_densest_subgraph(self):
        # 4-qubit device where qubits 0,1,2 form a triangle cluster
        coupling = {(0, 1), (0, 2), (1, 2), (2, 3)}
        # Want to place 3 logicals; densest 3-subset is {0,1,2}.
        pr = best_placement(3, [0, 1, 2, 3], coupling)
        self.assertEqual(sorted(pr.chosen_qubits), [0, 1, 2])
        # Mapping is logical->phys
        self.assertEqual(len(pr.mapping), 3)
        for i in range(3):
            self.assertIn(pr.mapping[i], pr.chosen_qubits)

    def test_trivial_placement(self):
        pr = best_placement(1, [0], set())
        self.assertEqual(pr.chosen_qubits, [0])
        self.assertEqual(pr.mapping, {0: 0})

    def test_too_few_physicals_raises(self):
        with self.assertRaises(ValueError):
            best_placement(3, [0, 1], {(0, 1)})


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

class TestScheduledGate(unittest.TestCase):
    def test_dataclass_fields(self):
        sg = ScheduledGate(gate_name="H", targets=[0],
                              layer=2, gate_id=7)
        self.assertEqual(sg.gate_name, "H")
        self.assertEqual(sg.targets, [0])
        self.assertEqual(sg.layer, 2)
        self.assertEqual(sg.gate_id, 7)


class TestScheduleCircuit(unittest.TestCase):
    def test_empty_gates(self):
        self.assertEqual(schedule_circuit([], 1), [])

    def test_single_gate_at_layer_zero(self):
        sched = schedule_circuit([("H", [0])], 1)
        self.assertEqual(len(sched), 1)
        self.assertEqual(sched[0].layer, 0)

    def test_same_qubit_gate_at_next_layer(self):
        sched = schedule_circuit([("H", [0]), ("X", [0])], 1)
        self.assertEqual(sched[0].layer, 0)
        self.assertEqual(sched[1].layer, 1)

    def test_different_qubit_gates_same_layer(self):
        sched = schedule_circuit([("H", [0]), ("X", [1])], 2)
        self.assertEqual(sched[0].layer, 0)
        self.assertEqual(sched[1].layer, 0)

    def test_respects_predecessor_dependencies(self):
        # (0, 1) in deps => gate 0 must precede gate 1
        sched = schedule_circuit([("a", [0]), ("b", [1]), ("c", [2])],
                                       3,
                                       deps=[(0, 1), (1, 2)])
        self.assertEqual(sched[0].layer, 0)  # gate 0
        self.assertEqual(sched[1].layer, 1)  # gate 1 (waits 0)
        self.assertEqual(sched[2].layer, 2)  # gate 2 (waits 1)

    def test_partial_dependency(self):
        # (0, 1) in deps => gate 1 waits gate 0; gate 2 not dependent
        sched = schedule_circuit([("a", [0]), ("b", [1]), ("c", [2])],
                                       3,
                                       deps=[(0, 1)])
        # gate 2 can run in layer 0 (no targets conflict, no deps)
        self.assertEqual(sched[2].layer, 0)
        self.assertEqual(sched[0].layer, 0)
        self.assertEqual(sched[1].layer, 1)


class TestCircuitDepth(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(circuit_depth([], 1), 0)

    def test_single_gate(self):
        self.assertEqual(circuit_depth([("H", [0])], 1), 1)

    def test_parallel_same_layer(self):
        self.assertEqual(circuit_depth([("H", [0]), ("X", [1])], 2),
                         1)

    def test_chain_of_same_qubit(self):
        self.assertEqual(circuit_depth(
            [("H", [0]), ("X", [0]), ("Z", [0])], 1), 3)

    def test_chain_with_deps(self):
        gates = [("H", [0]), ("H", [1]), ("H", [2])]
        self.assertEqual(circuit_depth(gates, 3,
                                            deps=[(0, 1), (1, 2)]), 3)


if __name__ == "__main__":
    unittest.main()
