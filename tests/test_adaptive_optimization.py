"""§5.2 — Adaptive Optimization tests, organised by the four
roadmap checkboxes."""
import unittest

from src.ir.ir_graph import EQIRGraph
from src.adaptive_optimization import (
    OptimizationLevel,
    HotPathInfo,
    HotPathRegistry,
    identify_hot_paths,
    PassProfileEntry,
    OptimizationProfile,
    record_pass_into_profile,
    CircuitDescription,
    describe_circuit,
    select_level,
    build_adaptive_pipeline,
    run_adaptive,
)


def _make_empty() -> EQIRGraph:
    return EQIRGraph()


def _make_small_clifford() -> EQIRGraph:
    g = EQIRGraph()
    g.add_operation("GATE", gate_name="H", args=[], targets=["q0"])
    g.add_operation("GATE", gate_name="CNOT", args=[], targets=["q0", "q1"])
    return g


def _make_small_with_rotation() -> EQIRGraph:
    g = EQIRGraph()
    g.add_operation("GATE", gate_name="H", args=[], targets=["q0"])
    g.add_operation("GATE", gate_name="RX", args=[0.5],
                      targets=["q0"])
    return g


def _make_large_mixed() -> EQIRGraph:
    """101 gates: a mix of Clifford and rotation gates."""
    g = EQIRGraph()
    for i in range(101):
        g.add_operation("GATE",
                          gate_name="H" if i % 2 == 0 else "RX",
                          args=[] if i % 2 == 0 else [0.1 * i],
                          targets=[f"q{i % 4}"])
    return g


def _make_measurement() -> EQIRGraph:
    g = EQIRGraph()
    g.add_operation("GATE", gate_name="H", args=[], targets=["q0"])
    g.add_operation("MEASURE", targets=["q0"], cbit_name="c0")
    return g


# ----- Level enum tests -----------------------------------------------

class TestOptimizationLevel(unittest.TestCase):
    def test_levels_are_distinct(self):
        levels = {OptimizationLevel.O0, OptimizationLevel.O1,
                  OptimizationLevel.O2, OptimizationLevel.O3,
                  OptimizationLevel.AUTO}
        self.assertEqual(len(levels), 5)

    def test_value_is_string(self):
        for l in OptimizationLevel:
            self.assertIsInstance(l.value, str)


# ----- Hot-path detection tests ---------------------------------------

class TestHotPathRegistry(unittest.TestCase):
    def test_record_increases_count(self):
        r = HotPathRegistry()
        r.record("block_a", duration_ns=100)
        r.record("block_a", duration_ns=200)
        self.assertEqual(r._counts["block_a"], 2)
        self.assertEqual(r._durations["block_a"], [100, 200])

    def test_top_k_returns_sorted_by_score(self):
        r = HotPathRegistry()
        r.record("a", duration_ns=10)  # score 10
        r.record("b", duration_ns=100)  # score 100
        r.record("b", duration_ns=100)  # score 200 total
        r.record("c", duration_ns=50)  # score 50
        top = r.top_k_hot_paths(2)
        self.assertEqual(len(top), 2)
        self.assertEqual(top[0].name, "b")
        # Second is either a (score 10) or c (score 50) — c wins.
        self.assertEqual(top[1].name, "c")

    def test_top_k_deterministic_for_tied_scores(self):
        r = HotPathRegistry()
        r.record("z", duration_ns=10)
        r.record("a", duration_ns=10)
        top = r.top_k_hot_paths(2)
        # Tie-broken by name ascending → "a" first.
        self.assertEqual(top[0].name, "a")
        self.assertEqual(top[1].name, "z")

    def test_record_loop_marked(self):
        r = HotPathRegistry()
        r.record("loop1", duration_ns=10, is_loop=True)
        r.record("loop1", duration_ns=10)  # is_loop defaults False; doesn't un-mark
        top = r.top_k_hot_paths(1)
        self.assertTrue(top[0].is_loop)

    def test_total_invocations(self):
        r = HotPathRegistry()
        r.record("a", duration_ns=1)
        r.record("b", duration_ns=1)
        r.record("a", duration_ns=1)
        self.assertEqual(r.total_invocations(), 3)

    def test_clear_resets(self):
        r = HotPathRegistry()
        r.record("a", duration_ns=1)
        r.clear()
        self.assertEqual(r.total_invocations(), 0)


class TestIdentifyHotPaths(unittest.TestCase):
    def test_returns_paths_above_threshold(self):
        profile = OptimizationProfile(hot_paths=[
            HotPathInfo("hot", invocation_count=200, average_duration_ns=10),
            HotPathInfo("cold", invocation_count=50, average_duration_ns=10),
        ])
        result = identify_hot_paths(profile, threshold=100)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "hot")

    def test_returns_empty_when_none_match(self):
        profile = OptimizationProfile(hot_paths=[
            HotPathInfo("cold", invocation_count=10, average_duration_ns=10),
        ])
        result = identify_hot_paths(profile, threshold=100)
        self.assertEqual(result, [])


# ----- Profile-guided tests -------------------------------------------

class TestOptimizationProfile(unittest.TestCase):
    def test_add_pass_entry_records_history(self):
        p = OptimizationProfile()
        p.add_pass_entry(PassProfileEntry(
            pass_name="eqir_optimization",
            gates_before=10, gates_after=5, duration_ns=100,
        ))
        self.assertEqual(len(p.pass_history["eqir_optimization"]), 1)

    def test_average_gates_removed(self):
        p = OptimizationProfile()
        p.add_pass_entry(PassProfileEntry(
            pass_name="x", gates_before=10, gates_after=5, duration_ns=1))
        p.add_pass_entry(PassProfileEntry(
            pass_name="x", gates_before=8, gates_after=2, duration_ns=1))
        # (10-5)+(8-2) = 11; /2 = 5.5
        self.assertAlmostEqual(p.average_gates_removed("x"), 5.5)

    def test_average_gates_removed_empty(self):
        p = OptimizationProfile()
        self.assertEqual(p.average_gates_removed("nope"), 0.0)

    def test_average_duration(self):
        p = OptimizationProfile()
        p.add_pass_entry(PassProfileEntry(
            pass_name="x", gates_before=0, gates_after=0, duration_ns=100))
        p.add_pass_entry(PassProfileEntry(
            pass_name="x", gates_before=0, gates_after=0, duration_ns=300))
        self.assertEqual(p.average_duration("x"), 200.0)

    def test_is_hot_path_program_with_threshold(self):
        p = OptimizationProfile(total_program_invocations=200)
        self.assertTrue(p.is_hot_path_program(threshold=100))

    def test_is_hot_path_program_false(self):
        p = OptimizationProfile(total_program_invocations=10)
        self.assertFalse(p.is_hot_path_program(threshold=100))

    def test_is_hot_path_program_via_hot_paths(self):
        p = OptimizationProfile(hot_paths=[
            HotPathInfo("loop", invocation_count=500, average_duration_ns=1),
        ])
        self.assertTrue(p.is_hot_path_program(threshold=100))


class TestRecordPassIntoProfile(unittest.TestCase):
    def test_records_a_pass_stats_object(self):
        from src.ir.pass_manager import PassStats
        profile = OptimizationProfile()
        record_pass_into_profile(
            profile, "eqir_optimization",
            PassStats(name="eqir_optimization",
                      gates_before=10, gates_after=5,
                      gates_removed=5, depth_before=2, depth_after=1,
                      depth_reduction=1, duration_ns=42))
        history = profile.pass_history["eqir_optimization"]
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].gates_before, 10)
        self.assertEqual(history[0].gates_after, 5)
        self.assertEqual(history[0].duration_ns, 42)


# ----- Circuit description tests -------------------------------------

class TestDescribeCircuit(unittest.TestCase):
    def test_empty_graph(self):
        d = describe_circuit(EQIRGraph())
        self.assertEqual(d.gate_count, 0)
        self.assertEqual(d.qubit_count, 0)

    def test_small_clifford(self):
        d = describe_circuit(_make_small_clifford())
        self.assertEqual(d.gate_count, 2)
        self.assertEqual(d.clifford_gate_count, 2)
        self.assertEqual(d.rotation_gate_count, 0)
        self.assertTrue(d.is_clifford_only())
        self.assertEqual(d.qubit_count, 2)

    def test_with_rotation(self):
        d = describe_circuit(_make_small_with_rotation())
        self.assertEqual(d.gate_count, 2)
        self.assertEqual(d.rotation_gate_count, 1)
        self.assertFalse(d.is_clifford_only())

    def test_measurement_detected(self):
        d = describe_circuit(_make_measurement())
        self.assertTrue(d.has_measurements)

    def test_rotation_fraction(self):
        d = describe_circuit(_make_small_with_rotation())
        self.assertAlmostEqual(d.rotation_fraction(), 0.5)

    def test_is_large(self):
        self.assertFalse(CircuitDescription(gate_count=50, depth=2).is_large())
        self.assertTrue(CircuitDescription(gate_count=101, depth=2).is_large())
        self.assertTrue(CircuitDescription(gate_count=50, depth=31).is_large())


# ----- Adaptive level selector ----------------------------------------

class TestSelectLevel(unittest.TestCase):
    def test_force_returns_force(self):
        self.assertEqual(select_level(CircuitDescription(),
                                        force=OptimizationLevel.O3),
                         OptimizationLevel.O3)

    def test_empty_circuit_returns_O0(self):
        self.assertEqual(select_level(CircuitDescription(gate_count=0)),
                         OptimizationLevel.O0)

    def test_small_clifford_returns_O1(self):
        d = describe_circuit(_make_small_clifford())
        self.assertEqual(select_level(d), OptimizationLevel.O1)

    def test_small_with_rotation_returns_O2(self):
        d = describe_circuit(_make_small_with_rotation())
        self.assertEqual(select_level(d), OptimizationLevel.O2)

    def test_large_clifford_returns_O2(self):
        d = CircuitDescription(gate_count=150, depth=10,
                                  clifford_gate_count=150,
                                  rotation_gate_count=0)
        # All Clifford → not large-mixed.
        self.assertEqual(select_level(d), OptimizationLevel.O2)

    def test_large_mixed_returns_O3(self):
        d = CircuitDescription(gate_count=150, depth=10,
                                  clifford_gate_count=50,
                                  rotation_gate_count=100,
                                  has_conditional_gates=True)
        self.assertEqual(select_level(d), OptimizationLevel.O3)

    def test_hot_profile_bumps_level(self):
        d = describe_circuit(_make_small_clifford())  # would be O1
        p = OptimizationProfile(total_program_invocations=500)
        self.assertEqual(select_level(d, p), OptimizationLevel.O2)


# ----- Adaptive pipeline builder -------------------------------------

class TestBuildAdaptivePipeline(unittest.TestCase):
    def test_O0_returns_empty_pm(self):
        pm = build_adaptive_pipeline(level=OptimizationLevel.O0)
        self.assertEqual(pm.names_in_execution_order(), [])

    def test_O1_returns_single_pass_pm(self):
        pm = build_adaptive_pipeline(level=OptimizationLevel.O1)
        names = pm.names_in_execution_order()
        self.assertEqual(len(names), 1)
        self.assertIn("eqir_optimization_o1", names)

    def test_O2_includes_eqir_optimization(self):
        pm = build_adaptive_pipeline(level=OptimizationLevel.O2)
        names = pm.names_in_execution_order()
        self.assertTrue(any("eqir_optimization_o2" in n for n in names))

    def test_O2_clifford_only_adds_clifford_sweep(self):
        circuit = describe_circuit(_make_small_clifford())
        pm = build_adaptive_pipeline(level=OptimizationLevel.O2,
                                       circuit=circuit)
        names = pm.names_in_execution_order()
        self.assertTrue(any("clifford_sweep" in n for n in names))

    def test_O2_with_rotation_adds_rotation_merge(self):
        circuit = describe_circuit(_make_small_with_rotation())
        pm = build_adaptive_pipeline(level=OptimizationLevel.O2,
                                       circuit=circuit)
        names = pm.names_in_execution_order()
        self.assertTrue(any("rotation_merge" in n for n in names))

    def test_O3_with_rotation_includes_all_passes(self):
        circuit = describe_circuit(_make_small_with_rotation())
        pm = build_adaptive_pipeline(level=OptimizationLevel.O3,
                                       circuit=circuit)
        names = pm.names_in_execution_order()
        # eqir + rotation_merge required; clifford_sweep is skipped
        # because the circuit is not Clifford-only.
        self.assertTrue(any("eqir_optimization_o3a" in n for n in names))
        self.assertTrue(any("rotation_merge" in n for n in names))

    def test_O3_with_profile_adds_second_eqir_pass(self):
        profile = OptimizationProfile()
        # Make profile think eqir_optimization removes gates.
        profile.add_pass_entry(PassProfileEntry(
            pass_name="eqir_optimization_o3a",
            gates_before=10, gates_after=5, duration_ns=1,
            optimizations=5))
        circuit = CircuitDescription(gate_count=150, depth=2,
                                       clifford_gate_count=100,
                                       rotation_gate_count=50)
        pm = build_adaptive_pipeline(level=OptimizationLevel.O3,
                                       profile=profile, circuit=circuit)
        names = pm.names_in_execution_order()
        self.assertTrue(any("o3b_profile" in n for n in names))

    def test_AUTO_with_no_circuit_returns_O1(self):
        pm = build_adaptive_pipeline(level=OptimizationLevel.AUTO)
        names = pm.names_in_execution_order()
        self.assertEqual(len(names), 1)

    def test_AUTO_with_large_mixed_uses_O3(self):
        circuit = describe_circuit(_make_large_mixed())
        pm = build_adaptive_pipeline(level=OptimizationLevel.AUTO,
                                       circuit=circuit)
        names = pm.names_in_execution_order()
        # O3 should include eqir_optimization_o3a + rotation_merge
        # (the large_mixed has many rotations).
        self.assertTrue(any("eqir_optimization_o3a" in n for n in names))


# ----- End-to-end run_adaptive ----------------------------------------

class TestRunAdaptive(unittest.TestCase):
    def test_run_adaptive_runs_pipeline(self):
        # HH graph passes; the optimizer's H-self-cancel rule
        # removes both H gates. After running, the gate count
        # should drop to 0.
        g = EQIRGraph()
        g.add_operation("GATE", gate_name="H", args=[], targets=["q0"])
        g.add_operation("GATE", gate_name="H", args=[], targets=["q0"])
        report = run_adaptive(g, level=OptimizationLevel.O2)
        self.assertEqual(g.gate_count() if hasattr(g, "gate_count") else
                          sum(1 for n in g.nodes.values() if n.type == "GATE"),
                         0)
        self.assertIsNotNone(report)

    def test_run_adaptive_O0_does_nothing(self):
        g = _make_small_clifford()
        before = sum(1 for n in g.nodes.values() if n.type == "GATE")
        run_adaptive(g, level=OptimizationLevel.O0)
        after = sum(1 for n in g.nodes.values() if n.type == "GATE")
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
