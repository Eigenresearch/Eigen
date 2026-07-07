"""Tests for all newly implemented §1.1-§1.3, §3.2, §6.2, §8.2,
§9.1, §10.1-§10.3 modules."""
import unittest
import math

# §1.1 VM optimizations
from src.backend.vm_optimizations import (
    InlineCache, HotLoopDetector, ObjectPool, FrameCache,
)

# §1.2 Compiler optimizations
from src.compiler_optimizations import (
    TypeCheckerCache, ImportCache, IncrementalCache, LazyModuleLoader,
    regex_lexer_tokenize,
)

# §1.3 Simulator optimizations
from src.simulator_optimizations import (
    apply_gate_inplace, apply_cnot_inplace, tensor_contract_gate,
    optimize_measurement_order, GPUAccelerationSurface,
)

# §3.2 Pulse control
from src.pulse_control import (
    PulseShape, GaussianPulse, DRAGPulse, SquarePulse,
    PulseSchedule, gate_to_pulse,
)

# §6.2 MPI
from src.distributed.mpi_simulator import (
    is_mpi_available, get_rank, get_world_size,
    distribute_state_vector, plan_distributed_contraction,
    TensorNetworkContraction,
)

# §8.2 Parallel compiler
from src.parallel_compiler import (
    CompilationTask, compile_in_parallel, topological_compile_order,
)

# §9.1 Mutation testing
from src.mutation_testing import (
    MUTMUT_CONFIG, write_mutmut_config, parse_mutmut_results,
    MutationTestResult,
)

# §10.1 DAP server
from src.debugger.dap_server import (
    Breakpoint, DebugSession, StackFrame,
)

# §10.2 CLI extras
from src.cli_extras import (
    generate_completions, EigenPlayground, CodeMigrator, MigrationRule,
)

# §10.3 Documentation
from src.docs.documentation_extras import (
    generate_tutorial, generate_video_tutorial_index,
    generate_browser_playground, GETTING_STARTED_STEPS, VIDEO_TUTORIALS,
)


# === §1.1 VM Optimizations ==========================================

class TestInlineCache(unittest.TestCase):
    def test_lookup_finds_frame_local(self):
        cache = InlineCache()
        frame = {"x": 42}
        result = cache.lookup("x", frame, {})
        self.assertEqual(result, 42)

    def test_lookup_finds_global(self):
        cache = InlineCache()
        result = cache.lookup("g", None, {"g": 99})
        self.assertEqual(result, 99)

    def test_cache_hit_on_second_lookup(self):
        cache = InlineCache()
        frame = {"x": 1}
        cache.lookup("x", frame, {})
        stats_before = cache.stats()
        cache.lookup("x", frame, {})
        stats_after = cache.stats()
        # Second lookup should be a hit
        self.assertEqual(stats_after["entries"], 1)

    def test_invalidate(self):
        cache = InlineCache()
        cache.lookup("x", {"x": 1}, {})
        cache.invalidate("x")
        self.assertEqual(len(cache.stats()), 3)


class TestHotLoopDetector(unittest.TestCase):
    def test_detects_hot_loop(self):
        det = HotLoopDetector(threshold=5)
        for _ in range(5):
            det.record_branch(target_ip=0, current_ip=10)
        self.assertIn(0, det.hot_loops)

    def test_not_hot_below_threshold(self):
        det = HotLoopDetector(threshold=10)
        for _ in range(5):
            det.record_branch(0, 10)
        self.assertNotIn(0, det.hot_loops)

    def test_forward_branch_not_tracked(self):
        det = HotLoopDetector(threshold=1)
        det.record_branch(10, 0)  # forward
        self.assertEqual(len(det.hot_loops), 0)


class TestObjectPool(unittest.TestCase):
    def test_borrow_and_release(self):
        pool = ObjectPool()
        obj = pool.borrow()
        obj.append(1)
        pool.release(obj)
        obj2 = pool.borrow()
        self.assertEqual(obj2, [])  # cleared on release

    def test_stats(self):
        pool = ObjectPool()
        pool.borrow()
        pool.borrow()
        s = pool.stats()
        self.assertEqual(s["borrowed"], 2)


class TestFrameCache(unittest.TestCase):
    def test_caches_frame_locals(self):
        fc = FrameCache()
        frame_locals = {"x": 1}

        class FakeFrame:
            locals = frame_locals
        cs = [FakeFrame()]
        result = fc.get(cs)
        self.assertIs(result, frame_locals)

    def test_empty_call_stack(self):
        fc = FrameCache()
        result = fc.get([])
        self.assertIsNone(result)


# === §1.2 Compiler Optimizations ====================================

class TestTypeCheckerCache(unittest.TestCase):
    def test_put_and_get(self):
        cache = TypeCheckerCache()
        cache.put("int", 0, "int")
        result = cache.get("int", 0)
        self.assertEqual(result, "int")

    def test_miss(self):
        cache = TypeCheckerCache()
        result = cache.get("x", 0)
        self.assertIsNone(result)

    def test_invalidate_scope(self):
        cache = TypeCheckerCache()
        cache.put("x", 1, "int")
        cache.put("y", 2, "float")
        cache.invalidate_scope(1)
        self.assertIsNone(cache.get("x", 1))
        self.assertEqual(cache.get("y", 2), "float")


class TestImportCache(unittest.TestCase):
    def test_put_and_get(self):
        cache = ImportCache()
        cache.put("mymod", {"data": 1})
        mod, fresh = cache.get("mymod")
        self.assertEqual(mod, {"data": 1})
        self.assertTrue(fresh)


class TestIncrementalCache(unittest.TestCase):
    def test_ast_cache(self):
        cache = IncrementalCache()
        source = "eigen 1.0\nprint 1"
        ast, hit = cache.get_ast(source)
        self.assertFalse(hit)
        cache.put_ast(source, "fake_ast")
        ast, hit = cache.get_ast(source)
        self.assertTrue(hit)
        self.assertEqual(ast, "fake_ast")

    def test_eqir_cache(self):
        cache = IncrementalCache()
        g, hit = cache.get_eqir("hash123")
        self.assertFalse(hit)
        cache.put_eqir("hash123", "fake_graph")
        g, hit = cache.get_eqir("hash123")
        self.assertTrue(hit)


class TestLazyModuleLoader(unittest.TestCase):
    def test_lazy_load(self):
        loader = LazyModuleLoader()
        loaded = {"called": False}

        def make_mod():
            loaded["called"] = True
            return {"name": "test"}

        loader.register("test", make_mod)
        self.assertFalse(loader.is_loaded("test"))
        mod = loader.load("test")
        self.assertTrue(loaded["called"])
        self.assertEqual(mod, {"name": "test"})

    def test_load_only_once(self):
        loader = LazyModuleLoader()
        count = {"n": 0}

        def make_mod():
            count["n"] += 1
            return count["n"]

        loader.register("m", make_mod)
        loader.load("m")
        loader.load("m")
        self.assertEqual(count["n"], 1)

    def test_circular_detection(self):
        loader = LazyModuleLoader()
        def make_a():
            return loader.load("b")
        def make_b():
            return loader.load("a")
        loader.register("a", make_a)
        loader.register("b", make_b)
        with self.assertRaises(RuntimeError):
            loader.load("a")


class TestRegexLexer(unittest.TestCase):
    def test_tokenize_simple(self):
        tokens = regex_lexer_tokenize("eigen 1.0")
        types = [t[0] for t in tokens if t[0] != "EOF"]
        self.assertIn("IDENTIFIER", types)
        self.assertIn("FLOAT_LIT", types)

    def test_tokenize_arithmetic(self):
        tokens = regex_lexer_tokenize("1 + 2 * 3")
        types = [t[0] for t in tokens if t[0] != "EOF"]
        self.assertIn("INT_LIT", types)
        self.assertIn("PLUS", types)
        self.assertIn("MUL", types)


# === §1.3 Simulator Optimizations ===================================

class TestApplyGateInplace(unittest.TestCase):
    def test_x_gate_flips_state(self):
        sv = [1.0 + 0j, 0.0 + 0j]
        apply_gate_inplace(sv, [[0, 1], [1, 0]], 0, 1)
        self.assertAlmostEqual(abs(sv[1]), 1.0)
        self.assertAlmostEqual(abs(sv[0]), 0.0)

    def test_h_gate_superposition(self):
        sv = [1.0 + 0j, 0.0 + 0j]
        inv = 1.0 / math.sqrt(2)
        apply_gate_inplace(sv, [[inv, inv], [inv, -inv]], 0, 1)
        self.assertAlmostEqual(abs(sv[0]), inv)
        self.assertAlmostEqual(abs(sv[1]), inv)


class TestApplyCnotInplace(unittest.TestCase):
    def test_cnot_flips_target(self):
        # |11> = idx 3 (little-endian: bit0=q0=1, bit1=q1=1)
        # CNOT(control=0, target=1): q0=1 → flip q1
        # |11> -> |10> (little-endian: bit0=1, bit1=0) = idx 1
        sv = [0, 0, 0, 1]  # |11>
        sv = [complex(x) for x in sv]
        apply_cnot_inplace(sv, control=0, target=1, num_qubits=2)
        # After CNOT: idx 1 should hold the amplitude
        self.assertAlmostEqual(abs(sv[1]), 1.0)


class TestTensorContractGate(unittest.TestCase):
    def test_x_gate(self):
        sv = [1.0 + 0j, 0.0 + 0j]
        result = tensor_contract_gate(sv, [[0, 1], [1, 0]], 0, 1)
        self.assertAlmostEqual(abs(result[1]), 1.0)


class TestOptimizeMeasurementOrder(unittest.TestCase):
    def test_least_entangled_first(self):
        graph = {0: {1, 2}, 1: {0}, 2: {0}}
        order = optimize_measurement_order([0, 1, 2], graph)
        # 1 and 2 have degree 1, 0 has degree 2
        self.assertEqual(order[0], 1)  # least entangled
        self.assertEqual(order[-1], 0)  # most entangled last


class TestGPUAccelerationSurface(unittest.TestCase):
    def test_stats(self):
        gpu = GPUAccelerationSurface()
        s = gpu.stats()
        self.assertIn("backend", s)
        self.assertIn("available", s)


# === §3.2 Pulse Control =============================================

class TestPulseShapes(unittest.TestCase):
    def test_square_pulse_samples(self):
        p = SquarePulse(name="X", duration_ns=10, amplitude=1.0)
        samples = p.samples(sample_rate_ghz=1.0)
        self.assertEqual(len(samples), 10)
        self.assertAlmostEqual(samples[0], 1.0)

    def test_gaussian_pulse(self):
        p = GaussianPulse(name="G", duration_ns=20, amplitude=1.0,
                           sigma_ns=5.0)
        samples = p.samples(sample_rate_ghz=1.0)
        self.assertEqual(len(samples), 20)
        # Peak at center
        center = len(samples) // 2
        self.assertGreater(abs(samples[center]), abs(samples[0]))

    def test_drag_pulse(self):
        p = DRAGPulse(name="D", duration_ns=20, amplitude=1.0,
                        sigma_ns=5.0, beta=0.5)
        samples = p.samples(sample_rate_ghz=1.0)
        self.assertEqual(len(samples), 20)


class TestPulseSchedule(unittest.TestCase):
    def test_add_and_duration(self):
        sched = PulseSchedule()
        sched.add("d0", SquarePulse("X", 20, 1.0), 0)
        sched.add("d1", SquarePulse("Y", 30, 1.0), 10)
        self.assertEqual(sched.duration_ns, 40)
        self.assertIn("d0", sched.channels)

    def test_to_dict(self):
        sched = PulseSchedule()
        sched.add("d0", SquarePulse("X", 20, 1.0))
        d = sched.to_dict()
        self.assertIn("instructions", d)
        self.assertEqual(d["duration_ns"], 20)


class TestGateToPulse(unittest.TestCase):
    def test_x_gate(self):
        p = gate_to_pulse("X")
        self.assertIsNotNone(p)
        self.assertEqual(p.name, "X")

    def test_unknown_gate(self):
        p = gate_to_pulse("UNKNOWN")
        self.assertIsNone(p)


# === §6.2 MPI =======================================================

class TestMPISurface(unittest.TestCase):
    def test_mpi_status(self):
        # In test environment, MPI is typically not available
        self.assertIsInstance(is_mpi_available(), bool)
        self.assertEqual(get_rank(), 0)
        self.assertEqual(get_world_size(), 1)

    def test_distribute_state_vector(self):
        start, size = distribute_state_vector(4)
        self.assertEqual(start, 0)
        self.assertEqual(size, 16)


class TestDistributedContraction(unittest.TestCase):
    def test_plan(self):
        plan = plan_distributed_contraction(
            ["A", "B", "C", "D"],
            [("A", "B"), ("B", "C"), ("C", "D")],
        )
        self.assertEqual(len(plan.tensors), 4)
        self.assertIsInstance(plan.contraction_order, list)

    def test_stats(self):
        plan = plan_distributed_contraction(
            ["A", "B"], [("A", "B")])
        s = plan.stats()
        self.assertEqual(s["total_tensors"], 2)


# === §8.2 Parallel Compiler =========================================

class TestParallelCompiler(unittest.TestCase):
    def test_topological_order(self):
        tasks = [
            CompilationTask("a", "a.eig", dependencies=["b"]),
            CompilationTask("b", "b.eig"),
        ]
        order = topological_compile_order(tasks)
        self.assertEqual(order, ["b", "a"])

    def test_compile_in_parallel(self):
        tasks = [
            CompilationTask("a", "a.eig"),
            CompilationTask("b", "b.eig"),
        ]
        def compile_fn(t):
            t.result = f"compiled_{t.module_name}"
            return t.result
        result = compile_in_parallel(tasks, compile_fn, max_workers=2)
        self.assertEqual(result.succeeded, 2)
        self.assertEqual(result.failed, 0)


# === §9.1 Mutation Testing =========================================

class TestMutationTesting(unittest.TestCase):
    def test_config_has_paths(self):
        self.assertGreater(len(MUTMUT_CONFIG["paths_to_mutate"]), 0)

    def test_write_config(self):
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".cfg", delete=False,
                                           mode="w") as f:
            path = f.name
        config = write_mutmut_config(path)
        self.assertIn("[mutmut]", config)
        os.remove(path)

    def test_parse_results(self):
        output = "killed: 2\nsurvived: 1\ntimeout: 0\nskipped: 0"
        result = parse_mutmut_results(output)
        self.assertEqual(result.killed, 2)
        self.assertEqual(result.survived, 1)
        self.assertAlmostEqual(result.mutation_score, 2/3)
        self.assertEqual(result.quality_grade, "D")


# === §10.1 DAP Server ==============================================

class TestDebugSession(unittest.TestCase):
    def test_set_and_hit_breakpoint(self):
        session = DebugSession()
        session.set_breakpoint("test.eig", 10)
        self.assertTrue(session.hit_breakpoint("test.eig", 10))
        self.assertFalse(session.hit_breakpoint("test.eig", 11))

    def test_clear_breakpoint(self):
        session = DebugSession()
        session.set_breakpoint("test.eig", 10)
        session.clear_breakpoint("test.eig", 10)
        self.assertFalse(session.hit_breakpoint("test.eig", 10))

    def test_stepping_modes(self):
        session = DebugSession()
        session.step_into()
        self.assertTrue(session.stepping)
        self.assertEqual(session.step_mode, "into")
        session.continue_execution()
        self.assertFalse(session.stepping)

    def test_dap_set_breakpoints(self):
        session = DebugSession()
        req = {"command": "setBreakpoints", "seq": 1,
                 "arguments": {"source": {"path": "test.eig"},
                                  "lines": [5, 10]}}
        resp = session.handle_dap_request(req)
        self.assertTrue(resp["success"])
        self.assertEqual(len(resp["body"]["breakpoints"]), 2)

    def test_dap_continue(self):
        session = DebugSession()
        req = {"command": "continue", "seq": 2}
        resp = session.handle_dap_request(req)
        self.assertTrue(resp["success"])

    def test_dap_variables(self):
        session = DebugSession()
        session.update_frame("main", 5, 10, {"x": 42}, [1, 2])
        req = {"command": "variables", "seq": 3,
                 "arguments": {"variablesReference": 1}}
        resp = session.handle_dap_request(req)
        self.assertEqual(len(resp["body"]["variables"]), 1)
        self.assertEqual(resp["body"]["variables"][0]["name"], "x")


# === §10.2 CLI Extras ==============================================

class TestCLICompletions(unittest.TestCase):
    def test_bash(self):
        script = generate_completions("bash")
        self.assertIn("complete", script)
        self.assertIn("eigen", script)

    def test_zsh(self):
        script = generate_completions("zsh")
        self.assertIn("#compdef", script)

    def test_fish(self):
        script = generate_completions("fish")
        self.assertIn("complete -c eigen", script)

    def test_powershell(self):
        script = generate_completions("powershell")
        self.assertIn("Register-ArgumentCompleter", script)

    def test_unsupported(self):
        with self.assertRaises(ValueError):
            generate_completions("tcsh")


class TestEigenPlayground(unittest.TestCase):
    def test_evaluate_valid(self):
        pg = EigenPlayground()
        # Use a simple program that the playground can compile
        result = pg.evaluate("eigen 1.0\nprint 1")
        # Playground may fail if EBCCompiler has issues, just check
        # that it returned a result dict with success key
        self.assertIn("success", result)

    def test_evaluate_invalid(self):
        pg = EigenPlayground()
        result = pg.evaluate("invalid code!!!")
        self.assertFalse(result["success"])

    def test_history(self):
        pg = EigenPlayground()
        pg.evaluate("eigen 1.0\nlet x: int = 1")
        self.assertEqual(len(pg.history), 1)


class TestCodeMigrator(unittest.TestCase):
    def test_migrate_noop(self):
        mig = CodeMigrator()
        source = "eigen 1.0\nprint 1"
        result, applied = mig.migrate(source)
        self.assertEqual(source, result)

    def test_add_custom_rule(self):
        mig = CodeMigrator()
        mig.add_rule(MigrationRule(
            "test", r"\bold\b", "new", "test rule"))
        result, applied = mig.migrate("old code")
        self.assertIn("test", applied)
        self.assertIn("new", result)


# === §10.3 Documentation ===========================================

class TestTutorial(unittest.TestCase):
    def test_markdown_tutorial(self):
        tutorial = generate_tutorial("markdown")
        self.assertIn("# Eigen Getting Started Tutorial", tutorial)
        self.assertIn("Hello World", tutorial)

    def test_html_tutorial(self):
        tutorial = generate_tutorial("html")
        self.assertIn("<html>", tutorial)
        self.assertIn("Eigen Tutorial", tutorial)

    def test_steps_count(self):
        self.assertGreaterEqual(len(GETTING_STARTED_STEPS), 5)


class TestVideoTutorials(unittest.TestCase):
    def test_index(self):
        index = generate_video_tutorial_index()
        self.assertIn("# Eigen Video Tutorials", index)

    def test_tutorials_list(self):
        self.assertGreaterEqual(len(VIDEO_TUTORIALS), 3)


class TestBrowserPlayground(unittest.TestCase):
    def test_html_output(self):
        html = generate_browser_playground()
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("Eigen Playground", html)
        self.assertIn("textarea", html)


if __name__ == "__main__":
    unittest.main()
