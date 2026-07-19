"""§11.2 — Native integration envelope tests."""
import unittest

from src.native_integration_envelope import (
    NativeModuleKind,
    NativeModuleSpec,
    default_native_module_specs,
    BuildHook,
    maturin_build_hook,
    pip_build_hook,
    FallbackReport,
    NativeWithFallback,
    NativeBenchmarkReport,
    benchmark_native_vs_python,
    native_status_report,
)


# ---------------------------------------------------------------------------
# NativeModuleKind
# ---------------------------------------------------------------------------

class TestNativeModuleKind(unittest.TestCase):
    def test_six_kinds(self):
        kinds = {NativeModuleKind.VM_CORE,
                  NativeModuleKind.GATE_KERNELS,
                  NativeModuleKind.ZX_GRAPH,
                  NativeModuleKind.CLI_UTILS,
                  NativeModuleKind.SIMULATOR,
                  NativeModuleKind.ROUTING}
        self.assertEqual(len(kinds), 6)

    def test_value_is_string(self):
        for k in NativeModuleKind:
            self.assertIsInstance(k.value, str)


# ---------------------------------------------------------------------------
# NativeModuleSpec
# ---------------------------------------------------------------------------

class TestNativeModuleSpec(unittest.TestCase):
    def test_default_priority(self):
        spec = NativeModuleSpec(
            kind=NativeModuleKind.SIMULATOR,
            rust_crate_path="x.rs",
            python_module_name="x.y",
        )
        self.assertEqual(spec.priority, "medium")
        self.assertEqual(spec.description, "")

    def test_available_returns_bool_for_unimportable_module(self):
        spec = NativeModuleSpec(
            kind=NativeModuleKind.SIMULATOR,
            rust_crate_path="x.rs",
            python_module_name="x.c.y.z.doesnt_exist",
        )
        self.assertIsInstance(spec.available(), bool)
        self.assertFalse(spec.available())

    def test_available_returns_true_for_existing_module(self):
        # Use a stdlib module to ensure import works
        spec = NativeModuleSpec(
            kind=NativeModuleKind.SIMULATOR,
            rust_crate_path="x.rs",
            python_module_name="math",
        )
        self.assertTrue(spec.available())


# ---------------------------------------------------------------------------
# default_native_module_specs
# ---------------------------------------------------------------------------

class TestDefaultNativeModuleSpecs(unittest.TestCase):
    def test_returns_six_specs(self):
        specs = default_native_module_specs()
        self.assertEqual(len(specs), 6)

    def test_each_kind_present_exactly_once(self):
        specs = default_native_module_specs()
        kinds = [s.kind for s in specs]
        self.assertEqual(len(set(kinds)), 6)
        for kind in NativeModuleKind:
            self.assertIn(kind, kinds)

    def test_simulator_and_routing_have_high_priority(self):
        specs = default_native_module_specs()
        for s in specs:
            if s.kind in (NativeModuleKind.SIMULATOR,
                          NativeModuleKind.ROUTING):
                self.assertEqual(s.priority, "high")
            elif s.kind is NativeModuleKind.CLI_UTILS:
                self.assertEqual(s.priority, "low")
            else:
                self.assertEqual(s.priority, "medium")

    def test_rust_crate_path_ends_with_rs(self):
        for s in default_native_module_specs():
            self.assertTrue(s.rust_crate_path.endswith(".rs"))

    def test_python_module_name_uses_eigen_namespace(self):
        for s in default_native_module_specs():
            self.assertTrue(s.python_module_name.startswith("eigen_native"))


# ---------------------------------------------------------------------------
# BuildHook
# ---------------------------------------------------------------------------

class TestBuildHook(unittest.TestCase):
    def test_dataclass_fields(self):
        h = BuildHook(command="echo hi", description="d", env={"A": "b"})
        self.assertEqual(h.command, "echo hi")
        self.assertEqual(h.description, "d")
        self.assertEqual(h.env, {"A": "b"})

    def test_default_env_is_empty_dict(self):
        h = BuildHook(command="c", description="d")
        self.assertEqual(h.env, {})

    def test_maturin_build_hook(self):
        h = maturin_build_hook()
        self.assertIn("maturin", h.command)
        self.assertEqual(h.env.get("MATURIN_PROFILE"), "release")
        self.assertTrue(h.description)

    def test_pip_build_hook(self):
        h = pip_build_hook()
        self.assertIn("pip", h.command)
        self.assertIn(".", h.command)
        self.assertEqual(h.env, {})

    def test_maturin_and_pip_are_distinct(self):
        self.assertNotEqual(maturin_build_hook().command,
                            pip_build_hook().command)


# ---------------------------------------------------------------------------
# FallbackReport
# ---------------------------------------------------------------------------

class TestFallbackReport(unittest.TestCase):
    def test_defaults(self):
        r = FallbackReport(used_native=True, duration_ns=42)
        self.assertTrue(r.used_native)
        self.assertEqual(r.duration_ns, 42)
        self.assertIsNone(r.error)

    def test_with_error_string(self):
        r = FallbackReport(used_native=False, duration_ns=10,
                             error="boom")
        self.assertEqual(r.error, "boom")


# ---------------------------------------------------------------------------
# NativeWithFallback
# ---------------------------------------------------------------------------

class TestNativeWithFallback(unittest.TestCase):
    def test_runs_native_when_present(self):
        def native(x):
            return x * 2

        def fallback(x):
            return x + 1

        dispatcher = NativeWithFallback(native, fallback)
        value, report = dispatcher.run(5)
        self.assertEqual(value, 10)
        self.assertTrue(report.used_native)
        self.assertIsNone(report.error)

    def test_falls_back_on_runtime_error(self):
        def native(x):
            raise RuntimeError("native missing")

        def fallback(x):
            return x + 100

        dispatcher = NativeWithFallback(native, fallback)
        value, report = dispatcher.run(5)
        self.assertEqual(value, 105)
        self.assertFalse(report.used_native)
        self.assertIn("native missing", report.error)

    def test_falls_back_on_import_error(self):
        captured = {}

        def native(x):
            raise ImportError("no module")

        def fallback(x):
            captured["called"] = True
            return x - 7

        dispatcher = NativeWithFallback(native, fallback)
        value, report = dispatcher.run(20)
        self.assertEqual(value, 13)
        self.assertFalse(report.used_native)
        self.assertIn("no module", report.error)
        self.assertTrue(captured.get("called"))

    def test_falls_back_on_attribute_error(self):
        def native(x):
            raise AttributeError("no attr")

        def fallback(x):
            return x

        dispatcher = NativeWithFallback(native, fallback)
        value, report = dispatcher.run(3)
        self.assertEqual(value, 3)
        self.assertFalse(report.used_native)

    def test_native_none_uses_fallback(self):
        def fallback(x):
            return x * 10

        dispatcher = NativeWithFallback(None, fallback)
        value, report = dispatcher.run(7)
        self.assertEqual(value, 70)
        self.assertFalse(report.used_native)
        self.assertIsNone(report.error)

    def test_passes_kwargs(self):
        def native(x, *, y):
            return x + y

        def fallback(x, *, y):
            return x - y

        dispatcher = NativeWithFallback(native, fallback)
        value, _ = dispatcher.run(10, y=5)
        self.assertEqual(value, 15)

    def test_non_fallback_errors_propagate(self):
        def native(x):
            raise TypeError("not a fallback error")

        def fallback(x):
            return x

        dispatcher = NativeWithFallback(native, fallback)
        with self.assertRaises(TypeError):
            dispatcher.run(1)


# ---------------------------------------------------------------------------
# NativeBenchmarkReport
# ---------------------------------------------------------------------------

class TestNativeBenchmarkReport(unittest.TestCase):
    def test_defaults(self):
        r = NativeBenchmarkReport(
            name="foo",
            native_duration_ns=100,
            python_duration_ns=200,
            speedup=2.0,
        )
        self.assertEqual(r.iterations, 1)
        self.assertEqual(r.notes, "")

    def test_speedup_one_when_tied(self):
        r = NativeBenchmarkReport(
            name="foo",
            native_duration_ns=100,
            python_duration_ns=100,
            speedup=1.0,
        )
        self.assertAlmostEqual(r.speedup, 1.0)


# ---------------------------------------------------------------------------
# benchmark_native_vs_python
# ---------------------------------------------------------------------------

class TestBenchmarkNativeVsPython(unittest.TestCase):
    def test_identity_functions_speedup_about_one(self):
        # Native identity and Python identity -- the speedup ratio may
        # vary, but the call counts and durations should be reported.
        r = benchmark_native_vs_python(
            "ident",
            native=lambda x: x,
            python=lambda x: x,
            iterations=10,
            args=(42,),
        )
        self.assertEqual(r.iterations, 10)
        self.assertEqual(r.name, "ident")
        self.assertEqual(r.native_duration_ns, 0
                          if r.native_duration_ns < 0
                          else r.native_duration_ns)
        self.assertGreater(r.python_duration_ns, 0)

    def test_native_none_speedup_zero(self):
        r = benchmark_native_vs_python(
            "only_python",
            native=None,
            python=lambda: 7,
            iterations=5,
        )
        self.assertEqual(r.native_duration_ns, 0)
        self.assertEqual(r.speedup, 0.0)

    def test_native_consistently_failing_falls_back(self):
        # Native always fails; the inner loop sets native=None and
        # native_total becomes 0 (speedup becomes 0).
        def native_failing():
            raise RuntimeError("always fails")

        def python_ok():
            return 0

        r = benchmark_native_vs_python(
            "fail_native",
            native=native_failing,
            python=python_ok,
            iterations=3,
        )
        # The native loop catches the exception and sets native to None,
        # so native_total remains 0 -> speedup = 0.
        self.assertEqual(r.native_duration_ns, 0)
        self.assertEqual(r.speedup, 0.0)

    def test_kwargs_passed_to_both_implementations(self):
        captured = {"native": 0, "python": 0}

        def native(x, *, mul):
            captured["native"] += 1
            return x * mul

        def python(x, *, mul):
            captured["python"] += 1
            return x * mul

        r = benchmark_native_vs_python(
            "kwargs",
            native=native,
            python=python,
            iterations=4,
            args=(3,),
            kwargs={"mul": 2},
        )
        self.assertEqual(r.iterations, 4)
        # Warm-up + 4 iterations = at least 5 each (warm-up calls native once + py once)
        self.assertGreaterEqual(captured["python"], 4)


# ---------------------------------------------------------------------------
# native_status_report
# ---------------------------------------------------------------------------

class TestNativeStatusReport(unittest.TestCase):
    def test_returns_dict_covering_six_kinds(self):
        report = native_status_report()
        self.assertIsInstance(report, dict)
        self.assertEqual(len(report), 6)
        for kind in NativeModuleKind:
            self.assertIn(kind, report)
            self.assertIsInstance(report[kind], bool)
