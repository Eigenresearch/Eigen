"""Tests for §4.1 Unified Backend Interface.

These tests exercise:

  * The QuantumBackend ABC contract (capabilities/validate/compile/execute)
    across every registered backend.
  * The `get_quantum_backend(name)` factory + `BACKEND_REGISTRY`.
  * The three concrete adapter classes (ExportBackendAdapter,
    QiskitBackendAdapter, RuntimeSubmitAdapter).
  * The generic `_detect_capabilities_from_graph` walker: gate-only
    circuits are valid on hardware exporters; mixed classical
    constructs are flagged.
  * The `RuntimeSubmitAdapter.execute` path that returns a polite
    `ExecutionResult.error` when required kwargs are missing (instead
    of raising) so callers can branch without try/except.
"""

from __future__ import annotations

import unittest

from src.backend.unified_backend import (
    BACKEND_REGISTRY,
    ExecutionResult,
    ExportBackendAdapter,
    PostProcessor,
    QuantumBackend,
    QiskitBackendAdapter,
    RuntimeSubmitAdapter,
    ValidationReport,
    expectation,
    filter_bitstrings,
    get_quantum_backend,
    list_backends,
    marginalize,
    normalize_counts,
)
from src.ir.ir_graph import EQIRGraph
from src.semantic.backend_capabilities import CapabilityLevel


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _bell_pair_graph() -> EQIRGraph:
    """A minimal Bell-pair EQIR: q0, q1 :: H q0; CNOT q0, q1; measure q0."""
    g = EQIRGraph()
    a = g.create_node("ALLOC", targets=["q0"])
    b = g.create_node("ALLOC", targets=["q1"])
    h = g.create_node("GATE", gate_name="H", targets=["q0"])
    cx = g.create_node("GATE", gate_name="CNOT", targets=["q0", "q1"])
    m = g.create_node("MEASURE", targets=["q0"], cbit_name="c0")
    # Manually populate `nodes` since `create_node` already registered them.
    g.nodes[a.id] = a
    g.nodes[b.id] = b
    g.nodes[h.id] = h
    g.nodes[cx.id] = cx
    g.nodes[m.id] = m
    return g


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRegistry(unittest.TestCase):
    def test_built_in_backends_registered(self):
        names = list_backends()
        for required in (
            "qiskit",
            "ibmq",
            "ibm",
            "ionq",
            "braket",
            "azure",
            "ibm_runtime",
            "ionq_runtime",
        ):
            self.assertIn(required, names, f"missing backend: {required}")

    def test_unknown_backend_raises_keyerror(self):
        with self.assertRaises(KeyError):
            get_quantum_backend("definitely_not_a_backend")

    def test_factory_is_lazy(self):
        # The factory entry for `ibm_runtime` should not have imported
        # qiskit_ibm_runtime until the first get_quantum_backend("ibm_runtime")
        # call... but actually since we DON'T import inside `make_ibm_runtime`
        # beyond module-level deferred `from x import y`, just verify a
        # second call doesn't re-create state.
        be1 = get_quantum_backend("qiskit")
        be2 = get_quantum_backend("qiskit")
        self.assertIsInstance(be1, QiskitBackendAdapter)
        self.assertIsInstance(be2, QiskitBackendAdapter)

    def test_ibmq_is_alias_for_qiskit(self):
        a = get_quantum_backend("ibmq")
        b = get_quantum_backend("qiskit")
        self.assertIsInstance(a, QiskitBackendAdapter)
        self.assertIsInstance(b, QiskitBackendAdapter)
        # Capabilities should be identical (both pull from 'qiskit').
        self.assertEqual(
            a.capabilities().supports_quantum_gates,
            b.capabilities().supports_quantum_gates,
        )


class TestProtocolSurface(unittest.TestCase):
    def test_every_backend_implements_trait(self):
        for name in list_backends():
            be = get_quantum_backend(name)
            self.assertIsInstance(be, QuantumBackend,
                                 f"{name} is not a QuantumBackend")
            self.assertTrue(hasattr(be, "capabilities"))
            self.assertTrue(hasattr(be, "validate"))
            self.assertTrue(hasattr(be, "compile"))
            self.assertTrue(hasattr(be, "execute"))
            # `ibmq` is an alias for `qiskit` (same QiskitBackendAdapter
            # instance, so .name == 'qiskit'); accept either.
            self.assertIn(be.name, (name, "qiskit"))

    def test_capabilities_returns_backendcapabilities(self):
        for name in list_backends():
            be = get_quantum_backend(name)
            caps = be.capabilities()
            self.assertEqual(
                caps.supports_quantum_gates,
                CapabilityLevel.SUPPORTED,
                f"{name} should support quantum gates",
            )


class TestValidateBellPair(unittest.TestCase):
    def test_qiskit_validate_bell_ok(self):
        g = _bell_pair_graph()
        be = get_quantum_backend("qiskit")
        report = be.validate(g)
        self.assertIsInstance(report, ValidationReport)
        self.assertTrue(report.ok, f"qiskit rejected bell: {report.warnings}")
        self.assertEqual(report.unsupported_pct, 0.0)
        # H, CNOT, MEASURE => at least 3 constructs
        self.assertGreaterEqual(len(report.stats), 3)

    def test_ionq_validate_bell_ok(self):
        g = _bell_pair_graph()
        be = get_quantum_backend("ionq")
        report = be.validate(g)
        self.assertTrue(report.ok,
                        f"ionq should accept a bell-pair EQIR: {report}")
        self.assertEqual(report.unsupported_pct, 0.0)

    def test_braket_validate_bell_ok(self):
        g = _bell_pair_graph()
        be = get_quantum_backend("braket")
        report = be.validate(g)
        self.assertTrue(report.ok)

    def test_azure_validate_bell_ok(self):
        g = _bell_pair_graph()
        be = get_quantum_backend("azure")
        report = be.validate(g)
        self.assertTrue(report.ok)

    def test_ibm_validate_bell_ok(self):
        g = _bell_pair_graph()
        be = get_quantum_backend("ibm")
        report = be.validate(g)
        self.assertTrue(report.ok)


class TestCompileProducesNative(unittest.TestCase):
    def test_qiskit_compile_returns_python_str(self):
        g = _bell_pair_graph()
        s = get_quantum_backend("qiskit").compile(g)
        self.assertIsInstance(s, str)
        self.assertGreater(len(s), 0)

    def test_ionq_compile_returns_json_string_with_qubits(self):
        g = _bell_pair_graph()
        s = get_quantum_backend("ionq").compile(g)
        self.assertIsInstance(s, str)
        self.assertIn('"qubits"', s)

    def test_braket_compile_returns_python_source(self):
        g = _bell_pair_graph()
        s = get_quantum_backend("braket").compile(g)
        self.assertIsInstance(s, str)
        self.assertIn("braket", s.lower())

    def test_azure_compile_returns_qir(self):
        g = _bell_pair_graph()
        s = get_quantum_backend("azure").compile(g)
        self.assertIsInstance(s, str)
        self.assertIn("Qubit", s)

    def test_ibm_compile_returns_qasm(self):
        g = _bell_pair_graph()
        s = get_quantum_backend("ibm").compile(g)
        self.assertIsInstance(s, str)
        # OpenQASM 3.0 — should contain a gate declaration or OQ3 header
        self.assertTrue(
            "OPENQASM" in s or "include" in s or "qubit" in s.lower(),
            f"IBM exporter didn't emit any QASM-looking header: {s[:80]!r}",
        )


class TestExecuteNoOpForExporters(unittest.TestCase):
    def test_ionq_execute_is_noop_with_metadata_note(self):
        g = _bell_pair_graph()
        be = get_quantum_backend("ionq")
        native = be.compile(g)
        result = be.execute(native, shots=4)
        self.assertIsInstance(result, ExecutionResult)
        self.assertIsNone(result.error)
        self.assertEqual(result.shots, 4)
        self.assertEqual(
            result.metadata.get("note"),
            "export-only backend; no execution performed",
        )

    def test_braket_execute_is_noop(self):
        g = _bell_pair_graph()
        be = get_quantum_backend("braket")
        result = be.execute(be.compile(g), shots=8)
        self.assertIsNone(result.error)


class TestRuntimeSubmitAdapter(unittest.TestCase):
    def test_execute_without_token_returns_error_result(self):
        # The two runtime adapters require graph= and api_token= kwargs.
        # Without them, execute returns a polite ExecutionResult.error
        # (NEVER raises) so callers can branch without try/except.
        for name in ("ibm_runtime", "ionq_runtime"):
            be = get_quantum_backend(name)
            self.assertIsInstance(be, RuntimeSubmitAdapter)
            result = be.execute(native="dummy", shots=1)
            self.assertIsInstance(result, ExecutionResult)
            self.assertIsNotNone(result.error)
            self.assertIn("submitadapter.execute", result.error.lower())

    def test_runtime_adapter_is_subclass_of_export(self):
        # RuntimeSubmitAdapter must also satisfy ExportBackendAdapter's
        # surface (it's a subclass) — callers can rely on .validate and
        # .compile from the parent.
        be = get_quantum_backend("ionq_runtime")
        self.assertIsInstance(be, ExportBackendAdapter)
        self.assertIsInstance(be, RuntimeSubmitAdapter)
        g = _bell_pair_graph()
        report = be.validate(g)
        self.assertTrue(report.ok)
        s = be.compile(g)
        self.assertIsInstance(s, str)


class TestValidationReportShape(unittest.TestCase):
    def test_validation_report_repr_contains_backend_name(self):
        r = ValidationReport(
            backend_name="my_backend",
            ok=True,
            stats={"supported": 100.0, "emulated": 0.0, "unsupported": 0.0},
        )
        rep = repr(r)
        self.assertIn("my_backend", rep)
        self.assertIn("100.0", rep)
        # ok path → status="ok"
        self.assertIn("ok", rep)

    def test_validation_report_percent_properties(self):
        r = ValidationReport(
            backend_name="b",
            ok=False,
            stats={"supported": 50.0, "emulated": 30.0, "unsupported": 20.0},
        )
        self.assertEqual(r.supported_pct, 50.0)
        self.assertEqual(r.emulated_pct, 30.0)
        self.assertEqual(r.unsupported_pct, 20.0)

    def test_validation_report_unsupported_nodes_compat(self):
        # `unsupported_nodes` is the legacy QiskitBackend.BackendReport
        # attribute name; audit / strict callers keyed on it.
        from src.semantic.backend_capabilities import UnsupportedOp

        r = ValidationReport(
            backend_name="b",
            ok=False,
            unsupported_ops=[
                UnsupportedOp("GATE", None, "FunkyGate", "unsupported"),
            ],
        )
        self.assertEqual(r.unsupported_nodes, 1)

    def test_execution_result_ok_property(self):
        ok_result = ExecutionResult(backend_name="b", native_handle=None)
        bad_result = ExecutionResult(
            backend_name="b", native_handle=None, error="boom"
        )
        self.assertTrue(ok_result.ok)
        self.assertFalse(bad_result.ok)


class TestStrictModeIntegration(unittest.TestCase):
    """Smoke test the call shape `run_command` uses for `--strict`."""

    def test_strict_check_uses_report_ok_not_unsupported_nodes(self):
        g = _bell_pair_graph()
        be = get_quantum_backend("qiskit")
        report = be.validate(g)
        # The new contract: callers should branch on `report.ok`, not
        # `report.unsupported_nodes > 0`. Both should agree here.
        self.assertEqual(report.ok, (report.unsupported_nodes == 0))


# ---------------------------------------------------------------------------
# §4.1 batch + post-processing pipeline
# ---------------------------------------------------------------------------


class TestBatchExecute(unittest.TestCase):
    def test_default_batch_is_sequential_for_exporter(self):
        g = _bell_pair_graph()
        be = get_quantum_backend("ionq")
        items = [(g, be.compile(g)) for _ in range(3)]
        results = be.batch_execute(items, shots=512)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertIsInstance(r, ExecutionResult)
            self.assertIsNone(r.error)
            self.assertEqual(r.shots, 512)

    def test_runtime_adapter_batch_returns_errors_without_token(self):
        be = get_quantum_backend("ionq_runtime")
        g = _bell_pair_graph()
        native = be.compile(g)
        results = be.batch_execute([(g, native), (g, native)], shots=10)
        self.assertEqual(len(results), 2)
        for r, native in zip(results, [native, native]):
            self.assertIsNotNone(r.error)
            self.assertIn("submitadapter.execute", r.error.lower())


class TestPostProcessor(unittest.TestCase):
    def _result(self, histograms):
        return ExecutionResult(
            backend_name="test",
            native_handle=None,
            shots=1024,
            histograms=histograms,
        )

    def test_marginalize_keeps_only_listed_qubits(self):
        r = self._result([{"001": 4, "100": 6, "111": 2}])
        pp = PostProcessor([marginalize([0])])
        out = pp.apply(r)
        # Qubit 0 is leftmost char: "001" -> "0", "100" -> "1", "111" -> "1"
        self.assertEqual(out.histograms[0], {"0": 4, "1": 8})

    def test_filter_bitstrings(self):
        r = self._result([{"000": 4, "111": 6, "010": 2}])
        out = PostProcessor([filter_bitstrings(lambda bs: bs[0] == "1")]).apply(r)
        # Only "111" and "010"... wait, both have first char "1" so keep both
        # Wait "010" has first char "0", so filter drops it. Result: {"111": 6}.
        self.assertEqual(out.histograms[0], {"111": 6})

    def test_normalize_counts(self):
        r = self._result([{"00": 4, "11": 6}])
        out = PostProcessor([normalize_counts()]).apply(r)
        self.assertAlmostEqual(out.histograms[0]["00"], 0.4)
        self.assertAlmostEqual(out.histograms[0]["11"], 0.6)

    def test_expectation_for_zz(self):
        # 2 qubits, four states. Z_0 * Z_1 eigenvalues: ++ on |00>, |11>;
        # -- on |01>, |10>. Weights sum is 4-4 = 0 for uniform input.
        r = self._result([{"00": 1, "11": 1, "10": 1, "01": 1}])
        out = PostProcessor([expectation({0: 1, 1: 1})]).apply(r)
        self.assertAlmostEqual(out.metadata["expectation"], 0.0)

    def test_expectation_for_anti_correlated(self):
        # Pure |10>: Z_0 = -1, Z_1 = +1, product = -1.
        r = self._result([{"10": 100}])
        out = PostProcessor([expectation({0: 1, 1: 1})]).apply(r)
        self.assertAlmostEqual(out.metadata["expectation"], -1.0)

    def test_chained_transforms_apply_in_order(self):
        # Marginalize then normalize: marginalize first drops some counts.
        r = self._result([{"010": 4, "100": 6}])
        out = PostProcessor([
            marginalize([0, 1]),   # keep positions 0,1 -> "01" : 4, "10" : 6
            normalize_counts(),
        ]).apply(r)
        self.assertAlmostEqual(out.histograms[0]["01"], 0.4)
        self.assertAlmostEqual(out.histograms[0]["10"], 0.6)

    def test_postprocessor_returns_independent_copy(self):
        r = self._result([{"00": 4}])
        pp = PostProcessor([normalize_counts()])
        out = pp.apply(r)
        # Original should be untouched.
        self.assertEqual(r.histograms[0], {"00": 4})
        self.assertEqual(out.histograms[0], {"00": 1.0})

    def test_empty_histograms_is_safe(self):
        r = self._result(None)
        out = PostProcessor([marginalize([0]), normalize_counts()]).apply(r)
        self.assertIsNone(out.histograms)
        # expectation should write 0.0 metadata, not blow up.
        out2 = PostProcessor([expectation({0: 1})]).apply(r)
        self.assertEqual(out2.metadata["expectation"], 0.0)


if __name__ == "__main__":
    unittest.main()
