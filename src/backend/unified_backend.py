"""§4.1 Unified Backend Interface — a single QuantumBackend contract over
every Eigen backend.

Existing concrete backends expose incompatible surfaces:

    QiskitBackend.transpile(graph, ast) -> (str, BackendReport)
        — QASM/Python script plus AST capability validation.

    IonQBackend.export(graph)            -> str                (IonQ JSON)
    AzureBackend.export(graph)           -> str                  (QIR text)
    IBMBackend.export(graph)             -> str            (OpenQASM 3.0)
    IBMBackend.export_qasm2(graph)       -> str            (legacy QASM 2.0)
    BraketBackend.export(graph)          -> str              (Braket Python)

    submit_to_ibm_quantum(graph, token, backend, shots=...) -> IBMJobResult
    submit_to_ionq(graph, token, ...)                       -> IonQ job
        — runtime submission functions that drive the actual network call.

§4.1 of sol.md proposes a single trait:

    trait QuantumBackend {
        fn compile(circuit)        -> NativeCircuit
        fn execute(native, shots)  -> Results
        fn capabilities()          -> BackendCapabilities
        fn validate(circuit)       -> ValidationReport
    }

`QuantumBackend` (ABC) below is that trait, plus three concrete adapter
classes (`ExportBackendAdapter`, `QiskitBackendAdapter`,
`RuntimeSubmitAdapter`) that wrap the heterogeneous concrete backends
behind the unified surface, and `get_quantum_backend(name)` factory +
`BACKEND_REGISTRY`.

Design:

  * `compile()` returns the backend's native serialization
    (OpenQASM string, IonQ JSON string, Azure QIR text, Braket Python
    source, ...). For runtimes, that native blob is what gets sent over
    the wire; for exporters, it IS the final artifact.

  * `execute()` defaults to a no-op (exporters don't execute). Runtime
    adapters override to perform network submission.

  * `validate()` walks the EQIR graph (and optionally AST) against the
    backend's `BackendCapabilities` and returns a `ValidationReport`
    with the SUPPORTED/EMULATED/UNSUPPORTED breakdown percentages and
    a list of offending constructs.

  * `capabilities()` returns the backend's static `BackendCapabilities`
    (looked up by backend name in `backend_capabilities.py`).

The adapter pattern keeps the existing concrete backend classes
(especially `QiskitBackend`, which is exercised by ~12 unittests)
untouched — we layer the unified contract on top of them, not inside.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Callable

from src.ir.ir_graph import EQIRGraph
from src.semantic.backend_capabilities import (
    BackendCapabilities,
    CapabilityLevel,
    get_backend_capabilities,
    UnsupportedOp,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ValidationReport:
    """Result of checking EQIR (and optional AST) against a backend's
    `BackendCapabilities`.

    `ok` is True iff nothing was classified `UNSUPPORTED`.

    `stats` carries the three percentage keys used by the audit/strict
    command and the existing `BackendReport`:

        stats["supported"]    in % (SUPPORTED construct share)
        stats["emulated"]     in % (EMULATED construct share)
        stats["unsupported"]  in % (UNSUPPORTED construct share)

    `unsupported_ops` is a list of `UnsupportedOp` instances; each has
    `kind`, `pretty_repr`, `reason`, optionally `source_span`.

    `warnings` lists human-readable strings (EMULATED hints).
    """

    backend_name: str
    ok: bool
    unsupported_ops: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def supported_pct(self) -> float:
        return self.stats.get("supported", 100.0 if self.ok else 0.0)

    @property
    def emulated_pct(self) -> float:
        return self.stats.get("emulated", 0.0)

    @property
    def unsupported_pct(self) -> float:
        return self.stats.get("unsupported", 0.0)

    @property
    def unsupported_nodes(self) -> int:
        """.unsupported_nodes — backward-compat shim for callers that
        used `QiskitBackend.BackendReport.unsupported_nodes`."""
        return len(self.unsupported_ops)

    def __repr__(self) -> str:
        status = "ok" if self.ok else "issues"
        out = [
            f"[{self.backend_name}] validation {status}: "
            f"{self.supported_pct:.1f}% supported, "
            f"{self.emulated_pct:.1f}% emulated, "
            f"{self.unsupported_pct:.1f}% unsupported",
        ]
        for w in self.warnings:
            out.append(f"  W: {w}")
        for op in self.unsupported_ops:
            out.append(f"  U: {op.pretty_repr} ({op.kind}): {op.reason}")
        return "\n".join(out)


@dataclass
class ExecutionResult:
    """Result of running a compiled circuit on a backend.

    For exporter adapters, `execute()` is a no-op; the result has
    `error=None`, `histograms=None`, and a metadata note flagging that
    no execution occurred.

    For runtime submission adapters, `execute()` performs the actual
    network submit. On success, `histograms` holds `[{bitstring: count}]`
    per shot-batch; on failure, `error` carries the exception message.
    """

    backend_name: str
    native_handle: Any
    shots: int = 0
    histograms: Optional[list] = None
    metadata: dict = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# The QuantumBackend Trait
# ---------------------------------------------------------------------------


class QuantumBackend(ABC):
    """Unified backend contract — §4.1.

    Concrete backends vary widely:

      * *Exporters* (IonQ, Azure, Braket, IBM QASM) only translate
        EQIR to a target-language blob and never execute. For those,
        `compile()` returns a string/dict and `execute()` is a no-op.

      * `QiskitBackend.transpile` returns both the script and a
        capability report. `QiskitBackendAdapter` delegates validation
        to that existing pass and surfaces a `ValidationReport`.

      * Runtime submission backends (IBM/IonQ cloud) require network and
        credentials. `execute()` is the network call itself, guarded by
        `MissingExtraError` (or a polite `ExecutionResult.error`)
        if the pip-extra isn't installed.
    """

    name: str = ""

    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """Return the static capability set for this backend."""

    @abstractmethod
    def validate(self, graph: EQIRGraph, ast=None) -> ValidationReport:
        """Walk graph (+AST) and check supports_*/emulated/unsupported."""

    @abstractmethod
    def compile(self, graph: EQIRGraph, ast=None, **kwargs):
        """Translate EQIR into the backend's native serialization
        (QASM string, IonQ JSON, Azure QIR text, Braket Python, ...)."""

    def execute(self, native, shots: int = 1024, **kwargs) -> ExecutionResult:
        """Default no-op execution (export-only backends).

        Submission backends override to actually fire jobs.

        Args:
            native: the artifact returned by `compile()`.
            shots: shots requested. Ignored by exporters.
        """
        return ExecutionResult(
            backend_name=self.name,
            native_handle=native,
            shots=shots,
            metadata={"note": "export-only backend; no execution performed"},
        )

    # ---- §4.1 batch + post-processing pipeline (optional surface) --------

    def batch_execute(
        self,
        circuits: list,
        shots: int = 1024,
        parallel: bool = False,
        **kwargs,
    ) -> list:
        """Run multiple circuits and return a `list[ExecutionResult]`.

        Default implementation is sequential — concrete runtime adapters
        whose SDK exposes true parallel submission (e.g. IBM Runtime
        sessions or IonQ batch endpoints) override this method.

        Args:
            circuits: list of (graph, native_handle) tuples — the graph
                      is for submission functions, the native handle is
                      what `compile()` already produced.
            shots: shots per circuit.
            parallel: hint that the caller prefers parallel submission.
                      Adapters that don't support real parallelism should
                      still return correct results, just sequentially.
        """
        results = []
        for graph, native in circuits:
            local_kwargs = dict(kwargs)
            local_kwargs["graph"] = graph
            results.append(self.execute(native, shots=shots, **local_kwargs))
        return results


# ---------------------------------------------------------------------------
# Result post-processing pipeline — §4.1
# ---------------------------------------------------------------------------


class PostProcessor:
    """Pluggable result post-processing pipeline.

    A `PostProcessor` holds an ordered list of `Transform`s; `apply(result)`
    runs them in sequence on a copy of the result and returns a new
    `ExecutionResult`. Transforms are pure functions
    `transform(result: ExecutionResult) -> ExecutionResult`.

    Built-in `Transform`s:

      * `marginalize(qubits)`  — drop counts outside the listed qubits
        from each histogram bitstring.
      * `filter_bitstrings(predicate)` — keep only bitstrings matching
        the predicate.
      * `normalize_counts()`             — divide each count by total.
      * `expectation(observable)`        — compute <Z...Z> expectation
        (Pauli-Z operators per qubit; `observable` is a dict mapping
        qubit_name -> +1/-1 weight; absent qubits = +1).
    """

    def __init__(self, transforms=None):
        self.transforms = list(transforms or [])

    def add(self, transform) -> "PostProcessor":
        self.transforms.append(transform)
        return self

    def apply(self, result: ExecutionResult) -> ExecutionResult:
        import copy as _copy
        r = _copy.deepcopy(result)
        for t in self.transforms:
            r = t(r)
        return r


def _bitstring_apply_mask(bitstring: str, mask_indices: list) -> str:
    """Keep only the characters at the listed indices, in order."""
    return "".join(bitstring[i] for i in mask_indices)


def marginalize(qubit_indices: list):
    """Drop all qubits not in `qubit_indices` from each bitstring.

    `qubit_indices` is a 0-indexed list of positions to keep (in the
    bitstring's natural left-to-right order).
    """
    keep = sorted(set(qubit_indices))

    def _transform(result: ExecutionResult) -> ExecutionResult:
        if not result.histograms:
            return result
        new_hists = []
        for hist in result.histograms:
            new_hist = {}
            for bs, count in hist.items():
                key = _bitstring_apply_mask(bs, keep)
                new_hist[key] = new_hist.get(key, 0) + count
            new_hists.append(new_hist)
        result.histograms = new_hists
        return result

    return _transform


def filter_bitstrings(predicate):
    """Keep only bitstrings matching the predicate `f(bitstring) -> bool`."""

    def _transform(result: ExecutionResult) -> ExecutionResult:
        if not result.histograms:
            return result
        new_hists = []
        for hist in result.histograms:
            new_hist = {bs: c for bs, c in hist.items() if predicate(bs)}
            new_hists.append(new_hist)
        result.histograms = new_hists
        return result

    return _transform


def normalize_counts():
    """Replace integer counts with normalized probabilities (0..1)."""

    def _transform(result: ExecutionResult) -> ExecutionResult:
        if not result.histograms:
            return result
        new_hists = []
        for hist in result.histograms:
            total = sum(hist.values()) or 1
            new_hists.append({bs: c / total for bs, c in hist.items()})
        result.histograms = new_hists
        return result

    return _transform


def expectation(observable: dict):
    """Compute <Σ Z_i^w_i> and stash the scalar in `metadata['expectation']`.

    `observable` maps bitstring index -> +1/-1 weight. Missing indices
    default to +1 (I).
    """
    # Sort indices descending so `keep` matches the bitstring layout.
    indices = sorted(observable.keys(), reverse=True)

    def _transform(result: ExecutionResult) -> ExecutionResult:
        if not result.histograms:
            result.metadata["expectation"] = 0.0
            return result
        total_exp = 0.0
        total_count = 0
        for hist in result.histograms:
            for bs, count in hist.items():
                bit_value = 1
                for i in indices:
                    # bitstring[0] is the leftmost char (qubit 0)
                    # Z eigenvalue: '0' -> +1, '1' -> -1
                    bit = bs[i] if i < len(bs) else "0"
                    z_eig = 1 if bit == "0" else -1
                    bit_value *= z_eig ** (1 if observable[i] > 0
                                           else 0)
                    # weight not 0 -> apply Z^1; weight 0 -> identity
                    if observable[i] < 0:
                        bit_value *= z_eig
                total_exp += bit_value * count
                total_count += count
        result.metadata["expectation"] = total_exp / max(1, total_count)
        return result

    return _transform


# ---------------------------------------------------------------------------
# Generic EQIR→capability walker (shared by exporter adapters)
# ---------------------------------------------------------------------------


def _detect_capabilities_from_graph(
    graph: EQIRGraph,
    caps: BackendCapabilities,
    backend_name: str,
    ast=None,
) -> ValidationReport:
    """Walk EQIR graph and check every construct against `caps`.

    EQIR node types we recognise:

      ALLOC      — qubit allocation (always SUPPORTED; capability is
                    about gates/measurements applied later, not allocation)
      GATE       — quantum gate under `supports_quantum_gates`
      MEASURE    — under `supports_measurements`
      TRACE      — under `supports_quantum_gates` (a partial trace still
                    manipulates the quantum state)
      PRINT      — under `supports_classical_functions`
      ASSERT     — under `supports_classical_functions`

    Any unknown node type counts as SUPPORTED (the IR-level construct is
    orthogonal to backend capabilities). The walker also visits optional
    AST statements (`ImportNode`, `FuncDeclNode`) for capability hints
    when the caller passes an AST.
    """

    unsupported = []
    warnings = []
    counts = {
        CapabilityLevel.SUPPORTED: 0,
        CapabilityLevel.EMULATED: 0,
        CapabilityLevel.UNSUPPORTED: 0,
    }

    def classify(level: CapabilityLevel, kind: str, label: str) -> None:
        counts[level] = counts.get(level, 0) + 1
        if level == CapabilityLevel.UNSUPPORTED:
            unsupported.append(
                UnsupportedOp(
                    kind=kind,
                    source_span=None,
                    pretty_repr=label,
                    reason=f"{label} unsupported by {backend_name} backend",
                )
            )
        elif level == CapabilityLevel.EMULATED:
            warnings.append(f"{label} emulated on {backend_name}")

    for node in graph.nodes.values():
        if node.type == "GATE":
            classify(
                caps.supports_quantum_gates, "GATE", f"{node.gate_name} gate"
            )
        elif node.type == "MEASURE":
            classify(caps.supports_measurements, "MEASURE", "measurement")
        elif node.type == "ALLOC":
            counts[CapabilityLevel.SUPPORTED] += 1
        elif node.type in ("TRACE",):
            classify(
                caps.supports_quantum_gates, "TRACE", "partial trace"
            )
        elif node.type == "PRINT":
            classify(
                caps.supports_classical_functions, "PRINT", "classical print"
            )
        elif node.type == "ASSERT":
            classify(
                caps.supports_classical_functions, "ASSERT", "assertion"
            )
        else:
            counts[CapabilityLevel.SUPPORTED] += 1

    if ast is not None:
        statements = getattr(ast, "statements", None)
        if statements is None:
            statements = getattr(ast, "body", None) or []
        for stmt in statements:
            cls = type(stmt).__name__
            if cls == "ImportNode":
                module_path = getattr(stmt, "module_path", "?")
                classify(
                    caps.supports_imports,
                    "ImportNode",
                    f"import {module_path}",
                )
            elif cls == "FuncDeclNode":
                classify(
                    caps.supports_recursion,
                    "FuncDeclNode",
                    "function declaration",
                )

    total = max(1, sum(counts.values()))
    stats = {
        "supported": 100.0 * counts[CapabilityLevel.SUPPORTED] / total,
        "emulated": 100.0 * counts[CapabilityLevel.EMULATED] / total,
        "unsupported": 100.0 * counts[CapabilityLevel.UNSUPPORTED] / total,
    }
    ok = counts[CapabilityLevel.UNSUPPORTED] == 0
    return ValidationReport(
        backend_name=backend_name,
        ok=ok,
        unsupported_ops=unsupported,
        warnings=warnings,
        stats=stats,
    )


# ---------------------------------------------------------------------------
# Concrete adapters
# ---------------------------------------------------------------------------


class ExportBackendAdapter(QuantumBackend):
    """Adapter for stateless text-export backends.

    Their concrete backend class exposes `export(graph) -> str` (IonQ,
    Azure, Braket, IBM QASM exporters). `execute()` is a no-op.
    """

    def __init__(
        self,
        name: str,
        exporter_cls,
        extra_capabilities: Optional[BackendCapabilities] = None,
    ):
        self.name = name
        self._exporter_cls = exporter_cls
        self._extra_caps = extra_capabilities

    def capabilities(self) -> BackendCapabilities:
        if self._extra_caps is not None:
            return self._extra_caps
        return get_backend_capabilities(self.name)

    def validate(self, graph: EQIRGraph, ast=None) -> ValidationReport:
        return _detect_capabilities_from_graph(
            graph, self.capabilities(), self.name, ast
        )

    def compile(self, graph: EQIRGraph, ast=None, **kwargs):
        exporter = self._exporter_cls()
        return exporter.export(graph)


class QiskitBackendAdapter(QuantumBackend):
    """Adapter for `QiskitBackend.transpile(graph, ast) -> (str, BackendReport)`.

    `QiskitBackend` already does capability checking inside `transpile`
    and returns its own `BackendReport`; we surface that report as the
    `ValidationReport` so callers see a uniform shape across every backend.
    """

    def __init__(self):
        from src.backend.qiskit_backend import QiskitBackend

        self.name = "qiskit"
        self._impl = QiskitBackend()

    def capabilities(self) -> BackendCapabilities:
        return self._impl.capabilities

    def validate(self, graph: EQIRGraph, ast=None) -> ValidationReport:
        _script, report = self._impl.transpile(graph, ast)
        return ValidationReport(
            backend_name=self.name,
            ok=(report.unsupported_nodes == 0),
            unsupported_ops=[
                UnsupportedOp(
                    kind="ASTNode",
                    source_span=None,
                    pretty_repr=w,
                    reason=w,
                )
                for w in report.warnings
            ],
            warnings=list(report.warnings),
            stats=dict(report.stats),
        )

    def compile(self, graph: EQIRGraph, ast=None, **kwargs):
        script, _report = self._impl.transpile(graph, ast)
        return script


class RuntimeSubmitAdapter(ExportBackendAdapter):
    """Adapter for *runtime submission* backends.

    Wraps a pair (`exporter_cls`, `submit_fn`):

      * `exporter_cls.export(graph) -> str`           — used by `compile()`.
      * `submit_fn(graph, api_token, *, shots, ...)`   — the actual network
        submission, used by `execute()`.

    `execute()` requires the caller to pass `graph=` and either
    `api_token=` or `token=` via kwargs so that the network submit can
    proceed; passing these via `kwargs` (NOT constructor) is deliberate —
    it keeps tokens out of the adapter's long-lived state.
    """

    def __init__(
        self,
        name: str,
        exporter_cls,
        submit_fn: Callable,
        extra_capabilities: Optional[BackendCapabilities] = None,
    ):
        super().__init__(name, exporter_cls, extra_capabilities)
        self._submit_fn = submit_fn

    def execute(self, native, shots: int = 1024, **kwargs) -> ExecutionResult:
        graph = kwargs.get("graph")
        token = kwargs.get("api_token") or kwargs.get("token")
        if graph is None or token is None:
            return ExecutionResult(
                backend_name=self.name,
                native_handle=native,
                shots=shots,
                error=(
                    "RuntimeSubmitAdapter.execute requires graph= and "
                    "api_token= kwargs for network submission"
                ),
            )
        real_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ("graph", "api_token", "token")
        }
        real_kwargs.setdefault("shots", shots)
        try:
            result = self._submit_fn(graph, token, **real_kwargs)
            histograms = []
            counts = getattr(result, "counts", None)
            if counts:
                histograms.append(dict(counts))
            job_id = getattr(result, "job_id", None)
            return ExecutionResult(
                backend_name=self.name,
                native_handle=native,
                shots=shots,
                histograms=histograms or None,
                metadata={
                    "job_id": job_id,
                    "backend_name": getattr(result, "backend_name", None),
                    "raw": getattr(result, "raw_result", None),
                },
            )
        except Exception as e:  # noqa: BLE001
            return ExecutionResult(
                backend_name=self.name,
                native_handle=native,
                shots=shots,
                error=f"{type(e).__name__}: {e}",
            )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


BACKEND_REGISTRY: dict = {}


def register_backend(name: str, factory: Callable[[], QuantumBackend]) -> None:
    BACKEND_REGISTRY[name] = factory


def get_quantum_backend(name: str) -> QuantumBackend:
    """Return the unified `QuantumBackend` adapter for `name`.

    Lazy: each registry entry is a zero-arg factory so we don't import
    optional heavy modules (qiskit_ibm_runtime, etc.) unless someone
    actually asks for that backend.

    Raises:
        KeyError: if `name` is not in the registry.
    """
    factory = BACKEND_REGISTRY.get(name)
    if factory is None:
        raise KeyError(
            f"Unknown quantum backend: '{name}'. "
            f"Known backends: {sorted(BACKEND_REGISTRY)}"
        )
    return factory()


def list_backends() -> list:
    """Return the sorted list of registered backend names."""
    return sorted(BACKEND_REGISTRY)


def _register_default_backends() -> None:
    """Populate the registry with the built-in backends.

    Qiskit / IBM QASM / IonQ / Azure / Braket are the export-only
    backends. Runtime submission (IBM Quantum, IonQ Cloud) is registered
    under names ending in `_runtime` so callers can request either the
    pure export or the full submit flow.
    """

    def make_qiskit() -> QuantumBackend:
        return QiskitBackendAdapter()

    register_backend("qiskit", make_qiskit)
    register_backend("ibmq", make_qiskit)

    def make_ibm() -> QuantumBackend:
        from src.backend.backends.ibm_backend import IBMBackend

        return ExportBackendAdapter("ibm", IBMBackend)

    register_backend("ibm", make_ibm)

    def make_ionq() -> QuantumBackend:
        from src.backend.backends.ionq_backend import IonQBackend

        return ExportBackendAdapter("ionq", IonQBackend)

    register_backend("ionq", make_ionq)

    def make_braket() -> QuantumBackend:
        from src.backend.backends.braket_backend import BraketBackend

        return ExportBackendAdapter("braket", BraketBackend)

    register_backend("braket", make_braket)

    def make_azure() -> QuantumBackend:
        from src.backend.backends.azure_backend import AzureBackend

        return ExportBackendAdapter("azure", AzureBackend)

    register_backend("azure", make_azure)

    def make_ibm_runtime() -> QuantumBackend:
        from src.backend.backends.ibm_backend import IBMBackend
        from src.backend.backends.ibm_runtime import submit_to_ibm_quantum

        return RuntimeSubmitAdapter(
            "ibm_runtime",
            IBMBackend,
            submit_to_ibm_quantum,
        )

    register_backend("ibm_runtime", make_ibm_runtime)

    def make_ionq_runtime() -> QuantumBackend:
        from src.backend.backends.ionq_backend import IonQBackend
        from src.backend.backends.ionq_runtime import submit_to_ionq

        return RuntimeSubmitAdapter(
            "ionq_runtime",
            IonQBackend,
            submit_to_ionq,
        )

    register_backend("ionq_runtime", make_ionq_runtime)

    def make_simulator(name: str = "simulator") -> QuantumBackend:
        from src.backend.backends.simulator_backend import SimulatorBackend

        return SimulatorBackend(name=name)

    register_backend("simulator", make_simulator)
    register_backend("local", lambda: make_simulator("local"))


_register_default_backends()
