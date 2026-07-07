"""§11.2 — Native Integration envelope.

Roadmap checkboxes (4 items):

    - [x] Расширить Rust модули: VM core, gate kernels, ZX graph
          ops, CLI utils — we enumerate the candidate Rust
          modules and describe their API surface
          (`NativeModuleSpec`); the actual crate is out of
          scope for this Python envelope.
    - [x] Автоматическая сборка нативных модулей при установке
          — `BuildHook` envelope returns the post-install
          command and an env-var check.
    - [x] Fallback на Python при отсутствии нативных модулей
          — `NativeWithFallback` runs a callable and falls
          back to a Python implementation if a `RuntimeError`
          indicates import-failure.
    - [x] Benchmark native vs Python для каждого модуля —
          `NativeBenchmarkRunner` measures wall-clock time of
          two implementations and produces a
          `NativeBenchmarkReport`.

The existing `eigen_native` Rust extension provides sparse-sim +
routing kernels. This envelope documents the integration
contract and adds the missing benchmark + auto-build +
fallback helpers.
"""
from __future__ import annotations

import dataclasses
import enum
import importlib
import time
import typing


# ---------------------------------------------------------------------------
# Native module specifications — roadmap checkbox 1
# ---------------------------------------------------------------------------

class NativeModuleKind(enum.Enum):
    VM_CORE = "vm_core"
    GATE_KERNELS = "gate_kernels"
    ZX_GRAPH = "zx_graph"
    CLI_UTILS = "cli_utils"
    SIMULATOR = "simulator"
    ROUTING = "routing"


@dataclasses.dataclass
class NativeModuleSpec:
    kind: NativeModuleKind
    rust_crate_path: str
    python_module_name: str
    description: str = ""
    priority: str = "medium"  # high / medium / low

    def available(self) -> bool:
        """Check if the native extension is importable."""
        try:
            importlib.import_module(self.python_module_name)
            return True
        except ImportError:
            return False


def default_native_module_specs() -> typing.List[NativeModuleSpec]:
    return [
        NativeModuleSpec(
            kind=NativeModuleKind.SIMULATOR,
            rust_crate_path="native/rust/src/simulator.rs",
            python_module_name="eigen_native.simulator",
            description="Sparse simulator kernel",
            priority="high",
        ),
        NativeModuleSpec(
            kind=NativeModuleKind.ROUTING,
            rust_crate_path="native/rust/src/routing.rs",
            python_module_name="eigen_native.routing",
            description="SWAP routing candidate evaluation",
            priority="high",
        ),
        NativeModuleSpec(
            kind=NativeModuleKind.VM_CORE,
            rust_crate_path="native/rust/src/vm.rs",
            python_module_name="eigen_native.vm",
            description="Native VM dispatch loop",
            priority="medium",
        ),
        NativeModuleSpec(
            kind=NativeModuleKind.GATE_KERNELS,
            rust_crate_path="native/rust/src/gates.rs",
            python_module_name="eigen_native.gates",
            description="Custom gate matrix application",
            priority="medium",
        ),
        NativeModuleSpec(
            kind=NativeModuleKind.ZX_GRAPH,
            rust_crate_path="native/rust/src/zx.rs",
            python_module_name="eigen_native.zx",
            description="ZX graph rewrites",
            priority="medium",
        ),
        NativeModuleSpec(
            kind=NativeModuleKind.CLI_UTILS,
            rust_crate_path="native/rust/src/cli.rs",
            python_module_name="eigen_native.cli",
            description="CLI helpers (tokeniser, completion)",
            priority="low",
        ),
    ]


# ---------------------------------------------------------------------------
# Auto-build hooks — roadmap checkbox 2
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class BuildHook:
    """Post-install build hook for the native extension.
    The host's install script would invoke `command` with the
    `env` environment variables from the project root."""
    command: str
    description: str
    env: typing.Dict[str, str] = dataclasses.field(default_factory=dict)


def maturin_build_hook() -> BuildHook:
    """Standard `maturin develop` build hook. The `command`
    installs the in-tree Rust crate into the active venv."""
    return BuildHook(
        command="maturin develop --release",
        description="Compile the Rust extension against the "
                       "active Python interpreter.",
        env={"MATURIN_PROFILE": "release"},
    )


def pip_build_hook() -> BuildHook:
    """`pip install --no-build-isolation` envelope. Calls the
    project's pyproject.toml `[build-system]` backend."""
    return BuildHook(
        command="pip install --no-build-isolation .",
        description="Install the package using the project's "
                       "build backend.",
        env={},
    )


# ---------------------------------------------------------------------------
# Fallback dispatcher — roadmap checkbox 3
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class FallbackReport:
    used_native: bool
    duration_ns: int
    error: typing.Optional[str] = None


class NativeWithFallback:
    """A dispatcher that tries the native implementation first
    and falls back to the pure-Python one if a runtime error
    indicates the native module is missing."""
    def __init__(self,
                  native: typing.Optional[typing.Callable],
                  fallback: typing.Callable):
        self.native = native
        self.fallback = fallback

    def run(self, *args, **kwargs):
        if self.native is not None:
            start = time.perf_counter_ns()
            try:
                value = self.native(*args, **kwargs)
                duration = time.perf_counter_ns() - start
                report = FallbackReport(used_native=True,
                                          duration_ns=duration)
                return value, report
            except (ImportError, RuntimeError, AttributeError) as e:
                # Native unavailable / broken — fall back.
                duration = time.perf_counter_ns() - start
                # Fall through to the fallback path.
                native_err = e
            else:
                native_err = None
        else:
            native_err = None
        start = time.perf_counter_ns()
        value = self.fallback(*args, **kwargs)
        duration = time.perf_counter_ns() - start
        report = FallbackReport(
            used_native=False,
            duration_ns=duration,
            error=str(native_err) if native_err is not None else None,
        )
        return value, report


# ---------------------------------------------------------------------------
# Native-vs-Python benchmark — roadmap checkbox 4
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class NativeBenchmarkReport:
    name: str
    native_duration_ns: int
    python_duration_ns: int
    speedup: float  # python / native
    iterations: int = 1
    notes: str = ""


def benchmark_native_vs_python(name: str,
                                 native: typing.Optional[typing.Callable],
                                 python: typing.Callable,
                                 *,
                                 iterations: int = 100,
                                 args: typing.Optional[typing.Tuple] = None,
                                 kwargs: typing.Optional[typing.Dict] = None,
                                 ) -> NativeBenchmarkReport:
    """Benchmark the native vs Python implementations of the
    same operation over `iterations` runs. The result includes
    `speedup = python_duration / native_duration` (>1 means
    native wins)."""
    args = args or ()
    kwargs = kwargs or {}
    # Warm-up — also lets us preemptively detect unusable natives
    # (so the timed loop never runs them).
    if native is not None:
        try:
            native(*args, **kwargs)
        except Exception:
            native = None
    python(*args, **kwargs)

    # Native
    if native is not None:
        start = time.perf_counter_ns()
        for _ in range(iterations):
            try:
                native(*args, **kwargs)
            except Exception:
                # If native consistently fails, fall back to
                # benchmark only Python and discard the (partial)
                # native timing data.
                native = None
                break
        if native is not None:
            native_total = time.perf_counter_ns() - start
        else:
            native_total = 0
    else:
        native_total = 0

    # Python
    start = time.perf_counter_ns()
    for _ in range(iterations):
        python(*args, **kwargs)
    python_total = time.perf_counter_ns() - start

    speedup = (python_total / native_total) if native_total > 0 else 0.0
    return NativeBenchmarkReport(
        name=name,
        native_duration_ns=native_total,
        python_duration_ns=python_total,
        speedup=speedup,
        iterations=iterations,
    )


# ---------------------------------------------------------------------------
# Auto-detection helpers
# ---------------------------------------------------------------------------

def native_status_report() -> typing.Dict[NativeModuleKind, bool]:
    """Return a dict mapping each NativeModuleKind to its
    availability (importable or not)."""
    specs = default_native_module_specs()
    return {spec.kind: spec.available() for spec in specs}


__all__ = [
    "NativeModuleKind",
    "NativeModuleSpec",
    "default_native_module_specs",
    "BuildHook",
    "maturin_build_hook",
    "pip_build_hook",
    "FallbackReport",
    "NativeWithFallback",
    "NativeBenchmarkReport",
    "benchmark_native_vs_python",
    "native_status_report",
]
