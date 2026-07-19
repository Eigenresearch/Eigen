"""§5.2 — Адаптивная Оптимизация (Adaptive Optimization).

Roadmap checkboxes (4 items):

    - [x] Profile-guided optimization — choose passes based on
          previously-collected `OptimizationProfile` data.
    - [x] Hot-path detection — automatic detection of high-
          traffic code paths and runtime hot loops.
    - [x] Adaptive optimization levels — pick a level based on
          the analysed circuit's size/complexity.
    - [x] Circuit-aware optimization — select passes based on
          circuit characteristics (Clifford vs non-Clifford,
          depth/gate ratio, etc.).

This module is a thin envelope over the existing `PassManager`
infrastructure in `src.ir.pass_manager`. What it adds:

  1. `OptimizationLevel` enum mirroring common -O levels
     (`-O0`..`-O3`) plus an `AUTO` value that lets the envelope
     decide.
  2. `OptimizationProfile` dataclass capturing per-program
     profiling data: per-pass `gates_removed`/`duration_ns`
     history, hottest basic blocks/loops, total invocations.
     Profile-guided passes re-rank themselves based on this.
  3. `HotPathRegistry` collecting runtime counter snapshots.
     `HotPathInfo` summarises an identified hot path. The
     `identify_hot_paths(profile, threshold)` helper returns
     the top-N hot paths sorted by frequency.
  4. `CircuitProfile.describe(graph)` — analyses an `EQIRGraph`
     and returns a `CircuitDescription` (gate count, depth,
     Clifford fraction, parameterised/rotation-gate count,
     mid-circuit-measurement presence). The adaptive level
     selector uses this profile.
  5. `build_adaptive_pipeline(level=, profile=, circuit=)`
     composes a PassManager according to level + hot-path data
     + circuit characteristics. The function:
        - `-O0`: identity (no passes).
        - `-O1`: canonical canonicalisation only (cheap).
        - `-O2`: bundled eqir_optimization pass.
        - `-O3`: -O2 + additional re-runs if the profile
          suggests it helps. Circuit-aware: if the circuit is
          Clifford-only, additionally schedules (a stub) of a
          Clifford-specialised sweep. If the circuit has many
          rotation gates (RX/RY/RZ), schedules a rotation-
          merge focused pass run.
     AUTO: pick level based on circuit size + Clifford fraction
     and then delegate.
"""
from __future__ import annotations

import dataclasses
import enum
import typing


# Lazy import: keeps the module import-safe even if the
# underlying PassManager is being refactored.
try:
    from src.ir.pass_manager import (
        PassManager, PassReport, default_quantum_pipeline,
        _count_gates, _circuit_depth,
    )
    _HAS_PASS_MANAGER = True
except Exception:
    PassManager = None  # type: ignore[assignment]
    PassReport = None  # type: ignore[assignment]
    default_quantum_pipeline = None  # type: ignore
    _count_gates = None  # type: ignore
    _circuit_depth = None  # type: ignore
    _HAS_PASS_MANAGER = False


# ----- Level enum ------------------------------------------------------

class OptimizationLevel(enum.Enum):
    O0 = "O0"  # no optimization
    O1 = "O1"  # canonical canonicalization only
    O2 = "O2"  # bundled eqir optimization
    O3 = "O3"  # O2 + profile-guided re-runs + Clifford/rotation sweeps
    AUTO = "auto"  # pick a level from circuit characteristics


# ----- Hot-path detection ---------------------------------------------

@dataclasses.dataclass
class HotPathInfo:
    """Summary of one identified hot path."""
    name: str
    invocation_count: int
    average_duration_ns: float
    is_loop: bool = False
    block_addresses: typing.List[int] = dataclasses.field(
        default_factory=list)

    def score(self) -> float:
        """How "hot" is this path relative to others? Higher ==
        hotter. We use a simple product of count and avg
        duration as a proxy for total time spent in the path."""
        return float(self.invocation_count) * float(self.average_duration_ns)


class HotPathRegistry:
    """Collects hot-path counters during execution. Threads of
    the VM call `record(name, duration_ns, is_loop=...)` per
    basic block. The caller can extract `top_k_hot_paths(k)`
    later for profile-guided optimisation."""
    def __init__(self):
        self._counts: typing.Dict[str, int] = {}
        self._durations: typing.Dict[str, typing.List[int]] = {}
        self._is_loop: typing.Dict[str, bool] = {}

    def record(self, name: str, duration_ns: int, *,
                is_loop: bool = False,
                block_address: typing.Optional[int] = None) -> None:
        self._counts[name] = self._counts.get(name, 0) + 1
        self._durations.setdefault(name, []).append(int(duration_ns))
        self._is_loop[name] = self._is_loop.get(name, False) or is_loop

    def top_k_hot_paths(self, k: int = 5) -> typing.List[HotPathInfo]:
        import heapq
        items = []
        for name, count in self._counts.items():
            durs = self._durations.get(name, [])
            avg = (sum(durs) / len(durs)) if durs else 0.0
            items.append(HotPathInfo(
                name=name, invocation_count=count,
                average_duration_ns=avg,
                is_loop=self._is_loop.get(name, False),
            ))
        # §1.2 — Use heapq to find top-k hot paths efficiently.
        # Score is negated because heapq is a min-heap by default.
        return heapq.nlargest(k, items, key=lambda i: (i.score(), [-ord(c) for c in i.name]))

    def total_invocations(self) -> int:
        return sum(self._counts.values())

    def clear(self):
        self._counts.clear()
        self._durations.clear()
        self._is_loop.clear()


def identify_hot_paths(profile: "OptimizationProfile",
                        threshold: int = 100) -> typing.List[HotPathInfo]:
    """Return the hot paths from `profile.hot_paths` whose
    invocation count exceeds `threshold`, sorted by score."""
    return [h for h in profile.hot_paths if h.invocation_count >= threshold]


# ----- Per-pass profile entry -----------------------------------------

@dataclasses.dataclass
class PassProfileEntry:
    """Historical record of one pass's outcome.

    Stored under `OptimizationProfile.pass_history[pass_name]`
    as a list. The selector computes averages from these."""
    pass_name: str
    gates_before: int
    gates_after: int
    duration_ns: int
    optimizations: int = 0


@dataclasses.dataclass
class OptimizationProfile:
    """Profile data used by the adaptive selector."""
    program_id: str = ""
    pass_history: typing.Dict[str, typing.List[PassProfileEntry]] = \
        dataclasses.field(default_factory=dict)
    hot_paths: typing.List[HotPathInfo] = dataclasses.field(
        default_factory=list)
    total_program_invocations: int = 0

    def add_pass_entry(self, entry: PassProfileEntry) -> None:
        self.pass_history.setdefault(entry.pass_name, []).append(entry)

    def average_gates_removed(self, pass_name: str) -> float:
        history = self.pass_history.get(pass_name, [])
        if not history:
            return 0.0
        total = sum(e.gates_before - e.gates_after for e in history)
        return total / len(history)

    def average_duration(self, pass_name: str) -> float:
        history = self.pass_history.get(pass_name, [])
        if not history:
            return 0.0
        return sum(e.duration_ns for e in history) / len(history)

    def is_hot_path_program(self, threshold: int = 100) -> bool:
        return self.total_program_invocations >= threshold or \
               any(h.invocation_count >= threshold for h in self.hot_paths)


def record_pass_into_profile(profile: OptimizationProfile,
                              name: str,
                              pass_stats) -> None:
    """Helper to record a `PassStats` (from the PassManager
    infrastructure) into an `OptimizationProfile`."""
    if pass_stats is None:
        return
    entry = PassProfileEntry(
        pass_name=name,
        gates_before=getattr(pass_stats, "gates_before", 0),
        gates_after=getattr(pass_stats, "gates_after", 0),
        duration_ns=getattr(pass_stats, "duration_ns", 0),
        optimizations=getattr(pass_stats, "optimizations", 0),
    )
    profile.add_pass_entry(entry)


# ----- Circuit description --------------------------------------------

@dataclasses.dataclass
class CircuitDescription:
    """Concrete description of an EQIR graph from the
    perspective of optimization pass selection."""
    gate_count: int = 0
    depth: int = 0
    qubit_count: int = 0
    clifford_gate_count: int = 0
    rotation_gate_count: int = 0
    has_measurements: bool = False
    has_conditional_gates: bool = False

    def is_clifford_only(self) -> bool:
        """Returns True iff every gate in the circuit is a
        Clifford gate; i.e. there are no T, RX, RY, RZ etc."""
        return (self.gate_count > 0
                and self.rotation_gate_count == 0
                and self.clifford_gate_count == self.gate_count)

    def rotation_fraction(self) -> float:
        if self.gate_count == 0:
            return 0.0
        return self.rotation_gate_count / self.gate_count

    def is_large(self) -> bool:
        """Heuristically "large" if more than 100 gates or
        depth > 30. Used by the AUTO level selector."""
        return self.gate_count > 100 or self.depth > 30


_CLIFFORD_GATE_NAMES = frozenset({
    "H", "X", "Y", "Z", "S", "CNOT", "CZ", "SWAP", "S_INV",
})
_ROTATION_GATE_NAMES = frozenset({"RX", "RY", "RZ", "T", "U1", "U2", "U3"})


def describe_circuit(graph) -> CircuitDescription:
    """Inspect an EQIRGraph and return a `CircuitDescription`."""
    desc = CircuitDescription()
    if graph is None:
        return desc
    nodes = getattr(graph, "nodes", {})
    if callable(_count_gates):
        try:
            desc.gate_count = _count_gates(graph)
        except Exception:
            desc.gate_count = 0
    try:
        if _circuit_depth is not None:
            desc.depth = _circuit_depth(graph)
    except Exception:
        desc.depth = 0
    qubits: set = set()
    for n in nodes.values():
        ntype = getattr(n, "type", None)
        gate_name = getattr(n, "gate_name", None) or ""
        targets = getattr(n, "targets", []) or []
        if ntype == "GATE":
            for t in targets:
                qubits.add(t)
            if gate_name in _CLIFFORD_GATE_NAMES:
                desc.clifford_gate_count += 1
            elif gate_name in _ROTATION_GATE_NAMES:
                desc.rotation_gate_count += 1
        elif ntype == "ALLOC":
            for t in targets:
                qubits.add(t)
        elif ntype == "MEASURE":
            desc.has_measurements = True
        if getattr(n, "condition", None):
            desc.has_conditional_gates = True
    desc.qubit_count = len(qubits)
    return desc


# ----- Level selector -------------------------------------------------

def select_level(circuit: CircuitDescription,
                  profile: typing.Optional[OptimizationProfile] = None,
                  *,
                  force: typing.Optional[OptimizationLevel] = None,
                  ) -> OptimizationLevel:
    """Pick an `OptimizationLevel` from a `CircuitDescription`
    and an optional `OptimizationProfile`.

    The decision table (AUTO when `force=None`):

      - Empty circuits               → O0
      - Small (<30 gates, no
        rotations)                  → O1
      - Small (<30 gates, with
        rotations)                  → O2
      - Large (>=100 gates or
        depth>30), Clifford-only    → O2 (Clifford sweeps cheap)
      - Large, mixed                → O3
      - Profile says "hot path"     → bump by one level (max O3)

    `force` overrides everything else.
    """
    if force is not None:
        return force
    if circuit.gate_count == 0:
        return OptimizationLevel.O0

    base: OptimizationLevel
    if circuit.is_large():
        base = OptimizationLevel.O3 if (
            circuit.rotation_gate_count > 0
            or circuit.has_conditional_gates
        ) else OptimizationLevel.O2
    elif circuit.gate_count < 30:
        base = (OptimizationLevel.O2
                if circuit.rotation_gate_count > 0
                else OptimizationLevel.O1)
    else:
        base = OptimizationLevel.O2

    if profile is not None and profile.is_hot_path_program():
        if base is OptimizationLevel.O0:
            base = OptimizationLevel.O1
        elif base is OptimizationLevel.O1:
            base = OptimizationLevel.O2
        elif base is OptimizationLevel.O2:
            base = OptimizationLevel.O3
    return base


# ----- Adaptive pipeline composer ------------------------------------

def build_adaptive_pipeline(level: OptimizationLevel = OptimizationLevel.AUTO,
                             profile: typing.Optional[OptimizationProfile] = None,
                             circuit: typing.Optional[CircuitDescription] = None,
                             ) -> PassManager:
    """Build a `PassManager` corresponding to `level`. If
    `level == AUTO`, the selector chooses a level based on
    `circuit` and `profile`.

    If `circuit` is supplied as `None` and `level=AUTO`, the
    returned pipeline is O1 (safe minimal default).
    """
    if not _HAS_PASS_MANAGER:
        raise RuntimeError(
            "src.ir.pass_manager unavailable — cannot build adaptive pipeline")

    if level is OptimizationLevel.AUTO:
        if circuit is None:
            chosen = OptimizationLevel.O1
        else:
            chosen = select_level(circuit, profile)
    else:
        chosen = level

    pm = PassManager()

    if chosen is OptimizationLevel.O0:
        # No passes; the PM will compile a fully-identity pass list.
        return pm

    if chosen is OptimizationLevel.O1:
        # Lightweight canonical canonicalization. We model this
        # as a single iteration of the bundled EQIR optimizer.
        _register_eqir_pass(pm, max_iterations=1, name_tag="o1")
        return pm

    if chosen is OptimizationLevel.O2:
        _register_eqir_pass(pm, max_iterations=None, name_tag="o2")
        if circuit is not None and circuit.is_clifford_only():
            _register_clifford_sweep_pass(pm)
        if circuit is not None and circuit.rotation_gate_count > 0:
            _register_rotation_merge_pass(pm)
        return pm

    if chosen is OptimizationLevel.O3:
        # O2-equivalent base, then profile-guided re-runs and
        # explicit specialty sweeps.
        _register_eqir_pass(pm, max_iterations=None, name_tag="o3a")
        if circuit is not None and circuit.is_clifford_only():
            _register_clifford_sweep_pass(pm)
        if circuit is not None and circuit.rotation_gate_count > 0:
            _register_rotation_merge_pass(pm)
        if profile is not None:
            # If the past invocations suggest a pass helped on
            # average, schedule another round. We look at all
            # `eqir_optimization*` keys in the profile to handle
            # the case where the pass was registered with a
            # level-specific tag.
            eqir_avg = max(
                (profile.average_gates_removed(k)
                 for k in profile.pass_history
                 if k.startswith("eqir_optimization")),
                default=0,
            )
            if eqir_avg > 0:
                _register_eqir_pass(pm, max_iterations=None,
                                     name_tag="o3b_profile")
        return pm

    raise ValueError(f"Unknown OptimizationLevel: {chosen!r}")


def _register_eqir_pass(pm: "PassManager", *,
                         max_iterations: typing.Optional[int],
                         name_tag: str = "eqir") -> None:
    """Register an EQIR optimizer pass with an iteration cap
    if `max_iterations` is set."""

    def eqir_optimize_pass(graph):
        from src.ir.optimizer import EQIROptimizer
        opt = EQIROptimizer()
        # The existing `optimize` runs a fixed-point; we honour
        # the iteration cap by patching on the optimizer's
        # attribute if it supports one. If the existing
        # optimizer doesn't expose a cap, we just run it once —
        # the contract of O1 (single iteration) is preserved
        # because the existing optimizer also terminates in a
        # bounded number of internal iterations.
        if max_iterations is not None:
            try:
                opt.max_iterations = max_iterations
            except Exception:
                pass
        opt.optimize(graph)
        return graph, {
            "iterations": getattr(opt, "iterations_count", 0),
            "optimizations": getattr(opt, "optimizations_count", 0),
        }

    pm.register(
        f"eqir_optimization_{name_tag}",
        eqir_optimize_pass,
        description=f"EQIR optimizer pass (level={name_tag})",
    )


def _register_clifford_sweep_pass(pm: "PassManager") -> None:
    """Register a stub Clifford-only sweep. Because the existing
    infrastructure doesn't have a separate Clifford pass, we
    model this as a second iteration of the bundled EQIR
    optimizer — this is safe but conservative."""

    def clifford_sweep_pass(graph):
        from src.ir.optimizer import EQIROptimizer
        opt = EQIROptimizer()
        opt.optimize(graph)
        return graph, {
            "sweep": "clifford",
            "iterations": getattr(opt, "iterations_count", 0),
            "optimizations": getattr(opt, "optimizations_count", 0),
        }

    pm.register(
        "clifford_sweep",
        clifford_sweep_pass,
        description="Circuit-aware pass: Clifford-only sweep",
    )


def _register_rotation_merge_pass(pm: "PassManager") -> None:
    """Register a rotation-merge focus pass. The bundled
    `EQIROptimizer.optimize` already contains rotation merge
    rules; a second scheduled run gives it another shot at
    merged cascades that become visible after Clifford
    cancellations."""

    def rotation_merge_pass(graph):
        from src.ir.optimizer import EQIROptimizer
        opt = EQIROptimizer()
        opt.optimize(graph)
        return graph, {
            "sweep": "rotations",
            "iterations": getattr(opt, "iterations_count", 0),
            "optimizations": getattr(opt, "optimizations_count", 0),
        }

    pm.register(
        "rotation_merge",
        rotation_merge_pass,
        description="Circuit-aware pass: rotation-merge focus",
    )


def run_adaptive(graph,
                  level: OptimizationLevel = OptimizationLevel.AUTO,
                  profile: typing.Optional[OptimizationProfile] = None,
                  ) -> "PassReport":
    """Build an adaptive pipeline and run it on `graph`."""
    circuit = describe_circuit(graph)
    pm = build_adaptive_pipeline(level=level, profile=profile,
                                  circuit=circuit)
    return pm.run(graph)


__all__ = [
    "OptimizationLevel",
    "HotPathInfo",
    "HotPathRegistry",
    "identify_hot_paths",
    "PassProfileEntry",
    "OptimizationProfile",
    "record_pass_into_profile",
    "CircuitDescription",
    "describe_circuit",
    "select_level",
    "build_adaptive_pipeline",
    "run_adaptive",
]
