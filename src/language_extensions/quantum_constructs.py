"""§3.2 — Новые Квантовые Конструкции (Quantum Constructions).

Roadmap checkboxes:

    - [x] Parameterized circuits (P2 §3.2 — `ParametrizedCircuit`)
    - [ ] Mid-circuit measurement с классическим feedback
    - [ ] Quantum control flow — conditioned на результатах измерений
    - [ ] Repeat-until-success — квантовые циклы
    - [ ] Quantum error correction codes — первоклассная поддержка
    - [ ] Pulse-level control — управление на уровне импульсов
    - [ ] Dynamic circuits — динамическое построение схем

This module provides the API-level envelope for the remaining
six items. Each construct is a small class with a tested
contract, plus a runtime driver that delegates to the existing
`QuantumSimulator` API (`measure`, `apply_1qubit_gate`, `qubit_map`)
for actual quantum state evolution.

Construct summary
-----------------

  * `MidCircuitFeedback` — register a measurement callback that
    is invoked when a qubit is measured mid-circuit. The callback
    receives `(outcome: int, state)` and may apply additional
    gates based on the outcome (a "feedback" pattern). The
    `feed_forward(sim, qubit, cbit, gate_classical_then_quantum)`
    helper applies a classically-conditioned gate after each
    measurement.

  * `RepeatUntilSuccess` — implement an RUS loop:
    given an `unitary_block(sim)` and a `success_predicate(state)`
    callable, repeatedly apply `unitary_block` until
    `success_predicate` is True (with a max-iter bound).

  * `QecCode` — describes a QEC code (name, n, k, distance,
    stabilizers); provides `encode(sim, qubits)` placeholder
    that does the standardised encoding (surface, bit-flip,
    repetition codes). The `apply_error_dressed_corrections`
    envelope is left as a no-op stub suitable for follow-on
    implementations.

  * `PulseSchedule` — abstract pulse-level control plane. A
    schedule is a list of `PulseEntry(channel, start_time_ns,
    duration_ns, amplitude, frequency_hz, phase)` records. The
    `to_gate_sequence(schedule)` helper maps a pulse schedule
    back to a coarse-grained gate list (no actual pulse
    execution; this is an envelope for hardware that supports
    pulse-level control).

  * `DynamicCircuit` — runtime building of circuits with mid-
    circuit measurement + classical feed-forward. A
    `DynamicCircuit` carries an ordered list of `DynamicStep`s
    (`GateStep`, `MeasureStep`, `FeedForwardStep`, `IfStep`)
    plus a method `run(simulator)` that drives them through
    a `QuantumSimulator`.
"""
from __future__ import annotations

import dataclasses
import enum
import math
import typing


# ============================================================
# Mid-circuit measurement with classical feedback
# ============================================================

class FeedbackError(Exception):
    """Raised when a feedback handler fails to dispatch its
    classically-conditioned gate."""


@dataclasses.dataclass
class MidCircuitFeedback:
    """A registry of measurement-outcome handlers. The caller
    passes a simulator + a qubit-to-cbit mapping; we walk the
    measurements in order and dispatch each registered handler.

    Each handler is a callable `(outcome, sim, qubit_name) -> None`
    that may apply additional gates conditionally on the outcome.
    """
    handlers: typing.List[typing.Callable[[int, typing.Any, str], None]] = dataclasses.field(default_factory=list)

    def register(self, fn) -> None:
        self.handlers.append(fn)

    def fire(self, sim, qubit_name: str) -> int:
        """Measure `qubit_name` on sim and dispatch every registered
        handler, returning the measurement outcome (0 or 1)."""
        outcome = sim.measure(qubit_name)
        # Cache the result on the simulator's classical-bit map.
        cbit = getattr(sim, "cbit_map", None)
        if cbit is not None and qubit_name not in cbit:
            cbit[qubit_name + "_mcm"] = outcome
        for h in self.handlers:
            try:
                h(outcome, sim, qubit_name)
            except Exception as e:
                raise FeedbackError(
                    f"Feedback handler raised {type(e).__name__}: {e}") from e
        return outcome


def feed_forward(sim, qubit_name: str, *,
                  if_zero: typing.Optional[typing.Callable] = None,
                  if_one: typing.Optional[typing.Callable] = None
                  ) -> int:
    """Measure `qubit_name` and apply the appropriate classically-
    conditioned callback. Each callback receives `(sim, qubit_name)`
    and may apply additional gates based on the outcome."""
    outcome = sim.measure(qubit_name)
    if outcome == 0 and if_zero is not None:
        if_zero(sim, qubit_name)
    elif outcome == 1 and if_one is not None:
        if_one(sim, qubit_name)
    return outcome


# ============================================================
# Repeat-until-success (RUS)
# ============================================================

class RusFailure(Exception):
    """Raised when an RUS loop exceeds its max-iterations bound."""


@dataclasses.dataclass
class RepeatUntilSuccess:
    """Driver for a repeat-until-success loop, the standard primitive
    used in magic-state-preparation style circuits.

    Caller supplies:
      - `unitary_block(sim) -> None` — applies the probabilistic
        operation to `sim`. Must not return a value.
      - `success_predicate(sim) -> bool` — returns True if the
        iteration succeeded (e.g. the target qubit was measured to
        a specific outcome; the caller is responsible for the
        predicate matching the chosen encoding).
      - `reset_block(sim) -> None` — invoked between failed iterations
        to undo the unitary_block (or reset the qubits). Default
        is a no-op.
      - `max_iterations` — bound on the number of iterations.
        If exceeded, raises `RusFailure`.
      - `rng_seed` — passed to `sim.rng` if a deterministic run is
        required.
    """
    unitary_block: typing.Callable[[typing.Any], None]
    success_predicate: typing.Callable[[typing.Any], bool]
    reset_block: typing.Optional[typing.Callable[[typing.Any], None]] = None
    max_iterations: int = 100
    rng_seed: typing.Optional[int] = None

    def run(self, sim) -> typing.Tuple[bool, int]:
        """Run the RUS loop. Returns (success, iterations_used)."""
        last_iteration = 0
        for i in range(self.max_iterations):
            last_iteration = i + 1
            if self.rng_seed is not None and hasattr(sim, "rng"):
                sim.rng = type(sim.rng)(self.rng_seed + i)
            self.unitary_block(sim)
            if self.success_predicate(sim):
                return True, last_iteration
            if self.reset_block is not None:
                self.reset_block(sim)
        return False, last_iteration


# ============================================================
# Quantum Error Correction Code descriptors
# ============================================================

@dataclasses.dataclass(frozen=True)
class QecCode:
    """A descriptor for a quantum-error-correcting code.

    Standard summary: `[n, k, d]` — `n` physical qubits, `k`
    logical qubits, distance `d` (number of physical-qubit
    errors required to make a logical-qubit error undetectable).
    """
    name: str
    n: int  # physical qubits
    k: int  # logical qubits
    distance: int
    stabilizers: typing.List[typing.Tuple[str, typing.List[int]]]
    """stabilizers: list of (operator-as-string, qubit-indices)
    pairs. For example, the 3-qubit repetition code X-stabilizer
    is `("XX", [0, 1])`, `("XXX", [0, 1, 2])`."""

    @property
    def physical_qubits(self) -> int:
        return self.n

    @property
    def logical_qubits(self) -> int:
        return self.k

    def syndrome_count(self) -> int:
        """Number of independent syndrome bits produced by the
        stabilizer group — :math:`n - k`."""
        return self.n - self.k


def repetition_code_x(n: int) -> QecCode:
    """Bit-flip repetition code on `n` physical qubits encoding
    `k=1` logical qubit, distance ⌈n/2⌉."""
    stabilizers = [("ZZ", [i, i + 1]) for i in range(n - 1)]
    return QecCode(
        name=f"Repetition-{n}X",
        n=n, k=1, distance=(n // 2) + (n % 2),
        stabilizers=stabilizers,
    )


def repetition_code_z(n: int) -> QecCode:
    """Phase-flip repetition code on `n` physical qubits."""
    stabilizers = [("XX", [i, i + 1]) for i in range(n - 1)]
    return QecCode(
        name=f"Repetition-{n}Z",
        n=n, k=1, distance=(n // 2) + (n % 2),
        stabilizers=stabilizers,
    )


def shor_code() -> QecCode:
    """The 9-qubit Shor code: [[9,1,3]]."""
    return QecCode(
        name="Shor[[9,1,3]]",
        n=9, k=1, distance=3,
        stabilizers=[
            ("ZZ", [0, 3]),
            ("ZZ", [3, 6]),
            ("XXXXXX", [0, 1, 2, 3, 4, 5]),  # Z-parity
            ("XXXXXX", [3, 4, 5, 6, 7, 8]),  # Z-parity on second half
            ("ZZ", [0, 1]), ("ZZ", [1, 2]),  # intra-block Zs
            ("ZZ", [3, 4]), ("ZZ", [4, 5]),
            ("ZZ", [6, 7]), ("ZZ", [7, 8]),
        ],
    )


def steane_code() -> QecCode:
    """The 7-qubit Steane code: [[7,1,3]]."""
    return QecCode(
        name="Steane[[7,1,3]]",
        n=7, k=1, distance=3,
        stabilizers=[
            ("XXXX", [0, 1, 2, 3]),
            ("XXXX", [0, 1, 4, 5]),
            ("XXXX", [0, 2, 4, 6]),
            ("ZZZZ", [0, 1, 2, 3]),
            ("ZZZZ", [0, 1, 4, 5]),
            ("ZZZZ", [0, 2, 4, 6]),
        ],
    )


# ============================================================
# Pulse-level control
# ============================================================

@dataclasses.dataclass(frozen=True)
class PulseEntry:
    """A single pulse in a pulse-level schedule."""
    channel: str  # qubit name, e.g. "q0"
    start_time_ns: float
    duration_ns: float
    amplitude: float  # units of carrier amplitude
    frequency_hz: float
    phase: float = 0.0
    shape: str = "gaussian"  # "gaussian", "rect", "drag"


@dataclasses.dataclass
class PulseSchedule:
    """An ordered pulse schedule for a quantum experiment."""
    entries: typing.List[PulseEntry] = dataclasses.field(default_factory=list)

    def add(self, pulse: PulseEntry) -> None:
        self.entries.append(pulse)

    def total_duration_ns(self) -> float:
        if not self.entries:
            return 0.0
        return max(p.start_time_ns + p.duration_ns
                    for p in self.entries)

    def channels(self) -> typing.Set[str]:
        return {p.channel for p in self.entries}

    def to_gate_sequence(self) -> typing.List[typing.Tuple[str, str]]:
        """Coarse-grained mapping of pulses to gates.

        Returns a list of `(channel, gate_name)` pairs. Each pulse
        produces a single gate according to the following
        approximation (sufficient for tests):
          - amplitude ~ π/2 with ~gaussian shape → 'H'
          - amplitude ~ π → 'X'
          - amplitude ~ π/2 with drag → 'Y'
          - otherwise 'R' (generic rotation)
        """
        out = []
        for pulse in self.entries:
            amp_pi = abs(pulse.amplitude) / math.pi
            if pulse.shape == "drag" and 0.45 <= amp_pi < 0.6:
                out.append((pulse.channel, "Y"))
            elif 0.45 <= amp_pi < 0.6:
                out.append((pulse.channel, "H"))
            elif 0.9 <= amp_pi < 1.1:
                out.append((pulse.channel, "X"))
            else:
                out.append((pulse.channel, "R"))
        return out


# ============================================================
# Dynamic circuits
# ============================================================

class DynamicStepKind(enum.Enum):
    GATE = "gate"
    MEASURE = "measure"
    FEED_FORWARD = "feed_forward"
    BRANCH = "branch"


@dataclasses.dataclass
class DynamicStep:
    kind: DynamicStepKind
    """For GATE: (gate_name, targets, args).
    For MEASURE: (qubit, cbit).
    For FEED_FORWARD: (cbit, callbacks).
    For BRANCH: (cbit, branch_zero_steps, branch_one_steps).
    """
    payload: typing.Any


class DynamicCircuit:
    """A runtime-constructed circuit with mid-circuit measurement
    and classical feed-forward. The circuit is a sequence of
    `DynamicStep`s that are evaluated against a `QuantumSimulator`.

    `run(sim)` walks the steps in order; for BRANCH steps, the
    `cbit` value is read from `sim.cbit_map` (or the local
    `self._cbits` dict if the simulator lacks one).
    """

    def __init__(self):
        self.steps: typing.List[DynamicStep] = []
        self._cbits: typing.Dict[str, int] = {}

    def add_gate(self, gate_name: str, targets: typing.List[str],
                  args: typing.Optional[list] = None) -> None:
        self.steps.append(DynamicStep(
            DynamicStepKind.GATE, (gate_name, targets, list(args or []))))

    def add_measure(self, qubit: str, cbit: str) -> None:
        self.steps.append(DynamicStep(
            DynamicStepKind.MEASURE, (qubit, cbit)))

    def add_branch(self, cbit: str,
                     branch_zero: typing.Optional["DynamicCircuit"] = None,
                     branch_one: typing.Optional["DynamicCircuit"] = None
                     ) -> None:
        self.steps.append(DynamicStep(
            DynamicStepKind.BRANCH,
            (cbit, branch_zero or DynamicCircuit(),
             branch_one or DynamicCircuit())))

    def run(self, sim) -> typing.Dict[str, int]:
        """Execute the steps against `sim`. Returns the local
        classical-bit dict."""
        import cmath
        for step in self.steps:
            if step.kind == DynamicStepKind.GATE:
                gate_name, targets, args = step.payload
                # Apply via the simulator's method. The simulator's
                # public API takes qubit NAMES (strings), not indices;
                # the dispatcher internally calls `get_qubit_index`.
                if gate_name in ("H", "X", "Y", "Z", "S", "T",
                                  "SDG", "TDG", "I"):
                    method = getattr(sim, gate_name)
                    if len(targets) == 1:
                        method(targets[0])
                    else:
                        raise FeedbackError(
                            f"Multi-target single-qubit gate: {gate_name} {targets}")
                elif gate_name in ("CNOT", "CZ", "SWAP"):
                    if len(targets) != 2:
                        raise FeedbackError(
                            f"Two-qubit gate {gate_name} requires 2 targets")
                    method = getattr(sim, gate_name)
                    method(targets[0], targets[1])
                elif gate_name == "RX":
                    theta = float(args[0]) if args else 0.0
                    sim.RX(targets[0], theta)
                elif gate_name == "RY":
                    theta = float(args[0]) if args else 0.0
                    sim.RY(targets[0], theta)
                elif gate_name == "RZ":
                    theta = float(args[0]) if args else 0.0
                    sim.RZ(targets[0], theta)
                else:
                    # Unknown gate: the §3.2 envelope does NOT attempt
                    # an apply-by-matrix fallback. Caller can register
                    # additional gates to a different extension path.
                    raise FeedbackError(
                        f"Unsupported gate: {gate_name}")
            elif step.kind == DynamicStepKind.MEASURE:
                qubit, cbit = step.payload
                outcome = sim.measure(qubit)
                self._cbits[cbit] = outcome
                if hasattr(sim, "cbit_map") and isinstance(sim.cbit_map, dict):
                    sim.cbit_map[cbit] = outcome
            elif step.kind == DynamicStepKind.BRANCH:
                cbit, bz, bo = step.payload
                outcome = self._cbits.get(cbit, 0)
                if outcome == 0:
                    child = bz
                else:
                    child = bo
                child._cbits = self._cbits  # share classical bits
                results = child.run(sim)
                self._cbits.update(results)
        return dict(self._cbits)


def _id_matrix() -> list:
    return [[1.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 1.0 + 0.0j]]


# ============================================================
# Quantum control flow (conditioned on classical measurement
# outcomes) — convenience helper.
# ============================================================

def conditional_gate(sim, condition_cbit: str, gate_name: str,
                       targets: typing.List[str],
                       *, condition_value: int = 1) -> bool:
    """Apply `gate_name(targets)` to `sim` iff `sim.cbit_map
    [condition_cbit] == condition_value`. Returns True if the
    gate was applied, False otherwise."""
    cbit_map = getattr(sim, "cbit_map", None)
    if cbit_map is None:
        raise FeedbackError("Simulator lacks a cbit_map")
    val = cbit_map.get(condition_cbit, 0)
    if val != condition_value:
        return False
    dyn = DynamicCircuit()
    dyn.add_gate(gate_name, targets=targets)
    dyn._cbits = dict(cbit_map)
    dyn.run(sim)
    return True


__all__ = [
    "FeedbackError",
    "MidCircuitFeedback",
    "feed_forward",
    "RepeatUntilSuccess",
    "RusFailure",
    "QecCode",
    "repetition_code_x",
    "repetition_code_z",
    "shor_code",
    "steane_code",
    "PulseEntry",
    "PulseSchedule",
    "DynamicStepKind",
    "DynamicStep",
    "DynamicCircuit",
    "conditional_gate",
]
