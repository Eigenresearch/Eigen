"""
P2 §3.2 — Parameterized circuits (surface-level API).

The roadmap lists parameterized circuits ("параметризованные схемы с
биндингом") as a P2 feature for Quantum Flexibility. Real Qiskit-style
parameter binding involves typed `Parameter` objects that flow through
the IR and emit a single bound circuit per set of concrete values.
Today Eigen has no first-class `Parameter` IR node; gate arguments are
floats passed at the simulator's `.RX(q, theta)` API.

This module provides the surface-level envelope today:

  * `Parameter(name)` — a symbolic placeholder with a name and an
    optional dtype hint. `Parameter` is hashable so users can index
    maps on it.
  * `ParametrizedCircuit(instructions)` — a small list of
    `(gate_name, qubits, parameter_or_value)` tuples. ``bind(params)``
    resolves any `Parameter` references to concrete `float`s and
    returns a resolved list suitable for hand-feeding to a backend.

The envelope intentionally does NOT depend on the IR/VM: down-stream
tooling today uses `ParametrizedCircuit.bind(...)` to produce the
concrete `(gate, qubits, angle)` triples which can be replayed through
the existing `QuantumSimulator` API. Real first-class IR integration
(`ParametricRX` opcode, type-checker support, optimiser awareness)
is future work.

Tests cover the bind / partial-bind / freeze / clone / serialise
behaviour of the envelope.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Iterable, Optional


class UnresolvedParameterError(ValueError):
    """Raised when `bind` is called without supplying every parameter
    that the circuit references."""


@dataclass(frozen=True)
class Parameter:
    """A symbolic placeholder for a numerical circuit parameter.

    Hashable + comparable by name. The optional ``dtype`` field is
    advisory — we don't enforce it at bind time today.
    """
    name: str
    dtype: str = "float"

    def __repr__(self) -> str:
        return f"Parameter({self.name!r})"

    def __str__(self) -> str:
        return f"${self.name}"


# A circuit instruction is a tuple of (gate_name, qubit_targets,
# parameter_or_value). The third element is either a `Parameter`
# (to be resolved by `bind`) or a concrete `float` / `int`.
Instruction = tuple


@dataclass
class ParametrizedCircuit:
    """A small list of gate instructions with optional `Parameter`
    placeholders. `bind(params)` resolves every placeholder to the
    value the caller supplies; an `UnresolvedParameterError` is
    raised when the caller omits a parameter the circuit references.
    """
    instructions: list[Instruction] = field(default_factory=list)
    name: str = ""

    def __init__(self, instructions: Optional[Iterable[Instruction]] = None,
                 name: str = ""):
        self.instructions = list(instructions) if instructions else []
        self.name = name

    def parameters(self) -> list[Parameter]:
        """Return the unique set of `Parameter` instances referenced
        by the circuit. The order is insertion order (first occurrence
        first) so callers can iterate predictably."""
        seen: list[Parameter] = []
        for instr in self.instructions:
            for arg in instr[2:]:
                if isinstance(arg, Parameter):
                    if arg not in seen:
                        seen.append(arg)
        return seen

    def bind(self, params: dict[Parameter | str, float]) -> "ResolvedCircuit":
        """Substitute every parameter in the circuit with a concrete
        value. Keys in ``params`` may be either ``Parameter`` instances
        or bare strings (matched against ``Parameter.name``). Reuses
        the existing Parameter objects where possible.

        Raises ``UnresolvedParameterError`` if any parameter has no
        supplied value."""
        # Normalise the params dict: every key becomes a `Parameter`
        # instance (or stays as-is if it already is one). When the user
        # supplied a string key, we look up the matching `Parameter`
        # in this circuit's parameter set.
        normalised: dict[Parameter, float] = {}
        str_keys = {p.name: p for p in self.parameters()}
        for k, v in params.items():
            if isinstance(k, Parameter):
                normalised[k] = v
            elif isinstance(k, str):
                if k not in str_keys:
                    raise UnresolvedParameterError(
                        f"Unknown parameter '{k}' (not in circuit)")
                normalised[str_keys[k]] = v
            else:
                raise TypeError(
                    f"bind() keys must be Parameter or str, got {type(k)}")

        # Verify every referenced parameter is in `normalised`.
        for p in self.parameters():
            if p not in normalised:
                raise UnresolvedParameterError(
                    f"Parameter '{p.name}' was not bound (supply"
                    f" bind({{{p.name}: <value>}}) or pass the Parameter"
                    f" object directly)")

        resolved: list[Instruction] = []
        for instr in self.instructions:
            new_args = []
            for arg in instr:
                if isinstance(arg, Parameter):
                    new_args.append(normalised[arg])
                else:
                    new_args.append(arg)
            # First two slots are gate name + targets list — they
            # never contain Parameter instances, but we still copy
            # them through unchanged.
            resolved.append(tuple(new_args))
        return ResolvedCircuit(instructions=resolved,
                               name=self.name,
                               binding=dict(normalised))

    def freeze(self, **kw) -> "ResolvedCircuit":
        """Convenience: bind by keyword using parameter names.
        Equivalent to `bind({Parameter(name): value})` for each kwarg.
        Useful for ergonomics in scripts."""
        if not kw:
            return self.bind({})
        return self.bind(kw)

    def clone(self) -> "ParametrizedCircuit":
        """Deep copy. We don't copy `Parameter` instances (they're
        frozen and intended to be shared by identity across circuits)."""
        return ParametrizedCircuit(
            instructions=[copy.copy(i) for i in self.instructions],
            name=self.name,
        )

    def __len__(self) -> int:
        return len(self.instructions)

    def __iter__(self):
        return iter(self.instructions)


@dataclass
class ResolvedCircuit:
    """The output of ``ParametrizedCircuit.bind`` — every parameter
    has been substituted with a concrete float/int, so instructions
    are immediately executable by the simulator.

    The `binding` field preserves the (Parameter, value) map used so
    downstream tooling (audit log, experiment tracker) can correlate
    resolved runs back to the parametrised circuit.
    """
    instructions: list[Instruction] = field(default_factory=list)
    name: str = ""
    binding: dict[Parameter, float] = field(default_factory=dict)

    @classmethod
    def of(cls, instructions, name: str = "") -> "ResolvedCircuit":
        return cls(instructions=instructions, name=name)

    def __len__(self) -> int:
        return len(self.instructions)

    def __iter__(self):
        return iter(self.instructions)


# ----- small helper: replay a resolved circuit via the simulator -----


def run_resolved_circuit(simulator, circuit: "ResolvedCircuit") -> None:
    """Apply every gate in `circuit` to `simulator` in order.

    Each instruction is `(gate_name, qubits, *[angles])`. We dispatch
    by gate name to the corresponding `QuantumSimulator` method. We
    intentionally keep the dispatch table closed — adding a new gate
    requires an explicit entry here so unknown gates raise rather than
    silently no-op. This is the P2 surface binding; the IR will later
    get a single `Q_GATE_NAME` opcode that does this lookup natively.
    """
    for instr in circuit.instructions:
        gate = instr[0]
        qubits = list(instr[1])
        rest = list(instr[2:])
        # Single-qubit gates
        if gate == "H": simulator.H(qubits[0])
        elif gate == "X": simulator.X(qubits[0])
        elif gate == "Y": simulator.Y(qubits[0])
        elif gate == "Z": simulator.Z(qubits[0])
        elif gate == "S": simulator.S(qubits[0])
        elif gate == "T": simulator.T(qubits[0])
        elif gate == "RX": simulator.RX(qubits[0], float(rest[0]))
        elif gate == "RY": simulator.RY(qubits[0], float(rest[0]))
        elif gate == "RZ": simulator.RZ(qubits[0], float(rest[0]))
        elif gate == "CNOT": simulator.CNOT(qubits[0], qubits[1])
        elif gate == "CZ": simulator.CZ(qubits[0], qubits[1])
        elif gate == "SWAP": simulator.SWAP(qubits[0], qubits[1])
        elif gate == "CCX": simulator.CCX(qubits[0], qubits[1], qubits[2])
        elif gate == "CSWAP":
            simulator.CSWAP(qubits[0], qubits[1], qubits[2])
        elif gate == "CP":
            simulator.CP(qubits[0], qubits[1], float(rest[0]))
        elif gate == "CRX":
            simulator.CRX(qubits[0], qubits[1], float(rest[0]))
        elif gate == "CRY":
            simulator.CRY(qubits[0], qubits[1], float(rest[0]))
        elif gate == "CRZ":
            simulator.CRZ(qubits[0], qubits[1], float(rest[0]))
        else:
            raise ValueError(f"Unknown gate '{gate}' in resolved circuit")
