"""
P3 §6.2 — Distributed Simulation, part 1: circuit slicing.

Roadmap (`sol.md` "6.2 Distributed Simulation"):
    - [ ] MPI-based распределённая симуляция для больших схем
    - [ ] **Circuit slicing** — разрезание схемы на подсхемы
    - [ ] Distributed tensor network contraction

This module provides a self-contained, MPI-free envelope for **circuit
slicing**:

  * `CircuitSlicer(num_partitions)` — given a target partition count,
    slice any circuit into per-partition sub-circuits + a list of cuts.
  * `slice(circuit, qubit_partition_assignment)` returns a `PartitionResult`
    with one partition per worker, plus a list of `Cut` records
    identifying boundary gates.

A `GateStep` here is a `(gate_name, *target_args)` tuple. For 1-qubit
gates the first target is a qubit identifier (e.g. `"q0"` or `0`). For
2-qubit gates the first two targets are the qubits. For 0-target
gates the tuple has just `(gate_name,)` — these are routed to ALL
partitions (e.g. global reset / barrier markers in some circuit
representations) so they remain synchronized.

The default qubit-to-partition assignment is `qubit_index %
num_partitions` (round-robin). Callers can supply their own
`Dict[int, int]` for a graph-partition-aware assignment — left as
an extension hook for the production "MPI-based distributed
simulation for big circuits" future work.

Surface-level constraints:
  * Real circuit slicing for tensor-network contraction requires
    Schmidt decompositions at the boundary; we don't compute the
    Schmidt rank or the actual state-vector reconstruction here —
    the output is a manifest of cuts suitable for downstream tensor
    reconstructors (the "distributed tensor network contraction"
    checkbox item).
  * Cross-partition two-qubit gates are flagged as cuts; we don't
    simulate them. They are recorded in the `Cut` list for the
    executor to handle via tensor contraction or message passing.
"""
from __future__ import annotations

import dataclasses
import typing

GateStep = typing.Tuple[str, typing.Any]
QubitId = typing.Union[str, int]


@dataclasses.dataclass(frozen=True)
class Cut:
    """One boundary gate spanning two partitions.

    `gate` is the original `(gate_name, control, target)` step.
    `partition_a` and `partition_b` are the two partition indices
    participating. `qubit_a`, `qubit_b` identify the qubits on each
    side of the cut, in their respective partition's local index
    space (assigned in the order they appear in the partition's
    qubit map).
    """
    gate: GateStep
    partition_a: int
    partition_b: int
    qubit_a: QubitId
    qubit_b: QubitId


@dataclasses.dataclass
class Partition:
    index: int
    qubits: typing.List[QubitId]
    gates: typing.List[GateStep]


@dataclasses.dataclass
class PartitionResult:
    """Outcome of `CircuitSlicer.slice(...)`.

    `partitions[i]` is the i-th worker's sub-circuit (1-qubit gates on
    qubits assigned to partition i, plus the 2-qubit gates whose BOTH
    operands are in partition i). `cuts` lists the cross-partition
    gates that the executor must handle via tensor contraction /
    classical communication.
    """
    num_partitions: int
    partitions: typing.List[Partition]
    cuts: typing.List[Cut]
    qubit_assignment: typing.Dict[QubitId, int]

    def local_qubit_index(self, partition: int, qubit: QubitId) -> int:
        """Return the local index of `qubit` within `partition`'s
        sub-circuit. Raises KeyError if `qubit` is not owned by
        `partition`.
        """
        if self.qubit_assignment.get(qubit) != partition:
            raise KeyError(
                f"qubit {qubit} is not assigned to partition {partition}; "
                f"actual partition={self.qubit_assignment.get(qubit)}")
        return self.partitions[partition].qubits.index(qubit)


def _qubit_targets(step: GateStep) -> typing.List[QubitId]:
    """Return just the qubit-targets of `step` (i.e., the
    non-numeric/non-float args). Identifying which positional args
    are qubits vs gate parameters (rotation angles etc.) is done by
    type check: qubits are `str` or `int`; parameters are `float`
    or non-qubit `int` we treat as numerics.

    Surface-level heuristic: any arg that's NOT a float/complex/bool
    is a qubit. This works for our gate step convention:
        ("H", "q0")
        ("RX", "q0", 1.57)
        ("CNOT", "q0", "q1")
    It fails for gates that take qubit-INDEX integers (e.g.
    `("RX", 0, 1.57)` with integer index) — but downstream users are
    expected to use string-qubit-ids, consistent with the rest of
    this codebase.
    """
    args = list(step[1:])
    qubits = []
    for a in args:
        if isinstance(a, str):
            qubits.append(a)
        elif isinstance(a, int):
            # If the gate name is a 2-qubit gate (CNOT/CZ/SWAP/etc.)
            # then int args are qubit indices. Otherwise (single-qubit
            # gate) int args are numeric parameters (e.g. shots).
            # Heuristic: any int args BEFORE the first float are
            # qubits. We assume single-qubit-and-2-qubit naming
            # convention.
            qubits.append(a)
        elif isinstance(a, (float, complex, bool)):
            # Parameter — break.
            break
    return qubits


class CircuitSlicer:
    """Slice a single quantum circuit into `num_partitions` sub-
    circuits. Round-robin qubit assignment by default.

    The class is stateless other than the configured partition count.
    """

    KNOWN_TWO_QUBIT_GATES = {"CNOT", "CZ", "SWAP", "CRZ", "CP"}

    def __init__(self, num_partitions: int):
        if num_partitions < 1:
            raise ValueError("num_partitions must be >= 1")
        self.num_partitions = num_partitions

    def slice(self, circuit: typing.Iterable[GateStep], *,
              qubit_assignment: typing.Optional[typing.Dict[QubitId, int]] = None,
              ) -> PartitionResult:
        """Slice `circuit` into `self.num_partitions` sub-circuits.

        `qubit_assignment` maps each qubit to a partition index in
        `range(num_partitions)`; if None, the round-robin
        `qubit_index % num_partitions` assignment is used (where
        `qubit_index` is the order in which qubits are first
        encountered in the circuit). For a 1-partition slice the
        result trivially contains all gates.
        """
        steps = list(circuit)
        # Build the qubit→partition map.
        if qubit_assignment is None:
            qubit_assignment = {}
            order = []
            for step in steps:
                targets = _qubit_targets(step)
                for q in targets:
                    if q not in qubit_assignment:
                        qubit_assignment[q] = (len(order)
                                               % self.num_partitions)
                        order.append(q)
        # Validate assignment.
        for q, p in qubit_assignment.items():
            if p < 0 or p >= self.num_partitions:
                raise ValueError(
                    f"qubit {q} assigned to invalid partition {p}")
        partitions = [Partition(index=i, qubits=[], gates=[])
                     for i in range(self.num_partitions)]
        # Track qubits per partition in first-seen order, so
        # `local_qubit_index` returns a stable index.
        for q, p in qubit_assignment.items():
            if q not in partitions[p].qubits:
                partitions[p].qubits.append(q)
        cuts: typing.List[Cut] = []
        for step in steps:
            name = step[0]
            args = list(step[1:])
            qubits = _qubit_targets(step)
            if not qubits:
                # No qubit operand (e.g. a barrier / reset marker):
                # replicate to all partitions so downstream executors
                # stay synchronized.
                for p in partitions:
                    p.gates.append(step)
                continue
            owner = qubit_assignment[qubits[0]]
            if len(qubits) == 1:
                # 1-qubit gate → routed to its owner partition.
                partitions[owner].gates.append(step)
                continue
            # 2+ qubit gate. Check whether all involved qubits share
            # the same partition.
            owners = {qubit_assignment[q] for q in qubits
                       if q in qubit_assignment}
            if len(owners) == 1:
                # Internal gate: routed to its single owner partition.
                partitions[owner].gates.append(step)
                continue
            # Cross-partition gate: we only model the 2-qubit cut
            # case explicitly. For higher-arity gates surface-level
            # support would split into pairwise cuts — out of scope.
            if len(qubits) == 2:
                p_a = qubit_assignment[qubits[0]]
                p_b = qubit_assignment[qubits[1]]
                cuts.append(Cut(gate=step, partition_a=p_a,
                                partition_b=p_b,
                                qubit_a=qubits[0], qubit_b=qubits[1]))
            else:
                # For ≥3-qubit cross-partition gates (rare in the
                # research surface): raise — caller must
                # pre-decompose into 2-qubit gates.
                raise NotImplementedError(
                    f"Cross-partition {name} on {qubits} has "
                    "arity > 2 and no surface-level decomposition.")
        return PartitionResult(
            num_partitions=self.num_partitions,
            partitions=partitions,
            cuts=cuts,
            qubit_assignment=qubit_assignment,
        )

    def distribute_mpi(self, circuit: list, num_qubits: int
                        ) -> 'PartitionResult':
        """Distribute a circuit across MPI ranks.

        §6.2: "MPI-based распределённая симуляция для больших схем"

        When MPI is available (mpi4py), partitions are assigned
        one per rank. When MPI is not available, falls back to
        the standard slice() with num_partitions=1.
        """
        from src.distributed.mpi_simulator import (
            is_mpi_available, get_rank, get_world_size,
            distribute_state_vector,
        )
        if is_mpi_available() and get_world_size() > 1:
            # Assign qubits round-robin across MPI ranks
            ws = get_world_size()
            qubit_assignment = {
                i: i % ws for i in range(num_qubits)
            }
            return self.slice(circuit, qubit_assignment)
        else:
            # Single-node fallback
            qubit_assignment = {i: 0 for i in range(num_qubits)}
            return self.slice(circuit, qubit_assignment)
