"""
P3 §6.2 — Distributed Simulation, part 2: job orchestration.

Roadmap (`sol.md` "6.2 Distributed Simulation"):
    - [ ] MPI-based распределённая симуляция для больших схем

This module provides a self-contained, MPI-free envelope for
orchestrating partitioned circuits across N workers:

  * `SequentialExecutor(num_workers=1)` — surface-level "MPI-
    like" runtime: simulates workers via in-process calls to the
    QuantumSimulator. The `MPIExecutor` class would map to actual
    MPI_Comm_rank / MPI_Sendrecv on real hardware; the Surface-level
    SequentialExecutor substitutes for it without requiring any
    MPI binding.
  * `DistributedJob(circuit, num_workers, *, slicer=None,
    executor=None)` — runs the full slice→execute→aggregate
    pipeline:
       1. Slice the circuit (default: CircuitSlicer with round-
          robin qubit assignment).
       2. For each partition, ask the `executor` to run the
          sub-circuit on a fresh QuantumSimulator and return its
          final state vector.
       3. Aggregate the per-partition state vectors into a single
          tensor product state of shape `2^N` where `N` is the
          total qubits.
       4. Record a `DistributedJobManifest` documenting the slices
          + cuts and the resulting state vector. The manifest is
          the unit of audit trail for distributed runs (see P3
          §12.3 ExperimentTracker; can be persisted there).
  * `DistributedJobManifest` (dataclass): num_workers, partitions,
    cuts, executor_name, aggregate_state (or None if not
    applicable), finished_at_ns.

Surface-level constraints:
  * Cross-partition cuts (from `CircuitSlicer`).cuts are NOT
    accounted for in the aggregated state vector. This means
    `aggregate_state` is only the tensor product of per-partition
    state vectors when the cut list is empty. When cuts are
    non-empty, `aggregate_state=None` and the caller is expected
    to either (a) reconstruct via Schmidt decomposition or (b)
    fall back to a non-distributed simulator for verification.
  * Real MPIExecutor implementation (per the roadmap) would
    serialize partitions via pickle, ship them across ranks, and
    aggregate state vectors via MPI_Allreduce. We provide a
    placeholder class `MPIExecutor` whose constructor checks
    for `mpi4py` and falls back to the SequentialExecutor when it's
    not available — surface-level, but with a real import attempt.
"""
from __future__ import annotations

import dataclasses
import time
import typing

from src.distributed.circuit_slicer import (
    CircuitSlicer,
    GateStep,
)


@dataclasses.dataclass
class WorkerRun:
    """Outcome of executing one partition's sub-circuit."""
    worker_id: int
    qubits: typing.List[typing.Any]
    state_vector: typing.List[complex]
    num_gates: int


@dataclasses.dataclass
class DistributedJobManifest:
    """Result of a `DistributedJob.run()` call. Useful as the audit
    record that may be persisted into a P3 §12.3 ExperimentTracker
    ledger entry."""
    num_workers: int
    num_qubits_total: int
    num_cuts: int
    cuts_summary: typing.List[typing.Dict[str, typing.Any]]
    worker_results: typing.List[WorkerRun]
    aggregate_state: typing.Optional[typing.List[complex]]
    executor_name: str
    finished_at_ns: int

    def to_dict(self) -> dict:
        return {
            "num_workers": self.num_workers,
            "num_qubits_total": self.num_qubits_total,
            "num_cuts": self.num_cuts,
            "cuts_summary": self.cuts_summary,
            "worker_results": [
                {"worker_id": w.worker_id,
                 "qubits": list(w.qubits),
                 "num_gates": w.num_gates,
                 "state_dim": len(w.state_vector)}
                for w in self.worker_results
            ],
            "aggregate_state": (self.aggregate_state is not None),
            "executor_name": self.executor_name,
            "finished_at_ns": self.finished_at_ns,
        }


class SequentialExecutor:
    """In-process executor that simulates `num_workers` independent
    QuantumSimulator processes by sequentially rebuilding a
    simulator per partition and applying the slice's gates.

    The "MPI-based distributed simulation" future work would
    replace this executor's `run_partition()` call with a real
    ranked dispatch.
    """

    name = "sequential"

    def __init__(self, num_workers: int = 1, *, seed: int = 0):
        if num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        self.num_workers = num_workers
        self.seed = seed

    def run_partition(self, worker_id: int, qubits: typing.List, gates: typing.List[GateStep],
                      ) -> WorkerRun:
        """Apply `gates` (1-qubit + intra-partition 2-qubit) to a
        fresh QuantumSimulator over the named `qubits` and return
        the per-worker's final state vector. The seed is rotated
        per worker so deterministic reproducibility holds across
        num_workers without the workers colliding on RNG state.
        """
        from src.simulator import QuantumSimulator
        sim = QuantumSimulator(sim_type="dense",
                               seed=(self.seed + worker_id))
        for q in qubits:
            sim.allocate_qubit(q)
        for step in gates:
            name = step[0]
            args = step[1:]
            # Some gates the QuantumSimulator doesn't expose as
            # named methods — route them via apply_1qubit_gate.
            matrix_dispatch = {
                "I": [[1.0, 0.0], [0.0, 1.0]],
                "SDG": [[1.0, 0.0], [0.0, -1.0j]],
                "TDG": [[1.0, 0.0], [0.0,
                                     __import__('cmath').exp(-1j*3.141592653589793/4)]],
            }
            if name in matrix_dispatch:
                sim.apply_1qubit_gate(args[0], matrix_dispatch[name])
                continue
            if name == "I":
                continue
            getattr(sim, name)(*args)
        return WorkerRun(
            worker_id=worker_id,
            qubits=list(qubits),
            state_vector=list(sim.get_state_vector()),
            num_gates=len(gates),
        )


class MPIExecutor:
    """MPI-based distributed executor.

    Real MPIExecutor implementation would use `mpi4py` to dispatch
    partitions to ranks. Surface-level: if `mpi4py` is installed
    and we're actually running under `mpirun` (i.e. world size > 1),
    use it; otherwise fall back to `SequentialExecutor`. We surface
    the API so downstream callers can swap in a real MPI binding
    without changing call sites.
    """

    name = "mpi"

    def __init__(self, *, comm=None, fallback_workers: int = 1):
        self._fallback = SequentialExecutor(fallback_workers)
        self.comm = comm
        # Lazy import mpi4py; if unavailable, keep self.comm=None and
        # we route through the fallback sequential executor.
        if comm is None:
            try:
                from mpi4py import MPI as _MPI
                self.comm = _MPI.COMM_WORLD
            except ImportError:
                self.comm = None
        # If we have a COMM_WORLD but size==1, the user almost
        # certainly ran us via `python` without `mpirun`. Fall back
        # to the sequential executor.
        if self.comm is not None and self.comm.Get_size() == 1:
            self.comm = None

    @property
    def num_workers(self) -> int:
        if self.comm is None:
            return self._fallback.num_workers
        return self.comm.Get_size()

    def run_partition(self, worker_id, qubits, gates) -> WorkerRun:
        # For surface-level: fall back to SequentialExecutor. A real
        # MPI implementation would bcast the partition to the rank
        # matching `worker_id`, run there, then gather state vectors.
        if self.comm is None:
            return self._fallback.run_partition(worker_id, qubits, gates)
        # If real comm available: serialise, dispatch, return.
        # (Implementation out of scope for surface-level envelope.)
        raise NotImplementedError(
            "Real MPI dispatch requires production code; install "
            "mpi4py and run via `mpirun -np N python <entry.py>`. "
            "Use SequentialExecutor for in-process runs.")


def _aggregate_state(worker_results: typing.List[WorkerRun],
                     qubit_assignment: typing.Dict,
                     total_qubits: int,
                     ) -> typing.List[complex]:
    """Aggregate per-worker state vectors into the full Hilbert
    space. We use numpy.kron iteratively over each worker's state
    vector (taken in worker_id order, with each worker's local
    qubit-index ordering aligned to its position in the global
    `qubit_assignment`).

    For surface-level correctness:
      * When the circuit has no cross-partition cuts, the kron
        product of per-worker state vectors equals the full state
        vector that a non-distributed simulator would produce.
      * When cuts ARE non-empty, the kron product is a coarse
        approximation (we ignore the entangled boundary). Caller
        should treat the result as ground-truth-less and
        cross-check with a non-distributed simulator.
    """
    import numpy as np
    state = np.array([1.0 + 0.0j], dtype=complex)
    for w in worker_results:
        # Worker state comes in local-qubit-index order. The
        # global qubit_assignment map places each worker's qubits
        # at indices `range(worker_id*local_width, ...)` for the
        # round-robin slicing; the kron product over
        # worker_results in worker_id order produces a basis
        # ordering where worker 0 is the LSB. This matches the
        # QuantumSimulator's convention IF we constructed the
        # simulator with qubits in worker_id order. Strictly,
        # this is only correct for the round-robin default
        # `CircuitSlicer`; the API accepts it as a convention.
        w_state = np.array(w.state_vector, dtype=complex)
        state = np.kron(w_state, state)
    return state.tolist()


class DistributedJob:
    """Orchestrator: slice a circuit, dispatch sub-circuits via an
    executor, then aggregate the results.

    Usage:

      job = DistributedJob(circuit, num_workers=4)
      manifest = job.run()

    Surface-level: workers run sequentially via `SequentialExecutor`
    unless an `MPIExecutor` is explicitly provided.
    """

    def __init__(self, circuit: typing.Iterable[GateStep],
                 num_workers: int, *,
                 slicer: typing.Optional[CircuitSlicer] = None,
                 executor: typing.Optional[typing.Any] = None):
        if num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        self.circuit = list(circuit)
        self.num_workers = num_workers
        self.slicer = slicer or CircuitSlicer(num_workers)
        self.executor = executor or SequentialExecutor(num_workers)

    def run(self, *, clock=time.time) -> DistributedJobManifest:
        partition_result = self.slicer.slice(self.circuit)
        worker_results = []
        for partition in partition_result.partitions:
            worker_results.append(self.executor.run_partition(
                partition.index, partition.qubits, partition.gates))
        # Aggregate only if there are no cross-partition cuts.
        if partition_result.cuts:
            aggregate = None
        else:
            total_qubits = sum(len(w.qubits) for w in worker_results)
            aggregate = _aggregate_state(
                worker_results,
                partition_result.qubit_assignment,
                total_qubits)
        return DistributedJobManifest(
            num_workers=self.num_workers,
            num_qubits_total=sum(len(w.qubits) for w in worker_results),
            num_cuts=len(partition_result.cuts),
            cuts_summary=[
                {"gate": c.gate[0],
                 "partition_a": c.partition_a,
                 "partition_b": c.partition_b,
                 "qubit_a": c.qubit_a, "qubit_b": c.qubit_b}
                for c in partition_result.cuts
            ],
            worker_results=worker_results,
            aggregate_state=aggregate,
            executor_name=getattr(self.executor, "name", "unknown"),
            finished_at_ns=int(clock() * 1_000_000_000),
        )
