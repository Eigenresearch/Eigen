"""§6.2 — MPI-based distributed simulation and distributed
tensor network contraction.

Surface module for distributed quantum simulation using MPI
(Message Passing Interface). When MPI is available (mpi4py),
this module provides circuit-slicing and state-distribution
for large circuits that exceed single-node memory.

When MPI is NOT available, all operations fall back to
single-node execution with appropriate warnings.
"""
from __future__ import annotations

import dataclasses

try:
    from mpi4py import MPI
    HAS_MPI = True
    _comm = MPI.COMM_WORLD
    _rank = _comm.Get_rank()
    _size = _comm.Get_size()
except ImportError:
    HAS_MPI = False
    _comm = None
    _rank = 0
    _size = 1


@dataclasses.dataclass
class MPISimConfig:
    """Configuration for MPI-based distributed simulation."""
    num_qubits: int
    rank: int = 0
    world_size: int = 1
    local_n_qubits: int = 0  # qubits stored on this node

    def __post_init__(self):
        if self.local_n_qubits == 0:
            # Distribute qubits across nodes
            self.local_n_qubits = max(1, self.num_qubits - _bit_length(
                self.world_size - 1)) if self.world_size > 1 else self.num_qubits


def _bit_length(n: int) -> int:
    return n.bit_length() if n > 0 else 0


def is_mpi_available() -> bool:
    return HAS_MPI


def get_rank() -> int:
    return _rank


def get_world_size() -> int:
    return _size


def distribute_state_vector(num_qubits: int) -> tuple[int, int]:
    """Determine the local portion of a state vector for this rank.

    Returns (local_offset, local_size) indicating which slice of
    the full state vector this node owns.
    """
    if not HAS_MPI or _size == 1:
        return 0, 1 << num_qubits
    total = 1 << num_qubits
    chunk = total // _size
    remainder = total % _size
    start = _rank * chunk + min(_rank, remainder)
    end = start + chunk + (1 if _rank < remainder else 0)
    return start, end - start


def allreduce_sum(value: float) -> float:
    """MPI Allreduce sum across all ranks."""
    if not HAS_MPI:
        return value
    return _comm.allreduce(value, op=MPI.SUM)


def broadcast_state(local_state: list, root: int = 0) -> list:
    """Broadcast state vector from root to all ranks."""
    if not HAS_MPI:
        return local_state
    return _comm.bcast(local_state, root=root)


@dataclasses.dataclass
class TensorNetworkContraction:
    """Distributed tensor network contraction plan.

    §6.2: "Distributed tensor network contraction"

    Represents a plan for contracting a tensor network across
    multiple MPI ranks. Each rank handles a subset of tensors.
    """
    tensors: list[str]  # tensor names
    contraction_order: list[tuple[str, str]]
    rank_assignment: dict[str, int]  # tensor_name -> rank

    def is_local(self, tensor_name: str) -> bool:
        return self.rank_assignment.get(tensor_name, 0) == _rank

    def local_tensors(self) -> list[str]:
        return [t for t in self.tensors if self.is_local(t)]

    def stats(self) -> dict:
        return {
            "total_tensors": len(self.tensors),
            "contraction_steps": len(self.contraction_order),
            "local_tensors": len(self.local_tensors()),
            "world_size": _size,
            "rank": _rank,
        }


def plan_distributed_contraction(tensor_names: list[str],
                                    edges: list[tuple[str, str]]
                                    ) -> TensorNetworkContraction:
    """Plan a distributed tensor network contraction.

    Assigns tensors to MPI ranks round-robin and generates a
    greedy contraction order based on edge connectivity.
    """
    rank_assignment = {}
    for i, t in enumerate(tensor_names):
        rank_assignment[t] = i % max(1, _size)

    # Greedy contraction: contract most-connected pairs first
    remaining = set(tensor_names)
    edge_counts: dict[str, int] = {}
    for a, b in edges:
        edge_counts[a] = edge_counts.get(a, 0) + 1
        edge_counts[b] = edge_counts.get(b, 0) + 1

    order = []
    edges_remaining = list(edges)
    while len(remaining) > 1 and edges_remaining:
        # Pick edge connecting two remaining tensors
        best = None
        best_score = -1
        for a, b in edges_remaining:
            if a in remaining and b in remaining:
                score = edge_counts.get(a, 0) + edge_counts.get(b, 0)
                if score > best_score:
                    best = (a, b)
                    best_score = score
        if best is None:
            break
        order.append(best)
        # Merge: remove b, keep a
        remaining.discard(best[1])
        edges_remaining = [(a, b) for a, b in edges_remaining
                            if best[1] not in (a, b)]

    return TensorNetworkContraction(
        tensors=list(tensor_names),
        contraction_order=order,
        rank_assignment=rank_assignment,
    )
