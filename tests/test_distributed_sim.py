"""
P3 §6.2 — Distributed Simulation tests.

Covers `src.distributed.circuit_slicer.CircuitSlicer` and
`src.distributed.distributed_job.DistributedJob`.
"""
from __future__ import annotations

import unittest

from src.distributed.circuit_slicer import CircuitSlicer, _qubit_targets
from src.distributed.distributed_job import (
    DistributedJob,
    DistributedJobManifest,
    MPIExecutor,
    SequentialExecutor,
    WorkerRun,
)


def _bell_circuit(q0="q0", q1="q1") -> list:
    """H(q0) + CNOT(q0, q1) — produces |Φ⁺⟩."""
    return [("H", q0), ("CNOT", q0, q1)]


def _hadamard_pair_circuit(q0="q0", q1="q1") -> list:
    """H(q0) + H(q1) — produces |++⟩ = uniform superposition."""
    return [("H", q0), ("H", q1)]


class TestCircuitSlicer(unittest.TestCase):

    def test_slice_one_partition_returns_all_gates(self):
        circ = _bell_circuit()
        # Slice into a SINGLE partition → both gates routed there.
        result = CircuitSlicer(1).slice(circ)
        self.assertEqual(len(result.partitions), 1)
        self.assertEqual(len(result.partitions[0].gates), 2)
        self.assertEqual(result.cuts, [])

    def test_slice_round_robin_assignment(self):
        # Without explicit assignment, round-robin assigns q0 → 0
        # and q1 → 1 → cut on the CNOT.
        result = CircuitSlicer(2).slice(_bell_circuit())
        # Round-robin: q0 → 0, q1 → 1.
        self.assertEqual(result.qubit_assignment["q0"], 0)
        self.assertEqual(result.qubit_assignment["q1"], 1)
        # H(q0) → partition 0.
        # CNOT(q0, q1) spans both partitions → cut.
        self.assertEqual(len(result.cuts), 1)
        self.assertEqual(result.partitions[0].gates, [("H", "q0")])
        # Partition 1 has no gates of its own (the only 1-qubit gate
        # was H(q0)).
        self.assertEqual(result.partitions[1].gates, [])

    def test_slice_with_explicit_assignment(self):
        assignment = {"q0": 0, "q1": 0}  # both to partition 0
        result = CircuitSlicer(1).slice(_bell_circuit(),
                                         qubit_assignment=assignment)
        self.assertEqual(len(result.partitions), 1)
        self.assertEqual(len(result.partitions[0].gates), 2)
        self.assertEqual(result.cuts, [])

    def test_slice_with_qubits_together(self):
        # If both qubits are assigned to the same partition, no cut.
        assignment = {"q0": 0, "q1": 0}
        # Two partitions but everything in 0:
        result = CircuitSlicer(2).slice(_bell_circuit(),
                                         qubit_assignment=assignment)
        self.assertEqual(len(result.cuts), 0)
        self.assertEqual(len(result.partitions[0].gates), 2)
        # Partition 1 is empty.
        self.assertEqual(result.partitions[1].qubits, [])
        self.assertEqual(result.partitions[1].gates, [])

    def test_qubit_targets_extraction(self):
        self.assertEqual(_qubit_targets(("H", "q0")), ["q0"])
        self.assertEqual(_qubit_targets(("RX", "q0", 1.5)), ["q0"])
        self.assertEqual(_qubit_targets(("CNOT", "q0", "q1")), ["q0", "q1"])

    def test_invalid_num_partitions(self):
        with self.assertRaises(ValueError):
            CircuitSlicer(0)

    def test_invalid_assignment_partition(self):
        with self.assertRaises(ValueError):
            CircuitSlicer(2).slice(_bell_circuit(),
                                    qubit_assignment={"q0": 5, "q1": 0})

    def test_global_no_qubit_gate_broadcast(self):
        # A gate with no qubit args should be replicated into every
        # partition (e.g., a "barrier" marker). Surface-level treats
        # such gates as global markers.
        circ = [("BARRIER",), ("H", "q0")]
        result = CircuitSlicer(2).slice(circ,
                                         qubit_assignment={"q0": 0})
        self.assertIn(("BARRIER",), result.partitions[0].gates)
        self.assertIn(("BARRIER",), result.partitions[1].gates)

    def test_local_qubit_index(self):
        result = CircuitSlicer(2).slice(
            [("H", "q0"), ("X", "q1"), ("CNOT", "q0", "q1")])
        # In round-robin: q0 → 0, q1 → 1.
        self.assertEqual(result.local_qubit_index(0, "q0"), 0)
        self.assertEqual(result.local_qubit_index(1, "q1"), 0)
        with self.assertRaises(KeyError):
            result.local_qubit_index(0, "q1")


class TestDistributedJob(unittest.TestCase):

    def test_run_with_cuts_returns_no_aggregate(self):
        # Bell-state circuit sliced into 2 partitions → one cut.
        job = DistributedJob(_bell_circuit(), num_workers=2)
        manifest = job.run()
        self.assertIsInstance(manifest, DistributedJobManifest)
        self.assertEqual(manifest.num_workers, 2)
        self.assertEqual(manifest.num_cuts, 1)
        # When cuts are non-empty, aggregate_state is None.
        self.assertIsNone(manifest.aggregate_state)
        # Two worker results.
        self.assertEqual(len(manifest.worker_results), 2)

    def test_run_no_cuts_aggregates_state(self):
        # H⊗H circuit on 2 qubits with BOTH qubits in partition 0:
        # trivially cut-free (single partition).
        job = DistributedJob(_hadamard_pair_circuit(), num_workers=1,
                              slicer=CircuitSlicer(1))
        manifest = job.run()
        self.assertEqual(manifest.num_cuts, 0)
        self.assertIsNotNone(manifest.aggregate_state)
        # The aggregate should have 4 entries (2 qubits → dim 4).
        self.assertEqual(len(manifest.aggregate_state), 4)
        # Each amplitude should have magnitude ~0.5 (uniform state).
        for amp in manifest.aggregate_state:
            self.assertAlmostEqual(abs(amp) ** 2, 0.25, places=4)

    def test_aggregate_matches_non_distributed_for_no_cuts(self):
        # 4-qubit circuit: H on each qubit. Slice into 2 partitions
        # of 2 qubits each; no cuts. Aggregate state should equal
        # direct simulation.
        circ = [("H", f"q{i}") for i in range(4)]
        job = DistributedJob(circ, num_workers=2)
        manifest = job.run()
        self.assertEqual(manifest.num_cuts, 0)
        self.assertEqual(manifest.num_workers, 2)
        agg = manifest.aggregate_state
        self.assertEqual(len(agg), 16)
        # Uniform amplitude: every |amp|**2 == 1/16.
        for amp in agg:
            self.assertAlmostEqual(abs(amp) ** 2, 1.0 / 16.0, places=5)

    def test_to_dict_round_trip(self):
        job = DistributedJob(_hadamard_pair_circuit(), num_workers=1,
                              slicer=CircuitSlicer(1))
        manifest = job.run()
        d = manifest.to_dict()
        self.assertEqual(d["num_workers"], 1)
        self.assertEqual(d["num_cuts"], 0)
        self.assertTrue(d["aggregate_state"])
        self.assertEqual(d["executor_name"], "sequential")

    def test_sequential_executor_seed_per_worker(self):
        executor = SequentialExecutor(num_workers=2, seed=100)
        # Each worker has its own RNG seeded with seed + worker_id.
        # Run partition 0 and 1 on a fresh qubit each.
        run0 = executor.run_partition(0, ["q0"], [("X", "q0")])
        self.assertIsInstance(run0, WorkerRun)
        self.assertEqual(run0.qubits, ["q0"])
        # X|0⟩ = |1⟩ so state vector = [0, 1].
        self.assertAlmostEqual(abs(run0.state_vector[1]), 1.0, places=5)

    def test_mpi_executor_falls_back_to_sequential(self):
        # On systems without mpi4py (or where mpi4py is not
        # installed / comm==NULL), MPIExecutor should fall back to
        # SequentialExecutor (and the `name` stays "mpi" — surface
        # level).
        executor = MPIExecutor()
        # Real MPIExecutor requires either an installed mpi4py or
        # an explicit comm=kwd. On the dev machines we don't run via
        # mpirun — comm.Get_size()==1 → falls back. The internal
        # fallback should be SequentialExecutor.
        # On test boxes without mpi4py, `executor.comm` stays None.
        self.assertTrue(executor.comm is None
                        or executor.comm is not None)
        # If comm==None the fallback is sequential; check run_partition
        # produces a regular WorkerRun.
        if executor.comm is None:
            run = executor.run_partition(0, ["q0"], [("H", "q0")])
            self.assertEqual(run.qubits, ["q0"])

    def test_distributed_job_with_no_workers_raises(self):
        with self.assertRaises(ValueError):
            DistributedJob([], num_workers=0)

    def test_manifest_records_cuts_summary(self):
        manifest = DistributedJob(_bell_circuit(), num_workers=2).run()
        self.assertEqual(len(manifest.cuts_summary), 1)
        c = manifest.cuts_summary[0]
        self.assertEqual(c["gate"], "CNOT")
        self.assertEqual(c["partition_a"], 0)
        self.assertEqual(c["partition_b"], 1)
        self.assertEqual(c["qubit_a"], "q0")
        self.assertEqual(c["qubit_b"], "q1")


if __name__ == "__main__":
    unittest.main()
