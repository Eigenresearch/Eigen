import unittest

import numpy as np

from src.tensor_network.mps import MPSSimulator
from src.simulator import QuantumSimulator


def _assert_vectors_close(mps_vec, dense_vec, tol=1e-6):
    a = np.array(mps_vec, dtype=complex)
    b = np.array(dense_vec, dtype=complex)
    if a.shape != b.shape:
        return False
    diff = np.linalg.norm(a - b)
    return diff <= tol


class TestMPSAdjacentQubitOrdering(unittest.TestCase):
    def test_cnot_adjacent_q0_q1(self):
        for seed in range(5):
            mps = MPSSimulator(max_bond_dim=64, seed=seed)
            dense = QuantumSimulator(sim_type="dense", seed=seed)
            for q in ["q0", "q1"]:
                mps.allocate_qubit(q)
                dense.allocate_qubit(q)
            mps.H("q0"); dense.H("q0")
            mps.CNOT("q0", "q1"); dense.CNOT("q0", "q1")
            mv = mps.get_state_vector()
            dv = dense.get_state_vector()
            self.assertLessEqual(
                np.linalg.norm(np.array(mv) - np.array(dv)),
                1e-6,
                msg=f"MPS != dense for CNOT(q0,q1) seed={seed}",
            )

    def test_cnot_adjacent_q1_q0(self):
        for seed in range(5):
            mps = MPSSimulator(max_bond_dim=64, seed=seed)
            dense = QuantumSimulator(sim_type="dense", seed=seed)
            for q in ["q0", "q1"]:
                mps.allocate_qubit(q)
                dense.allocate_qubit(q)
            mps.H("q1"); dense.H("q1")
            mps.CNOT("q1", "q0"); dense.CNOT("q1", "q0")
            mv = mps.get_state_vector()
            dv = dense.get_state_vector()
            self.assertLessEqual(
                np.linalg.norm(np.array(mv) - np.array(dv)),
                1e-6,
                msg=f"MPS != dense for CNOT(q1,q0) seed={seed}",
            )

    def test_cz_adjacent_q0_q1(self):
        mps = MPSSimulator(max_bond_dim=64, seed=42)
        dense = QuantumSimulator(sim_type="dense", seed=42)
        for q in ["q0", "q1"]:
            mps.allocate_qubit(q)
            dense.allocate_qubit(q)
        mps.H("q0"); dense.H("q0")
        mps.CZ("q0", "q1"); dense.CZ("q0", "q1")
        mv = mps.get_state_vector()
        dv = dense.get_state_vector()
        self.assertLessEqual(np.linalg.norm(np.array(mv) - np.array(dv)), 1e-6)


class TestMPSNonAdjacentQubits(unittest.TestCase):
    def test_cnot_non_adjacent_q0_q2(self):
        for seed in range(5):
            mps = MPSSimulator(max_bond_dim=64, seed=seed)
            dense = QuantumSimulator(sim_type="dense", seed=seed)
            for q in ["q0", "q1", "q2"]:
                mps.allocate_qubit(q)
                dense.allocate_qubit(q)
            mps.H("q0"); dense.H("q0")
            mps.CNOT("q0", "q2"); dense.CNOT("q0", "q2")
            mv = mps.get_state_vector()
            dv = dense.get_state_vector()
            self.assertLessEqual(
                np.linalg.norm(np.array(mv) - np.array(dv)),
                1e-6,
                msg=f"MPS != dense for CNOT(q0,q2) seed={seed}",
            )

    def test_cnot_non_adjacent_q2_q0(self):
        for seed in range(3):
            mps = MPSSimulator(max_bond_dim=64, seed=seed)
            dense = QuantumSimulator(sim_type="dense", seed=seed)
            for q in ["q0", "q1", "q2"]:
                mps.allocate_qubit(q)
                dense.allocate_qubit(q)
            mps.H("q2"); dense.H("q2")
            mps.CNOT("q2", "q0"); dense.CNOT("q2", "q0")
            mv = mps.get_state_vector()
            dv = dense.get_state_vector()
            self.assertLessEqual(np.linalg.norm(np.array(mv) - np.array(dv)), 1e-6)

    def test_cnot_non_adjacent_q0_q3(self):
        mps = MPSSimulator(max_bond_dim=64, seed=42)
        dense = QuantumSimulator(sim_type="dense", seed=42)
        for q in ["q0", "q1", "q2", "q3"]:
            mps.allocate_qubit(q)
            dense.allocate_qubit(q)
        mps.H("q0"); dense.H("q0")
        mps.CNOT("q0", "q3"); dense.CNOT("q0", "q3")
        mv = mps.get_state_vector()
        dv = dense.get_state_vector()
        self.assertLessEqual(np.linalg.norm(np.array(mv) - np.array(dv)), 1e-6)

    def test_cnot_reverse_non_adjacent_q3_q0(self):
        mps = MPSSimulator(max_bond_dim=64, seed=42)
        dense = QuantumSimulator(sim_type="dense", seed=42)
        for q in ["q0", "q1", "q2", "q3"]:
            mps.allocate_qubit(q)
            dense.allocate_qubit(q)
        mps.H("q3"); dense.H("q3")
        mps.CNOT("q3", "q0"); dense.CNOT("q3", "q0")
        mv = mps.get_state_vector()
        dv = dense.get_state_vector()
        self.assertLessEqual(np.linalg.norm(np.array(mv) - np.array(dv)), 1e-6)

    def test_cz_non_adjacent_q0_q2(self):
        mps = MPSSimulator(max_bond_dim=64, seed=11)
        dense = QuantumSimulator(sim_type="dense", seed=11)
        for q in ["q0", "q1", "q2"]:
            mps.allocate_qubit(q)
            dense.allocate_qubit(q)
        mps.H("q0"); dense.H("q0")
        mps.CZ("q0", "q2"); dense.CZ("q0", "q2")
        mv = mps.get_state_vector()
        dv = dense.get_state_vector()
        self.assertLessEqual(np.linalg.norm(np.array(mv) - np.array(dv)), 1e-6)


class TestMPSSwapAndCNOT(unittest.TestCase):
    def test_swap_then_cnot_adjacent(self):
        mps = MPSSimulator(max_bond_dim=64, seed=42)
        dense = QuantumSimulator(sim_type="dense", seed=42)
        for q in ["q0", "q1", "q2"]:
            mps.allocate_qubit(q)
            dense.allocate_qubit(q)
        mps.SWAP("q0", "q1"); dense.SWAP("q0", "q1")
        mps.CNOT("q1", "q2"); dense.CNOT("q1", "q2")
        mv = mps.get_state_vector()
        dv = dense.get_state_vector()
        self.assertLessEqual(np.linalg.norm(np.array(mv) - np.array(dv)), 1e-6)

    def test_double_swap_preserves_state(self):
        mps = MPSSimulator(max_bond_dim=64, seed=42)
        dense = QuantumSimulator(sim_type="dense", seed=42)
        for q in ["q0", "q1", "q2"]:
            mps.allocate_qubit(q)
            dense.allocate_qubit(q)
        mps.SWAP("q0", "q2"); dense.SWAP("q0", "q2")
        mps.SWAP("q0", "q2"); dense.SWAP("q0", "q2")
        mv = mps.get_state_vector()
        dv = dense.get_state_vector()
        self.assertLessEqual(np.linalg.norm(np.array(mv) - np.array(dv)), 1e-6)

    def test_swap_correctness_basic(self):
        mps = MPSSimulator(max_bond_dim=64, seed=42)
        dense = QuantumSimulator(sim_type="dense", seed=42)
        for q in ["q0", "q1"]:
            mps.allocate_qubit(q)
            dense.allocate_qubit(q)
        mps.X("q0"); dense.X("q0")
        mps.SWAP("q0", "q1"); dense.SWAP("q0", "q1")
        mv = mps.get_state_vector()
        dv = dense.get_state_vector()
        self.assertLessEqual(np.linalg.norm(np.array(mv) - np.array(dv)), 1e-6)


class TestMPSReorderingAfterSwap(unittest.TestCase):
    def test_chain_of_cnots(self):
        mps = MPSSimulator(max_bond_dim=64, seed=42)
        dense = QuantumSimulator(sim_type="dense", seed=42)
        for q in ["q0", "q1", "q2", "q3"]:
            mps.allocate_qubit(q)
            dense.allocate_qubit(q)
        mps.H("q0"); dense.H("q0")
        for tgt in ["q1", "q2", "q3"]:
            mps.CNOT("q0", tgt)
            dense.CNOT("q0", tgt)
        mv = mps.get_state_vector()
        dv = dense.get_state_vector()
        self.assertLessEqual(np.linalg.norm(np.array(mv) - np.array(dv)), 1e-6)

    def test_cnot_in_both_directions(self):
        mps = MPSSimulator(max_bond_dim=64, seed=42)
        dense = QuantumSimulator(sim_type="dense", seed=42)
        for q in ["q0", "q1", "q2"]:
            mps.allocate_qubit(q)
            dense.allocate_qubit(q)
        mps.H("q1"); dense.H("q1")
        mps.CNOT("q0", "q2"); dense.CNOT("q0", "q2")
        mps.CNOT("q2", "q0"); dense.CNOT("q2", "q0")
        mv = mps.get_state_vector()
        dv = dense.get_state_vector()
        self.assertLessEqual(np.linalg.norm(np.array(mv) - np.array(dv)), 1e-6)

    def test_qubit_index_unchanged_after_swap_matrix(self):
        mps = MPSSimulator(max_bond_dim=64, seed=42)
        for q in ["q0", "q1"]:
            mps.allocate_qubit(q)
        self.assertEqual(mps.qubits, ["q0", "q1"])
        mps.SWAP("q0", "q1")
        self.assertEqual(mps.qubits, ["q0", "q1"])

    def test_get_qubit_index_after_swap_matrix(self):
        mps = MPSSimulator(max_bond_dim=64, seed=42)
        for q in ["q0", "q1"]:
            mps.allocate_qubit(q)
        mps.SWAP("q0", "q1")
        self.assertEqual(mps.get_qubit_index("q0"), 0)
        self.assertEqual(mps.get_qubit_index("q1"), 1)


class TestMPSStateVectorOrdering(unittest.TestCase):
    def test_state_vector_zero_qubit_initialized(self):
        mps = MPSSimulator(max_bond_dim=64)
        mps.allocate_qubit("q0")
        sv = mps.get_state_vector()
        self.assertEqual(len(sv), 2)
        self.assertAlmostEqual(sv[0].real, 1.0)
        self.assertAlmostEqual(sv[1].real, 0.0)

    def test_state_vector_x_on_q0(self):
        mps = MPSSimulator(max_bond_dim=64)
        dense = QuantumSimulator(sim_type="dense")
        for q in ["q0", "q1"]:
            mps.allocate_qubit(q)
            dense.allocate_qubit(q)
        mps.X("q0"); dense.X("q0")
        mv = mps.get_state_vector()
        dv = dense.get_state_vector()
        self.assertLessEqual(np.linalg.norm(np.array(mv) - np.array(dv)), 1e-6)

    def test_state_vector_with_swap(self):
        mps = MPSSimulator(max_bond_dim=64)
        dense = QuantumSimulator(sim_type="dense")
        for q in ["q0", "q1", "q2"]:
            mps.allocate_qubit(q)
            dense.allocate_qubit(q)
        mps.X("q0"); dense.X("q0")
        mps.SWAP("q0", "q2"); dense.SWAP("q0", "q2")
        mv = mps.get_state_vector()
        dv = dense.get_state_vector()
        self.assertLessEqual(np.linalg.norm(np.array(mv) - np.array(dv)), 1e-6)

    def test_norm_squared_is_one_after_swap(self):
        mps = MPSSimulator(max_bond_dim=64, seed=42)
        for q in ["q0", "q1", "q2"]:
            mps.allocate_qubit(q)
        mps.H("q0")
        mps.CNOT("q0", "q2")
        mps.SWAP("q0", "q2")
        self.assertAlmostEqual(mps.norm_squared(), 1.0, places=6)


class TestMPSMultipleQubitCircuits(unittest.TestCase):
    def test_ghz_state_4_qubits(self):
        mps = MPSSimulator(max_bond_dim=128, seed=42)
        dense = QuantumSimulator(sim_type="dense", seed=42)
        for q in ["q0", "q1", "q2", "q3"]:
            mps.allocate_qubit(q)
            dense.allocate_qubit(q)
        mps.H("q0"); dense.H("q0")
        for tgt in ["q1", "q2", "q3"]:
            mps.CNOT("q0", tgt)
            dense.CNOT("q0", tgt)
        mv = mps.get_state_vector()
        dv = dense.get_state_vector()
        self.assertLessEqual(np.linalg.norm(np.array(mv) - np.array(dv)), 1e-6)

    def test_qubit_order_independent_for_isolated_gates(self):
        for order in [["a", "b"], ["b", "a"]]:
            mps = MPSSimulator(max_bond_dim=64, seed=42)
            for q in order:
                mps.allocate_qubit(q)
            mps.X(order[0])
            sv = mps.get_state_vector()
            self.assertEqual(len(sv), 4)


if __name__ == "__main__":
    unittest.main()
