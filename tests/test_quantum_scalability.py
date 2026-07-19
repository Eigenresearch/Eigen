"""§8.1 — Quantum Scalability tests, organised by the rows in
the roadmap table."""
import math
import unittest

from src.quantum_scalability import (
    ScalabilityMode,
    ScalabilityStrategy,
    get_strategy,
    pick_scalability_mode,
    apply_gate_in_place,
    SparseGateMatrix,
    apply_sparse_gate_to_state,
    SparseDensityMatrix,
    MPSSwapRouter,
    SWAPMove,
    OptimisedTableau,
)


# ---------------------------------------------------------------------------
# Helper gate matrices
# ---------------------------------------------------------------------------

SQRT2 = 1.0 / math.sqrt(2)
H_MATRIX = [[SQRT2, SQRT2], [SQRT2, -SQRT2]]
X_MATRIX = [[0, 1], [1, 0]]
Z_MATRIX = [[1, 0], [0, -1]]
IDENTITY_2 = [[1, 0], [0, 1]]
CNOT_MATRIX = [[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]]

# Control-on-q0 (LSB), target-on-q1 (MSB) variant — flips q1
# when q0=1.
CNOT_Q0_CONTROL = [[1, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0],
                    [0, 1, 0, 0]]


# ---------------------------------------------------------------------------
# ScalabilityMode enum
# ---------------------------------------------------------------------------

class TestScalabilityMode(unittest.TestCase):
    def test_five_modes(self):
        self.assertEqual(len(list(ScalabilityMode)), 5)
        names = {m.value for m in ScalabilityMode}
        self.assertEqual(names, {"statevector", "sparse",
                                    "stabilizer", "mps",
                                    "density_matrix"})


class TestScalabilityStrategy(unittest.TestCase):
    def test_statevector_strategy(self):
        s = get_strategy(ScalabilityMode.STATEVECTOR)
        self.assertEqual(s.current_limit, 25)
        self.assertEqual(s.target_limit, 32)
        self.assertIn("in-place updates", s.approaches)

    def test_sparse_strategy(self):
        s = get_strategy(ScalabilityMode.SPARSE)
        self.assertIn("sparse gate matrices", s.approaches)

    def test_stabilizer_strategy(self):
        s = get_strategy(ScalabilityMode.STABILIZER)
        self.assertEqual(s.target_limit, 10000)
        self.assertIn("optimised tableau", s.approaches)

    def test_mps_strategy(self):
        s = get_strategy(ScalabilityMode.MPS)
        self.assertEqual(s.target_limit, 200)
        self.assertIn("SWAP routing", s.approaches)

    def test_density_matrix_strategy(self):
        s = get_strategy(ScalabilityMode.DENSITY_MATRIX)
        self.assertEqual(s.target_limit, 18)
        self.assertIn("sparse density", s.approaches)

    def test_within_target_limit(self):
        s = ScalabilityStrategy(mode=ScalabilityMode.MPS,
                                   current_limit=100, target_limit=200,
                                   approaches=[])
        self.assertTrue(s.within_target_limit(150))
        self.assertTrue(s.within_target_limit(50))
        self.assertFalse(s.within_target_limit(201))
        self.assertTrue(s.within_current_limit(50))
        self.assertFalse(s.within_current_limit(150))


# ---------------------------------------------------------------------------
# Mode picker
# ---------------------------------------------------------------------------

class TestPickScalabilityMode(unittest.TestCase):
    def test_small_circuit_returns_statevector(self):
        m = pick_scalability_mode(n_qubits=10,
                                   has_non_clifford=True,
                                   entanglement=0.5)
        self.assertEqual(m, ScalabilityMode.STATEVECTOR)

    def test_clifford_large_returns_stabilizer(self):
        m = pick_scalability_mode(n_qubits=200,
                                   has_non_clifford=False,
                                   entanglement=0.5)
        self.assertEqual(m, ScalabilityMode.STABILIZER)

    def test_clifford_small_returns_statevector(self):
        m = pick_scalability_mode(n_qubits=10,
                                   has_non_clifford=False)
        self.assertEqual(m, ScalabilityMode.STATEVECTOR)

    def test_high_sparsity_large_returns_sparse(self):
        m = pick_scalability_mode(n_qubits=28,
                                   has_non_clifford=True,
                                   sparsity=0.8)
        self.assertEqual(m, ScalabilityMode.SPARSE)

    def test_low_entanglement_large_returns_mps(self):
        m = pick_scalability_mode(n_qubits=50,
                                   has_non_clifford=True,
                                   entanglement=0.1)
        self.assertEqual(m, ScalabilityMode.MPS)

    def test_mixed_state_small_returns_density_matrix(self):
        m = pick_scalability_mode(n_qubits=12,
                                   is_mixed_state=True)
        self.assertEqual(m, ScalabilityMode.DENSITY_MATRIX)

    def test_mixed_state_large_returns_mps(self):
        m = pick_scalability_mode(n_qubits=20,
                                   is_mixed_state=True)
        self.assertEqual(m, ScalabilityMode.MPS)

    def test_explicit_density_matrix_request_returns_density(self):
        m = pick_scalability_mode(n_qubits=16,
                                   is_density_matrix=True)
        self.assertEqual(m, ScalabilityMode.DENSITY_MATRIX)


# ---------------------------------------------------------------------------
# In-place gate application
# ---------------------------------------------------------------------------

class TestApplyGateInPlace(unittest.TestCase):
    def test_identity_gate_preserves_state(self):
        sv = [1+0j, 0+0j, 0+0j, 0+0j]  # |00>
        apply_gate_in_place(sv, IDENTITY_2, [0], 2)
        # Single-qubit identity on qubit 0 of a 2-qubit space
        for a, b in zip(sv, [1+0j, 0+0j, 0+0j, 0+0j], strict=False):
            self.assertAlmostEqual(a, b)

    def test_h_gate_creates_superposition(self):
        sv = [1+0j, 0+0j]  # |0>
        apply_gate_in_place(sv, H_MATRIX, [0], 1)
        self.assertAlmostEqual(sv[0], SQRT2)
        self.assertAlmostEqual(sv[1], SQRT2)

    def test_x_gate_flips_qubit(self):
        sv = [1+0j, 0+0j]  # |0>
        apply_gate_in_place(sv, X_MATRIX, [0], 1)
        self.assertEqual(sv, [0+0j, 1+0j])  # |1>

    def test_x_gate_on_qubit_1_of_two(self):
        sv = [1+0j, 0+0j, 0+0j, 0+0j]  # |00>
        apply_gate_in_place(sv, X_MATRIX, [1], 2)
        # |10> → index with bit 1 set: index 2
        self.assertEqual(sv, [0+0j, 0+0j, 1+0j, 0+0j])

    def test_h_gate_on_qubit_1_of_two(self):
        sv = [1+0j, 0+0j, 0+0j, 0+0j]
        apply_gate_in_place(sv, H_MATRIX, [1], 2)
        # |0+> = (|00> + |10>) / sqrt(2): indices 0 and 2
        self.assertAlmostEqual(sv[0], SQRT2)
        self.assertAlmostEqual(sv[2], SQRT2)
        self.assertEqual(sv[1], 0)
        self.assertEqual(sv[3], 0)

    def test_cnot_two_qubits(self):
        # Bell state via H on qubit 0 then CNOT(q0=control, q1=target).
        # Using CNOT_Q0_CONTROL which flips q1 when q0=1.
        sv = [1+0j, 0+0j, 0+0j, 0+0j]  # |00>
        apply_gate_in_place(sv, H_MATRIX, [0], 2)
        apply_gate_in_place(sv, CNOT_Q0_CONTROL, [0, 1], 2)
        # Bell state: (|00>+|11>)/sqrt(2). |11> in LSB-convention
        # (q0 LSB, q1 MSB) is physical index 3.
        inv_sqrt2 = 1.0 / math.sqrt(2)
        self.assertAlmostEqual(sv[0], inv_sqrt2)
        self.assertAlmostEqual(sv[3], inv_sqrt2)
        self.assertAlmostEqual(sv[1], 0)
        self.assertAlmostEqual(sv[2], 0)

    def test_zero_qubits_returns_input(self):
        sv = [1+0j]
        result = apply_gate_in_place(sv, IDENTITY_2, [0], 0)
        self.assertEqual(result, [1+0j])

    def test_zero_targets_returns_input(self):
        sv = [1+0j, 2+0j]
        result = apply_gate_in_place(sv, IDENTITY_2, [], 2)
        self.assertEqual(result, [1+0j, 2+0j])

    def test_wrong_gate_dim_raises(self):
        sv = [1+0j, 0+0j]
        with self.assertRaises(ValueError):
            apply_gate_in_place(sv, CNOT_MATRIX, [0], 1)


# ---------------------------------------------------------------------------
# Sparse gate matrix
# ---------------------------------------------------------------------------

class TestSparseGateMatrix(unittest.TestCase):
    def test_dim_property(self):
        s = SparseGateMatrix(n_targets=1)
        self.assertEqual(s.dim, 2)
        s = SparseGateMatrix(n_targets=2)
        self.assertEqual(s.dim, 4)

    def test_add_and_nnz(self):
        s = SparseGateMatrix(n_targets=1)
        s.add(0, 0, 1)
        s.add(1, 1, 1)
        self.assertEqual(s.nnz(), 2)

    def test_add_out_of_bounds_raises(self):
        s = SparseGateMatrix(n_targets=1)
        with self.assertRaises(IndexError):
            s.add(0, 2, 1)  # col 2 out of bounds for dim 2

    def test_apply_identity_sparse(self):
        s = SparseGateMatrix(n_targets=1)
        s.add(0, 0, 1)
        s.add(1, 1, 1)
        sv = [3+0j, 4+1j]
        apply_sparse_gate_to_state(s, sv, [0], 1)
        # Should be unchanged
        self.assertEqual(sv, [3+0j, 4+1j])

    def test_apply_x_sparse(self):
        s = SparseGateMatrix(n_targets=1)
        s.add(0, 1, 1)
        s.add(1, 0, 1)
        sv = [1+0j, 0+0j]  # |0>
        apply_sparse_gate_to_state(s, sv, [0], 1)
        self.assertEqual(sv, [0+0j, 1+0j])  # |1>

    def test_apply_off_diag_sparse(self):
        s = SparseGateMatrix(n_targets=1)
        s.add(0, 1, 0.5)
        s.add(1, 0, 0.5)
        sv = [1+0j, 1+0j]  # superposition
        apply_sparse_gate_to_state(s, sv, [0], 1)
        self.assertEqual(sv, [0.5+0j, 0.5+0j])

    def test_apply_to_two_qubit_gate(self):
        # Sparse diagonal of a 2-qubit Z⊗I gate.
        s = SparseGateMatrix(n_targets=2)
        s.add(0, 0, 1)
        s.add(1, 1, 1)
        s.add(2, 2, -1)
        s.add(3, 3, -1)
        sv = [1+0j, 2+0j, 3+0j, 4+0j]
        apply_sparse_gate_to_state(s, sv, [0, 1], 2)
        # Z⊗I on |10> (idx 2) → -|10>; on |11> (idx 3) → -|11>
        self.assertEqual(sv, [1+0j, 2+0j, -3+0j, -4+0j])

    def test_target_count_mismatch_raises(self):
        s = SparseGateMatrix(n_targets=1)
        sv = [1+0j, 0+0j]
        with self.assertRaises(ValueError):
            apply_sparse_gate_to_state(s, sv, [0, 1], 2)


# ---------------------------------------------------------------------------
# Sparse density matrix
# ---------------------------------------------------------------------------

class TestSparseDensityMatrix(unittest.TestCase):
    def test_dim(self):
        s = SparseDensityMatrix(n_qubits=1)
        self.assertEqual(s.dim, 2)
        s = SparseDensityMatrix(n_qubits=2)
        self.assertEqual(s.dim, 4)

    def test_add_and_nnz(self):
        s = SparseDensityMatrix(n_qubits=1)
        s.add(0, 0, 1)
        self.assertEqual(s.nnz(), 1)

    def test_trace_of_pure_diagonal(self):
        s = SparseDensityMatrix(n_qubits=1)
        s.add(0, 0, 0.5)
        s.add(1, 1, 0.5)
        self.assertAlmostEqual(s.trace(), 1.0)

    def test_to_dense_round_trip(self):
        dense = [[1, 0], [0, 0]]  # |0><0|
        s = SparseDensityMatrix.from_dense(dense, n_qubits=1)
        self.assertEqual(s.to_dense(), dense)

    def test_apply_identity_unitary_unchanged(self):
        s = SparseDensityMatrix(n_qubits=1)
        s.add(0, 0, 1)
        s.apply_unitary(IDENTITY_2)
        self.assertEqual(s.entries, [(0, 0, 1)])

    def test_apply_x_unitary_to_pure_0(self):
        # ρ = |0><0| = [[1, 0], [0, 0]]
        # After X: ρ' = (X|0>)(<0|X†) = |1><1| = [[0, 0], [0, 1]]
        s = SparseDensityMatrix(n_qubits=1)
        s.add(0, 0, 1)
        s.apply_unitary(X_MATRIX)
        dense = s.to_dense()
        # Need to flatten comparison: result should be (1, 1) entry.
        self.assertEqual(dense[1][1], 1)
        self.assertEqual(dense[0][0], 0)

    def test_apply_unitary_dimension_mismatch_raises(self):
        s = SparseDensityMatrix(n_qubits=1)
        with self.assertRaises(ValueError):
            s.apply_unitary(CNOT_MATRIX)  # 4x4 matrix on 2x2 density

    def test_expectation_with_pauli_z(self):
        # ρ = |0><0| = [[1,0,0], [0,0,0]], O = Z = [[1, 0], [0, -1]]
        # <Z> = tr(ρ Z) = 1
        s = SparseDensityMatrix(n_qubits=1)
        s.add(0, 0, 1)
        z = s.expectation(Z_MATRIX)
        self.assertAlmostEqual(z, 1)

    def test_expectation_with_pauli_z_on_excited(self):
        s = SparseDensityMatrix(n_qubits=1)
        s.add(1, 1, 1)
        z = s.expectation(Z_MATRIX)
        self.assertAlmostEqual(z, -1)

    def test_expectation_superposition(self):
        # ρ = |+><+| = [[1/2, 1/2], [1/2, 1/2]]
        # <Z> in |+> is 0
        s = SparseDensityMatrix(n_qubits=1)
        s.add(0, 0, 0.5)
        s.add(0, 1, 0.5)
        s.add(1, 0, 0.5)
        s.add(1, 1, 0.5)
        z = s.expectation(Z_MATRIX)
        self.assertAlmostEqual(z, 0)


# ---------------------------------------------------------------------------
# SWAP router for MPS
# ---------------------------------------------------------------------------

class TestMPSSwapRouter(unittest.TestCase):
    def test_zero_distance_raises(self):
        r = MPSSwapRouter(4)
        with self.assertRaises(ValueError):
            r.route_two_qubit_gate(0, 0)

    def test_out_of_range_raises(self):
        r = MPSSwapRouter(4)
        with self.assertRaises(IndexError):
            r.route_two_qubit_gate(0, 4)
        with self.assertRaises(IndexError):
            r.route_two_qubit_gate(5, 1)

    def test_adjacent_qubits_need_no_swaps(self):
        r = MPSSwapRouter(4)
        result = r.route_two_qubit_gate(1, 2)
        self.assertEqual(result.forward_swaps, [])
        self.assertEqual(result.backward_swaps, [])
        self.assertEqual(result.effective_targets, (1, 2))

    def test_distance_two(self):
        r = MPSSwapRouter(4)
        result = r.route_two_qubit_gate(0, 2)
        self.assertEqual(len(result.forward_swaps), 1)
        self.assertEqual(result.forward_swaps[0], SWAPMove(0, 1))
        self.assertEqual(result.backward_swaps[0], SWAPMove(0, 1))
        self.assertEqual(result.effective_targets, (1, 2))
        self.assertEqual(result.total_swaps(), 2)

    def test_distance_three(self):
        r = MPSSwapRouter(5)
        result = r.route_two_qubit_gate(0, 3)
        # Forward: swap 0,1 then 1,2 → qubit 0 moves two positions
        self.assertEqual(result.forward_swaps,
                          [SWAPMove(0, 1), SWAPMove(1, 2)])
        # Backward: undo, in reverse order.
        self.assertEqual(result.backward_swaps,
                          [SWAPMove(1, 2), SWAPMove(0, 1)])
        self.assertEqual(result.effective_targets, (2, 3))


# ---------------------------------------------------------------------------
# Optimised tableau envelope (surface)
# ---------------------------------------------------------------------------

class TestOptimisedTableau(unittest.TestCase):
    def test_default_settings(self):
        t = OptimisedTableau()
        self.assertTrue(t.bit_pack)
        self.assertTrue(t.cache_pauli_frames)
        self.assertTrue(t.lazy_measurement)

    def test_estimate_memory_with_bit_packing(self):
        t = OptimisedTableau(bit_pack=True)
        m = t.estimate_memory_bytes(100)
        # ~ (2 * 100 * 100 / 8) + 100 = 2500 + 100 = 2600 bytes
        self.assertLess(m, 3000)

    def test_estimate_memory_without_bit_packing(self):
        t = OptimisedTableau(bit_pack=False)
        m = t.estimate_memory_bytes(100)
        # ~ 16 * 100 * 100 = 160000 bytes
        self.assertEqual(m, 16 * 100 * 100)

    def test_bit_packing_saves_memory(self):
        t_packed = OptimisedTableau(bit_pack=True)
        t_unpacked = OptimisedTableau(bit_pack=False)
        n = 50
        self.assertGreater(t_unpacked.estimate_memory_bytes(n),
                             t_packed.estimate_memory_bytes(n))


if __name__ == "__main__":
    unittest.main()
