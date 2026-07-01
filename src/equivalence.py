"""Mathematical equivalence checker for Eigen quantum circuits.

This module provides the EquivalenceChecker class which verifies whether
two EQIR graphs produce the same quantum operation (up to global phase).

IMPORTANT: The canonical hash comparison (used in the IR graph layer) is
a necessary but NOT sufficient condition for circuit equivalence.
Canonical hash equality means the circuits have the same structural
representation, but does not prove mathematical equivalence. Two
circuits with different canonical hashes are definitely NOT equivalent
(fast reject), but two circuits with the same canonical hash may still
be non-equivalent due to hash collisions or incomplete canonicalization.

For a full proof of equivalence, use the `are_equivalent()` method which
performs unitary matrix comparison (for small circuits) or ZX-calculus
rewriting (for larger circuits).
"""
from src.ir.ir_graph import EQIRGraph
from src.simulator import QuantumSimulator
from src.zx.exceptions import IndeterminateEquivalenceError

class EquivalenceChecker:
    def __init__(self):
        pass

    def get_all_qubits(self, graph: EQIRGraph) -> set[str]:
        qubits = set()
        for node in graph.nodes.values():
            if node.type == 'ALLOC':
                qubits.add(node.targets[0])
            elif node.type == 'GATE':
                for t in node.targets:
                    qubits.add(t)
        return qubits

    def generate_unitary(self, graph: EQIRGraph, qubit_order: list[str]) -> list[list[complex]]:
        n_qubits = len(qubit_order)
        dim = 2 ** n_qubits
        
        # Initialize U as dim x dim matrix of zeros
        U = [[0.0j for _ in range(dim)] for _ in range(dim)]
        
        # Sort graph nodes topologically
        nodes = graph.topological_sort()
        
        # Create one simulator and reuse it across columns
        sim = QuantumSimulator()
        for qubit in qubit_order:
            sim.allocate_qubit(qubit)
        
        for col in range(dim):
            # Reset simulator state vector to basis state |col>
            sim.state_vector = [0.0j] * dim
            sim.state_vector[col] = 1.0 + 0.0j
            
            # Apply only the gate operations from the graph
            for node in nodes:
                # We skip ALLOC (already pre-allocated), MEASURE, and classical nodes
                if node.type != 'GATE':
                    continue
                
                # Execute the gate on the simulator
                g_name = node.gate_name
                targets = node.targets
                args = node.args
                
                if g_name == 'H':
                    sim.H(targets[0])
                elif g_name == 'X':
                    sim.X(targets[0])
                elif g_name == 'Y':
                    sim.Y(targets[0])
                elif g_name == 'Z':
                    sim.Z(targets[0])
                elif g_name == 'S':
                    sim.S(targets[0])
                elif g_name == 'T':
                    sim.T(targets[0])
                elif g_name == 'RX':
                    sim.RX(targets[0], args[0])
                elif g_name == 'RY':
                    sim.RY(targets[0], args[0])
                elif g_name == 'RZ':
                    sim.RZ(targets[0], args[0])
                elif g_name == 'CNOT':
                    sim.CNOT(targets[0], targets[1])
                elif g_name == 'CZ':
                    sim.CZ(targets[0], targets[1])
                elif g_name == 'SWAP':
                    sim.SWAP(targets[0], targets[1])
                elif g_name == 'CCX':
                    sim.CCX(targets[0], targets[1], targets[2])
                elif g_name == 'CSWAP':
                    sim.CSWAP(targets[0], targets[1], targets[2])
                elif g_name == 'CP':
                    sim.CP(targets[0], targets[1], args[0])
                elif g_name == 'CRX':
                    sim.CRX(targets[0], targets[1], args[0])
                elif g_name == 'CRY':
                    sim.CRY(targets[0], targets[1], args[0])
                elif g_name == 'CRZ':
                    sim.CRZ(targets[0], targets[1], args[0])
                else:
                    raise ValueError(f"Unknown gate in equivalence check: {g_name}")
            
            # Write final state vector to the col-th column of U
            final_state = sim.get_state_vector()
            for row in range(dim):
                U[row][col] = final_state[row]
                
        return U

    def are_equivalent(self, graph1: EQIRGraph, graph2: EQIRGraph) -> bool:
        """Check whether two EQIR graphs are mathematically equivalent.

        This method performs a full unitary matrix comparison (for circuits
        with <= 8 qubits) or ZX-calculus rewriting (for larger circuits).

        Note: Canonical hash equality (available via graph.canonical_hash())
        is a fast-reject mechanism, NOT a proof of equivalence. If canonical
        hashes differ, the circuits are definitely not equivalent. If they
        match, this method should be called for a definitive answer.

        Args:
            graph1: First EQIR graph.
            graph2: Second EQIR graph.

        Returns:
            True if the circuits are equivalent up to global phase.
        """
        # Get set of all qubits across both graphs
        qubits1 = self.get_all_qubits(graph1)
        qubits2 = self.get_all_qubits(graph2)
        all_qubits = sorted(list(qubits1.union(qubits2)))
        
        n_qubits = len(all_qubits)
        if n_qubits > 8:
            from src.zx.zx_equivalence import ZXEquivalenceChecker
            zx_checker = ZXEquivalenceChecker()
            return zx_checker.are_equivalent(graph1, graph2)
            
        if n_qubits == 0:
            return True  # Trivially equivalent
            
        U1 = self.generate_unitary(graph1, all_qubits)
        U2 = self.generate_unitary(graph2, all_qubits)
        
        dim = len(U1)
        
        # Find the entry in U2 with maximum magnitude to compute global phase ratio
        max_val = 0.0
        max_idx = (0, 0)
        for r in range(dim):
            for c in range(dim):
                val = abs(U2[r][c])
                if val > max_val:
                    max_val = val
                    max_idx = (r, c)
                    
        r_max, c_max = max_idx
        v2 = U2[r_max][c_max]
        
        if abs(v2) < 1e-12:
            return False  # All zeros (should not happen for a valid unitary)
            
        v1 = U1[r_max][c_max]
        
        # Calculate phase ratio: v1 / v2
        global_phase = v1 / v2
        
        # The phase ratio must have absolute value close to 1.0 (since U1, U2 are unitary, scaling is 1)
        if abs(abs(global_phase) - 1.0) > 1e-9:
            return False
            
        # Check that U1_ij = global_phase * U2_ij for all i, j
        for r in range(dim):
            for c in range(dim):
                diff = abs(U1[r][c] - global_phase * U2[r][c])
                if diff > 1e-9:
                    return False
                    
        return True
