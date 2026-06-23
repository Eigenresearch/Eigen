from src.ir.ir_graph import EQIRGraph, EQIRNode
from src.simulator import QuantumSimulator

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
        
        for col in range(dim):
            # Create a simulator for this column (input basis state |col>)
            sim = QuantumSimulator()
            
            # Allocate all qubits in the specified order to guarantee same internal mapping
            for qubit in qubit_order:
                sim.allocate_qubit(qubit)
                
            # Set simulator state vector to basis state |col>
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
                else:
                    raise ValueError(f"Unknown gate in equivalence check: {g_name}")
            
            # Write final state vector to the col-th column of U
            final_state = sim.get_state_vector()
            for row in range(dim):
                U[row][col] = final_state[row]
                
        return U

    def are_equivalent(self, graph1: EQIRGraph, graph2: EQIRGraph) -> bool:
        # Get set of all qubits across both graphs
        qubits1 = self.get_all_qubits(graph1)
        qubits2 = self.get_all_qubits(graph2)
        all_qubits = sorted(list(qubits1.union(qubits2)))
        
        n_qubits = len(all_qubits)
        if n_qubits > 8:
            # Fallback: Rewrite-Based / Canonicalization Equivalence Check
            from src.ir.optimizer import EQIROptimizer
            opt = EQIROptimizer()
            g1_opt = opt.optimize(graph1)
            g2_opt = opt.optimize(graph2)
            
            gates1 = [n for n in g1_opt.topological_sort() if n.type == 'GATE']
            gates2 = [n for n in g2_opt.topological_sort() if n.type == 'GATE']
            
            if len(gates1) != len(gates2):
                return False
                
            for gt1, gt2 in zip(gates1, gates2):
                if gt1.gate_name != gt2.gate_name or gt1.targets != gt2.targets or gt1.args != gt2.args or gt1.condition != gt2.condition:
                    return False
            return True
            
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
        if abs(abs(global_phase) - 1.0) > 1e-5:
            return False
            
        # Check that U1_ij = global_phase * U2_ij for all i, j
        for r in range(dim):
            for c in range(dim):
                diff = abs(U1[r][c] - global_phase * U2[r][c])
                if diff > 1e-5:
                    return False
                    
        return True
