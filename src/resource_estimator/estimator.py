# Resource Estimator for Eigen programs
from src.ir.ir_graph import EQIRGraph, EQIRNode

class ResourceEstimator:
    def estimate(self, graph: EQIRGraph) -> dict:
        nodes = graph.topological_sort()
        
        qubits = set()
        cnot_count = 0
        clifford_count = 0
        t_count = 0
        measurement_count = 0
        gate_count = 0
        
        # Clifford gates set
        clifford_gates = {'H', 'X', 'Y', 'Z', 'S', 'CNOT', 'CZ', 'SWAP'}
        
        # Initialize depths tracking
        depths = {node.id: 0 for node in nodes}
        t_depths = {node.id: 0 for node in nodes}
        
        for node in nodes:
            # Track logical qubits
            if node.type == 'ALLOC':
                qubits.add(node.targets[0])
                continue
                
            # Track quantum gates/measures
            is_quantum = node.type in ('GATE', 'MEASURE')
            is_t = (node.type == 'GATE' and node.gate_name == 'T')
            
            # Compute depth and T-depth in DAG
            parent_ids = [p.id for p in node.parents if p.id in depths]
            
            if parent_ids:
                max_depth = max(depths[pid] for pid in parent_ids)
                max_t_depth = max(t_depths[pid] for pid in parent_ids)
            else:
                max_depth = 0
                max_t_depth = 0
                
            depths[node.id] = max_depth + (1 if is_quantum else 0)
            t_depths[node.id] = max_t_depth + (1 if is_t else 0)
            
            if node.type == 'GATE':
                gate_count += 1
                g_name = node.gate_name
                if g_name == 'CNOT':
                    cnot_count += 1
                    clifford_count += 1
                elif g_name in clifford_gates:
                    clifford_count += 1
                elif g_name == 'T':
                    t_count += 1
            elif node.type == 'MEASURE':
                measurement_count += 1

        total_depth = max(depths.values()) if depths else 0
        total_t_depth = max(t_depths.values()) if t_depths else 0
        
        return {
            'logical_qubits': len(qubits),
            'circuit_depth': total_depth,
            'gate_count': gate_count,
            'cnot_count': cnot_count,
            't_count': t_count,
            't_depth': total_t_depth,
            'clifford_count': clifford_count,
            'measurements': measurement_count
        }
