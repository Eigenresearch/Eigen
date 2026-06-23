# IonQ backend exporter for Eigen EQIR
import json
from src.ir.ir_graph import EQIRGraph

class IonQBackend:
    def export(self, graph: EQIRGraph) -> str:
        qubits = set()
        for node in graph.nodes.values():
            if node.type == 'ALLOC':
                qubits.add(node.targets[0])
                
        sorted_qubits = sorted(list(qubits))
        q_map = {q: idx for idx, q in enumerate(sorted_qubits)}
        
        circuit = {
            "qubits": len(sorted_qubits),
            "circuit": []
        }
        
        for node in graph.topological_sort():
            if node.type != 'GATE':
                continue
                
            g_name = node.gate_name.lower()
            targets = [q_map[t] for t in node.targets]
            
            # Map common gates to IonQ native gate set
            # IonQ supports: gpi, gpi2, ms (Mølmer-Sørensen), etc., or standard gates
            gate_entry = {
                "gate": g_name,
                "targets": targets
            }
            if node.args:
                gate_entry["phase"] = node.args[0]
                
            circuit["circuit"].append(gate_entry)
            
        return json.dumps(circuit, indent=2)
