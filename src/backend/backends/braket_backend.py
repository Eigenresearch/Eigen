# AWS Braket backend exporter for Eigen EQIR
from src.ir.ir_graph import EQIRGraph

class BraketBackend:
    def export(self, graph: EQIRGraph) -> str:
        lines = [
            "from braket.circuits import Circuit",
            "import math",
            "",
            "device_circuit = Circuit()",
            ""
        ]
        
        qubits = set()
        for node in graph.nodes.values():
            if node.type == 'ALLOC':
                qubits.add(node.targets[0])
                
        sorted_qubits = sorted(list(qubits))
        q_map = {q: idx for idx, q in enumerate(sorted_qubits)}
        
        for node in graph.topological_sort():
            if node.type != 'GATE':
                continue
                
            g_name = node.gate_name.lower()
            targets = [q_map[t] for t in node.targets]
            args = node.args
            
            if g_name == 'h':
                lines.append(f"device_circuit.h({targets[0]})")
            elif g_name == 'x':
                lines.append(f"device_circuit.x({targets[0]})")
            elif g_name == 'y':
                lines.append(f"device_circuit.y({targets[0]})")
            elif g_name == 'z':
                lines.append(f"device_circuit.z({targets[0]})")
            elif g_name == 's':
                lines.append(f"device_circuit.s({targets[0]})")
            elif g_name == 't':
                lines.append(f"device_circuit.t({targets[0]})")
            elif g_name == 'rx':
                lines.append(f"device_circuit.rx({targets[0]}, {args[0]})")
            elif g_name == 'ry':
                lines.append(f"device_circuit.ry({targets[0]}, {args[0]})")
            elif g_name == 'rz':
                lines.append(f"device_circuit.rz({targets[0]}, {args[0]})")
            elif g_name == 'cnot':
                lines.append(f"device_circuit.cnot({targets[0]}, {targets[1]})")
            elif g_name == 'cz':
                lines.append(f"device_circuit.cz({targets[0]}, {targets[1]})")
            elif g_name == 'swap':
                lines.append(f"device_circuit.swap({targets[0]}, {targets[1]})")
                
        return "\n".join(lines)
