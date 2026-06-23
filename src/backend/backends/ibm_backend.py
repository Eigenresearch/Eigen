# IBM backend exporter for Eigen EQIR
import json
from src.ir.ir_graph import EQIRGraph

class IBMBackend:
    def export(self, graph: EQIRGraph) -> str:
        lines = [
            "OPENQASM 2.0;",
            'include "qelib1.inc";',
            ""
        ]
        
        # 1. Allocate registers
        qubits = set()
        measures = []
        
        for node in graph.nodes.values():
            if node.type == 'ALLOC':
                qubits.add(node.targets[0])
            elif node.type == 'MEASURE':
                measures.append((node.targets[0], node.cbit_name))
                
        sorted_qubits = sorted(list(qubits))
        q_map = {q: idx for idx, q in enumerate(sorted_qubits)}
        
        lines.append(f"qreg q[{len(sorted_qubits)}];")
        if measures:
            lines.append(f"creg c[{len(measures)}];")
        lines.append("")
        
        # 2. Translate gates
        for node in graph.topological_sort():
            if node.type != 'GATE':
                continue
            
            g_name = node.gate_name.lower()
            targets = [f"q[{q_map[t]}]" for t in node.targets]
            args = node.args
            
            if g_name == 'h':
                lines.append(f"h {targets[0]};")
            elif g_name == 'x':
                lines.append(f"x {targets[0]};")
            elif g_name == 'y':
                lines.append(f"y {targets[0]};")
            elif g_name == 'z':
                lines.append(f"z {targets[0]};")
            elif g_name == 's':
                lines.append(f"s {targets[0]};")
            elif g_name == 't':
                lines.append(f"t {targets[0]};")
            elif g_name == 'rx':
                lines.append(f"rx({args[0]}) {targets[0]};")
            elif g_name == 'ry':
                lines.append(f"ry({args[0]}) {targets[0]};")
            elif g_name == 'rz':
                lines.append(f"rz({args[0]}) {targets[0]};")
            elif g_name == 'cnot':
                lines.append(f"cx {targets[0]}, {targets[1]};")
            elif g_name == 'cz':
                lines.append(f"cz {targets[0]}, {targets[1]};")
            elif g_name == 'swap':
                lines.append(f"swap {targets[0]}, {targets[1]};")
                
        # 3. Translate measures
        for idx, (q, c) in enumerate(measures):
            lines.append(f"measure q[{q_map[q]}] -> c[{idx}];")
            
        return "\n".join(lines)
