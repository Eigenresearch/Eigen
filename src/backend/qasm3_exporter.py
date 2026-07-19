# OpenQASM 3.0 backend exporter for Eigen EQIR
from src.ir.ir_graph import EQIRGraph

class Qasm3Exporter:
    def export(self, graph: EQIRGraph) -> str:
        lines = [
            "OPENQASM 3.0;",
            'include "stdgates.inc";',
            ""
        ]
        
        qubits = set()
        measures = []
        
        for node in graph.nodes.values():
            if node.type == 'ALLOC':
                qubits.add(node.targets[0])
            elif node.type == 'MEASURE':
                measures.append((node.targets[0], node.cbit_name))
                
        sorted_qubits = sorted(list(qubits))
        q_map = {q: idx for idx, q in enumerate(sorted_qubits)}
        
        lines.append(f"qubit[{len(sorted_qubits)}] q;")
        if measures:
            lines.append(f"bit[{len(measures)}] c;")
        lines.append("")
        
        # Translate gates
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
            elif g_name in ('cnot', 'cx'):
                lines.append(f"cx {targets[0]}, {targets[1]};")
            elif g_name == 'cz':
                lines.append(f"cz {targets[0]}, {targets[1]};")
            elif g_name == 'swap':
                lines.append(f"swap {targets[0]}, {targets[1]};")
            elif g_name == 'ccx':
                lines.append(f"ccx {targets[0]}, {targets[1]}, {targets[2]};")
            elif g_name == 'cswap':
                lines.append(f"cswap {targets[0]}, {targets[1]}, {targets[2]};")
            elif g_name == 'cp':
                lines.append(f"ctrl @ phase({args[0]}) {targets[0]}, {targets[1]};")
            elif g_name == 'crx':
                lines.append(f"ctrl @ rx({args[0]}) {targets[0]}, {targets[1]};")
            elif g_name == 'cry':
                lines.append(f"ctrl @ ry({args[0]}) {targets[0]}, {targets[1]};")
            elif g_name == 'crz':
                lines.append(f"ctrl @ rz({args[0]}) {targets[0]}, {targets[1]};")
            else:
                lines.append(f"# Unsupported gate in OpenQASM 3: {node.gate_name} on {node.targets}")
        
        # Translate measurements
        if measures:
            lines.append("")
            for idx, (q, _c) in enumerate(measures):
                lines.append(f"c[{idx}] = measure q[{q_map[q]}];")
                
        return "\n".join(lines)
