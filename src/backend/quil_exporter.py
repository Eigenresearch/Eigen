# Quil backend exporter for Eigen EQIR
from src.ir.ir_graph import EQIRGraph

class QuilExporter:
    def export(self, graph: EQIRGraph) -> str:
        lines = []
        
        qubits = set()
        measures = []
        
        for node in graph.nodes.values():
            if node.type == 'ALLOC':
                qubits.add(node.targets[0])
            elif node.type == 'MEASURE':
                measures.append((node.targets[0], node.cbit_name))
                
        sorted_qubits = sorted(list(qubits))
        q_map = {q: idx for idx, q in enumerate(sorted_qubits)}
        
        if measures:
            lines.append(f"DECLARE ro BIT[{len(measures)}]")
            lines.append("")
            
        # Translate gates
        for node in graph.topological_sort():
            if node.type != 'GATE':
                continue
            
            g_name = node.gate_name.lower()
            targets = [str(q_map[t]) for t in node.targets]
            args = node.args
            
            if g_name == 'h':
                lines.append(f"H {targets[0]}")
            elif g_name == 'x':
                lines.append(f"X {targets[0]}")
            elif g_name == 'y':
                lines.append(f"Y {targets[0]}")
            elif g_name == 'z':
                lines.append(f"Z {targets[0]}")
            elif g_name == 's':
                lines.append(f"S {targets[0]}")
            elif g_name == 't':
                lines.append(f"T {targets[0]}")
            elif g_name == 'rx':
                lines.append(f"RX({args[0]}) {targets[0]}")
            elif g_name == 'ry':
                lines.append(f"RY({args[0]}) {targets[0]}")
            elif g_name == 'rz':
                lines.append(f"RZ({args[0]}) {targets[0]}")
            elif g_name in ('cnot', 'cx'):
                lines.append(f"CNOT {targets[0]} {targets[1]}")
            elif g_name == 'cz':
                lines.append(f"CZ {targets[0]} {targets[1]}")
            elif g_name == 'swap':
                lines.append(f"SWAP {targets[0]} {targets[1]}")
            elif g_name == 'ccx':
                lines.append(f"CCNOT {targets[0]} {targets[1]} {targets[2]}")
            elif g_name == 'cswap':
                lines.append(f"CSWAP {targets[0]} {targets[1]} {targets[2]}")
            elif g_name == 'cp':
                lines.append(f"CPHASE({args[0]}) {targets[0]} {targets[1]}")
            elif g_name == 'crx':
                lines.append(f"CONTROLLED RX({args[0]}) {targets[0]} {targets[1]}")
            elif g_name == 'cry':
                lines.append(f"CONTROLLED RY({args[0]}) {targets[0]} {targets[1]}")
            elif g_name == 'crz':
                lines.append(f"CONTROLLED RZ({args[0]}) {targets[0]} {targets[1]}")
            else:
                lines.append(f"# Unsupported gate in Quil: {node.gate_name} on {node.targets}")
                
        # Translate measurements
        if measures:
            lines.append("")
            for idx, (q, _c) in enumerate(measures):
                lines.append(f"MEASURE {q_map[q]} ro[{idx}]")
                
        return "\n".join(lines)
