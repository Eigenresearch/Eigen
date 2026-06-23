# Azure Quantum QIR exporter for Eigen EQIR
from src.ir.ir_graph import EQIRGraph

class AzureBackend:
    def export(self, graph: EQIRGraph) -> str:
        lines = [
            "; ModuleID = 'EigenQIRModule'",
            "source_filename = \"eigen_qir\"",
            "",
            "%Qubit = type opaque",
            "%Result = type opaque",
            "",
            "define void @main() #0 {",
            "entry:"
        ]
        
        qubits = set()
        for node in graph.nodes.values():
            if node.type == 'ALLOC':
                qubits.add(node.targets[0])
                
        sorted_qubits = sorted(list(qubits))
        q_map = {q: idx for idx, q in enumerate(sorted_qubits)}
        
        # 1. Declare qubits in QIR format
        for q, idx in q_map.items():
            lines.append(f"  %q{idx} = call %Qubit* @__quantum__rt__qubit_allocate()")
            
        # 2. Gate operations
        for node in graph.topological_sort():
            if node.type != 'GATE':
                continue
                
            g_name = node.gate_name.upper()
            targets = [f"%q{q_map[t]}" for t in node.targets]
            args = node.args
            
            if g_name == 'H':
                lines.append(f"  call void @__quantum__qis__h__body(%Qubit* {targets[0]})")
            elif g_name == 'X':
                lines.append(f"  call void @__quantum__qis__x__body(%Qubit* {targets[0]})")
            elif g_name == 'Y':
                lines.append(f"  call void @__quantum__qis__y__body(%Qubit* {targets[0]})")
            elif g_name == 'Z':
                lines.append(f"  call void @__quantum__qis__z__body(%Qubit* {targets[0]})")
            elif g_name == 'S':
                lines.append(f"  call void @__quantum__qis__s__body(%Qubit* {targets[0]})")
            elif g_name == 'T':
                lines.append(f"  call void @__quantum__qis__t__body(%Qubit* {targets[0]})")
            elif g_name == 'RX':
                lines.append(f"  call void @__quantum__qis__rx__body(double {args[0]}, %Qubit* {targets[0]})")
            elif g_name == 'RY':
                lines.append(f"  call void @__quantum__qis__ry__body(double {args[0]}, %Qubit* {targets[0]})")
            elif g_name == 'RZ':
                lines.append(f"  call void @__quantum__qis__rz__body(double {args[0]}, %Qubit* {targets[0]})")
            elif g_name == 'CNOT':
                lines.append(f"  call void @__quantum__qis__cnot__body(%Qubit* {targets[0]}, %Qubit* {targets[1]})")
            elif g_name == 'CZ':
                lines.append(f"  call void @__quantum__qis__cz__body(%Qubit* {targets[0]}, %Qubit* {targets[1]})")
                
        # 3. Free qubits
        for q, idx in q_map.items():
            lines.append(f"  call void @__quantum__rt__qubit_release(%Qubit* %q{idx})")
            
        lines.append("  ret void")
        lines.append("}")
        lines.append("")
        
        # Declarations of QIS functions
        lines.append("declare %Qubit* @__quantum__rt__qubit_allocate()")
        lines.append("declare void @__quantum__rt__qubit_release(%Qubit*)")
        lines.append("declare void @__quantum__qis__h__body(%Qubit*)")
        lines.append("declare void @__quantum__qis__x__body(%Qubit*)")
        lines.append("declare void @__quantum__qis__y__body(%Qubit*)")
        lines.append("declare void @__quantum__qis__z__body(%Qubit*)")
        lines.append("declare void @__quantum__qis__s__body(%Qubit*)")
        lines.append("declare void @__quantum__qis__t__body(%Qubit*)")
        lines.append("declare void @__quantum__qis__rx__body(double, %Qubit*)")
        lines.append("declare void @__quantum__qis__ry__body(double, %Qubit*)")
        lines.append("declare void @__quantum__qis__rz__body(double, %Qubit*)")
        lines.append("declare void @__quantum__qis__cnot__body(%Qubit*, %Qubit*)")
        lines.append("declare void @__quantum__qis__cz__body(%Qubit*, %Qubit*)")
        
        return "\n".join(lines)
