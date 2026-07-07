# Azure Quantum QIR exporter for Eigen EQIR
#
# Audit §5: Previously this exporter carried an if/elif chain for each gate.
# Notably, SWAP was missing from the chain entirely, so circuits containing
# a SWAP gate silently dropped the operation when exported to QIR.
# Now routes through the shared `gate_registry` table so adding a new gate
# to one exporter is reflected in all four. The SWAP gate is now emitted
# as `__quantum__qis__swap__body(%Qubit*, %Qubit*)`, and the QIR declaration
# for it is printed alongside the others.
from src.ir.ir_graph import EQIRGraph
from src.backend.gate_registry import get_gate_spec, all_registered_gates


class AzureBackend:
    def export(self, graph: EQIRGraph) -> str:
        lines = [
            "; ModuleID = 'EigenQIRModule'",
            'source_filename = "eigen_qir"',
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

        sorted_qubits = sorted(qubits)
        q_map = {q: idx for idx, q in enumerate(sorted_qubits)}

        # 1. Declare qubits in QIR format
        for q, idx in q_map.items():
            lines.append(f"  %q{idx} = call %Qubit* @__quantum__rt__qubit_allocate()")

        # 2. Gate operations - dispatch via the shared gate registry so that
        # newly-added gates are reflected here automatically.
        # We collect the *set* of QIR callee functions actually used and emit
        # only those declarations, which keeps QIR concise and ensures
        # unsupported gates don't silently pass through.
        used_qir_funcs: list[str] = []
        used_qir_func_names: set[str] = set()
        unsupported: list[str] = []

        for node in graph.topological_sort():
            if node.type != 'GATE':
                continue
            spec = get_gate_spec(node.gate_name)
            if spec is None or not spec.qir_func:
                unsupported.append(node.gate_name)
                lines.append(
                    f"  ; Unsupported gate {node.gate_name} on {node.targets} - dropped"
                )
                continue
            targets = [f"%q{q_map[t]}" for t in node.targets]
            # QIR single-qubit rotations take (double, %Qubit*); two- and
            # three-qubit gates take only the qubit pointers.
            if spec.takes_angle:
                arg_list = [f"double {node.args[0]}"] + targets
            else:
                arg_list = targets
            callee = spec.qir_callee()
            lines.append(f"  call void @{callee}({', '.join(arg_list)})")
            if spec.qir_func not in used_qir_func_names:
                used_qir_func_names.add(spec.qir_func)
                used_qir_funcs.append(spec.qir_func)

        # 3. Free qubits
        for q, idx in q_map.items():
            lines.append(f"  call void @__quantum__rt__qubit_release(%Qubit* %q{idx})")

        lines.append("  ret void")
        lines.append("}")
        lines.append("")

        # Declarations of QIS functions - emit only for what was actually
        # used, in deterministic (sorted) order.
        lines.append("declare %Qubit* @__quantum__rt__qubit_allocate()")
        lines.append("declare void @__quantum__rt__qubit_release(%Qubit*)")
        for func in sorted(used_qir_func_names):
            # Build the parameter list: (double, %Qubit*) for rotation/phase
            # gates (1 angle + qubits), (%Qubit* x N) for non-angle gates.
            # The angle position is inferred from the GateSpec table to be
            # robust against future additions.
            spec = next(s for s in [get_gate_spec(g) for g in all_registered_gates()]
                         if s and s.qir_func == func)
            if spec.takes_angle:
                arg_types = "double, " + ", ".join(["%Qubit*"] * spec.qubit_count)
            else:
                arg_types = ", ".join(["%Qubit*"] * spec.qubit_count)
            lines.append(f"declare void @{spec.qir_callee()}({arg_types})")

        if unsupported:
            lines.append(
                f"; NOTE: {len(unsupported)} unsupported gate(s): "
                f"{sorted(set(unsupported))}"
            )

        return "\n".join(lines)
