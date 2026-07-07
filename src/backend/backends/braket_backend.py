# AWS Braket backend exporter for Eigen EQIR
#
# Audit §5: Previously this exporter carried an if/elif chain duplicated
# across the four hardware exporter modules. It now routes through the
# shared `gate_registry.GateSpec` table so adding a gate to one exporter
# is reflected in all four.
from src.ir.ir_graph import EQIRGraph
from src.backend.gate_registry import get_gate_spec


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

        sorted_qubits = sorted(qubits)
        q_map = {q: idx for idx, q in enumerate(sorted_qubits)}

        unsupported: list[str] = []
        for node in graph.topological_sort():
            if node.type != 'GATE':
                continue
            spec = get_gate_spec(node.gate_name)
            if spec is None or not spec.braket_method:
                unsupported.append(node.gate_name)
                lines.append(f"# Unsupported gate: {node.gate_name} on {node.targets}")
                continue
            targets = [q_map[t] for t in node.targets]
            # Braket single-qubit rotations are .rx(q, angle); single-qubit
            # gates are .h(q); two-qubit are .cnot(c, t); three-qubit .ccx(c, t1, t2).
            if spec.takes_angle:
                args = [str(targets[0]), repr(node.args[0])] + [str(t) for t in targets[1:]]
            else:
                args = [str(t) for t in targets]
            args_str = ", ".join(args)
            lines.append(f"device_circuit.{spec.braket_method}({args_str})")

        if unsupported:
            lines.append("")
            lines.append(
                f"# NOTE: {len(unsupported)} unsupported gate(s): "
                f"{sorted(set(unsupported))}"
            )
        return "\n".join(lines)
