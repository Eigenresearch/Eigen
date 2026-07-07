# IonQ backend exporter for Eigen EQIR
#
# Audit §5: Previously this exporter emitted the bare lowercase Eigen gate
# name as the IonQ JSON `gate` field, without consulting a central
# registry. It now routes through the shared `gate_registry.GateSpec`
# table. IonQ accepts the standard gate names (h, x, y, z, rx, ry, rz, cnot,
# cz, swap, etc.) - the spec table's `ionq_gate` field carries those.
import json
from src.ir.ir_graph import EQIRGraph
from src.backend.gate_registry import get_gate_spec


class IonQBackend:
    def export(self, graph: EQIRGraph) -> str:
        qubits = set()
        for node in graph.nodes.values():
            if node.type == 'ALLOC':
                qubits.add(node.targets[0])

        sorted_qubits = sorted(qubits)
        q_map = {q: idx for idx, q in enumerate(sorted_qubits)}

        circuit = {
            "qubits": len(sorted_qubits),
            "circuit": [],
        }

        unsupported: list[str] = []
        for node in graph.topological_sort():
            if node.type != 'GATE':
                continue
            spec = get_gate_spec(node.gate_name)
            if spec is None or not spec.ionq_gate:
                unsupported.append(node.gate_name)
                continue
            targets = [q_map[t] for t in node.targets]
            gate_entry = {"gate": spec.ionq_gate, "targets": targets}
            if node.args and spec.takes_angle:
                gate_entry["phase"] = node.args[0]
            circuit["circuit"].append(gate_entry)

        if unsupported:
            # Surface unsupported gates as an extra JSON key so the caller
            # can detect them and the audit's "no silent drops" rule is met.
            circuit["unsupported_gates"] = sorted(set(unsupported))

        return json.dumps(circuit, indent=2)
