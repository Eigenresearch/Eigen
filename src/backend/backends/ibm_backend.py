# IBM backend exporter for Eigen EQIR
#
# Audit §5: Previously this exporter emitted OpenQASM 2.0 with a long
# per-gate `if/elif` chain duplicated across all four hardware exporter
# modules. It now routes the same shared `GateSpec` table from
# `gate_registry.py` and emits OpenQASM 3.0 - the format Qiskit itself
# has migrated to as its primary interchange format. The legacy 2.0 string
# is reachable via the `export_qasm2()` method for callers that explicitly
# need it.
from src.ir.ir_graph import EQIRGraph
from src.backend.gate_registry import get_gate_spec, all_registered_gates


def _collect_qubits_and_measures(graph: EQIRGraph):
    qubits = set()
    measures = []
    for node in graph.nodes.values():
        if node.type == 'ALLOC':
            qubits.add(node.targets[0])
        elif node.type == 'MEASURE':
            measures.append((node.targets[0], node.cbit_name))
    sorted_qubits = sorted(qubits)
    return sorted_qubits, measures


def _render_qasm3_gate(spec_qasm_name: str, spec_qasm3_rendering, args, targets_str_list):
    """Return the text line for a GATE node in QASM3 form."""
    angle_str = f"({args[0]})" if args else ""
    if args and spec_qasm3_rendering:
        # Modern OpenQASM 3.0 controlled-rotation spelling: `ctrl @ phase(0.5) q[0], q[1];`
        head = f"{spec_qasm3_rendering}{angle_str}"
    else:
        head = f"{spec_qasm_name}{angle_str}"
    return f"{head} {', '.join(targets_str_list)};"


class IBMBackend:
    """IBM Quantum (OpenQASM) exporter."""

    def export(self, graph: EQIRGraph) -> str:
        """Emit OpenQASM 3.0 (the current Qiskit primary interchange format)."""
        lines = [
            "OPENQASM 3.0;",
            'include "stdgates.inc";',
            "",
        ]
        sorted_qubits, measures = _collect_qubits_and_measures(graph)
        q_map = {q: idx for idx, q in enumerate(sorted_qubits)}
        lines.append(f"qubit[{len(sorted_qubits)}] q;")
        if measures:
            lines.append(f"bit[{len(measures)}] c;")
        lines.append("")

        unsupported: list[str] = []
        for node in graph.topological_sort():
            if node.type != 'GATE':
                continue
            spec = get_gate_spec(node.gate_name)
            if spec is None:
                unsupported.append(node.gate_name)
                lines.append(f"// Unsupported gate in OpenQASM 3: {node.gate_name} on {node.targets}")
                continue
            targets_str_list = [f"q[{q_map[t]}]" for t in node.targets]
            args = [a for a in node.args if a is not None] if spec.takes_angle else []
            lines.append(_render_qasm3_gate(spec.qasm_name, spec.qasm3_rendering, args, targets_str_list))

        if measures:
            lines.append("")
            for idx, (q, _c) in enumerate(measures):
                lines.append(f"c[{idx}] = measure q[{q_map[q]}];")

        if unsupported:
            # Audit explicitly notes the IBM exporter must never silently
            # drop an unknown gate - surface it to the caller via a comment
            # trail in the emitted program.
            lines.append("")
            lines.append(
                f"// NOTE: {len(unsupported)} unsupported gate(s) emitted: "
                f"{sorted(set(unsupported))}"
            )
        return "\n".join(lines)

    def export_qasm2(self, graph: EQIRGraph) -> str:
        """Emit OpenQASM 2.0 for compatibility with older runtimes.

        Uses the legacy `qreg`/`creg` declarations and `qelib1.inc` include,
        plus 2.0-only spelling of controlled rotations (`crx/cry/crz/cp`)
        rather than the 3.0 `ctrl @ ...` form."""
        lines = [
            "OPENQASM 2.0;",
            'include "qelib1.inc";',
            "",
        ]
        sorted_qubits, measures = _collect_qubits_and_measures(graph)
        q_map = {q: idx for idx, q in enumerate(sorted_qubits)}
        lines.append(f"qreg q[{len(sorted_qubits)}];")
        if measures:
            lines.append(f"creg c[{len(measures)}];")
        lines.append("")

        for node in graph.topological_sort():
            if node.type != 'GATE':
                continue
            spec = get_gate_spec(node.gate_name)
            if spec is None:
                lines.append(f"// Unsupported gate in OpenQASM 2: {node.gate_name} on {node.targets}")
                continue
            targets_str_list = [f"q[{q_map[t]}]" for t in node.targets]
            angle_str = f"({node.args[0]})" if spec.takes_angle and node.args else ""
            # The legacy qelib1.inc spelling is the bare `qasm_name`
            # (e.g. `crx`, `cp`) rather than the 3.0 `ctrl @ ...` form.
            lines.append(f"{spec.qasm_name}{angle_str} {', '.join(targets_str_list)};")

        for idx, (q, _c) in enumerate(measures):
            lines.append(f"measure q[{q_map[q]}] -> c[{idx}];")

        return "\n".join(lines)
