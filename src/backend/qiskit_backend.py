from src.ir.ir_graph import EQIRGraph, EQIRNode
from src.semantic.backend_capabilities import get_backend_capabilities, CapabilityLevel, UnsupportedOp
from src.diagnostics import DiagnosticEngine, Diagnostic, DiagnosticSeverity, SourceLocation
from src.frontend.ast import (
    ProgramNode, ASTNode, FuncDeclNode, StructDeclNode, ForNode, WhileNode,
    TryCatchNode, ThrowNode, StructLiteralNode, ArrayLiteralNode, DotAccessNode,
    IndexAccessNode, CallNode, LetNode, AssignmentNode
)

class BackendReport:
    def __init__(self, backend_name: str, warnings: list[str], unsupported_nodes: int, generated_lines: int, stats: dict = None):
        self.backend_name = backend_name
        self.warnings = warnings
        self.unsupported_nodes = unsupported_nodes
        self.generated_lines = generated_lines
        self.stats = stats or {"supported": 100.0, "emulated": 0.0, "unsupported": 0.0}

    def __repr__(self) -> str:
        warn_str = "\n".join(f"- {w}" for w in self.warnings) if self.warnings else "None"
        return (
            f"Backend: {self.backend_name}\n"
            f"Supported: {self.stats['supported']:.1f}%\n"
            f"Emulated: {self.stats['emulated']:.1f}%\n"
            f"Unsupported: {self.stats['unsupported']:.1f}%\n"
            f"Warnings:\n{warn_str}\n"
            f"Unsupported constructs: {self.unsupported_nodes}\n"
            f"Generated lines: {self.generated_lines}"
        )

class QiskitBackend:
    def __init__(self):
        self.capabilities = get_backend_capabilities("qiskit")

    def transpile(self, graph: EQIRGraph, ast: ProgramNode | None = None) -> tuple[str, BackendReport]:
        if ast is None:
            ast = ProgramNode(1.0, None, [], [])
        diag_engine = DiagnosticEngine()
        unsupported_ops = []

        node_stats = {
            CapabilityLevel.SUPPORTED: 0,
            CapabilityLevel.EMULATED: 0,
            CapabilityLevel.UNSUPPORTED: 0
        }

        # 1. Capability Checks on AST
        if ast.imports and self.capabilities.supports_imports == CapabilityLevel.EMULATED:
            diag_engine.emit(
                DiagnosticSeverity.WARNING,
                "Imports are only partially supported in Qiskit backend (inlining is simulated).",
                SourceLocation("ast", 1, 0)
            )
        elif ast.imports and self.capabilities.supports_imports == CapabilityLevel.UNSUPPORTED:
            diag_engine.emit(
                DiagnosticSeverity.ERROR,
                "Imports are not supported in Qiskit backend.",
                SourceLocation("ast", 1, 0)
            )
            for imp in ast.imports:
                unsupported_ops.append(UnsupportedOp("ImportNode", None, imp.module_path, "Imports not supported"))

        for imp in ast.imports:
            node_stats[self.capabilities.supports_imports] += 1

        def visit(node: ASTNode):
            if node is None:
                return
            
            # Match capabilities
            level = CapabilityLevel.SUPPORTED
            if isinstance(node, (FuncDeclNode, CallNode)):
                level = self.capabilities.supports_classical_functions
            elif isinstance(node, (StructDeclNode, StructLiteralNode)):
                level = self.capabilities.supports_structs
            elif isinstance(node, (ForNode, WhileNode)):
                level = self.capabilities.supports_loops
            elif isinstance(node, (TryCatchNode, ThrowNode)):
                level = self.capabilities.supports_exceptions
            elif isinstance(node, ArrayLiteralNode):
                level = self.capabilities.supports_arrays
            elif isinstance(node, DotAccessNode):
                level = self.capabilities.supports_field_access
            elif isinstance(node, IndexAccessNode):
                level = self.capabilities.supports_index_access

            node_stats[level] += 1
            if isinstance(node, FuncDeclNode):
                if self.capabilities.supports_classical_functions == CapabilityLevel.UNSUPPORTED:
                    diag_engine.emit(
                        DiagnosticSeverity.WARNING,
                        f"Classical function declaration '{node.name}' is not supported by Qiskit backend.",
                        SourceLocation("ast")
                    )
                    unsupported_ops.append(UnsupportedOp("FuncDeclNode", None, f"func {node.name}", "Classical functions not supported"))
            elif isinstance(node, StructDeclNode):
                if self.capabilities.supports_structs == CapabilityLevel.UNSUPPORTED:
                    diag_engine.emit(
                        DiagnosticSeverity.WARNING,
                        f"Struct declaration '{node.name}' is not supported by Qiskit backend.",
                        SourceLocation("ast")
                    )
                    unsupported_ops.append(UnsupportedOp("StructDeclNode", None, f"struct {node.name}", "Structs not supported"))
            elif isinstance(node, (ForNode, WhileNode)):
                if self.capabilities.supports_loops == CapabilityLevel.UNSUPPORTED:
                    diag_engine.emit(
                        DiagnosticSeverity.WARNING,
                        "Control loop is not supported by Qiskit backend.",
                        SourceLocation("ast")
                    )
                    unsupported_ops.append(UnsupportedOp(type(node).__name__, None, "loop", "Loops not supported"))
            elif isinstance(node, TryCatchNode):
                if self.capabilities.supports_exceptions == CapabilityLevel.UNSUPPORTED:
                    diag_engine.emit(
                        DiagnosticSeverity.WARNING,
                        "Exception handling try-catch block is not supported by Qiskit backend.",
                        SourceLocation("ast")
                    )
                    unsupported_ops.append(UnsupportedOp("TryCatchNode", None, "try-catch", "Exceptions not supported"))
            elif isinstance(node, ThrowNode):
                if self.capabilities.supports_exceptions == CapabilityLevel.UNSUPPORTED:
                    diag_engine.emit(
                        DiagnosticSeverity.WARNING,
                        "Exception throw statement is not supported by Qiskit backend.",
                        SourceLocation("ast")
                    )
                    unsupported_ops.append(UnsupportedOp("ThrowNode", None, "throw", "Exceptions not supported"))
            elif isinstance(node, StructLiteralNode):
                if self.capabilities.supports_structs == CapabilityLevel.UNSUPPORTED:
                    diag_engine.emit(
                        DiagnosticSeverity.WARNING,
                        f"Struct instantiation of '{node.struct_name}' is not supported by Qiskit backend.",
                        SourceLocation("ast")
                    )
                    unsupported_ops.append(UnsupportedOp("StructLiteralNode", None, f"{node.struct_name} {{...}}", "Structs not supported"))
            elif isinstance(node, ArrayLiteralNode):
                if self.capabilities.supports_arrays == CapabilityLevel.UNSUPPORTED:
                    diag_engine.emit(
                        DiagnosticSeverity.WARNING,
                        "Array literal is not supported by Qiskit backend.",
                        SourceLocation("ast")
                    )
                    unsupported_ops.append(UnsupportedOp("ArrayLiteralNode", None, "array", "Arrays not supported"))
                elif self.capabilities.supports_arrays == CapabilityLevel.EMULATED:
                    diag_engine.emit(
                        DiagnosticSeverity.WARNING,
                        "Array literal is only partially supported by Qiskit backend.",
                        SourceLocation("ast")
                    )
            elif isinstance(node, DotAccessNode):
                if self.capabilities.supports_field_access == CapabilityLevel.UNSUPPORTED:
                    diag_engine.emit(
                        DiagnosticSeverity.WARNING,
                        "Struct member field access is not supported by Qiskit backend.",
                        SourceLocation("ast")
                    )
                    unsupported_ops.append(UnsupportedOp("DotAccessNode", None, f".{node.member}", "Field access not supported"))
            elif isinstance(node, IndexAccessNode):
                if self.capabilities.supports_index_access == CapabilityLevel.UNSUPPORTED:
                    diag_engine.emit(
                        DiagnosticSeverity.WARNING,
                        "Index access is not supported by Qiskit backend.",
                        SourceLocation("ast")
                    )
                    unsupported_ops.append(UnsupportedOp("IndexAccessNode", None, "[]", "Index access not supported"))
            elif isinstance(node, CallNode):
                if self.capabilities.supports_classical_functions == CapabilityLevel.UNSUPPORTED:
                    diag_engine.emit(
                        DiagnosticSeverity.WARNING,
                        "Classical function call is not supported by Qiskit backend.",
                        SourceLocation("ast")
                    )
                    unsupported_ops.append(UnsupportedOp("CallNode", None, "call", "Classical calls not supported"))
            elif isinstance(node, LetNode):
                visit(node.value)
            elif isinstance(node, AssignmentNode):
                visit(node.target)
                visit(node.value)

            # Traverse child nodes
            if hasattr(node, 'body') and isinstance(node.body, list):
                for child in node.body:
                    visit(child)
            if hasattr(node, 'try_body') and isinstance(node.try_body, list):
                for child in node.try_body:
                    visit(child)
            if hasattr(node, 'catch_body') and isinstance(node.catch_body, list):
                for child in node.catch_body:
                    visit(child)
            if hasattr(node, 'left') and isinstance(node.left, ASTNode):
                visit(node.left)
            if hasattr(node, 'right') and isinstance(node.right, ASTNode):
                visit(node.right)
            if hasattr(node, 'expr') and isinstance(node.expr, ASTNode):
                visit(node.expr)

        for stmt in ast.body:
            visit(stmt)

        # 2. Identify all allocated qubits and classical bits from EQIR Graph
        nodes = graph.topological_sort()
        qubits = []
        cbits = []
        
        for node in nodes:
            if node.type == 'ALLOC':
                q_name = node.targets[0]
                if q_name not in qubits:
                    qubits.append(q_name)
            elif node.type == 'MEASURE':
                c_name = node.cbit_name
                if c_name not in cbits:
                    cbits.append(c_name)
                q_name = node.targets[0]
                if q_name not in qubits:
                    qubits.append(q_name)
                    
        for node in nodes:
            for q_name in node.targets:
                if q_name not in qubits:
                    qubits.append(q_name)
            if node.cbit_name and node.cbit_name not in cbits:
                cbits.append(node.cbit_name)
            if node.condition:
                c_name = node.condition[0]
                if c_name not in cbits:
                    cbits.append(c_name)

        qubits.sort()
        cbits.sort()
        qubit_map = {name: i for i, name in enumerate(qubits)}
        cbit_map = {name: i for i, name in enumerate(cbits)}

        # 3. Build the python script lines
        lines = [
            "# Transpiled from Eigen EQIR v1.1 Graph to Qiskit Script",
            "import numpy as np",
            "from qiskit import QuantumCircuit, transpile",
            "from qiskit_aer import AerSimulator",
            ""
        ]

        # Add comments for unsupported ops at the top
        if unsupported_ops:
            lines.append("# " + "=" * 50)
            lines.append("# WARNING: Unsupported classical constructs encountered:")
            for op in unsupported_ops:
                lines.append(f"# - {op.kind} (pretty: '{op.pretty_repr}'): {op.reason}")
            lines.append("# " + "=" * 50)
            lines.append("")

        lines.extend([
            f"# Allocate circuit with {len(qubits)} qubits and {len(cbits)} classical bits",
            f"qc = QuantumCircuit({len(qubits)}, {len(cbits)})",
            ""
        ])

        # Translate operations
        for node in nodes:
            if node.type == 'ALLOC':
                lines.append(f"# Allocated qubit: {node.targets[0]}")
                continue
                
            cond_suffix = ""
            if node.condition:
                c_name, op, val = node.condition
                if op == '==':
                    c_idx = cbit_map.get(c_name, 0)
                    if isinstance(val, int):
                        cond_suffix = f".c_if({c_idx}, {val})"
                    else:
                        lines.append(f"# Skipped classical condition on non-integer value {c_name} {op} {val}")
                        continue
                else:
                    lines.append(f"# Skipped classical condition {c_name} {op} {val} (Qiskit only supports == comparison in c_if)")
                    continue

            if node.type == 'GATE':
                g_name = node.gate_name.lower()
                q_indices = [qubit_map[t] for t in node.targets]
                
                if g_name == 'h':
                    lines.append(f"qc.h({q_indices[0]}){cond_suffix}")
                elif g_name == 'x':
                    lines.append(f"qc.x({q_indices[0]}){cond_suffix}")
                elif g_name == 'y':
                    lines.append(f"qc.y({q_indices[0]}){cond_suffix}")
                elif g_name == 'z':
                    lines.append(f"qc.z({q_indices[0]}){cond_suffix}")
                elif g_name == 's':
                    lines.append(f"qc.s({q_indices[0]}){cond_suffix}")
                elif g_name == 't':
                    lines.append(f"qc.t({q_indices[0]}){cond_suffix}")
                elif g_name == 'cnot':
                    lines.append(f"qc.cx({q_indices[0]}, {q_indices[1]}){cond_suffix}")
                elif g_name == 'cz':
                    lines.append(f"qc.cz({q_indices[0]}, {q_indices[1]}){cond_suffix}")
                elif g_name == 'swap':
                    lines.append(f"qc.swap({q_indices[0]}, {q_indices[1]}){cond_suffix}")
                elif g_name in ('rx', 'ry', 'rz'):
                    angle = node.args[0] if node.args else 0.0
                    # Check if angle is unsupported placeholder
                    if isinstance(angle, str) and angle.startswith("__") and angle.endswith("__"):
                        lines.append(f"# Skipped gate {node.gate_name} due to unsupported classical angle expression: {angle}")
                    else:
                        lines.append(f"qc.{g_name}({angle}, {q_indices[0]}){cond_suffix}")
                else:
                    lines.append(f"# Unsupported gate: {node.gate_name} on {node.targets}")

            elif node.type == 'MEASURE':
                q_idx = qubit_map[node.targets[0]]
                c_idx = cbit_map[node.cbit_name]
                lines.append(f"qc.measure({q_idx}, {c_idx}){cond_suffix}")

            elif node.type == 'TRACE':
                lines.append(f"# TRACE: Save statevector snapshot at this barrier")
                lines.append("qc.save_statevector() if hasattr(qc, 'save_statevector') else None")

            elif node.type == 'PRINT':
                val = node.print_expr
                if isinstance(val, str) and val.startswith("__") and val.endswith("__"):
                    lines.append(f"# Eigen warning: unsupported print expression omitted ({val})")
                elif isinstance(val, str) and val.isidentifier():
                    lines.append(f"print('[PRINT DIRECTIVE] cbit {val}:', {val} if '{val}' in locals() else '{val}')")
                else:
                    if isinstance(val, str):
                        escaped_val = val.replace("\\", "\\\\").replace("'", "\\'")
                        lines.append(f"print('[PRINT DIRECTIVE] {escaped_val}')")
                    else:
                        lines.append(f"print('[PRINT DIRECTIVE] {val}')")

            elif node.type == 'ASSERT':
                left, op, right = node.assert_cond
                left_unsupported = isinstance(left, str) and left.startswith("__") and left.endswith("__")
                right_unsupported = isinstance(right, str) and right.startswith("__") and right.endswith("__")
                if left_unsupported or right_unsupported:
                    lines.append(f"# Eigen warning: unsupported assert omitted ({left} {op} {right})")
                else:
                    lines.append(f"# Assert condition: {left} {op} {right}")

        # Add execution footer
        lines.extend([
            "",
            "# Execute the circuit using Qiskit Aer",
            "simulator = AerSimulator()",
            "compiled_circuit = transpile(qc, simulator)",
            "result = simulator.run(compiled_circuit, shots=1024).result()",
            "counts = result.get_counts(qc)",
            "print('Simulation results counts:', counts)"
        ])

        script_code = "\n".join(lines)
        
        # Build Report
        warn_msgs = [d.message for d in diag_engine.get_warnings()]
        
        total_nodes = sum(node_stats.values())
        if total_nodes > 0:
            supported_pct = (node_stats[CapabilityLevel.SUPPORTED] / total_nodes) * 100.0
            emulated_pct = (node_stats[CapabilityLevel.EMULATED] / total_nodes) * 100.0
            unsupported_pct = (node_stats[CapabilityLevel.UNSUPPORTED] / total_nodes) * 100.0
        else:
            supported_pct = 100.0
            emulated_pct = 0.0
            unsupported_pct = 0.0
            
        stats = {
            "supported": supported_pct,
            "emulated": emulated_pct,
            "unsupported": unsupported_pct
        }

        report = BackendReport(
            backend_name="Qiskit",
            warnings=warn_msgs,
            unsupported_nodes=len(unsupported_ops),
            generated_lines=len(lines),
            stats=stats
        )
        
        return script_code, report
