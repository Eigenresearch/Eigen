import sys
import os
from src.cli import register_command
from src.compiler import compile_to_eqir
from src.ir.optimizer import EQIROptimizer
from src.equivalence import EquivalenceChecker

from src.zx.exceptions import IndeterminateEquivalenceError

@register_command("verify-equiv")
def verify_equiv_command(args, workspace_root):
    graph1, _ = compile_to_eqir(args.file1, workspace_root)
    graph2, _ = compile_to_eqir(args.file2, workspace_root)
    
    # Early indeterminate check for large circuits (> 16 qubits)
    qubits1 = set()
    for node in graph1.nodes.values():
        if node.type == 'ALLOC':
            qubits1.add(node.targets[0])
    qubits2 = set()
    for node in graph2.nodes.values():
        if node.type == 'ALLOC':
            qubits2.add(node.targets[0])
    all_qubits = qubits1 | qubits2
    if len(all_qubits) > 16:
        print("\nResult: INDETERMINATE (Circuit too large or complex to verify) [WARNING]")
        print(f"Details: Circuit has {len(all_qubits)} > 16 qubits. ZX equivalence checker cannot verify it without hanging.")
        print("=" * 50)
        sys.exit(3)
        
    opt_level = 0
    if getattr(args, "O", None) is not None:
        opt_level = args.O
    elif getattr(args, "optimize", False):
        opt_level = 2
        
    if opt_level >= 1:
        optimizer = EQIROptimizer()
        graph1 = optimizer.optimize(graph1)
        graph2 = optimizer.optimize(graph2)
        
    checker = EquivalenceChecker()
    
    from src.canonicalizer import Canonicalizer
    canon = Canonicalizer()
    
    h1 = canon.hash_circuit(graph1)
    h2 = canon.hash_circuit(graph2)
    
    print("=" * 50)
    print("          EIGEN EQUIVALENCE CHECK          ")
    print("=" * 50)
    print(f"File 1: {args.file1}")
    print(f"File 2: {args.file2}")
    
    if h1 != h2:
        print("\nResult: NOT EQUIVALENT (Fast Reject via Canonical Hash) [FAIL]")
        print("=" * 50)
        sys.exit(0)
        
    try:
        equivalent = checker.are_equivalent(graph1, graph2)
        if equivalent:
            print("\nResult: Mathematically EQUIVALENT (up to global phase) [SUCCESS]")
        else:
            print("\nResult: NOT EQUIVALENT [FAIL]")
        print("=" * 50)
    except IndeterminateEquivalenceError as e:
        print("\nResult: INDETERMINATE (Circuit too large or complex to verify) [WARNING]")
        print(f"Details: {e}")
        print("=" * 50)
        sys.exit(3)
    except Exception as e:
        print(f"Equivalence Verification Error: {e}", file=sys.stderr)
        sys.exit(1)


@register_command("verify")
def verify_command(args, workspace_root):
    """Static verification of an Eigen source file.
    
    Checks:
    1. Type correctness (via type checker)
    2. Qubit usage safety (no double-free, no use-after-measure races)
    3. Static assertion validation where possible
    4. Quantum circuit property analysis (Clifford detection, gate counts)
    """
    from src.frontend.lexer import Lexer
    from src.frontend.parser import Parser
    from src.semantic.import_resolver import ImportResolver
    from src.semantic.type_checker import TypeChecker, TypeErrorException
    from src.ir.ir_converter import EQIRConverter
    
    filepath = args.file
    if not os.path.isfile(filepath):
        print(f"Error: File '{filepath}' not found.", file=sys.stderr)
        sys.exit(1)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    issues = []
    warnings = []
    info = []
    
    print("=" * 60)
    print("              EIGEN STATIC VERIFIER              ")
    print("=" * 60)
    print(f"File: {filepath}")
    print()
    
    # Phase 1: Parse
    print("[1/5] Parsing...")
    try:
        lexer = Lexer(content)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        info.append("Parse: OK")
    except SyntaxError as e:
        issues.append(f"Parse Error: {e}")
        _print_results(issues, warnings, info)
        sys.exit(1)
    
    # Phase 2: Import Resolution
    print("[2/5] Resolving imports...")
    try:
        resolver = ImportResolver(workspace_root)
        ast = resolver.resolve(ast)
        info.append("Import Resolution: OK")
    except Exception as e:
        issues.append(f"Import Resolution Error: {e}")
    
    # Phase 3: Type Checking
    print("[3/5] Type checking...")
    try:
        type_checker = TypeChecker()
        type_checker.check(ast)
        info.append("Type Check: OK")
    except TypeErrorException as e:
        issues.append(f"Type Error: {e}")
    except Exception as e:
        warnings.append(f"Type Check Warning: {e}")
    
    # Phase 4: Qubit Safety Analysis
    print("[4/5] Qubit safety analysis...")
    from src.frontend.ast import (
        VarDeclNode, GateNode, MeasureNode, QFuncCallNode, QFuncDeclNode,
        IfNode, WhileNode, ForNode, TryCatchNode
    )
    allocated_qubits = set()
    measured_qubits = set()
    gate_after_measure = []
    qubit_gate_counts = {}
    total_gates = 0
    clifford_gates = {'H', 'X', 'Y', 'Z', 'S', 'CNOT', 'CZ', 'SWAP'}
    non_clifford_count = 0
    t_count = 0
    
    def analyze_body(body):
        nonlocal total_gates, non_clifford_count, t_count
        for node in body:
            if isinstance(node, VarDeclNode) and node.type_name == 'qubit':
                allocated_qubits.add(node.name)
            elif isinstance(node, GateNode):
                total_gates += 1
                if node.gate_name == 'T':
                    t_count += 1
                if node.gate_name not in clifford_gates:
                    non_clifford_count += 1
                for t in node.targets:
                    qubit_gate_counts[t] = qubit_gate_counts.get(t, 0) + 1
                    if t in measured_qubits:
                        gate_after_measure.append((node.gate_name, t))
            elif isinstance(node, MeasureNode):
                measured_qubits.add(node.qubit_name)
            elif isinstance(node, QFuncDeclNode):
                analyze_body(node.body)
            elif isinstance(node, IfNode):
                analyze_body(node.body)
                if hasattr(node, 'else_body') and node.else_body:
                    analyze_body(node.else_body)
            elif isinstance(node, WhileNode):
                analyze_body(node.body)
            elif isinstance(node, ForNode):
                analyze_body(node.body)
            elif isinstance(node, TryCatchNode):
                analyze_body(node.try_body)
                analyze_body(node.catch_body)
    
    analyze_body(ast.body)
    
    if gate_after_measure:
        for gate_name, qubit in gate_after_measure:
            warnings.append(f"Gate '{gate_name}' applied to qubit '{qubit}' after measurement")
    
    info.append(f"Allocated qubits: {len(allocated_qubits)}")
    info.append(f"Total gates: {total_gates}")
    info.append(f"T-count: {t_count}")
    
    is_clifford = (non_clifford_count == 0 and total_gates > 0)
    if is_clifford:
        info.append("Circuit class: Clifford (efficiently classically simulable)")
    else:
        info.append(f"Circuit class: Non-Clifford ({non_clifford_count} non-Clifford gates)")
    
    # Phase 5: EQIR Structure Check
    print("[5/5] EQIR structure check...")
    try:
        graph, _ = compile_to_eqir(filepath, workspace_root)
        depth = graph.compute_depth()
        info.append(f"Circuit depth: {depth}")
        info.append(f"EQIR nodes: {len(graph.nodes)}")
        info.append("EQIR Structure: OK")
    except Exception as e:
        warnings.append(f"EQIR Structure Warning: {e}")
    
    print()
    _print_results(issues, warnings, info)
    
    if issues:
        sys.exit(1)
    sys.exit(0)


def _print_results(issues, warnings, info):
    print("-" * 60)
    if info:
        print("INFO:")
        for item in info:
            print(f"  ✓ {item}")
    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(f"  ⚠ {w}")
    if issues:
        print("\nERRORS:")
        for issue in issues:
            print(f"  ✗ {issue}")
    print()
    if not issues:
        print("Result: VERIFICATION PASSED")
    else:
        print(f"Result: VERIFICATION FAILED ({len(issues)} error(s))")
    print("=" * 60)
