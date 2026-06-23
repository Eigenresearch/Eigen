from src.cli import register_command
from src.compiler import compile_to_eqir
from src.resource_estimator.estimator import ResourceEstimator

@register_command("estimate")
def estimate_command(args, workspace_root):
    graph, _ = compile_to_eqir(args.file, workspace_root)
    estimator = ResourceEstimator()
    metrics = estimator.estimate(graph)
    
    print("=" * 50)
    print("          EIGEN RESOURCE ESTIMATION          ")
    print("=" * 50)
    print(f"File: {args.file}")
    print("-" * 50)
    print(f"Logical Qubits:   {metrics['logical_qubits']}")
    print(f"Circuit Depth:    {metrics['circuit_depth']}")
    print(f"Gate Count:       {metrics['gate_count']}")
    print(f"CNOT Count:       {metrics['cnot_count']}")
    print(f"T Count:          {metrics['t_count']}")
    print(f"T Depth:          {metrics['t_depth']}")
    print(f"Clifford Count:   {metrics['clifford_count']}")
    print(f"Measurements:     {metrics['measurements']}")
    print("=" * 50)
