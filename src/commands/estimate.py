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
    print(f"Logical Qubits:       {metrics['logical_qubits']}")
    print(f"Classical Bits:        {metrics['classical_bits']}")
    print(f"Circuit Depth:         {metrics['circuit_depth']}")
    print(f"Two-Qubit Depth:       {metrics['two_qubit_depth']}")
    print(f"T Depth:               {metrics['t_depth']}")
    print(f"Measurement Depth:     {metrics['measurement_depth']}")
    print("-" * 50)
    print(f"Total Gate Count:      {metrics['gate_count']}")
    print(f"  Single-Qubit Gates:   {metrics['single_qubit_count']}")
    print(f"  Two-Qubit Gates:      {metrics['two_qubit_count']}")
    print(f"    - CNOTs:            {metrics['cnot_count']}")
    print(f"    - SWAPs:            {metrics['swap_count']}")
    print(f"  Three-Qubit Gates:    {metrics['three_qubit_count']}")
    print(f"    - Toffoli (CCX):    {metrics['toffoli_count']}")
    print(f"Clifford Gates:        {metrics['clifford_count']}")
    print(f"T Gates:               {metrics['t_count']}")
    print(f"Rotation Gates:        {metrics['rotation_count']} "
          f"(T synthesis est: {metrics['rotation_t_estimate']} T)")
    print("-" * 50)
    print(f"Measurements:         {metrics['measurements']}")
    print("=" * 50)
