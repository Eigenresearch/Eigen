import time
from src.ir_graph import EQIRGraph

class EQIRProfiler:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.execution_time_ms = 0.0

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        self.end_time = time.perf_counter()
        if self.start_time:
            self.execution_time_ms = (self.end_time - self.start_time) * 1000.0

    def profile(self, graph: EQIRGraph) -> dict:
        gates_count = 0
        gate_types_count = {}
        entangling_gates_count = 0
        
        entangling_types = {"CNOT", "CZ", "SWAP"}
        rotation_types = {"RX", "RY", "RZ"}
        
        qubits = set()
        
        for node in graph.nodes.values():
            if node.type == 'ALLOC':
                qubits.add(node.targets[0])
            elif node.type == 'GATE':
                gates_count += 1
                g_name = node.gate_name
                gate_types_count[g_name] = gate_types_count.get(g_name, 0) + 1
                
                # Check if it's an entangling gate
                if g_name in entangling_types:
                    entangling_gates_count += 1
                    
                # Track qubits
                for t in node.targets:
                    qubits.add(t)
            elif node.type == 'MEASURE':
                qubits.add(node.targets[0])

        active_qubit_count = len(qubits)
        state_vector_size = 2 ** active_qubit_count if active_qubit_count > 0 else 0
        circuit_depth = graph.compute_depth()

        return {
            "execution_time_ms": self.execution_time_ms,
            "total_gates": gates_count,
            "gate_breakdown": gate_types_count,
            "entangling_gates": entangling_gates_count,
            "active_qubits": active_qubit_count,
            "circuit_depth": circuit_depth,
            "state_vector_size": state_vector_size,
        }

    def print_profile_report(self, stats: dict):
        print("=" * 40)
        print("          EIGEN RUNTIME PROFILE          ")
        print("=" * 40)
        print(f"Execution Time:      {stats['execution_time_ms']:.3f} ms")
        print(f"Active Qubits:       {stats['active_qubits']}")
        print(f"Circuit Depth:       {stats['circuit_depth']}")
        print(f"Total Gates:         {stats['total_gates']}")
        print(f"  Entangling Gates:  {stats['entangling_gates']}")
        print(f"State Vector Size:   {stats['state_vector_size']} complex amplitudes")
        if stats['gate_breakdown']:
            print("Gate Breakdown:")
            for g_name, count in sorted(stats['gate_breakdown'].items()):
                print(f"  {g_name}: {count}")
        print("=" * 40)
