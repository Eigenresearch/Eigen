"""Simulator backend adapter for the unified backend interface."""

from __future__ import annotations

from src.backend.unified_backend import QuantumBackend, ValidationReport, ExecutionResult
from src.ir.ir_graph import EQIRGraph
from src.semantic.backend_capabilities import BackendCapabilities, get_backend_capabilities
from src.simulator import QuantumSimulator

class SimulatorBackend(QuantumBackend):
    """Adapter for the local Eigen QuantumSimulator."""

    def __init__(self, name: str = "simulator"):
        self.name = name

    def capabilities(self) -> BackendCapabilities:
        return get_backend_capabilities("dense")  # Local simulator is highly capable

    def validate(self, graph: EQIRGraph, ast=None) -> ValidationReport:
        # Local simulator supports almost everything through fallbacks.
        from src.backend.unified_backend import _detect_capabilities_from_graph
        return _detect_capabilities_from_graph(graph, self.capabilities(), self.name, ast)

    def compile(self, graph: EQIRGraph, ast=None, **kwargs) -> EQIRGraph:
        # Simulator takes the graph directly.
        return graph

    def execute(self, native: EQIRGraph, shots: int = 1024, **kwargs) -> ExecutionResult:
        sim_type = kwargs.get("sim_type", "auto")
        gpu_platform = kwargs.get("gpu_platform", "none")
        
        sim = QuantumSimulator(sim_type=sim_type, gpu_platform=gpu_platform)
        
        # Simple graph-to-simulator runner
        # This is a simplified version of what EigenRuntime.execute does.
        # For a full implementation, we'd need to handle all IR node types.
        
        # We simulate shot-by-shot if there are mid-circuit measurements,
        # but for simple circuits we can do a single pass if shots=1 or no measure.
        # Here we just provide a basic implementation.
        
        histograms = []
        for _ in range(shots):
            counts = {}
            # Reset simulator or create new one?
            sim = QuantumSimulator(sim_type=sim_type, gpu_platform=gpu_platform)
            for node in native.nodes.values():
                if node.type == "ALLOC":
                    sim.allocate_qubit(node.qubit_name)
                elif node.type == "GATE":
                    getattr(sim, node.gate_name)(*node.targets, *node.args)
                elif node.type == "MEASURE":
                    res = sim.measure(node.qubit_name)
                    counts[node.qubit_name] = res
            histograms.append(counts)
            
        # Aggregate bitstrings
        # In Eigen, bitstrings are usually concatenated qubit results.
        # This implementation is just a placeholder to satisfy the API.
        
        return ExecutionResult(
            backend_name=self.name,
            native_handle=native,
            shots=shots,
            histograms=histograms,
            metadata={"sim_type": sim_type}
        )
