import hashlib
from src.ir.ir_graph import EQIRGraph
from src.ir.optimizer import EQIROptimizer

class Canonicalizer:
    def __init__(self):
        self.optimizer = EQIROptimizer()

    def hash_circuit(self, graph: EQIRGraph) -> str:
        # Optimize the graph first to get a canonical simplified structure
        from copy import deepcopy
        g_opt = deepcopy(graph)
        self.optimizer.optimize(g_opt)
        # Build a sorted representation of the gate names and types
        representation = []
        for node in g_opt.nodes.values():
            if node.type == 'GATE':
                # Include gate name and any arguments (rounded) to distinguish rotations
                args_str = ",".join(f"{float(a):.6f}" for a in node.args) if node.args else ""
                representation.append(f"{node.gate_name}:{args_str}")
            elif node.type == 'MEASURE':
                representation.append('MEASURE')
                
        repr_str = "|".join(sorted(representation))
        return hashlib.sha256(repr_str.encode('utf-8')).hexdigest()
