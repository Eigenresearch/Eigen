import hashlib
from copy import deepcopy
from src.ir.ir_graph import EQIRGraph
from src.ir.optimizer import EQIROptimizer

class Canonicalizer:
    def __init__(self):
        self.optimizer = EQIROptimizer()

    def hash_circuit(self, graph: EQIRGraph) -> str:
        g_opt = deepcopy(graph)
        self.optimizer.optimize(g_opt)
        
        representation = []
        for node in g_opt.topological_sort():
            if node.type == 'GATE':
                args_str = ",".join(f"{float(a):.6f}" if isinstance(a, (int, float)) else str(a) for a in node.args) if node.args else ""
                targets_str = ",".join(node.targets)
                cond_str = f" if {node.condition[0]}{node.condition[1]}{node.condition[2]}" if node.condition else ""
                representation.append(f"GATE:{node.gate_name}:{targets_str}:{args_str}{cond_str}")
            elif node.type == 'MEASURE':
                targets_str = ",".join(node.targets)
                cond_str = f" if {node.condition[0]}{node.condition[1]}{node.condition[2]}" if node.condition else ""
                representation.append(f"MEASURE:{targets_str}:{node.cbit_name}{cond_str}")
            elif node.type == 'ALLOC':
                targets_str = ",".join(node.targets)
                representation.append(f"ALLOC:{targets_str}")
            elif node.type == 'PRINT':
                cond_str = f" if {node.condition[0]}{node.condition[1]}{node.condition[2]}" if node.condition else ""
                representation.append(f"PRINT:{node.print_expr}{cond_str}")
            elif node.type == 'ASSERT':
                cond_str = f" if {node.condition[0]}{node.condition[1]}{node.condition[2]}" if node.condition else ""
                representation.append(f"ASSERT:{node.assert_cond[0]}{node.assert_cond[1]}{node.assert_cond[2]}{cond_str}")
            elif node.type == 'TRACE':
                cond_str = f" if {node.condition[0]}{node.condition[1]}{node.condition[2]}" if node.condition else ""
                representation.append(f"TRACE{cond_str}")
                
        repr_str = "|".join(representation)
        return hashlib.sha256(repr_str.encode('utf-8')).hexdigest()
