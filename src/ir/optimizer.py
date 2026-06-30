from src.ir.ir_graph import EQIRGraph, EQIRNode
import math

class EQIROptimizer:
    def __init__(self):
        self.optimizations_count = 0

    def optimize(self, graph: EQIRGraph) -> EQIRGraph:
        """
        Runs optimization passes using a worklist-based rewrite loop with loop guards.
        """
        try:
            import eigen_native
            dict_data = graph.to_dict()
            opt_dict = eigen_native.optimize_eqir_native(dict_data)
            
            # Reconstruct optimized graph
            opt_graph = EQIRGraph.from_dict(opt_dict)
            
            # Mutate in-place
            graph.nodes = opt_graph.nodes
            graph.next_node_id = opt_graph.next_node_id
            graph.qubit_last_writer = opt_graph.qubit_last_writer
            graph.cbit_last_writer = opt_graph.cbit_last_writer
            
            self.iterations_count = opt_dict.get("iterations_count", 0)
            self.optimizations_count = opt_dict.get("optimizations_count", 0)
            return graph
        except (ImportError, AttributeError):
            pass

        worklist = set(graph.nodes.keys())
        max_iterations = len(graph.nodes) * 5 + 1000
        iterations = 0
        
        self_inverse_gates = {"H", "X", "Y", "Z"}
        rotation_gates = {"RX", "RY", "RZ"}
        
        while worklist and iterations < max_iterations:
            node_id = worklist.pop()
            if node_id not in graph.nodes:
                continue
                
            node = graph.nodes[node_id]
            if node.type != 'GATE':
                continue
                
            iterations += 1
            
            # Rule 1: Self-inverse cancellation (High Priority)
            if node.gate_name in self_inverse_gates:
                target_qubit = node.targets[0]
                next_node = None
                for child in node.children:
                    if target_qubit in child.targets:
                        next_node = child
                        break
                if (next_node and next_node.id in graph.nodes and 
                    next_node.type == 'GATE' and 
                    next_node.gate_name == node.gate_name and 
                    next_node.targets[0] == target_qubit and 
                    next_node.condition == node.condition):
                    
                    affected = {p.id for p in node.parents} | {c.id for c in next_node.children}
                    self._cancel_nodes(graph, node, next_node)
                    worklist.update(affected)
                    self.optimizations_count += 1
                    continue
                    
            # Rule 2: Rotation merging (High Priority)
            if node.gate_name in rotation_gates:
                target_qubit = node.targets[0]
                next_node = None
                for child in node.children:
                    if target_qubit in child.targets:
                        next_node = child
                        break
                if (next_node and next_node.id in graph.nodes and 
                    next_node.type == 'GATE' and 
                    next_node.gate_name == node.gate_name and 
                    next_node.targets[0] == target_qubit and 
                    next_node.condition == node.condition):
                    
                    angle1 = node.args[0]
                    angle2 = next_node.args[0]
                    if not isinstance(angle1, (int, float)) or not isinstance(angle2, (int, float)):
                        continue
                    new_angle = (angle1 + angle2) % (2 * math.pi)
                    node.args[0] = new_angle
                    
                    affected = {p.id for p in next_node.parents} | {c.id for c in next_node.children} | {node.id}
                    self._bypass_node(graph, next_node)
                    worklist.update(affected)
                    self.optimizations_count += 1
                    continue
            
            # Rule 3: Dead gate elimination (Medium Priority)
            if node.gate_name in rotation_gates and len(node.args) > 0 and abs(node.args[0]) < 1e-9:
                affected = {p.id for p in node.parents} | {c.id for c in node.children}
                self._bypass_node(graph, node)
                worklist.update(affected)
                self.optimizations_count += 1
                continue
                
            # Rule 4: Peephole optimizations (H -> X/Z -> H) (Medium Priority)
            if node.gate_name == 'H':
                q = node.targets[0]
                n2 = None
                for child in node.children:
                    if child.targets and child.targets[0] == q:
                        n2 = child
                        break
                if n2 and n2.id in graph.nodes and n2.type == 'GATE' and n2.gate_name in ('X', 'Z'):
                    n3 = None
                    for child in n2.children:
                        if child.targets and child.targets[0] == q:
                            n3 = child
                            break
                    if n3 and n3.id in graph.nodes and n3.type == 'GATE' and n3.gate_name == 'H':
                        target_gate = 'Z' if n2.gate_name == 'X' else 'X'
                        n2.gate_name = target_gate
                        
                        affected = {p.id for p in node.parents} | {c.id for c in n3.children} | {n2.id}
                        self._bypass_node(graph, node)
                        self._bypass_node(graph, n3)
                        worklist.update(affected)
                        self.optimizations_count += 1
                        continue
                        
            # Rule 5: Peephole optimizations (S -> S -> Z  and  T -> T -> S) (Medium Priority)
            if node.gate_name in ('S', 'T'):
                q = node.targets[0]
                n2 = None
                for child in node.children:
                    if child.targets and child.targets[0] == q:
                        n2 = child
                        break
                if n2 and n2.id in graph.nodes and n2.type == 'GATE' and n2.gate_name == node.gate_name:
                    target_gate = 'Z' if node.gate_name == 'S' else 'S'
                    n2.gate_name = target_gate
                    
                    affected = {p.id for p in node.parents} | {c.id for c in n2.children} | {n2.id}
                    self._bypass_node(graph, node)
                    worklist.update(affected)
                    self.optimizations_count += 1
                    continue
                    
            # Rule 6: Commutation cancellation (Case 1: Z q0 -> CNOT q0, q1 -> Z q0) (Low Priority)
            if node.gate_name == 'Z':
                q0 = node.targets[0]
                n2 = None
                for child in node.children:
                    if child.targets and child.targets[0] == q0:
                        n2 = child
                        break
                if n2 and n2.id in graph.nodes and n2.type == 'GATE' and n2.gate_name == 'CNOT' and n2.targets[0] == q0:
                    n3 = None
                    for child in n2.children:
                        if child.targets and child.targets[0] == q0:
                            n3 = child
                            break
                    if n3 and n3.id in graph.nodes and n3.type == 'GATE' and n3.gate_name == 'Z' and n3.targets[0] == q0:
                        affected = {p.id for p in node.parents} | {c.id for c in n3.children} | {n2.id}
                        self._bypass_node(graph, node)
                        self._bypass_node(graph, n3)
                        worklist.update(affected)
                        self.optimizations_count += 1
                        continue
                        
            # Rule 7: Commutation cancellation (Case 2: X q1 -> CNOT q0, q1 -> X q1) (Low Priority)
            if node.gate_name == 'X':
                q1 = node.targets[0]
                n2 = None
                for child in node.children:
                    if child.targets and len(child.targets) > 1 and child.targets[1] == q1:
                        n2 = child
                        break
                if n2 and n2.id in graph.nodes and n2.type == 'GATE' and n2.gate_name == 'CNOT' and n2.targets[1] == q1:
                    n3 = None
                    for child in n2.children:
                        if child.targets and child.targets[0] == q1:
                            n3 = child
                            break
                    if n3 and n3.id in graph.nodes and n3.type == 'GATE' and n3.gate_name == 'X' and n3.targets[0] == q1:
                        affected = {p.id for p in node.parents} | {c.id for c in n3.children} | {n2.id}
                        self._bypass_node(graph, node)
                        self._bypass_node(graph, n3)
                        worklist.update(affected)
                        self.optimizations_count += 1
                        continue
                        
        self.iterations_count = iterations
        return graph

    def _cancel_nodes(self, graph: EQIRGraph, node1: EQIRNode, node2: EQIRNode):
        parents = list(node1.parents)
        children = list(node2.children)
        for parent in parents:
            parent.remove_child(node1)
            for child in children:
                parent.add_child(child)
        for child in children:
            node2.remove_child(child)
        node1.parents.clear()
        node1.children.clear()
        node2.parents.clear()
        node2.children.clear()
        if node1.id in graph.nodes:
            del graph.nodes[node1.id]
        if node2.id in graph.nodes:
            del graph.nodes[node2.id]

    def _bypass_node(self, graph: EQIRGraph, node: EQIRNode):
        parents = list(node.parents)
        children = list(node.children)
        for parent in parents:
            parent.remove_child(node)
            for child in children:
                parent.add_child(child)
        for child in children:
            node.remove_child(child)
        node.parents.clear()
        node.children.clear()
        if node.id in graph.nodes:
            del graph.nodes[node.id]

