from src.ir_graph import EQIRGraph, EQIRNode

class EQIROptimizer:
    def __init__(self):
        self.optimizations_count = 0

    def optimize(self, graph: EQIRGraph) -> EQIRGraph:
        """
        Runs optimization passes repeatedly until no more improvements are made (fixed-point).
        """
        while True:
            improved = False
            # Run self-inverse cancellation pass
            if self._pass_cancel_self_inverse(graph):
                improved = True
            # Run rotation merging pass
            if self._pass_merge_rotations(graph):
                improved = True
            
            if not improved:
                break
        return graph

    def _pass_cancel_self_inverse(self, graph: EQIRGraph) -> bool:
        nodes = graph.topological_sort()
        self_inverse_gates = {"H", "X", "Y", "Z"}

        for node in nodes:
            if node.id not in graph.nodes:
                continue  # Already deleted
                
            if node.type == 'GATE' and node.gate_name in self_inverse_gates:
                # Since it's a 1-qubit gate, it has 1 target qubit
                target_qubit = node.targets[0]
                
                # Find the next node that touches this target qubit
                # The next node is a child of the current node
                next_node = None
                for child in node.children:
                    if target_qubit in child.targets:
                        # Since it's the immediate next operation on this qubit wire,
                        # this child must have this target qubit
                        next_node = child
                        break
                
                if next_node is not None:
                    # Check if next_node is a gate, has the same gate name, same target, and same condition
                    if (next_node.type == 'GATE' and 
                        next_node.gate_name == node.gate_name and 
                        next_node.targets[0] == target_qubit and 
                        next_node.condition == node.condition):
                        
                        # We can cancel node and next_node!
                        self._cancel_nodes(graph, node, next_node)
                        self.optimizations_count += 1
                        return True
        return False

    def _pass_merge_rotations(self, graph: EQIRGraph) -> bool:
        nodes = graph.topological_sort()
        rotation_gates = {"RX", "RY", "RZ"}

        for node in nodes:
            if node.id not in graph.nodes:
                continue  # Already deleted
                
            if node.type == 'GATE' and node.gate_name in rotation_gates:
                target_qubit = node.targets[0]
                
                # Find next node touching this qubit
                next_node = None
                for child in node.children:
                    if target_qubit in child.targets:
                        next_node = child
                        break
                
                if next_node is not None:
                    if (next_node.type == 'GATE' and 
                        next_node.gate_name == node.gate_name and 
                        next_node.targets[0] == target_qubit and 
                        next_node.condition == node.condition):
                        
                        # Merge next_node into node
                        angle1 = node.args[0]
                        angle2 = next_node.args[0]
                        new_angle = (angle1 + angle2) % (2 * 3.141592653589793)  # Keep in [0, 2PI)
                        
                        # Update current node's angle
                        node.args[0] = new_angle
                        
                        # Bypass and remove next_node
                        self._bypass_node(graph, next_node)
                        self.optimizations_count += 1
                        return True
        return False

    def _cancel_nodes(self, graph: EQIRGraph, node1: EQIRNode, node2: EQIRNode):
        # We need to remove node1 and node2 (where node1 -> node2)
        # Connect parents of node1 to children of node2
        parents = list(node1.parents)
        children = list(node2.children)
        
        for parent in parents:
            # Disconnect parent from node1
            parent.remove_child(node1)
            # Connect parent to children of node2
            for child in children:
                parent.add_child(child)
                
        # For children of node2, disconnect them from node2
        for child in children:
            node2.remove_child(child)
            
        # Clean up node1 and node2 references
        node1.parents.clear()
        node1.children.clear()
        node2.parents.clear()
        node2.children.clear()
        
        # Remove from graph
        if node1.id in graph.nodes:
            del graph.nodes[node1.id]
        if node2.id in graph.nodes:
            del graph.nodes[node2.id]

    def _bypass_node(self, graph: EQIRGraph, node: EQIRNode):
        # Connect parents of node to children of node
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
