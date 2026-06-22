class EQIRNode:
    def __init__(self, node_id: int, node_type: str, **kwargs):
        self.id = node_id
        self.type = node_type  # 'ALLOC', 'GATE', 'MEASURE', 'TRACE', 'PRINT', 'ASSERT'
        self.gate_name = kwargs.get('gate_name')       # e.g., 'H', 'CNOT', 'RX'
        self.targets = kwargs.get('targets', [])       # list of qubit names
        self.args = kwargs.get('args', [])             # list of evaluated float/int
        self.cbit_name = kwargs.get('cbit_name')       # for MEASURE
        self.condition = kwargs.get('condition')       # None or (cbit_name, op, val)
        self.print_expr = kwargs.get('print_expr')     # ASTNode or evaluated value
        self.assert_cond = kwargs.get('assert_cond')   # None or (left_ref, op, right_val)
        
        self.parents = set()   # set of EQIRNode
        self.children = set()  # set of EQIRNode

    def add_child(self, child_node: 'EQIRNode'):
        self.children.add(child_node)
        child_node.parents.add(self)

    def remove_child(self, child_node: 'EQIRNode'):
        if child_node in self.children:
            self.children.remove(child_node)
        if self in child_node.parents:
            child_node.parents.remove(self)

    def __repr__(self):
        cond_str = f" if {self.condition[0]}{self.condition[1]}{self.condition[2]}" if self.condition else ""
        if self.type == 'ALLOC':
            return f"[{self.id}] ALLOC {self.targets[0]}"
        elif self.type == 'GATE':
            args_str = f"({', '.join(map(str, self.args))})" if self.args else ""
            targets_str = ", ".join(self.targets)
            return f"[{self.id}] {self.gate_name}{args_str} {targets_str}{cond_str}"
        elif self.type == 'MEASURE':
            return f"[{self.id}] MEASURE {self.targets[0]} -> {self.cbit_name}{cond_str}"
        elif self.type == 'TRACE':
            return f"[{self.id}] TRACE{cond_str}"
        elif self.type == 'PRINT':
            return f"[{self.id}] PRINT {self.print_expr}{cond_str}"
        elif self.type == 'ASSERT':
            return f"[{self.id}] ASSERT {self.assert_cond[0]} {self.assert_cond[1]} {self.assert_cond[2]}{cond_str}"
        return f"[{self.id}] {self.type}{cond_str}"


class EQIRGraph:
    def __init__(self):
        self.nodes = {}  # node_id -> EQIRNode
        self.next_node_id = 0
        self.qubit_last_writer = {}  # qubit_name -> EQIRNode (last operation that touched this qubit)
        self.cbit_last_writer = {}   # cbit_name -> EQIRNode (last operation that wrote/read this cbit)

    def create_node(self, node_type: str, **kwargs) -> EQIRNode:
        node_id = self.next_node_id
        self.next_node_id += 1
        node = EQIRNode(node_id, node_type, **kwargs)
        self.nodes[node_id] = node
        return node

    def add_operation(self, node_type: str, **kwargs) -> EQIRNode:
        node = self.create_node(node_type, **kwargs)
        
        # Build dependency edges based on qubits touched
        if node.type == 'TRACE':
            # TRACE is a barrier: depends on the last operation of all active qubits
            for qubit in list(self.qubit_last_writer.keys()):
                parent = self.qubit_last_writer[qubit]
                parent.add_child(node)
                self.qubit_last_writer[qubit] = node
        else:
            for qubit in node.targets:
                if qubit in self.qubit_last_writer:
                    parent = self.qubit_last_writer[qubit]
                    parent.add_child(node)
                self.qubit_last_writer[qubit] = node

        # Build dependency edges based on classical bits read/written
        cbit_deps = []
        if node.cbit_name:
            cbit_deps.append(node.cbit_name)
        if node.condition:
            cbit_deps.append(node.condition[0])
            
        if node.type == 'PRINT':
            if isinstance(node.print_expr, str):
                cbit_deps.append(node.print_expr)
                
        if node.type == 'ASSERT' and node.assert_cond:
            left, op, right = node.assert_cond
            if isinstance(left, str):
                cbit_deps.append(left)
            if isinstance(right, str):
                cbit_deps.append(right)

        for cbit in cbit_deps:
            if cbit in self.cbit_last_writer:
                parent = self.cbit_last_writer[cbit]
                parent.add_child(node)
            self.cbit_last_writer[cbit] = node

        return node

    def get_sources(self) -> list[EQIRNode]:
        """Returns all nodes that have no parents."""
        return [node for node in self.nodes.values() if not node.parents]

    def get_sinks(self) -> list[EQIRNode]:
        """Returns all nodes that have no children."""
        return [node for node in self.nodes.values() if not node.children]

    def topological_sort(self) -> list[EQIRNode]:
        """Returns nodes in topologically sorted order."""
        visited = set()
        stack = []

        def dfs(node):
            visited.add(node.id)
            for child in sorted(node.children, key=lambda n: n.id):
                if child.id not in visited:
                    dfs(child)
            stack.append(node)

        # Sort sources by ID to be deterministic
        for source in sorted(self.get_sources(), key=lambda n: n.id):
            if source.id not in visited:
                dfs(source)

        return list(reversed(stack))

    def compute_depth(self) -> int:
        """
        Computes the quantum depth of the circuit (length of longest path of quantum gates).
        """
        # We only count GATE and MEASURE nodes in quantum depth. ALLOC nodes have depth 0.
        memo = {}

        def get_node_depth(node: EQIRNode) -> int:
            if node.id in memo:
                return memo[node.id]
            
            # Depth of current node
            self_weight = 1 if node.type in ('GATE', 'MEASURE') else 0
            
            max_parent_depth = 0
            for parent in node.parents:
                max_parent_depth = max(max_parent_depth, get_node_depth(parent))
                
            memo[node.id] = max_parent_depth + self_weight
            return memo[node.id]

        max_depth = 0
        for sink in self.get_sinks():
            max_depth = max(max_depth, get_node_depth(sink))
        return max_depth

    def print_graph(self):
        """Prints a human-readable list of nodes in topological order with their connections."""
        for node in self.topological_sort():
            parent_ids = [p.id for p in sorted(node.parents, key=lambda n: n.id)]
            child_ids = [c.id for c in sorted(node.children, key=lambda n: n.id)]
            print(f"Node {node} | Parents: {parent_ids} | Children: {child_ids}")

    def to_dict(self) -> dict:
        nodes_data = []
        for n in self.nodes.values():
            nodes_data.append({
                "id": n.id,
                "type": n.type,
                "gate_name": n.gate_name,
                "targets": n.targets,
                "args": n.args,
                "cbit_name": n.cbit_name,
                "condition": n.condition,
                "print_expr": n.print_expr,
                "assert_cond": n.assert_cond,
                "children_ids": [c.id for c in n.children]
            })
        return {
            "next_node_id": self.next_node_id,
            "nodes": nodes_data
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EQIRGraph":
        graph = cls()
        graph.next_node_id = data["next_node_id"]
        
        # Create all nodes
        nodes_map = {}
        for nd in data["nodes"]:
            node = EQIRNode(
                nd["id"], nd["type"],
                gate_name=nd["gate_name"],
                targets=nd["targets"],
                args=nd["args"],
                cbit_name=nd["cbit_name"],
                condition=nd["condition"],
                print_expr=nd["print_expr"],
                assert_cond=nd["assert_cond"]
            )
            graph.nodes[nd["id"]] = node
            nodes_map[nd["id"]] = (node, nd["children_ids"])
            
        # Re-establish parent-child links
        for nid, (node, children_ids) in nodes_map.items():
            for cid in children_ids:
                if cid in graph.nodes:
                    node.add_child(graph.nodes[cid])
                    
        return graph
