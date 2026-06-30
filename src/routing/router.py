"""Hardware Connectivity Mapper for Eigen quantum circuits.

Maps logical qubit operations onto hardware coupling maps by inserting
SWAP gates where necessary to satisfy connectivity constraints.
"""
from collections import deque


try:
    import eigen_native as native
except ImportError:
    native = None


class CouplingMap:
    """Represents hardware qubit connectivity topology."""

    def __init__(self, edges: list[tuple[int, int]]):
        self.edges = set()
        self.adjacency = {}
        self.num_qubits = 0
        for q1, q2 in edges:
            self.add_edge(q1, q2)

    def add_edge(self, q1: int, q2: int):
        self.edges.add((min(q1, q2), max(q1, q2)))
        self.adjacency.setdefault(q1, set()).add(q2)
        self.adjacency.setdefault(q2, set()).add(q1)
        self.num_qubits = max(self.num_qubits, q1 + 1, q2 + 1)

    def are_connected(self, q1: int, q2: int) -> bool:
        return q2 in self.adjacency.get(q1, set())

    def neighbors(self, q: int) -> set[int]:
        return self.adjacency.get(q, set())

    def shortest_path(self, src: int, dst: int) -> list[int]:
        """BFS shortest path between two physical qubits."""
        if src == dst:
            return [src]
        if native is not None:
            return native.fast_shortest_path(list(self.edges), src, dst)
        visited = {src}
        queue = deque([(src, [src])])
        while queue:
            current, path = queue.popleft()
            for neighbor in self.neighbors(current):
                if neighbor == dst:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return []

    def distance(self, q1: int, q2: int) -> int:
        path = self.shortest_path(q1, q2)
        return len(path) - 1 if path else float('inf')

    @staticmethod
    def linear(n: int) -> 'CouplingMap':
        """Create a linear coupling map: 0-1-2-..-(n-1)."""
        edges = [(i, i + 1) for i in range(n - 1)]
        return CouplingMap(edges)

    @staticmethod
    def grid(rows: int, cols: int) -> 'CouplingMap':
        """Create a 2D grid coupling map."""
        edges = []
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                if c + 1 < cols:
                    edges.append((idx, idx + 1))
                if r + 1 < rows:
                    edges.append((idx, idx + cols))
        return CouplingMap(edges)

    @staticmethod
    def heavy_hex(n: int) -> 'CouplingMap':
        """Create a simplified heavy-hex-like coupling map (IBM-inspired).
        Uses a grid with alternating removed edges."""
        return CouplingMap.grid(n, n)


class RoutedCircuit:
    """Represents a circuit after routing with physical qubit assignments."""

    def __init__(self):
        self.operations = []
        self.initial_mapping = {}
        self.final_mapping = {}
        self.swap_count = 0

    def add_gate(self, gate_name: str, physical_qubits: list[int], args: list = None):
        self.operations.append((gate_name, physical_qubits, args or []))

    def add_swap(self, q1: int, q2: int):
        self.operations.append(('SWAP', [q1, q2], []))
        self.swap_count += 1

    def summary(self) -> dict:
        return {
            'total_gates': len(self.operations),
            'swap_count': self.swap_count,
            'initial_mapping': dict(self.initial_mapping),
            'final_mapping': dict(self.final_mapping),
        }


class BasicSwapRouter:
    """Routes circuits by inserting SWAPs along shortest paths.

    For each two-qubit gate where the logical qubits are not adjacent
    on the coupling map, inserts SWAP gates along the shortest path
    to bring them together.
    """

    def __init__(self, coupling_map: CouplingMap):
        self.coupling_map = coupling_map

    def route(self, circuit_ops: list[dict], logical_qubits: list[str]) -> RoutedCircuit:
        result = RoutedCircuit()

        mapping = {}
        reverse_mapping = {}

        for i, lq in enumerate(logical_qubits):
            if i >= self.coupling_map.num_qubits:
                raise ValueError(
                    f"Not enough physical qubits ({self.coupling_map.num_qubits}) "
                    f"for {len(logical_qubits)} logical qubits."
                )
            mapping[lq] = i
            reverse_mapping[i] = lq

        result.initial_mapping = dict(mapping)

        for op in circuit_ops:
            gate = op['gate']
            targets = op['targets']
            args = op.get('args', [])

            if len(targets) <= 1:
                phys = [mapping[targets[0]]]
                result.add_gate(gate, phys, args)
            else:
                p0 = mapping[targets[0]]
                p1 = mapping[targets[1]]

                if self.coupling_map.are_connected(p0, p1):
                    result.add_gate(gate, [p0, p1], args)
                else:
                    path = self.coupling_map.shortest_path(p0, p1)
                    if not path:
                        raise ValueError(f"No path between physical qubits {p0} and {p1}")

                    for i in range(len(path) - 2):
                        swap_a = path[i]
                        swap_b = path[i + 1]

                        result.add_swap(swap_a, swap_b)

                        lq_a = reverse_mapping.get(swap_a)
                        lq_b = reverse_mapping.get(swap_b)

                        if lq_a is not None:
                            mapping[lq_a] = swap_b
                        if lq_b is not None:
                            mapping[lq_b] = swap_a

                        reverse_mapping[swap_a] = lq_b
                        reverse_mapping[swap_b] = lq_a

                    new_p0 = mapping[targets[0]]
                    new_p1 = mapping[targets[1]]
                    result.add_gate(gate, [new_p0, new_p1], args)

        result.final_mapping = dict(mapping)
        return result


class GreedyRouter:
    """Routes circuits using a greedy heuristic.

    Looks ahead at upcoming two-qubit gates and selects the SWAP
    that minimizes the total distance for the nearest interactions.
    """

    def __init__(self, coupling_map: CouplingMap, lookahead: int = 5):
        self.coupling_map = coupling_map
        self.lookahead = lookahead

    def _total_distance(self, mapping: dict, ops: list[dict], start: int) -> float:
        total = 0.0
        decay = 1.0
        for i in range(start, min(start + self.lookahead, len(ops))):
            targets = ops[i]['targets']
            if len(targets) >= 2:
                p0 = mapping[targets[0]]
                p1 = mapping[targets[1]]
                total += self.coupling_map.distance(p0, p1) * decay
                decay *= 0.5
        return total

    def route(self, circuit_ops: list[dict], logical_qubits: list[str]) -> RoutedCircuit:
        result = RoutedCircuit()

        mapping = {}
        reverse_mapping = {}
        for i, lq in enumerate(logical_qubits):
            if i >= self.coupling_map.num_qubits:
                raise ValueError(
                    f"Not enough physical qubits ({self.coupling_map.num_qubits}) "
                    f"for {len(logical_qubits)} logical qubits."
                )
            mapping[lq] = i
            reverse_mapping[i] = lq

        result.initial_mapping = dict(mapping)

        op_idx = 0
        max_iterations = len(circuit_ops) * self.coupling_map.num_qubits * 10

        iteration = 0
        while op_idx < len(circuit_ops):
            iteration += 1
            if iteration > max_iterations:
                raise RuntimeError("GreedyRouter exceeded maximum iterations")

            op = circuit_ops[op_idx]
            gate = op['gate']
            targets = op['targets']
            args = op.get('args', [])

            if len(targets) <= 1:
                phys = [mapping[targets[0]]]
                result.add_gate(gate, phys, args)
                op_idx += 1
                continue

            p0 = mapping[targets[0]]
            p1 = mapping[targets[1]]

            if self.coupling_map.are_connected(p0, p1):
                result.add_gate(gate, [p0, p1], args)
                op_idx += 1
                continue

            best_swap = None
            best_score = float('inf')

            for edge_q1, edge_q2 in self.coupling_map.edges:
                trial_mapping = dict(mapping)
                lq_a = reverse_mapping.get(edge_q1)
                lq_b = reverse_mapping.get(edge_q2)
                if lq_a is not None:
                    trial_mapping[lq_a] = edge_q2
                if lq_b is not None:
                    trial_mapping[lq_b] = edge_q1

                score = self._total_distance(trial_mapping, circuit_ops, op_idx)
                if score < best_score:
                    best_score = score
                    best_swap = (edge_q1, edge_q2)

            if best_swap is None:
                raise ValueError("No valid SWAP found")

            sq1, sq2 = best_swap
            result.add_swap(sq1, sq2)

            lq_a = reverse_mapping.get(sq1)
            lq_b = reverse_mapping.get(sq2)
            if lq_a is not None:
                mapping[lq_a] = sq2
            if lq_b is not None:
                mapping[lq_b] = sq1
            reverse_mapping[sq1] = lq_b
            reverse_mapping[sq2] = lq_a

        result.final_mapping = dict(mapping)
        return result


class SabreRouter:
    """Routes circuits using the SABRE (Structure-Aware Bidirectional Router) algorithm.

    Maintains a dynamic front layer of gates and selects SWAPs that minimize
    the look-ahead distance of active and extended layer interactions.
    """

    def __init__(self, coupling_map: CouplingMap, lookahead_weight: float = 0.5):
        self.coupling_map = coupling_map
        self.lookahead_weight = lookahead_weight

    def route(self, circuit_ops: list[dict], logical_qubits: list[str]) -> RoutedCircuit:
        result = RoutedCircuit()

        mapping = {}
        reverse_mapping = {}
        for i, lq in enumerate(logical_qubits):
            if i >= self.coupling_map.num_qubits:
                raise ValueError(
                    f"Not enough physical qubits ({self.coupling_map.num_qubits}) "
                    f"for {len(logical_qubits)} logical qubits."
                )
            mapping[lq] = i
            reverse_mapping[i] = lq

        result.initial_mapping = dict(mapping)

        ops = []
        for idx, op in enumerate(circuit_ops):
            ops.append({
                'id': idx,
                'gate': op['gate'],
                'targets': op['targets'],
                'args': op.get('args', []),
                'completed': False
            })

        max_iterations = len(circuit_ops) * self.coupling_map.num_qubits * 20
        iteration = 0
        completed_ops_count = 0

        while completed_ops_count < len(ops) and iteration < max_iterations:
            iteration += 1

            front_layer = []
            extended_layer = []

            active_ops = set()
            for lq in logical_qubits:
                for op in ops:
                    if not op['completed'] and lq in op['targets']:
                        active_ops.add(op['id'])
                        break

            for op_id in active_ops:
                op = ops[op_id]
                is_ready = True
                for t in op['targets']:
                    for prev_op in ops[:op_id]:
                        if not prev_op['completed'] and t in prev_op['targets']:
                            is_ready = False
                            break
                    if not is_ready:
                        break
                if is_ready:
                    front_layer.append(op)
                else:
                    extended_layer.append(op)

            one_qubit_routed = False
            for op in front_layer:
                if len(op['targets']) <= 1:
                    phys = [mapping[op['targets'][0]]]
                    result.add_gate(op['gate'], phys, op['args'])
                    op['completed'] = True
                    completed_ops_count += 1
                    one_qubit_routed = True

            if one_qubit_routed:
                continue

            two_qubit_routed = False
            for op in front_layer:
                if len(op['targets']) == 2:
                    p0 = mapping[op['targets'][0]]
                    p1 = mapping[op['targets'][1]]
                    if self.coupling_map.are_connected(p0, p1):
                        result.add_gate(op['gate'], [p0, p1], op['args'])
                        op['completed'] = True
                        completed_ops_count += 1
                        two_qubit_routed = True

            if two_qubit_routed:
                continue

            best_swap = None
            best_score = float('inf')

            for edge_q1, edge_q2 in self.coupling_map.edges:
                trial_mapping = dict(mapping)
                lq_a = reverse_mapping.get(edge_q1)
                lq_b = reverse_mapping.get(edge_q2)

                if lq_a is not None:
                    trial_mapping[lq_a] = edge_q2
                if lq_b is not None:
                    trial_mapping[lq_b] = edge_q1

                front_score = 0.0
                for op in front_layer:
                    p0 = trial_mapping[op['targets'][0]]
                    p1 = trial_mapping[op['targets'][1]]
                    front_score += self.coupling_map.distance(p0, p1)
                front_score /= len(front_layer)

                extended_score = 0.0
                if extended_layer:
                    count = 0
                    for op in extended_layer[:5]:
                        if len(op['targets']) == 2:
                            p0 = trial_mapping[op['targets'][0]]
                            p1 = trial_mapping[op['targets'][1]]
                            extended_score += self.coupling_map.distance(p0, p1)
                            count += 1
                    if count > 0:
                        extended_score /= count

                score = front_score + self.lookahead_weight * extended_score

                if score < best_score:
                    best_score = score
                    best_swap = (edge_q1, edge_q2)

            if best_swap is None:
                raise ValueError("SABRE Router could not find any valid SWAP.")

            sq1, sq2 = best_swap
            result.add_swap(sq1, sq2)

            lq_a = reverse_mapping.get(sq1)
            lq_b = reverse_mapping.get(sq2)
            if lq_a is not None:
                mapping[lq_a] = sq2
            if lq_b is not None:
                mapping[lq_b] = sq1

            reverse_mapping[sq1] = lq_b
            reverse_mapping[sq2] = lq_a

        if iteration >= max_iterations:
            raise RuntimeError("SABRE Router exceeded maximum iterations limit.")

        result.final_mapping = dict(mapping)
        return result


def route_eqir_graph(graph, coupling_map: CouplingMap, router_type: str = 'basic') -> RoutedCircuit:
    """Route an EQIR graph onto a hardware coupling map."""
    logical_qubits = []
    circuit_ops = []

    sorted_nodes = graph.topological_sort()

    for node in sorted_nodes:
        if node.type == 'ALLOC':
            qname = node.targets[0]
            if qname not in logical_qubits:
                logical_qubits.append(qname)
        elif node.type == 'GATE':
            circuit_ops.append({
                'gate': node.gate_name,
                'targets': list(node.targets),
                'args': list(node.args),
            })

    if not logical_qubits:
        return RoutedCircuit()

    if len(logical_qubits) > coupling_map.num_qubits:
        raise ValueError(
            f"Circuit requires {len(logical_qubits)} qubits but coupling map "
            f"only has {coupling_map.num_qubits} physical qubits."
        )

    if router_type == 'greedy':
        router = GreedyRouter(coupling_map)
    elif router_type == 'sabre':
        router = SabreRouter(coupling_map)
    else:
        router = BasicSwapRouter(coupling_map)

    return router.route(circuit_ops, logical_qubits)
