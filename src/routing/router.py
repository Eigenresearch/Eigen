"""Hardware Connectivity Mapper for Eigen quantum circuits.

Maps logical qubit operations onto hardware coupling maps by inserting
SWAP gates where necessary to satisfy connectivity constraints.
"""
from collections import deque

DEFAULT_LOOKAHEAD = 5
DEFAULT_LOOKAHEAD_WEIGHT = 0.5


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
    def heavy_hex(d: int) -> 'CouplingMap':
        """Create a real IBM heavy-hex coupling map with d qubits per side.

        The heavy-hex topology is IBM's flagship qubit topology (used in Eagle,
        Heron processors). It consists of staggered hexagonal rings where each
        hexagon shares edges with its neighbours, giving every qubit at most
        3 neighbours — the "heavy" part comes from the fact that two sides of
        each hexagon have an extra (anchor) qubit, making them "heavier" than
        a simple hexagonal lattice.

        Parameter d is the number of qubits along one side of the hexagonal
        pattern. The total qubit count is approximately d * (2*d - 1).

        For a concrete IBM device, prefer ibm_eagle() or ibm_condor().
        """
        if d < 2:
            raise ValueError("heavy_hex requires d >= 2")

        edges = []
        num_qubits = 0

        rows = d
        cols = 2 * d - 1

        for r in range(rows):
            for c in range(cols):
                q = r * cols + c
                if q + 1 > num_qubits:
                    num_qubits = q + 1

                if c + 1 < cols:
                    if r % 2 == 0:
                        if c % 2 == 0:
                            edges.append((q, q + 1))
                    else:
                        if c % 2 == 1:
                            edges.append((q, q + 1))

                if r + 1 < rows:
                    if c % 2 == 0:
                        edges.append((q, q + cols))

        cm = CouplingMap(edges)

        anchors = []
        for r in range(rows - 1):
            for c in range(cols):
                q = r * cols + c
                q_below = (r + 1) * cols + c
                if c % 2 == 1 and r % 2 == 0:
                    anchor = num_qubits + len(anchors)
                    anchors.append(anchor)
                    cm.add_edge(q, anchor)
                    cm.add_edge(anchor, q_below)

        return cm

    @staticmethod
    def ibm_eagle() -> 'CouplingMap':
        """IBM Eagle processor topology (127 qubits).

        Used in ibm_sherbrooke, ibm_brisbane, ibm_kyiv, etc.
        Based on the heavy-hex architecture with a single ring of 7x7 hexagons.
        """
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),
            (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13),
            (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 19),
            (19, 20), (20, 21), (21, 22), (22, 23), (23, 24), (24, 25),
            (25, 26), (26, 27), (27, 28), (28, 29), (29, 30), (30, 31),
            (31, 32), (32, 33), (33, 34), (34, 35), (35, 36), (36, 37),
            (1, 38), (38, 39), (39, 40), (4, 41), (41, 42), (42, 43),
            (7, 44), (44, 45), (45, 46), (10, 47), (47, 48), (48, 49),
            (13, 50), (50, 51), (51, 52), (16, 53), (53, 54), (54, 55),
            (19, 56), (56, 57), (57, 58), (22, 59), (59, 60), (60, 61),
            (25, 62), (62, 63), (63, 64), (28, 65), (65, 66), (66, 67),
            (31, 68), (68, 69), (69, 70), (34, 71), (71, 72), (72, 73),
            (37, 74), (74, 75), (75, 76),
            (40, 77), (43, 77), (46, 78), (49, 78), (52, 79), (55, 79),
            (58, 80), (61, 80), (64, 81), (67, 81), (70, 82), (73, 82),
            (76, 83),
            (77, 84), (78, 85), (79, 86), (80, 87), (81, 88), (82, 89),
            (83, 90), (84, 91), (85, 92), (86, 93), (87, 94), (88, 95),
            (89, 96), (90, 97),
            (91, 98), (92, 99), (93, 100), (94, 101), (95, 102), (96, 103),
            (97, 104),
            (98, 105), (99, 106), (100, 107), (101, 108), (102, 109),
            (103, 110), (104, 111),
            (105, 112), (106, 113), (107, 114), (108, 115), (109, 116),
            (110, 117), (111, 118),
            (112, 119), (113, 120), (114, 121), (115, 122), (116, 123),
            (117, 124), (118, 125),
            (119, 126),
        ]
        return CouplingMap(edges)

    @staticmethod
    def ibm_condor() -> 'CouplingMap':
        """IBM Condor processor topology (1121 qubits).

        A large-scale heavy-hex topology. For testing purposes, this returns
        a 12-ring heavy-hex structure scaled to approximately 1121 qubits.
        Due to the size, a simplified heavy-hex is generated.
        """
        return CouplingMap.heavy_hex(24)

    @staticmethod
    def ionq_alltoall(n: int) -> 'CouplingMap':
        """IonQ all-to-all connectivity topology.

        IonQ trapped-ion quantum computers support full all-to-all
        connectivity — every qubit can interact directly with every other.
        """
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        return CouplingMap(edges)

    @staticmethod
    def rigetti_ring(n: int) -> 'CouplingMap':
        """Rigetti ring topology.

        Rigetti superconducting processors typically use a ring/linear
        topology with an additional connection closing the ring.
        """
        edges = [(i, i + 1) for i in range(n - 1)]
        if n > 2:
            edges.append((0, n - 1))
        return CouplingMap(edges)

    @staticmethod
    def google_sycamore(rows: int = 9, cols: int = 6) -> 'CouplingMap':
        """Google Sycamore grid topology.

        Google's Sycamore and related processors use a 2D grid with
        specific nearest-neighbour connectivity.
        """
        return CouplingMap.grid(rows, cols)


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

    def __init__(self, coupling_map: CouplingMap, lookahead: int = DEFAULT_LOOKAHEAD):
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

    def __init__(self, coupling_map: CouplingMap, lookahead_weight: float = DEFAULT_LOOKAHEAD_WEIGHT):
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

            # Audit §1.2: prefer the Rust kernel `eigen_native.fast_sabre_swap_score`
            # for the inner swap-scoring loop when available (it implements the
            # same arithmetic but with deterministic lexicographic tie-breaking,
            # which keeps routing output byte-stable w.r.t. input edge order).
            try:
                import eigen_native as native
                rust_kernel = native.fast_sabre_swap_score
            except (ImportError, AttributeError):
                rust_kernel = None

            if rust_kernel is not None:
                # Convert front/extended layers to (l0, l1) tuples of
                # logical indices that the Rust kernel expects. Logical
                # qubit *keys* can be either ints (`0`, `1`, …) or strings
                # (`'q0'`, `'q1'`, …); we build a stable integer-to-name
                # mapping so the Rust kernel only ever sees integer
                # indices into `mapping_list`.
                logical_names = sorted(mapping.keys())
                name_to_idx = {name: i for i, name in enumerate(logical_names)}
                mapping_list = [mapping[name] for name in logical_names]
                front_pairs = [
                    (name_to_idx[op['targets'][0]], name_to_idx[op['targets'][1]])
                    for op in front_layer
                    if len(op['targets']) == 2
                    and op['targets'][0] in name_to_idx
                    and op['targets'][1] in name_to_idx
                ]
                extended_pairs = [
                    (name_to_idx[op['targets'][0]], name_to_idx[op['targets'][1]])
                    for op in extended_layer
                    if len(op['targets']) == 2
                    and op['targets'][0] in name_to_idx
                    and op['targets'][1] in name_to_idx
                ]
                # Distance matrix: the Rust kernel needs the full
                # `num_physical × num_physical` distance matrix. The
                # `CouplingMap.distance` method supports caching — call
                # it once per pair to populate the cache, then expose the
                # matrix as a list of lists. `n_phys` is derived from all
                # physical indices appearing either in the coupling-map
                # edges or in the current mapping values, so it covers
                # every reachable qubit.
                all_phys = set()
                for (p0, p1) in self.coupling_map.edges:
                    all_phys.add(p0)
                    all_phys.add(p1)
                all_phys.update(mapping.values())
                n_phys = (max(all_phys) + 1) if all_phys else 0
                # `CouplingMap.distance` returns `float('inf')` for pairs that
                # have no path between them in the coupling graph. The Rust
                # kernel expects `usize`, so translate `inf` to a large but
                # finite sentinel (chosen to comfortably exceed the maximum
                # theoretical shortest-path length on a 10_000-node
                # coupling graph) — this preserves ranking while keeping the
                # sum finite for f64 comparison.
                INF_SENTINEL = 99999
                def _dist_to_int(d):
                    if d == float('inf'):
                        return INF_SENTINEL
                    return int(d)
                dist_matrix = [
                    [_dist_to_int(self.coupling_map.distance(i, j)) for j in range(n_phys)]
                    for i in range(n_phys)
                ]
                # `CouplingMap.edges` is already stored as a set of
                # `(min, max)` pairs; preserve that ordering for the
                # Rust kernel (which iterates them in input order).
                phys_edges = list(self.coupling_map.edges)
                result_pair = rust_kernel(
                    phys_edges,
                    dist_matrix,
                    mapping_list,
                    front_pairs,
                    extended_pairs,
                    self.lookahead_weight,
                )
                if result_pair is not None:
                    edge_q1, edge_q2, score = result_pair
                    best_swap = (edge_q1, edge_q2)
                    best_score = score
            else:
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

                    # Deterministic tie-break: strictly smaller score wins;
                    # equal scores go to the lexicographically smallest
                    # (edge_q1, edge_q2), matching `fast_sabre_swap_score`.
                    if best_swap is None or score < best_score or (
                        score == best_score and (edge_q1, edge_q2) < best_swap
                    ):
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
